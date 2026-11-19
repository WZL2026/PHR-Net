"""
NETracer Training Dataset  (optimised)
======================================
Wraps a standard vessel segmentation dataset and generates NETracer
training labels.  All skeleton-walk information is pre-computed during
__init__ so that __getitem__ only does fast array operations.

Labels produced per sample
--------------------------
  walked_path   [1,ps,ps]  — history of previous positions along the branch
  centreline    [1,ps,ps]  — skeleton of all vessels in the patch
  future_node   [1,ps,ps]  — Gaussian blob at the next tracing position
  future_edge   [1,ps,ps]  — line from centre → next node
  local_branch  [1,ps,ps]  — only the current branch  (for L_conn)
  is_positive   bool       — always True (negative samples disabled by default)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.morphology import skeletonize
from skimage.draw import line
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
#  Branch extraction  (called once per image during init)
# ---------------------------------------------------------------------------

def _extract_branches(skeleton: np.ndarray,
                      min_length: int = 10) -> List[List[Tuple[int, int]]]:
    """Extract branches (sequences of connected skeleton pixels) from a 2D skeleton."""
    H, W = skeleton.shape
    pts = np.argwhere(skeleton > 0)
    if len(pts) < min_length:
        return []

    sp = np.pad(skeleton, 1, mode='constant')
    nbrs: Dict[Tuple[int, int], int] = {}
    for x, y in pts:
        nbrs[(x, y)] = int(sp[x:x + 3, y:y + 3].sum()) - 1

    endpoints = {p for p, c in nbrs.items() if c == 1}
    junctions = {p for p, c in nbrs.items() if c > 2}
    branches: List[List[Tuple[int, int]]] = []
    visited: set = set()

    def _follow(cur: Tuple[int, int]) -> List[Tuple[int, int]]:
        seg = [cur]
        visited.add(cur)
        while cur not in junctions:
            x, y = cur
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if skeleton[nx, ny] and (nx, ny) not in visited:
                            seg.append((nx, ny))
                            visited.add((nx, ny))
                            cur = (nx, ny)
                            found = True
                            break
                if found:
                    break
            if not found:
                break
        return seg

    for ep in sorted(endpoints):
        if ep not in visited:
            br = _follow(ep)
            if len(br) >= min_length:
                branches.append(br)

    for jn in sorted(junctions):
        if jn in visited:
            continue
        x, y = jn
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if skeleton[nx, ny] and (nx, ny) not in visited:
                        br = _follow(jn)
                        if len(br) >= min_length:
                            branches.append(br)
                        break
            else:
                continue
            break

    return branches


# ---------------------------------------------------------------------------
#  Fast rendering helpers
# ---------------------------------------------------------------------------

def _draw_line_fast(out: np.ndarray, x0: int, y0: int,
                    x1: int, y1: int, val: float = 1.0) -> None:
    """Draw a line on a 2D array (mutates in-place)."""
    rr, cc = line(x0, y0, x1, y1)
    valid = (rr >= 0) & (rr < out.shape[0]) & (cc >= 0) & (cc < out.shape[1])
    out[rr[valid], cc[valid]] = val


def _gaussian_fast(shape: Tuple[int, int], cx: float, cy: float,
                   sigma: float = 3.0) -> np.ndarray:
    """2D Gaussian blob, returns float32 [H, W]."""
    xs = np.arange(shape[0], dtype=np.float32)
    ys = np.arange(shape[1], dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
#  Main dataset
# ---------------------------------------------------------------------------

class NETracerDatasetWrapper(Dataset):
    """
    Wraps a standard vessel dataset and produces NETracer training samples.

    All skeleton-walk data is pre-computed at init time, so __getitem__
    only does array slicing and fast line/Gaussian rendering.

    Args:
        base_dataset:  DRIVEDataset / CHASEDB1Dataset / STAREDataset.
        patch_size:    Crop size  (default 256;  NETracer paper uses 64).
        neg_ratio:     Fraction of negative samples  (0 = disabled).
        walk_len:      Number of history steps in walked-path mask.
        sample_step:   Sample every N pixels along each branch.
    """

    def __init__(self, base_dataset: Dataset, patch_size: int = 256,
                 neg_ratio: float = 0.0, walk_len: int = 10,
                 sample_step: int = 5, centreline_target: str = "skeleton"):
        """centreline_target 控制"中心线头"的监督目标：

          - "skeleton"（默认，等同官方定义）：目标 = 血管骨架（1 px）。
            与官方 `train_2D.py` 一致（那里用 `y_skl` 监督 centreline 头），
            但这样训出来的模型输出是细骨架，**不能**直接拿去和血管掩膜算 Dice；
          - "vessel"：目标 = 血管掩膜。这是把 NETracer 放进"稠密血管分割统一协议"里
            做公平对比时必须选的模式（否则指标与其它模型不可比）。

        注意：node / edge / local-branch 这三个辅助目标始终来自骨架，不受此参数影响。
        """
        if centreline_target not in ("skeleton", "vessel"):
            raise ValueError("centreline_target 只能是 'skeleton' 或 'vessel'")
        self.centreline_target = centreline_target

        self.base = base_dataset
        self.patch_size = patch_size
        self.neg_ratio = neg_ratio
        self.walk_len = walk_len
        self.sample_step = sample_step

        # ---- pre-compute ----
        self.skeletons: List[np.ndarray] = []
        self.vessel_masks: List[np.ndarray] = []
        self.branches: List[List[List[Tuple[int, int]]]] = []

        # Each positive sample: (img_idx, x, y, br_idx, pos_in_br)
        self.pos_samples: List[Tuple[int, int, int, int, int]] = []

        self._precompute()

        self.n_pos = len(self.pos_samples)
        self.n_neg = (int(self.n_pos * neg_ratio / (1.0 - neg_ratio))
                      if neg_ratio > 0 and neg_ratio < 1.0 else 0)

    # ------------------------------------------------------------------
    def _precompute(self) -> None:
        for img_idx in range(len(self.base)):
            _, mask, _ = self.base[img_idx]
            mask_np = mask.squeeze().numpy()
            mask_bin = (mask_np > 0.5).astype(np.uint8)

            skel = skeletonize(mask_bin).astype(np.uint8)
            self.skeletons.append(skel)
            self.vessel_masks.append(mask_bin)

            brs = _extract_branches(skel, min_length=self.walk_len + 2)
            self.branches.append(brs)

            for bi, br in enumerate(brs):
                # sample every sample_step pixels
                for pos in range(0, len(br), self.sample_step):
                    self.pos_samples.append(
                        (img_idx, br[pos][0], br[pos][1], bi, pos))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_pos + self.n_neg

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if idx < self.n_pos:
            return self._make_positive(idx)
        return self._make_negative()

    # ------------------------------------------------------------------
    #  Positive sample
    # ------------------------------------------------------------------

    def _make_positive(self, idx: int) -> Tuple[torch.Tensor, ...]:
        img_idx, cx, cy, br_idx, pos_in_br = self.pos_samples[idx % self.n_pos]
        image, _, _ = self.base[img_idx]          # image is already a tensor
        skeleton = self.skeletons[img_idx]
        branch = self.branches[img_idx][br_idx]

        ps = self.patch_size
        hs = ps // 2

        # ---- path correction: small random offset ----
        noise_x = np.random.randint(-3, 4)
        noise_y = np.random.randint(-3, 4)
        cx_n = max(0, min(skeleton.shape[0] - 1, cx + noise_x))
        cy_n = max(0, min(skeleton.shape[1] - 1, cy + noise_y))

        # ---- crop window ----
        x0, x1 = cx_n - hs, cx_n + hs
        y0, y1 = cy_n - hs, cy_n + hs

        # ---- image patch ----
        img_np = image.numpy()
        img_patch = self._crop_pad(img_np, x0, x1, y0, y1)

        # ---- centreline patch（目标取决于 centreline_target）----
        if self.centreline_target == "vessel":
            cl_src = self.vessel_masks[img_idx]
        else:
            cl_src = skeleton
        cl_patch = self._crop_pad(cl_src[np.newaxis, ...], x0, x1, y0, y1)

        # ---- walked path: previous positions (pre-computed from branch) ----
        prev_start = max(0, pos_in_br - self.walk_len)
        prev_positions = branch[prev_start:pos_in_br]  # already computed!

        wp = np.zeros((1, ps, ps), dtype=np.float32)
        for px, py in prev_positions:
            lx, ly = px - x0, py - y0
            if 0 <= lx < ps and 0 <= ly < ps:
                wp[0, lx, ly] = 1.0
        for i in range(len(prev_positions) - 1):
            _draw_line_fast(wp[0],
                            prev_positions[i][0] - x0,
                            prev_positions[i][1] - y0,
                            prev_positions[i + 1][0] - x0,
                            prev_positions[i + 1][1] - y0)

        # ---- future node: next position along branch ----
        next_pos_in_br = min(pos_in_br + 1, len(branch) - 1)
        next_node = branch[next_pos_in_br]
        next_dx = next_node[0] - cx_n
        next_dy = next_node[1] - cy_n

        node_patch = _gaussian_fast((ps, ps), float(hs + next_dx),
                                     float(hs + next_dy), sigma=3.0)
        node_patch = node_patch[np.newaxis, ...]

        # ---- future edge: line from centre → next node ----
        edge_patch = np.zeros((1, ps, ps), dtype=np.float32)
        _draw_line_fast(edge_patch[0], hs, hs, hs + next_dx, hs + next_dy)

        # ---- local branch: only THIS branch (for L_conn) ----
        # Use branch segment visible in the patch
        lb_patch = np.zeros((1, ps, ps), dtype=np.float32)
        # render the whole branch segment
        br_start = max(0, pos_in_br - 30)
        br_end = min(len(branch), pos_in_br + 30)
        for i in range(br_start, br_end):
            px, py = branch[i]
            lx, ly = px - x0, py - y0
            if 0 <= lx < ps and 0 <= ly < ps:
                lb_patch[0, lx, ly] = 1.0
        for i in range(br_start, br_end - 1):
            _draw_line_fast(lb_patch[0],
                            branch[i][0] - x0, branch[i][1] - y0,
                            branch[i + 1][0] - x0, branch[i + 1][1] - y0)

        return (
            torch.from_numpy(img_patch).float(),
            torch.from_numpy(wp).float(),
            torch.from_numpy(cl_patch).float(),
            torch.from_numpy(node_patch).float(),
            torch.from_numpy(edge_patch).float(),
            torch.from_numpy(lb_patch).float(),
            torch.tensor(True),
        )

    # ------------------------------------------------------------------
    #  Negative sample
    # ------------------------------------------------------------------

    def _make_negative(self) -> Tuple[torch.Tensor, ...]:
        img_idx = np.random.randint(len(self.base))
        image, _, _ = self.base[img_idx]
        H, W = image.shape[1], image.shape[2]
        ps = self.patch_size
        hs = ps // 2

        cx = np.random.randint(hs, max(hs + 1, H - hs))
        cy = np.random.randint(hs, max(hs + 1, W - hs))
        x0, x1 = cx - hs, cx + hs
        y0, y1 = cy - hs, cy + hs

        img_patch = self._crop_pad(image.numpy(), x0, x1, y0, y1)
        z = np.zeros((1, ps, ps), dtype=np.float32)

        return (
            torch.from_numpy(img_patch).float(),
            torch.from_numpy(z).float(),
            torch.from_numpy(z).float(),
            torch.from_numpy(z).float(),
            torch.from_numpy(z).float(),
            torch.from_numpy(z).float(),
            torch.tensor(False),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _crop_pad(arr: np.ndarray, x0: int, x1: int, y0: int, y1: int
                  ) -> np.ndarray:
        C, H, W = arr.shape
        pw, ph = x1 - x0, y1 - y0
        out = np.zeros((C, pw, ph), dtype=arr.dtype)

        sx0, sx1 = max(0, x0), min(H, x1)
        sy0, sy1 = max(0, y0), min(W, y1)
        dx0, dx1 = sx0 - x0, sx0 - x0 + (sx1 - sx0)
        dy0, dy1 = sy0 - y0, sy0 - y0 + (sy1 - sy0)

        out[:, dx0:dx1, dy0:dy1] = arr[:, sx0:sx1, sy0:sy1]
        return out
