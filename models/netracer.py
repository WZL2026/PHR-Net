"""
NETracer: A Topology-Aware Iterative Tracing Approach for Tubular Structure Extraction
=======================================================================================
Liu et al., ICCV 2025  |  https://github.com/CSDLLab/NETracer

This module provides the complete NETracer model as a single importable file,
following the same convention as other models in the PHRNet project.

Contents
--------
  Building Blocks      – ConvReLU, res_conv_block_2d, up/down_conv_2d
  PE_Net_2D            – NENet (Node-Edge Estimation Network) with 3 output heads
  NETracer_Net         – Wrapper with dual interface (segmentation / full tracing)
  Loss Functions       – L_cl, L_node, L_edge, L_conn  (Section 3.2)
  GDS Strategy         – Geodesic Distance-based Search    (Section 3.3)

Usage (segmentation – comparison framework)
-------------------------------------------
    from models.netracer import NETracer_Net
    model = NETracer_Net(in_channels=3, out_channels=1)
    logits = model(image_tensor)          # [B, 1, H, W]  centreline logits

Usage (full NETracer – iterative tracing)
------------------------------------------
    from models.netracer import NETracer_Net, compute_netracer_loss
    model = NETracer_Net(in_channels=3, out_channels=1)
    cl, pt, edge = model.forward_tracing(image, walked_path)   # 3 × [B,1,H,W]
    loss, d = compute_netracer_loss(cl, pt, edge, ...)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import generate_binary_structure
from typing import List, Tuple, Optional


# ======================================================================
#  Building Blocks  (V-Net style encoder-decoder components)
# ======================================================================

class ConvReLU(nn.Module):
    """Conv3x3 → BN → ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class up_conv_2d(nn.Module):
    """Transposed conv for up-sampling (stride-2)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2,
                               padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class down_conv_2d(nn.Module):
    """Strided conv for down-sampling (stride-2)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class res_conv_block_2d(nn.Module):
    """Residual block:  Conv3×3→BN→ReLU→Conv3×3→BN  +  skip connection."""
    def __init__(self, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv1(x) + x)


# ======================================================================
#  PE_Net_2D  —  Node-Edge Estimation Network  (NENet)
# ======================================================================

class PE_Net_2D(nn.Module):
    """
    V-Net backbone with three output heads:

      centreline_block  →  vessel centreline probability map
      edge_block        →  future-edge probability map (connectivity cue)
      point_block       →  future-node probability map (next tracing position)

    Input:
      input_image  [B, 3, H, W]    RGB / gray image tile
      walked_path  [B, 1, H, W]    mask of previously traced positions

    Output:
      centreline_out  [B, 1, H, W]   after Sigmoid
      point_out       [B, 1, H, W]   after Sigmoid
      edge_out        [B, 1, H, W]   after Sigmoid
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # ---- encoder ----
        self.conv_input = ConvReLU(in_ch, filters[0])
        self.Conv1 = res_conv_block_2d(filters[0])
        self.Down1 = down_conv_2d(filters[0], filters[1])
        self.Conv2 = res_conv_block_2d(filters[1])
        self.Down2 = down_conv_2d(filters[1], filters[2])
        self.Conv3 = res_conv_block_2d(filters[2])
        self.Down3 = down_conv_2d(filters[2], filters[3])
        self.Conv4 = res_conv_block_2d(filters[3])
        self.Down4 = down_conv_2d(filters[3], filters[4])
        self.Conv5_1 = res_conv_block_2d(filters[4])

        # ---- decoder ----
        self.Up5 = up_conv_2d(filters[4], filters[3])
        self.Up_conv5 = res_conv_block_2d(filters[3])
        self.Up4 = up_conv_2d(filters[3], filters[2])
        self.Up_conv4 = res_conv_block_2d(filters[2])
        self.Up3 = up_conv_2d(filters[2], filters[1])
        self.Up_conv3 = res_conv_block_2d(filters[1])
        self.Up2 = up_conv_2d(filters[1], filters[0])
        self.Up_conv2 = res_conv_block_2d(filters[0])

        # ---- centreline head ----
        self.centerline_block = res_conv_block_2d(filters[0])
        self.Conv_centerline = nn.Sequential(
            res_conv_block_2d(filters[0]),
            res_conv_block_2d(filters[0]),
        )
        self.Conv_centerline_out = nn.Conv2d(filters[0], out_ch,
                                              kernel_size=1, stride=1, padding=0)

        # ---- walked-path encoder ----
        self.walked_path_input = ConvReLU(1, filters[0])

        # ---- point-edge fusion ----
        self.point_edge_block = ConvReLU(filters[0] + filters[0], filters[0])

        # ---- edge head ----
        self.Conv_edge = nn.Sequential(
            res_conv_block_2d(filters[0]),
            res_conv_block_2d(filters[0]),
        )
        self.Conv_edge_out = nn.Sequential(
            res_conv_block_2d(filters[0]),
            nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0),
        )

        # ---- point (node) head ----
        self.Conv_point = nn.Sequential(
            res_conv_block_2d(filters[0]),
            res_conv_block_2d(filters[0]),
            nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0),
        )

        self.active_Sigmoid = nn.Sigmoid()

    def forward(self, input_image: torch.Tensor,
                walked_path: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Full 3-output forward pass."""

        # ---- encoder ----
        x = self.conv_input(input_image)
        e1 = self.Conv1(x)
        e1_d = self.Down1(e1)
        e2 = self.Conv2(e1_d)
        e2_d = self.Down2(e2)
        e3 = self.Conv3(e2_d)
        e3_d = self.Down3(e3)
        e4 = self.Conv4(e3_d)
        e4_d = self.Down4(e4)
        e5_2 = self.Conv5_1(e4_d)

        # ---- decoder ----
        d5 = self.Up5(e5_2)
        d5 = torch.add(e4, d5)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.add(e3, d4)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.add(e2, d3)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.add(e1, d2)
        stage_fuse = self.Up_conv2(d2)

        # ---- centreline ----
        centreline_fts = self.centerline_block(stage_fuse)
        d0_cl = self.Conv_centerline(centreline_fts)
        centreline_out = self.active_Sigmoid(self.Conv_centerline_out(d0_cl))

        # ---- walked-path features ----
        walked_path_f = self.walked_path_input(walked_path)
        stage_fuse_cat = torch.cat([centreline_fts, walked_path_f], dim=1)

        # ---- point-edge shared encoder ----
        point_edge_fts = self.point_edge_block(stage_fuse_cat)

        # ---- edge ----
        edge_fts = self.Conv_edge(point_edge_fts)
        edge_out = self.active_Sigmoid(self.Conv_edge_out(edge_fts))

        # ---- point (node) ----
        point_fts_input = torch.add(walked_path_f, edge_fts)
        point_fts = self.Conv_point(point_fts_input)
        point_out = self.active_Sigmoid(point_fts)

        return centreline_out, point_out, edge_out


# ======================================================================
#  NETracer_Net  —  Dual-interface wrapper
# ======================================================================

class NETracer_Net(PE_Net_2D):
    """
    NETracer model with two interfaces:

      forward(x)            Segmentation mode  — image → centreline logits.
                             Used by the unified comparison / test pipeline.

      forward_tracing(x, w) Full NETracer mode — image + walked_path →
                             (centreline, point, edge) all after Sigmoid.
                             Used during NETracer-specific training.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__(in_ch=in_channels, out_ch=out_channels)

    # ------------------------------------------------------------------
    #  Segmentation interface  (compatible with comparison framework)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  [B, C, H, W]  input image.

        Returns:
            [B, 1, H, W]  centreline logits (BEFORE Sigmoid).
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        B, _, H, W = x.shape
        walked_path = torch.zeros(B, 1, H, W, device=x.device)

        # ---- encoder (same as PE_Net_2D) ----
        x1 = self.conv_input(x)
        e1 = self.Conv1(x1);          e1_d = self.Down1(e1)
        e2 = self.Conv2(e1_d);        e2_d = self.Down2(e2)
        e3 = self.Conv3(e2_d);        e3_d = self.Down3(e3)
        e4 = self.Conv4(e3_d);        e4_d = self.Down4(e4)
        e5_2 = self.Conv5_1(e4_d)

        # ---- decoder ----
        d5 = self.Up5(e5_2);          d5 = torch.add(e4, d5);   d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5);            d4 = torch.add(e3, d4);   d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4);            d3 = torch.add(e2, d3);   d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3);            d2 = torch.add(e1, d2)
        stage_fuse = self.Up_conv2(d2)

        # ---- centreline head only ----
        cl_fts = self.centerline_block(stage_fuse)
        d0 = self.Conv_centerline(cl_fts)
        return self.Conv_centerline_out(d0)  # raw logits

    # ------------------------------------------------------------------
    #  Full NETracer interface
    # ------------------------------------------------------------------

    def forward_tracing(self, x: torch.Tensor,
                        walked_path: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x:           [B, 3, H, W]  input image patch.
            walked_path: [B, 1, H, W]  walked-path mask.

        Returns:
            (centreline, point, edge)  — each [B, 1, H, W] after Sigmoid.
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if walked_path.size(1) != 1:
            walked_path = walked_path[:, :1, :, :]

        return super().forward(x, walked_path)


# ======================================================================
#  Loss Functions  (Section 3.2 of the NETracer paper)
# ======================================================================

def weighted_bce_loss(pred: torch.Tensor,
                      target: torch.Tensor,
                      w: float = 0.999) -> torch.Tensor:
    """
    Weighted BCE with foreground weight `w` (L_cl, L_node, L_e).

        weight[foreground] = w
        weight[background] = 1 − w

    Default w=0.999 (999:1 fg:bg ratio).  This high value compensates for the
    extreme sparsity of skeleton pixels in 256×256 patches (~0.2-0.5 %).
    The original paper used w=0.9 on 64×64 patches where skeleton density
    is roughly 10× higher, so w=0.9 was sufficient for that setting.
    """
    weight = torch.zeros_like(target)
    weight.fill_(1.0 - w)
    weight[target > 0] = w
    return nn.BCELoss(weight=weight, reduction='mean')(pred, target.float())


def local_connectivity_loss(edge_pred: torch.Tensor,
                            local_branch_label: torch.Tensor) -> torch.Tensor:
    """
    L_conn = |ŷ_e ⊙ (1 − y_b)|₁   (Eq. 9)

    Penalises edge predictions that fall OUTSIDE the correct local branch.
    """
    penalty = edge_pred * (1.0 - local_branch_label)
    return penalty.abs().mean()


def compute_netracer_loss(
    centreline_pred: torch.Tensor,
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    centreline_gt: torch.Tensor,
    node_gt: torch.Tensor,
    edge_gt: torch.Tensor,
    local_branch_gt: torch.Tensor,
    positive_mask: torch.Tensor = None,
    lambda_cl: float = 1.0,
    lambda_node: float = 1.0,
    lambda_edge: float = 1.0,
    lambda_conn: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Total NETracer loss:

        L = λ_cl·L_cl + λ_node·L_node + λ_edge·L_e + λ_conn·L_conn

    L_cl is computed on ALL samples.
    L_node, L_e, L_conn are only computed on positive (on-vessel) samples.

    Returns:
        (total_loss, {L_cl, L_node, L_e, L_conn, total})
    """
    # L_cl — all samples
    L_cl = weighted_bce_loss(centreline_pred, centreline_gt, w=0.9)

    # remaining losses — only positive samples
    if positive_mask is not None and positive_mask.any():
        pos_idx = positive_mask.nonzero(as_tuple=False).squeeze(-1)

        cn = centreline_pred[pos_idx]; cg = centreline_gt[pos_idx]
        pn = node_pred[pos_idx];       ng = node_gt[pos_idx]
        en = edge_pred[pos_idx];       eg = edge_gt[pos_idx]
        lb = local_branch_gt[pos_idx]

        L_node = weighted_bce_loss(pn, ng, w=0.9) if ng.sum() > 0 else \
            torch.tensor(0.0, device=centreline_pred.device)
        L_e = weighted_bce_loss(en, eg, w=0.9) if eg.sum() > 0 else \
            torch.tensor(0.0, device=centreline_pred.device)
        L_conn = local_connectivity_loss(en, lb) if lb.sum() > 0 else \
            torch.tensor(0.0, device=centreline_pred.device)
    else:
        L_node = torch.tensor(0.0, device=centreline_pred.device)
        L_e = torch.tensor(0.0, device=centreline_pred.device)
        L_conn = torch.tensor(0.0, device=centreline_pred.device)

    total = (lambda_cl * L_cl + lambda_node * L_node +
             lambda_edge * L_e + lambda_conn * L_conn)

    loss_dict = {
        'L_cl': L_cl.item(),
        'L_node': L_node.item() if isinstance(L_node, torch.Tensor) else L_node,
        'L_e': L_e.item() if isinstance(L_e, torch.Tensor) else L_e,
        'L_conn': L_conn.item() if isinstance(L_conn, torch.Tensor) else L_conn,
        'total': total.item(),
    }
    return total, loss_dict


def dice_score(pred: torch.Tensor, target: torch.Tensor,
               smooth: float = 1.0) -> float:
    """Dice similarity coefficient (for monitoring)."""
    p = pred.contiguous().view(-1)
    t = target.contiguous().view(-1)
    intersection = (p * t).sum()
    return (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)


# ======================================================================
#  GDS — Geodesic Distance-based Search  (Section 3.3)
# ======================================================================

def extract_candidate_nodes(
    point_map: np.ndarray,
    r_mask: int = 3,
    threshold_ratio: float = 0.5,
    border: int = 4,
) -> List[Tuple[int, int, int]]:
    """
    Extract candidate future nodes from the point probability map.

    Iteratively finds local maxima, suppressing a circular neighbourhood
    after each, stopping when max < original_max × threshold_ratio.
    """
    H, W = point_map.shape
    pm = point_map.copy().astype(np.float32)

    if H > 2 * border and W > 2 * border:
        pm[:border, :] = 0; pm[-border:, :] = 0
        pm[:, :border] = 0; pm[:, -border:] = 0

    orig_max = pm.max()
    if orig_max == 0:
        return [(0, H // 2, W // 2)]

    threshold = orig_max * threshold_ratio
    candidates = []

    while pm.max() > threshold and len(candidates) < 20:
        idx = np.argmax(pm)
        mx, my = np.unravel_index(idx, pm.shape)
        candidates.append((0, int(mx), int(my)))

        x0 = max(0, mx - r_mask); x1 = min(H, mx + r_mask + 1)
        y0 = max(0, my - r_mask); y1 = min(W, my + r_mask + 1)
        pm[x0:x1, y0:y1] = 0

    return candidates if candidates else [(0, H // 2, W // 2)]


def gds_search_2d(
    edge_map: np.ndarray,
    seed_pos: Tuple[int, int, int],
    candidate_nodes: List[Tuple[int, int, int]],
    gamma: float = 10.0,
    max_iter: int = 10000,
) -> Tuple[Tuple[int, int, int], np.ndarray]:
    """
    Dijkstra-like geodesic distance search on the edge probability map.

    Traversal cost:

        e(x,y) = (g(x) + g(y)) / 2
        g(p)   = exp(γ · (ŷ_max − ŷ(p)) / range) − 1

    Returns the first candidate node reached by the shortest path.
    """
    H, W = edge_map.shape
    em = edge_map.astype(np.float32)
    INF = np.finfo(np.float32).max

    score = np.full((H, W), INF, dtype=np.float32)
    score[seed_pos[1], seed_pos[2]] = 0.0

    visited = np.full((H + 2, W + 2), INF, dtype=np.float32)
    visited[1:-1, 1:-1] = 0.0
    visited[seed_pos[1] + 1, seed_pos[2] + 1] = INF

    struct = generate_binary_structure(2, 1).astype(np.float32)

    em_max, em_min = em.max(), em.min()
    em_range = em_max - em_min if em_max > em_min else 1.0

    cur = seed_pos

    for _ in range(max_iter):
        cx, cy = cur[1], cur[2]

        patch = visited[cx:cx + 3, cy:cy + 3] * struct
        uy, ux = np.where(patch == 0.0)

        for i in range(len(ux)):
            nx, ny = cx + ux[i] - 1, cy + uy[i] - 1

            gn = np.exp(gamma * (em_max - em[nx, ny]) / em_range) - 1.0
            gc = np.exp(gamma * (em_max - em[cx, cy]) / em_range) - 1.0
            ns = score[cx, cy] + (gn + gc) / 2.0

            if ns < score[nx, ny]:
                score[nx, ny] = ns
            visited[nx + 1, ny + 1] = INF

        active = (visited[1:-1, 1:-1] == INF) & np.isfinite(score)
        if not active.any():
            break

        masked = np.where(active, score, INF)
        nx_new, ny_new = np.unravel_index(np.argmin(masked), score.shape)

        visited[nx_new + 1, ny_new + 1] = INF
        cur = (0, nx_new, ny_new)

        for cand in candidate_nodes:
            if cand[1] == nx_new and cand[2] == ny_new:
                return cand, score

        if score[nx_new, ny_new] >= 255.0:
            break

    best, best_s = candidate_nodes[0], INF
    for cand in candidate_nodes:
        s = score[cand[1], cand[2]]
        if s < best_s:
            best_s, best = s, cand
    return best, score


def select_future_node(
    point_map: np.ndarray,
    edge_map: np.ndarray,
    seed_pos: Tuple[int, int, int],
    r_mask: int = 3,
) -> Tuple[Optional[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Full GDS pipeline: extract candidates → geodesic search → best node.
    """
    candidates = extract_candidate_nodes(point_map, r_mask=r_mask)
    if len(candidates) == 1:
        return candidates[0], candidates
    best, _ = gds_search_2d(edge_map, seed_pos, candidates)
    return best, candidates
