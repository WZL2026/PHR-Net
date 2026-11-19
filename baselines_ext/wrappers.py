# -*- coding: utf-8 -*-
"""第三方基线模型的统一适配层。

目标：把各论文官方仓库的网络接成统一接口——输入 RGB 3 通道、输出**单通道 logits**且与输入同分辨率，
从而可以直接挂到本项目的统一训练/测试框架（`train_comparison.py` / `test_comparison.py`）。

工作流：
    1) 把第三方仓库放到项目根目录的 `third_party/`（旧目录名 `对比模型源码/` 也兼容，
    2) `python baselines_ext/apply_ext_patch.py` 把本层的模型名注册进 get_model()；
    3) `python baselines_ext/smoke_test.py` 逐个构建 + 参数量 + 前向形状自检（CPU）。

也可单独自检本层所有构建函数：
    python baselines_ext/wrappers.py
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- 仓库路径
def _default_root() -> Path:
    """第三方仓库根目录：优先 third_party/，兼容旧目录名 对比模型源码/。"""
    base = Path(__file__).resolve().parents[1]
    for name in ("third_party", "对比模型源码"):
        cand = base / name
        if cand.is_dir():
            return cand
    return base / "third_party"


DEFAULT_ROOT = _default_root()
REPO_ROOT = Path(os.environ.get("EXT_BASELINES_ROOT", DEFAULT_ROOT))
DIRS = {
    "cenet": "CE-Net-master/CE-Net-master",
    "saunet": "SA-UNet-master-new/SA-UNet-master",     # 官方只有 Keras 版，本适配层不支持
    "iternet": "IterNet-master/IterNet-master",         # 官方只有 Keras 版，本适配层不支持
    "litehrnet": "Lite-HRNet-hrnet/Lite-HRNet-hrnet",   # 依赖 mmcv+mmpose，且只是姿态估计主干
    "m2unet": "M2U-Net-master/M2U-Net-master",          # 视网膜血管版（含官方预训练权重）
    "frunet": "FR-UNet-master/FR-UNet-master",
    "serpmamba": "Serp-Mamba-main/Serp-Mamba-main",
    "swin_unet": "Swin-Unet-main/Swin-Unet-main",
    "transunet": "TransUNet-main/TransUNet-main",
    "umamba": "U-Mamba-main/U-Mamba-main",
    "swin_umamba": "Swin-UMamba-main/Swin-UMamba-main",
    "vmunet": "VM-UNet-main/VM-UNet-main",
}


def repo(kind: str) -> Path:
    p = REPO_ROOT / DIRS[kind]
    if not p.is_dir():
        raise FileNotFoundError(f"找不到第三方仓库目录：{p}")
    return p


def _syspath(p: Path):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------- 通用容器
class ToLogits(nn.Module):
    """把各种输出（概率/多通道/多尺度 list）统一成单通道 logits。

    tag 非空时，**第一次前向**会打印一行 `[adapter] <tag> output shape=...` 诊断，
    便于在服务器日志里核对输出形状/分辨率（每个模型只打一次，不刷屏）。

    尺寸处理（三者优先级）：
      1. **切块（推荐给固定尺寸的官方实现）**：`tile` 非空且输入不小于 tile 时，把大图切成
         带重叠的 tile×tile 小块，各自在**原分辨率**上前向，重叠区按次数平均拼接。
         相比"整图缩放到 net_size"不丢细节（TransUNet 的 ViT pos-embed 与 img_size 绑定，
         缩放到 256 会把 512 窗口的细血管抹平；切 256 块则与训练分辨率完全一致）。
      2. **缩放**：`net_size` 非空且输入尺寸不等于它时，缩放到 net_size 前向，输出再插值回原尺寸。
      3. **原样**：两者都不触发时直接前向。
    """

    def __init__(self, net: nn.Module, mode: str = "raw", cls_index: int = 1,
                 net_size: int | None = None, tag: str | None = None,
                 tile: int | None = None, tile_overlap: int = 32):
        super().__init__()
        self.net = net
        self.mode = mode          # raw | prob2logit | channel
        self.cls_index = cls_index
        self.net_size = net_size  # 非空：先把输入缩放到 net_size 再前向（用于固定尺寸的官方实现）
        self.tag = tag
        self.tile = tile          # 非空：切成 tile×tile 小块前向（不缩放，保留细节）
        self.tile_overlap = max(0, int(tile_overlap))
        self._diag_done = False

    def _post(self, y: torch.Tensor) -> torch.Tensor:
        """多尺度/3 维/通道/sigmoid 的统一后处理（不含尺寸插值）。"""
        if isinstance(y, (list, tuple)):           # 官方多尺度输出 -> 取主分支（第 0 个）
            y = y[0]
        if y.dim() == 3:                           # 个别实现漏掉 batch 维 -> [B,H,W] 补成 [B,1,H,W]
            y = y.unsqueeze(1)
        if self.mode == "channel":                 # 2 类 softmax -> 取血管通道
            y = y[:, self.cls_index:self.cls_index + 1]
        elif self.mode == "prob2logit":            # 模型内部已 sigmoid
            y = torch.logit(y.clamp(1e-6, 1 - 1e-6))
        return y

    def _forward_tiled(self, x: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
        """重叠切块前向 + 按覆盖次数平均拼接（不改变分辨率）。"""
        b, _c, h, w = x.shape
        stride = max(1, tile - overlap)
        grid = lambda n: sorted({0, *range(0, n - tile + 1, stride), n - tile})  # noqa: E731
        ys: torch.Tensor | None = None
        wsum = torch.zeros((1, 1, h, w), dtype=torch.float32, device=x.device)
        for i in grid(h):
            for j in grid(w):
                patch = x[:, :, i:i + tile, j:j + tile]
                out = self._post(self.net(patch))
                if out.shape[-2:] != (tile, tile):
                    out = F.interpolate(out, size=(tile, tile), mode="bilinear", align_corners=False)
                out = out.float()
                if ys is None:
                    ys = torch.zeros((b, out.shape[1], h, w), dtype=torch.float32, device=x.device)
                ys[:, :, i:i + tile, j:j + tile] += out
                wsum[:, :, i:i + tile, j:j + tile] += 1.0
        assert ys is not None
        return ys / wsum.clamp_min(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape[-2:]
        resized = tiled = False
        if self.net_size is not None and tuple(orig) != (self.net_size, self.net_size):
            if self.tile is not None and min(orig) >= self.tile:
                y = self._forward_tiled(x, self.tile, self.tile_overlap)
                tiled = True
            else:  # 输入比 tile 还小 -> 退回缩放（保持旧行为）
                x = F.interpolate(x, size=(self.net_size, self.net_size),
                                  mode="bilinear", align_corners=False)
                resized = True
                y = self._post(self.net(x))
        else:
            y = self._post(self.net(x))
        if not tiled:
            if y.shape[-2:] != orig:               # 保证与输入同分辨率
                y = F.interpolate(y, size=orig, mode="bilinear", align_corners=False)
        if self.tag and not self._diag_done:
            self._diag_done = True
            how = "tiled" if tiled else ("resized" if resized else "native")
            print(f"[adapter] {self.tag} output shape={tuple(y.shape)}"
                  f"  in={orig[0]}x{orig[1]} -> out={y.shape[-2]}x{y.shape[-1]}"
                  f"  (net_size={self.net_size}, mode={how})",
                  flush=True)
        return y


def _load_ext_package(pkg_name: str, pkg_dir: Path, mod_name: str):
    """以**独立包名**加载第三方 models/ 目录里的模块。

    为什么需要：FR-UNet / VM-UNet 等仓库里也有一个 `models` 包，而主项目同样有 `models`
    且已被导入，`import models.xxx` 会命中主项目那个包 -> ModuleNotFoundError。
    这里用 importlib 以唯一包名（如 frunet_models）加载，互不干扰，同时保留相对导入能力。
    """
    import importlib
    import importlib.util

    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name, pkg_dir / "__init__.py",
            submodule_search_locations=[str(pkg_dir)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{pkg_name}.{mod_name}")


# ---------------------------------------------------------------- 诊断 / 权重工具
_DIAG_PRINTED: set[str] = set()


def _diag(msg: str, once_key: str | None = None) -> None:
    """适配层统一日志：行首固定 `[adapter]`，同一条（once_key）只打印一次。"""
    key = once_key or msg
    if key in _DIAG_PRINTED:
        return
    _DIAG_PRINTED.add(key)
    print(f"[adapter] {msg}", flush=True)


def _use_pretrained() -> bool:
    """`USE_PRETRAINED` 开关（与 build_swin_umamba 里原有的三态判断同一语义）。

    - 未设置 / `1` / `true` / `yes` / `on` -> True（加载预训练，保持历史行为）；
    - `0` / `false` / `no` / `off` / 空字符串 -> False（"从零训练"对照：**不查找、不加载**
      任何预训练权重，随机初始化）。

    说明：这只管**隐式的 ImageNet/官方预训练权重**；显式传入的 `ckpt=` 路径（用户自己指定的
    检查点/resume）不在此开关范围内，行为不变。
    """
    return os.environ.get("USE_PRETRAINED", "1").strip().lower() \
        not in ("0", "false", "no", "off", "")


def _first_existing(cands) -> Path | None:
    for c in cands:
        if not c:
            continue
        p = Path(c)
        if p.is_file():
            return p
    return None


def _rglob_first(root: Path, pattern: str) -> Path | None:
    try:
        for p in sorted(root.rglob(pattern)):
            if p.is_file():
                return p
    except Exception:  # noqa: BLE001  目录不存在/无权限都不该让构建失败
        pass
    return None


def _load_pretrained(net: nn.Module, sd, tag: str, prefixes=("", "swin_unet.", "model.", "module.")):
    """按“名字 + shape 匹配”把第三方预训练权重装进 net（strict=False），并打印加载数量。

    返回 (loaded, total)。找不到任何匹配张量时不会假装成功，只打印警告。
    """
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):  # 常见包裹层
            if isinstance(sd.get(key), dict):
                sd = sd[key]
                break
    if not isinstance(sd, dict):
        raise TypeError(f"无法识别的权重格式：{type(sd)}")
    clean = {k.replace("module.", "").replace("model.", ""): v
             for k, v in sd.items() if hasattr(v, "shape")}
    model_sd = net.state_dict()
    best_n, best = 0, {}
    for pre in prefixes:
        cand = {pre + k: v for k, v in clean.items()}
        ok = {k: v for k, v in cand.items()
              if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
        if len(ok) > best_n:
            best_n, best = len(ok), ok
    if best_n:
        net.load_state_dict(best, strict=False)
        _diag(f"{tag} 预训练加载：{best_n}/{len(model_sd)} 个张量"
              f"（源权重 {len(clean)} 个，按名字+shape 匹配，strict=False）")
    else:
        _diag(f"{tag} [WARN] 预训练权重与模型没有任何名字+shape 匹配的张量，保持随机初始化"
              f"（源 {len(clean)} 个 / 模型 {len(model_sd)} 个）")
    return best_n, len(model_sd)


# ---------------------------------------------------------------- mamba / selective scan
def _mamba_ssm_selective_scan():
    """尝试取 mamba_ssm 的 CUDA selective_scan；失败返回 (None, 原因)。

    刻意放在函数内部导入：mamba_ssm 导入时会触发 triton 的编译/JIT，在登录节点（无 GPU）
    或 gcc/triton 编译失败的环境会直接抛异常。适配层需要区分“环境问题”与“代码问题”，
    所以这里把异常捕获成一段诊断文本，而不是让整个构建崩掉。
    """
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore
        return selective_scan_fn, ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:200]}"


def _selective_scan_fallback(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                             delta_softplus=False, return_last_state=False):
    """纯 PyTorch（CPU/GPU 通用）的 selective scan，接口等价 mamba_ssm.selective_scan_fn。

    形状约定（同 mamba_ssm）：
        u, delta: (b, d, l)   A: (d, n)   B, C: (b, n, l) 或 (b, g, n, l)
        D: (d,)   delta_bias: (d,)   z: (b, d, l)        -> 返回 y: (b, d, l)

    做法：序列分块 + 块内 cumsum（等价于逐步递推，但不写 Python 逐点循环 ——
    VM-UNet 在 576x608 整图上前向时 L 可达 6 万，逐点循环在服务器上会慢到不可用）：
        dA_i = Σ_{j<=i} delta_j ⊙ A          （<= 0，对数域的衰减）
        x_i  = exp(dA_i) ⊙ h0 + Σ_{k<=i} exp(dA_i - dA_k) ⊙ (delta_k u_k ⊗ B_k)
    取块内 c = dA_{T-1}（最小）做平移：exp(c - dA_k) ∈ (0,1]，exp(dA_i - c) ∈ [1, exp(span)]，
    两者相乘恰为真实衰减因子。全程 float64 + 自适应块长（保证块内跨度 span <= 600），
    因此不会溢出/上溢。
    """
    dtype_in = u.dtype
    # 工作精度固定 float32：块内指数跨度被限制在 60 以内（见下方 T 的取法），
    # exp(60)=1.1e26 仍在 float32 范围内，因此比 float64 省一半内存。
    work = torch.float32
    ud = u.to(work)
    dd = delta.to(work)
    if delta_bias is not None:
        dd = dd + delta_bias.to(work).view(1, -1, 1)
    if delta_softplus:
        dd = F.softplus(dd)

    b, d, L = ud.shape
    n = A.shape[-1]
    Af = A.to(work)
    grouped = (B.dim() == 4)
    rep = (d // B.shape[1]) if grouped else 1
    Bf = B.to(work)
    Cf = C.to(work)

    # 内存护栏：回退实现要同时保存 (b,d,L) 的输入副本与逐块中间量，
    # 在 CPU 上 L=65536(256x256) 的整图会吃掉数 GB；超限时给出可读报错，
    # 而不是让进程被 OOM 杀掉（表现为 0xC0000005 段错误，无从排查）。
    max_elems = int(os.environ.get("SELECTIVE_SCAN_MAX_ELEMS", "30000000"))
    if b * d * L > max_elems:
        raise RuntimeError(
            f"[selective_scan/fallback] 需要处理 b*d*L={b*d*L} 个元素，超过上限 {max_elems}。"
            f" 这是纯 PyTorch 回退的自我保护（避免 CPU OOM 段错误）。"
            f" 解决办法：① 使用 mamba_ssm 的 CUDA 实现（推荐，服务器已修 gcc/triton 即可）；"
            f" ② 或设 SELECTIVE_SCAN_MAX_ELEMS={b*d*L} 放宽上限（需足够内存）。"
        )

    # 每步的最大对数衰减量（用于决定块长）
    span_per_step = 0.0
    if L:
        span_per_step = float(dd.abs().max().item()) * float(Af.abs().max().item())

    # 自适应块长：两个约束共同决定
    #   (a) 数值安全：块内对数域跨度 span = |delta*A|max * T <= 60，保证 float32 下 exp() 不溢出；
    #   (b) 计算图深度/内存：单个 (b,d,T,n) 张量的元素数不超过 elem_budget，
    #       避免 L 很大时把序列切成上千块 —— 计算图过深会在反向/图析构时栈溢出
    #       （实测 VMUNet：原实现 T 固定 32，序列长时会被切成上千块 => 0xC0000005 段错误）。
    #       注：L 是 SS2D 展平后的 token 数，不等于像素数；VM-UNet 峰值 b*d*L 约 4.2e6（576x608）
    #       与 7.9e5（256x256），因此下面的护栏对它是兜底，不保护模型自身激活的显存/内存占用。
    # 说明：span 越小允许的 T 越大；两者取小，再按 L 截断。
    elem_budget = 8_000_000                      # 约 64 MB/张量（float64）
    per_step = max(1, b * d * n)
    T_mem = max(1, elem_budget // per_step)
    if L <= 0:
        T = 32
    elif span_per_step <= 0:
        T = min(L, T_mem)
    else:
        T = max(1, min(int(60.0 / span_per_step), T_mem, L))

    h = torch.zeros((b, d, n), dtype=work, device=u.device)
    ys = []
    for s in range(0, L, T):
        e = min(s + T, L)
        dt = dd[:, :, s:e]                                    # (b,d,t)
        ut = ud[:, :, s:e]                                    # (b,d,t)
        if grouped:
            Bt = Bf[:, :, :, s:e].repeat_interleave(rep, dim=1)   # (b,d,n,t)
            Ct = Cf[:, :, :, s:e].repeat_interleave(rep, dim=1)
        else:
            Bt = Bf[:, :, s:e].unsqueeze(1).expand(b, d, n, e - s)
            Ct = Cf[:, :, s:e].unsqueeze(1).expand(b, d, n, e - s)
        dA = torch.cumsum(dt.unsqueeze(-1) * Af.view(1, d, 1, n), dim=2)      # (b,d,t,n)
        c = dA[:, :, -1:, :]                                                  # (b,d,1,n)
        bterm = (dt * ut).unsqueeze(-1) * Bt.transpose(2, 3)                  # (b,d,t,n)
        S = torch.cumsum(torch.exp(c - dA) * bterm, dim=2)                    # (b,d,t,n)
        x = torch.exp(dA - c) * S + torch.exp(dA) * h.unsqueeze(2)            # (b,d,t,n)
        h = x[:, :, -1, :]
        ys.append((x * Ct.transpose(2, 3)).sum(-1))                           # (b,d,t)

    y = torch.cat(ys, dim=2) if ys else ud.new_zeros(b, d, 0)
    if D is not None:
        y = y + ud * D.to(work).view(1, -1, 1)
    if z is not None:
        y = y * F.silu(z.to(work))
    out = y.to(dtype_in)
    return (out, h.to(dtype_in)) if return_last_state else out


def _ensure_selective_scan(vmamba_mod) -> str:
    """保证 vmamba.SS2D.forward_corev0 用到的全局名 `selective_scan_fn` 一定有定义。

    官方 vmamba.py 第 12-15 行是 `try: from mamba_ssm... except: pass`，扩展不可用时导入被
    静默吞掉，于是 forward 里立刻 `NameError: name 'selective_scan_fn' is not defined`
    （实测 VMUNet/DRIVE 就是这个）。这里显式补上：优先 mamba_ssm，否则接纯 PyTorch 回退，
    绝不静默跳过 SSM 模块。
    """
    already = getattr(vmamba_mod, "selective_scan_fn", None)
    if already is not None:
        # 名字已被定义，有两种来源，必须按**身份**区分：
        #   ① 官方 vmamba.py 第 12-15 行的 import 成功 -> 真 mamba_ssm CUDA 实现；
        #   ② 本函数**上一轮**装上的纯 PyTorch 回退实现。
        # ②确实会发生：train_comparison.py 先在 364 行 build 一次数参数量，再在 381 行按
        # 5 折各 build 一次。若不区分，第 2 次起的 tag 会变成 VMUNet[mamba_ssm]，而 ToLogits
        # 每次前向都独立打印该 tag -> 日志里出现"其实在跑回退、却写着 mamba_ssm"的假象，
        # 直接破坏"看日志判断走了哪条路径"这一核查手段。故此处只修诊断标签，不动任何数值逻辑。
        if already is _selective_scan_fallback:
            backend = "pytorch-fallback"
            _diag("VMUNet selective_scan backend = 纯 PyTorch 回退实现（本进程内先前已装好，复用）",
                  "vmunet_backend")
        else:
            backend = "mamba_ssm"
            _diag("VMUNet selective_scan backend = mamba_ssm（官方 import 成功）", "vmunet_backend")
        return backend

    fn, err = _mamba_ssm_selective_scan()
    if fn is not None:
        vmamba_mod.selective_scan_fn = fn
        backend = "mamba_ssm"
        _diag("VMUNet selective_scan backend = mamba_ssm（repo 内 import 被吞，已在此显式补上）",
              "vmunet_backend")
    else:
        vmamba_mod.selective_scan_fn = _selective_scan_fallback
        backend = "pytorch-fallback"
        _diag(f"VMUNet selective_scan backend = 纯 PyTorch 回退实现"
              f"（mamba_ssm 不可用：{err}）", "vmunet_backend")
    if getattr(vmamba_mod, "selective_scan_ref", None) is None:
        vmamba_mod.selective_scan_ref = _selective_scan_fallback
    return backend


def _require_mamba_ssm(model_name: str) -> str:
    """确保 `mamba_ssm` 可用：优先真 CUDA 扩展，不可用时启用**纯 PyTorch 回退**。

    返回后端名（"mamba_ssm" 或 "pytorch-fallback"）。

    背景：集群上现成的 selective_scan_cuda 是在 glibc>=2.32 上编的，计算节点 glibc 更旧
    -> `ImportError: GLIBC_2.32 not found`，MambaVesselNet / U-Mamba / Swin-UMamba 全跑不了。
    回退实现（baselines_ext/mamba_fallback.py）数值与官方一致（选择子扫描已逐步比对），
    只是慢一些；**论文中需注明该实现差异**。只有连回退都装不上时才报错。
    """
    from baselines_ext.mamba_fallback import install_mamba_fallback
    backend = install_mamba_fallback(verbose=True)
    if backend != "mamba_ssm":
        _diag(f"{model_name}：mamba_ssm CUDA 扩展不可用 -> 已启用纯 PyTorch 回退实现"
              f"（数值等价、速度较慢；论文需注明）", f"{model_name}_mamba_fallback")
    return backend


# ---------------------------------------------------------------- 各模型构建
def build_cenet(in_channels: int = 3, num_classes: int = 1) -> nn.Module:
    """CE-Net (TMI 2019)。入口 CE_Net_(num_classes, num_channels)，输出单通道 logits。

    注意：cenet.py 内部用相对导入（from .backbones...），必须**按包**导入，
    因此把仓库的 src/ 加进 sys.path，再 from lib.models.networks.cenet import。

    预训练：CE-Net 的 ResNet34 主干预训练**不在本层**——`cenet.py:177` 写死
    `get_resnet_backbone('resnet34')(pretrain=True)`。`USE_PRETRAINED=0` 时在这里临时
    屏蔽该加载（详见下方注释），其它情况行为与历史完全一致。
    """
    r = repo("cenet")
    _syspath(r / "src")
    from lib.models.networks.cenet import CE_Net_  # type: ignore
    # [WARN] CE-Net 官方 forward 末尾是 `return torch.sigmoid(out)`（cenet.py:229）。
    # 若按 raw 处理，BCEWithLogitsLoss 会在概率上再做一次 sigmoid -> 训练几乎不收敛
    # （典型现象：train_dice 爬到 0.5 左右、val_dice 卡在 ~0.55）。
    # 这里把概率用 logit() 转回 logits，与 sigmoid 完全互逆，数学上等价于原实现。
    if _use_pretrained():
        net = CE_Net_(num_classes=num_classes, num_channels=in_channels)
    else:
        _diag("CENet USE_PRETRAINED=0 -> 跳过预训练加载（随机初始化）", "cenet_pre")
        # CE-Net 的 ImageNet 预训练由第三方代码无条件执行：
        #   cenet.py:177 get_resnet_backbone('resnet34')(pretrain=True)
        #   -> build_resnet.get_resnet_34 -> build_resnet.build_resnet.init_weights
        #   -> model_zoo.load_url(resnet34-333f7ec4.pth)（离线节点还会直接抛异常）
        # 这里只在构建 CE_Net_ 期间把 init_weights 临时替换成空实现（构建后立即还原），
        # 从而真正做到"ResNet34 主干随机初始化"：不改第三方代码、不下载权重，
        # 对 USE_PRETRAINED=1 的路径零影响。
        from lib.models.networks.backbones.resnet.build_resnet import (  # type: ignore
            build_resnet as _build_resnet_cls)
        _init_weights_bak = _build_resnet_cls.init_weights
        _build_resnet_cls.init_weights = lambda self, resnet_model_name: None
        try:
            net = CE_Net_(num_classes=num_classes, num_channels=in_channels)
        finally:
            _build_resnet_cls.init_weights = _init_weights_bak
    return ToLogits(net, mode="prob2logit")


def build_vmunet(in_channels: int = 3, num_classes: int = 1) -> nn.Module:
    """VM-UNet (2024)。注意：官方 forward 在 num_classes==1 时返回 sigmoid 结果，
    这里改用 2 类头并取血管通道，得到干净的 logits（训练用 BCE 必须如此）。

    另外（实测崩溃修复）：官方 vmamba.py 把 `selective_scan_fn` 的导入放在 try/except 里，
    服务器上 mamba_ssm 扩展不可用时名字未定义 -> forward 直接 NameError。
    这里在构建模型前显式补齐该全局名（优先 mamba_ssm，否则纯 PyTorch 回退实现）。
    """
    r = repo("vmunet")
    _syspath(r)
    from models.vmunet import vmamba as vmamba_mod  # type: ignore
    backend = _ensure_selective_scan(vmamba_mod)
    from models.vmunet.vmunet import VMUNet  # type: ignore
    net = VMUNet(input_channels=in_channels, num_classes=2, depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2], drop_path_rate=0.2)
    return ToLogits(net, mode="channel", cls_index=1, tag=f"VMUNet[{backend}]")


def _mobilenetv2_features(ckpt: str | None = None):
    """取 torchvision MobileNetV2 的 features（ImageNet 预训练），失败返回 (None, 原因)。

    官方 M2U-Net（CVPRW 2019）的编码器就是 MobileNetV2 的前 14 层；论文用 ImageNet 预训练
    权重初始化。本地有 .pth 就优先用（离线节点友好），否则用 torchvision 的官方权重。
    """
    import torchvision.models as tvm
    # 注意：REPO_ROOT 指向 third_party（EXT_BASELINES_ROOT），项目自己的 weights/ 在
    # REPO_ROOT.parent/weights —— 之前漏了这一条，导致服务器上明明有
    # weights/mobilenet_v2-b0353104.pth 却找不到、退回 torchvision 联网下载（无外网必失败）。
    path = (ckpt or os.environ.get("M2UNET_MOBILENET_CKPT")
            or _first_existing([REPO_ROOT.parent / "weights" / "mobilenet_v2-b0353104.pth",
                                REPO_ROOT.parent / "weights" / "mobilenet_v2.pth"])
            or _rglob_first(REPO_ROOT.parent / "weights", "mobilenet_v2*.pth")
            or _rglob_first(REPO_ROOT / "weights", "mobilenet_v2*.pth")
            or _rglob_first(REPO_ROOT / "weights", "*mobilenet*v2*.pth")
            # 项目内自带一份（整包拷到服务器时无需再手动放到 ~/.cache，离线节点也能用）
            or _first_existing([DEFAULT_ROOT / "_backups" / "pretrained" / "mobilenet_v2-b0353104.pth",
                                REPO_ROOT.parent / "_backups" / "pretrained" / "mobilenet_v2-b0353104.pth",
                                REPO_ROOT / "_backups" / "pretrained" / "mobilenet_v2-b0353104.pth"]))
    try:
        net = tvm.mobilenet_v2(weights=None)
    except Exception as e:  # noqa: BLE001
        return None, f"torchvision.mobilenet_v2 不可用: {e}"
    if path:
        try:
            sd = torch.load(path, map_location="cpu")
            if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
                sd = sd["state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items() if hasattr(v, "shape")}
            missing, _ = net.load_state_dict(sd, strict=False)
            return net.features, f"本地权重 {Path(path).name}（missing={len(missing)}）"
        except Exception as e:  # noqa: BLE001
            return None, f"本地权重 {path} 读取失败：{type(e).__name__}: {str(e)[:120]}"
    try:
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1
        return tvm.mobilenet_v2(weights=weights).features, "torchvision ImageNet IMAGENET1K_V1"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def _load_mobilenetv2_encoder(net: nn.Module, ckpt: str | None = None) -> tuple[int, int]:
    """把 MobileNetV2 前 14 层的 ImageNet 权重按官方层映射装进 M2U-Net 的编码器。

    M2U-Net 的 Encoder（m2unet.py:89-117）与 torchvision MobileNetV2.features 的前 14 个
    layer 逐层同构（结构、顺序、通道全一致），因此按「叶子模块（Conv2d/BatchNorm2d）出现顺序」
    一一对应拷贝即可，并逐个校验 type+shape。打印 loaded/total 供服务器日志核对。
    """
    feats, desc = _mobilenetv2_features(ckpt)
    encoder_layers = [net.conv1, net.conv2, net.conv3, net.conv4]   # = 官方 Encoder 的 14 层
    dst_leaves = [m for blk in encoder_layers for m in blk.modules()
                  if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
    total = sum(len(m.state_dict()) for m in dst_leaves)
    if feats is None:
        _diag(f"M2UNet [WARN] 未能加载 MobileNetV2 预训练编码器（{desc}）-> 编码器随机初始化，"
              f"指标会明显偏低（论文要求 ImageNet 预训练）", "m2unet_pretrained")
        return 0, total

    src_leaves = [m for m in feats.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
    loaded = 0
    mismatched = 0
    for s, d in zip(src_leaves, dst_leaves):
        if type(s) is not type(d):
            mismatched += 1
            continue
        ss, ds = s.state_dict(), d.state_dict()
        if set(ss) != set(ds) or any(tuple(ss[k].shape) != tuple(ds[k].shape) for k in ss):
            mismatched += 1
            continue
        d.load_state_dict(ss)
        loaded += len(ss)
    _diag(f"M2UNet loaded {loaded}/{total} pretrained MobileNetV2 encoder tensors "
          f"（{desc}，src_layers={len(src_leaves)} dst_layers={len(dst_leaves)}，"
          f"结构不匹配 {mismatched} 层）", "m2unet_pretrained")
    return loaded, total


def build_m2unet(in_channels: int = 3, num_classes: int = 1, ckpt: str | None = None,
                 mobilenet_ckpt: str | None = None, pretrained_encoder: bool = True) -> nn.Module:
    """M2U-Net (CVPRW 2019)——面向资源受限设备的轻量视网膜血管分割（约 0.55 M）。

    仓库自带官方权重：models/M2UNet{DRIVE,CHASEDB1,HRF}.pth（按数据集训练，非通用初始化）。
    实测问题：编码器（MobileNetV2 前 14 层）**从未加载过 ImageNet 预训练**，官方 _initialize_weights()
    会把整网随机初始化 -> 训练崩盘（val_dice 卡在 DRIVE 0.63 / CHASE 0.58 / STARE 0.40）。
    修复：构建后按官方层映射把 torchvision MobileNetV2 的 ImageNet 权重装进编码器。
    注意：原始 forward 输出 logits（benchmark 脚本才做 sigmoid），这里可**直接原样用**。
    上/下采样：编码器 stride 2 x3 + 解码器 Upsample(scale_factor=2) x3 + LastDecoderBlock 再 x2，
    对任意偶数尺寸都能回到输入分辨率（已自测 256x256 与 576x608）。

    预训练开关：`USE_PRETRAINED=0` 时跳过 MobileNetV2 ImageNet 编码器加载（随机初始化），
    用于"从零训练"对照；未设置或 `1` 时行为与历史完全一致。
    """
    r = repo("m2unet")
    _syspath(r)
    from m2unet import m2unet  # type: ignore
    net = ToLogits(m2unet(), tag="M2UNet")
    if pretrained_encoder:
        if _use_pretrained():
            _load_mobilenetv2_encoder(net.net, mobilenet_ckpt)
        else:
            _diag("M2UNet USE_PRETRAINED=0 -> 跳过预训练加载（随机初始化）", "m2unet_pre")
    if ckpt and Path(ckpt).is_file():
        miss, unexp = load_ckpt_flexible(net, ckpt)
        print(f"  [M2UNet] 载入 {Path(ckpt).name}: missing={miss} unexpected={unexp}")
    return net



def _erfnet_encoder_pretrained(explicit: str | None = None) -> Path | None:
    """ERFNet 的 ImageNet 预训练**编码器**（官方仓库 trained_models/erfnet_encoder_pretrained.pth.tar）。

    查找顺序：显式参数 > 环境变量 ERFNET_ENCODER_CKPT > weights/ > _backups/pretrained/ > 仓库内 rglob。
    """
    return _first_existing([
        explicit, os.environ.get("ERFNET_ENCODER_CKPT"),
        REPO_ROOT / "weights" / "erfnet_encoder_pretrained.pth.tar",
        REPO_ROOT.parent / "weights" / "erfnet_encoder_pretrained.pth.tar",
        REPO_ROOT.parent / "_backups" / "pretrained" / "erfnet_encoder_pretrained.pth.tar",
    ]) or _rglob_first(REPO_ROOT, "erfnet_encoder_pretrained*") \
       or _rglob_first(DEFAULT_ROOT, "erfnet_encoder_pretrained*")


def _vgg16_imagenet_state(explicit: str | None = None):
    """取 torchvision VGG-16 的 ImageNet state_dict，返回 (state_dict 或 None, 来源说明)。

    本地文件优先（计算节点没有外网，torchvision 自动下载会失败）；找不到才退回 torchvision。
    """
    p = _first_existing([
        explicit, os.environ.get("DRIU_VGG16_CKPT"),
        REPO_ROOT / "weights" / "vgg16-397923af.pth",
        REPO_ROOT.parent / "weights" / "vgg16-397923af.pth",
        REPO_ROOT.parent / "_backups" / "pretrained" / "vgg16-397923af.pth",
    ]) or _rglob_first(REPO_ROOT, "vgg16-*.pth") or _rglob_first(DEFAULT_ROOT, "vgg16-*.pth")
    if p:
        return torch.load(p, map_location="cpu"), f"本地文件 {p}"
    try:
        from torchvision.models import VGG16_Weights, vgg16
        return vgg16(weights=VGG16_Weights.IMAGENET1K_V1).state_dict(), "torchvision IMAGENET1K_V1"
    except Exception as e:  # noqa: BLE001
        return None, f"torchvision 加载失败（{type(e).__name__}: {str(e)[:110]}）"


# DRIU 里 VGG-16 features 的分段：子模块名 -> 在 torchvision features 里的起始下标
# 见 third_party/M2U-Net-master/.../driu.py:87-91
#   encoder = list(models.vgg16().children())[0][:23]
#   conv1 = encoder[0:4]; conv2 = encoder[4:9]; conv3 = encoder[9:16]; conv4 = encoder[16:23]
_DRIU_VGG16_SLICES = (("conv1", 0), ("conv2", 4), ("conv3", 9), ("conv4", 16))

def build_erfnet(in_channels: int = 3, num_classes: int = 1, ckpt: str | None = None,
                 encoder_ckpt: str | None = None) -> nn.Module:
    """ERFNet——M2U-Net 论文里做 ARM 端侧对照的模型（约 2.06 M），其权重也在同一仓库。

    它对我们的"边缘部署"叙事有用：M2U-Net 原论文给出 ERFNet 在 RK3399 上的实测延迟。
    分辨率对齐（已核查）：编码器 DownsamplerBlock 用 conv(stride=2,pad=1) + MaxPool(2,2)，
    偶数输入下两者输出一致；解码器 UpsampleBlock(ConvTranspose, output_padding=1) x2 两次 +
    输出层 ConvTranspose(2,stride2) 一次 -> 输出恰好回到输入分辨率（256/512 都对齐）。
    因此这里不改结构，只加一条首帧诊断，便于对日志核对。
    """
    r = repo("m2unet")
    _syspath(r)
    from erfnet import Net as ERFNet  # type: ignore
    model = ERFNet(num_classes=num_classes)

    # 隐式 ImageNet 预训练**编码器**：官方 ERFNet 仓库提供 erfnet_encoder_pretrained.pth.tar，
    # M2U-Net 论文里 ERFNet 的 0.8022 就是"ImageNet 预训练编码器 + 544x544 整图"。
    # 不加这段就是从零训练（我们之前 DRIVE 只有 0.7644，差的就是这个）。
    if _use_pretrained():
        p = _erfnet_encoder_pretrained(encoder_ckpt)
        if p:
            # 官方 erfnet_encoder_pretrained.pth.tar 的键形如
            #   module.features.encoder.initial_block.conv.weight
            # 比 encoder 自身的 state_dict 多了 "(module.)features.encoder." 前缀，
            # 需要**先剥离**该前缀（_load_pretrained 只能加前缀，不能剥）。
            sd = torch.load(p, map_location="cpu")
            if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
                sd = sd["state_dict"]
            sd = {k.replace("module.features.encoder.", "").replace("features.encoder.", ""): v
                  for k, v in sd.items()}
            _load_pretrained(model.encoder, sd, "ERFNet-encoder",
                             prefixes=("", "module.", "encoder."))
            _diag(f"ERFNet ImageNet 编码器来源：{p}", "erfnet_pretrained")
        else:
            _diag("ERFNet [WARN] 未找到 ImageNet 预训练编码器（erfnet_encoder_pretrained.pth.tar）"
                  "-> 从零训练，指标会偏低；可用 ERFNET_ENCODER_CKPT=<path> 指定", "erfnet_pretrained")
    else:
        _diag("ERFNet USE_PRETRAINED=0 -> 跳过预训练加载（随机初始化）", "erfnet_pre")

    net = ToLogits(model, tag="ERFNet")
    if ckpt and Path(ckpt).is_file():
        miss, unexp = load_ckpt_flexible(net, ckpt)
        print(f"  [ERFNet] 载入 {Path(ckpt).name}: missing={miss} unexpected={unexp}")
    return net


def _driu_upsample_forward(self, x_in, input_res):
    """batch-safe 版 UpsampleBlock.forward（官方实现写死 batch=1）。

    官方 driu.py:70 是 `x = x[0:1, 0:16, h_s:h_e, w_s:w_e]` —— 只对 batch=1 成立。
    统一框架训练 batch=8 时，conv1_2_16 是 (8,16,H,W)，而 3 个 side 分支被裁成 (1,16,H,W)，
    于是 concatfuse 里 `torch.cat([...], dim=1)` 报
    "Sizes of tensors must match except in dimension 1. Expected size 8 but got size 1"（实测）。
    这里按同样公式做中心裁剪，但保留 batch 维。
    """
    img_h, img_w = int(input_res[0]), int(input_res[1])
    x = self.upconv(self.conv(x_in))
    up_h, up_w = x.shape[2], x.shape[3]
    if up_h < img_h or up_w < img_w:      # 输入过小（上采样不够）-> 直接插值到目标分辨率
        return F.interpolate(x, size=(img_h, img_w), mode="bilinear", align_corners=False)
    h_s = (up_h - img_h) // 2
    w_s = (up_w - img_w) // 2
    return x[:, :, h_s:h_s + img_h, w_s:w_s + img_w]   # 与官方中心裁剪公式等价，但不丢 batch


def build_driu(in_channels: int = 3, num_classes: int = 1, ckpt: str | None = None,
               vgg16_ckpt: str | None = None) -> nn.Module:
    """DRIU (2016)——血管分割的早期深度基线（VGG16 主干），同一仓库提供官方权重。

    实测崩溃修复：官方 3 个 UpsampleBlock 的裁剪写死了 `x[0:1]`，batch>1 时与主分支 cat 失败。
    这里把 UpsampleBlock 换成 batch-safe 版本；DRIU 的 concatfuse 本来就是 1 通道输出，
    再由 ToLogits 保证 [B,1,H,W] 且与输入同分辨率。
    """
    r = repo("m2unet")
    _syspath(r)
    from driu import DRIU  # type: ignore
    net = DRIU()
    patched = 0
    for m in net.modules():
        if type(m).__name__ == "UpsampleBlock":
            m.forward = types.MethodType(_driu_upsample_forward, m)
            patched += 1
    _diag(f"DRIU: {patched} 个 UpsampleBlock 已替换为 batch-safe 裁剪"
          f"（官方写死 x[0:1]，batch=8 时 cat 报错）", "driu_patch")

    # 隐式 ImageNet VGG-16：官方 DRIU 主干是 ImageNet 预训练 VGG-16（DRIVE Dice 0.822）；
    # driu.py 用的是 torchvision models.vgg16()（默认 pretrained=False）-> 不加这段就是从零训练。
    if _use_pretrained():
        sd, src = _vgg16_imagenet_state(vgg16_ckpt)
        if sd is None:
            _diag(f"DRIU [WARN] 未能加载 ImageNet VGG-16（{src}）-> 主干随机初始化，指标会明显偏低",
                  "driu_pretrained")
        else:
            loaded = 0
            for attr, off in _DRIU_VGG16_SLICES:
                seq = getattr(net, attr, None)
                if seq is None:
                    continue
                for li, layer in enumerate(seq):
                    if not isinstance(layer, nn.Conv2d):
                        continue
                    for pname, p in layer.named_parameters(recurse=False):
                        key = f"features.{off + li}.{pname}"
                        v = sd.get(key)
                        if v is not None and tuple(v.shape) == tuple(p.shape):
                            with torch.no_grad():
                                p.copy_(v)
                            loaded += 1
            _diag(f"DRIU VGG-16 预训练加载：{loaded} 个张量（来源：{src}）", "driu_pretrained")
    else:
        _diag("DRIU USE_PRETRAINED=0 -> 跳过预训练加载（随机初始化）", "driu_pre")

    wrapped = ToLogits(net, tag="DRIU")
    if ckpt and Path(ckpt).is_file():
        miss, unexp = load_ckpt_flexible(wrapped, ckpt)
        print(f"  [DRIU] 载入 {Path(ckpt).name}: missing={miss} unexpected={unexp}")
    return wrapped


def build_frunet(in_channels: int = 3, num_classes: int = 1,
                 feature_scale: int = 2, out_ave: bool = True) -> nn.Module:
    """FR-UNet (JBHI 2022)——全程全分辨率的血管分割网络，无下采样。

    纯 PyTorch、自包含（models/fr_unet.py + utils.py），`out_ave=True` 时把 5 个全分辨率
    输出取平均（官方默认），输出单通道 logits、与输入同分辨率，可直接用。

    坑：该仓库也有一个 `models` 包，与主项目的 `models` 重名，直接 `import models.fr_unet`
    会命中主项目那个包 -> ModuleNotFoundError。这里用 importlib 以 `frunet_models` 之名加载。
    """
    r = repo("frunet")
    fr_unet = _load_ext_package("frunet_models", r / "models", "fr_unet")
    return ToLogits(fr_unet.FR_UNet(num_classes=num_classes, num_channels=in_channels,
                                    feature_scale=feature_scale, dropout=0.2, fuse=True, out_ave=out_ave))


def _swin_unet_pretrained(pre: str | None = None):
    """找 Swin-T 的 ImageNet 预训练权重：显式参数 > 环境变量 > 本地文件 > (可选) timm 下载。

    返回 (state_dict 或 None, 来源说明)。
    """
    env = os.environ.get("SWIN_UNET_PRETRAINED")
    path = _first_existing([
        pre, env,
        REPO_ROOT / "Swin-Unet-main" / "Swin-Unet-main" / "swin_tiny_patch4_window7_224.pth",
        DEFAULT_ROOT / "Swin-Unet-main" / "Swin-Unet-main" / "swin_tiny_patch4_window7_224.pth",
    ]) or _rglob_first(REPO_ROOT / "weights", "*swin_tiny*.pth") \
       or _rglob_first(DEFAULT_ROOT, "swin_tiny_patch4_window7_224.pth")
    if path:
        return torch.load(path, map_location="cpu"), f"本地文件 {path}"
    if os.environ.get("SWIN_UNET_ALLOW_DOWNLOAD") == "1":   # 默认不联网下载，避免 job 卡在网络上
        try:
            import timm
            return (timm.create_model("swin_tiny_patch4_window7_224",
                                      pretrained=True).state_dict(),
                    "timm swin_tiny_patch4_window7_224 (ImageNet)")
        except Exception as e:  # noqa: BLE001
            return None, f"timm 下载失败：{type(e).__name__}: {str(e)[:160]}"
    return None, ("未找到本地权重（找过 weights/ 与 third_party/，可用 SWIN_UNET_PRETRAINED=<path> "
                  "指定，或 SWIN_UNET_ALLOW_DOWNLOAD=1 让 timm 下载）")


def build_swin_unet(in_channels: int = 3, num_classes: int = 1, img_size: int = 224,
                    pretrained: str | None = None) -> nn.Module:
    """Swin-Unet (2021)。

    三个必须注意的点：
      1) 官方 `get_config(args)` 会先读 args.cfg 指向的 yaml（仓库自带 configs/*.yaml），
         再按 args 上的若干属性 merge，所以这里用 SimpleNamespace 把所需属性全部置为默认；
      2) **输入尺寸必须让 patch(4)xwindow(7) 整除**。官方代码在 `__init__` 里就用
         `input_resolution` 建好了 SW-MSA 的 attention mask，且 `window_partition` 要求
         H、W 恰为 window_size 的整数倍。224 时 224/4=56、56/7=8 [OK]；256 时 64/7 除不尽
         -> 构造阶段即报 `shape '[1, 9, 7, 9, 7, 1]' is invalid for input of size 4096`。
         因此这里固定按官方 224 分辨率前向，再由 ToLogits 把输入缩放进来、输出插回原分辨率；
      3) 官方依赖 **ImageNet 预训练 Swin-T 权重**（config.MODEL.PRETRAIN_CKPT）。这里给本地权重
         留好加载入口（参数 > SWIN_UNET_PRETRAINED > weights/ 或 third_party/ 下的
         swin_tiny_patch4_window7_224.pth），找到就按名字+shape 加载并打印数量；
         `USE_PRETRAINED=0` 时整段跳过（不查找、不下载、不加载），用于"从零训练"对照；
         找不到只打印警告（绝不休默认为“已加载”）。
    """
    from types import SimpleNamespace

    r = repo("swin_unet")
    _syspath(r)
    from config import get_config  # type: ignore
    from networks.vision_transformer import SwinUnet  # type: ignore

    yaml_cfg = r / "configs" / "swin_tiny_patch4_window7_224_lite.yaml"
    args = SimpleNamespace(
        cfg=str(yaml_cfg), opts=None, batch_size=None, zip=False, cache_mode=None,
        resume=None, accumulation_steps=None, use_checkpoint=None, amp_opt_level=None,
        tag=None, eval=False, throughput=False, local_rank=0, num_classes=None, img_size=None,
    )
    cfg = get_config(args)
    cfg.defrost()
    cfg.MODEL.SWIN.IN_CHANS = in_channels
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.DATA.IMG_SIZE = img_size
    cfg.freeze()
    net = SwinUnet(cfg, img_size=img_size, num_classes=num_classes, zero_head=True)

    if _use_pretrained():
        sd, src = _swin_unet_pretrained(pretrained)
        if sd is not None:
            _load_pretrained(net, sd, "SwinUnet", prefixes=("swin_unet.", "", "model.", "module."))
            _diag(f"SwinUnet 预训练来源：{src}", "swin_src")
        else:
            _diag(f"SwinUnet [WARN] 未加载 ImageNet 预训练 Swin-T 权重（{src}）"
                  f"-> 随机初始化，指标会明显偏低", "swin_src")
    else:
        _diag("SwinUnet USE_PRETRAINED=0 -> 跳过预训练加载（随机初始化）", "swin_unet_pre")
    return ToLogits(net, net_size=img_size, tag="SwinUnet")


def _transunet_pretrained(pre: str | None = None) -> Path | None:
    """找 TransUNet 官方 ImageNet21k 的 R50+ViT-B_16.npz（显式参数 > 环境变量 > 本地文件）。"""
    return _first_existing([
        pre, os.environ.get("TRANSUNET_PRETRAINED"),
        REPO_ROOT / "TransUNet-main" / "TransUNet-main" / "model" / "vit_checkpoint"
        / "imagenet21k" / "R50+ViT-B_16.npz",
        DEFAULT_ROOT / "TransUNet-main" / "TransUNet-main" / "model" / "vit_checkpoint"
        / "imagenet21k" / "R50+ViT-B_16.npz",
    ]) or _rglob_first(REPO_ROOT / "weights", "R50+ViT-B_16.npz") \
       or _rglob_first(DEFAULT_ROOT, "R50+ViT-B_16.npz")


def build_transunet(in_channels: int = 3, num_classes: int = 1, img_size: int = 256,
                    pretrained_vit: str | None = None) -> nn.Module:
    """TransUNet (2021)。R50-ViT-B/16 主干；强烈建议加载 ImageNet 预训练权重，
    否则原始 ViT 部分基本没学到东西，指标会明显偏低（论文里必须注明是否用了预训练）。

    实测崩溃修复：TransUNet **不是全卷积网络** —— Embeddings 里的
    `position_embeddings` 是按 img_size 固定下来的 patch 数（vit_seg_modeling.py:149、163），
    而统一框架的训练用 256 裁剪、验证/测试用**整图**（dataset.py 补零到 32 的倍数，DRIVE 是
    576x608 -> 16x16 patch 后是 1368 个 token），于是
        x + self.position_embeddings   ->  "size of tensor a (1368) must match ... b (256)"
    （CHASE/STARE 则在 ResNetV2 的 non-square padding 分支
      vit_seg_modeling_resnet_skip.py:155 直接 shape 崩掉）。
    修复：wrapper 内固定按 img_size 前向（ToLogits(net_size=img_size)），输出再插值回输入分辨率，
    从而无论 256 裁剪还是整图推理都满足"输出与输入同分辨率、单通道 logits"的契约。

    预训练开关：`USE_PRETRAINED=0` 时跳过 ImageNet21k 权重的查找与加载（随机初始化），
    用于"从零训练"对照；未设置或 `1` 时行为与历史完全一致。
    """
    r = repo("transunet")
    _syspath(r)
    from networks.vit_seg_configs import get_r50_b16_config  # type: ignore
    from networks.vit_seg_modeling import VisionTransformer  # type: ignore
    cfg = get_r50_b16_config()
    cfg.n_classes = num_classes
    cfg.n_skip = 3
    net = VisionTransformer(cfg, img_size=img_size, num_classes=num_classes, zero_head=True, vis=False)
    if _use_pretrained():
        vit = _transunet_pretrained(pretrained_vit)
        if vit:
            import numpy as np
            net.load_from(weights=np.load(vit))  # 官方 load_from：pos-embed 会自动 zoom 到当前 img_size
            _diag(f"TransUNet 已加载 ImageNet21k 预训练权重：{vit}", "transunet_pretrained")
        else:
            _diag("TransUNet [WARN] 未找到 ImageNet21k 预训练权重 R50+ViT-B_16.npz（随机初始化，"
                  "指标会明显偏低）；可用 TRANSUNET_PRETRAINED=<path> 指定", "transunet_pretrained")
    else:
        _diag("TransUNet USE_PRETRAINED=0 -> 跳过预训练加载（随机初始化）", "transunet_pre")
    # 切块推理：测试窗 512x512 会被切成若干 256x256 小块（各占 1/8 重叠）分别前向，
    # 重叠区按覆盖次数平均。这样既不缩放（保住细血管），又让 ViT pos-embed 与训练分辨率一致。
    return ToLogits(net, net_size=img_size, tile=img_size,
                    tile_overlap=max(16, img_size // 8), tag="TransUNet")


def _nnunet_2d_kwargs(in_channels: int, num_classes: int, features=(32, 64, 128, 256, 320)):
    """nnU-Net 系（U-Mamba / Swin-UMamba）需要的显式 2D 配置。"""
    n = len(features)
    return dict(
        input_channels=in_channels,
        n_stages=n,
        features_per_stage=list(features),
        conv_op=nn.Conv2d,
        kernel_sizes=[[3, 3]] * n,
        strides=[[1, 1]] + [[2, 2]] * (n - 1),
        n_conv_per_stage=[2] * n,
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2] * (n - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


class _PadTo16(nn.Module):
    """前向时把 H/W 在右下补零到 16 的整数倍，输出裁回原尺寸。

    为什么需要：U-Mamba 的解码器用 `torch.cat` 拼接上采样特征与同层 skip 特征
    （UMambaBot_2d.py:335），4 次 stride-2 下采样 + `F.interpolate(scale_factor=2)`
    上采样要求 H/W 是 16 的倍数；不满足时官方代码在**解码器内部**直接抛
    `RuntimeError: Sizes of tensors must match ... Expected size 14 but got size 13`
    （本地实测 100x124），适配层来不及插值回原分辨率。

    本项目 `utils/dataset.py` 已把图像补到 32 的倍数（训练裁剪 256x256），因此现有
    训练/测试路径下 H/W 恒为 16 的倍数，本包装是**恒等变换**（不改变数值、不改变
    参数量与层名之外的任何东西）；它只是让适配层「输入任意尺寸 -> 输出同分辨率单通道
    logits」的契约对非 16 倍数输入也成立。
    """

    def __init__(self, net: nn.Module, mult: int = 16):
        super().__init__()
        self.net = net
        self.mult = mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % self.mult, (-w) % self.mult
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))      # 常数 0，与 dataset.py 的补零一致
        return self.net(x)[..., :h, :w]


def build_umamba(in_channels: int = 3, num_classes: int = 1) -> nn.Module:
    """U-Mamba (2024)。需要 nnunetv2 + dynamic_network_architectures + mamba_ssm。

    mamba_ssm 的导入**保持惰性**（只在本函数内触发），并在这里显式检查一次：
    缺扩展时抛带 `[adapter]` 前缀的清晰错误，便于区分服务器环境问题 vs 适配代码问题。

    注意：U-Mamba **没有**纯 PyTorch 回退（不像 VM-UNet），所以只要构建+前向成功，
    跑的就一定是 mamba_ssm 的 CUDA 实现。
    """
    _require_mamba_ssm("UMamba")
    r = repo("umamba")
    _syspath(r)
    from umamba.nnunetv2.nets.UMambaBot_2d import UMambaBot  # type: ignore
    net = UMambaBot(**_nnunet_2d_kwargs(in_channels, num_classes))
    return ToLogits(_PadTo16(net), tag="UMamba")


def _swin_umamba_pretrained_ckpt(pre: str | None = None) -> Path | None:
    """找 Swin-UMamba 官方要求的 ImageNet 预训练 VMamba-Tiny 权重。

    官方 README「Prepare data & pretrained model」：下载 VMamba releases（`#v0cls`，2024-02-18）
    的 `vssmtiny_dp01_ckpt_epoch_292.pth`，放到 `data/pretrained/vmamba/vmamba_tiny_e292.pth`。
    实查该 ckpt（91.6 MB / 212 个张量，顶层 `{'model': OrderedDict}`）：键为
    `layers.N.blocks.M.self_attention.*` / `layers.N.downsample.*` / `patch_embed.*` / `norm.*` / `head.*`。

    查找顺序：显式参数 > `SWIN_UMAMBA_PRETRAINED` > 官方路径 > 项目 `weights/` 下的同名/同源文件。
    """
    r = REPO_ROOT / DIRS["swin_umamba"] / "data" / "pretrained" / "vmamba"
    return _first_existing([
        pre, os.environ.get("SWIN_UMAMBA_PRETRAINED"),
        r / "vmamba_tiny_e292.pth",
        REPO_ROOT / "weights" / "vmamba_tiny_e292.pth",
        REPO_ROOT / "weights" / "vssmtiny_dp01_ckpt_epoch_292.pth",
    ]) or _rglob_first(REPO_ROOT / "weights", "vmamba_tiny_e292.pth") \
       or _rglob_first(REPO_ROOT / "weights", "vssmtiny*.pth")


def _load_swin_umamba_pretrained(net: nn.Module, ckpt: Path) -> tuple[int, int]:
    """按官方 `load_pretrained_ckpt`（SwinUMamba.py:628-657）的映射装 VMamba 编码器权重。

    官方映射只有两步：加前缀 `vssm_encoder.` + `layers.<i>.downsample.*` -> `downsamples.<i>.*`
    （我们的 VSSMEncoder 把 patch merging 放在独立的 ModuleList 里，见 SwinUMamba.py:385/400）。
    差异说明：官方实现用 `assert` 校形状、且 `skip_params` 把 `patch_embed.norm.weight` 写了两次
    （`patch_embed.norm.bias` 漏掉，LayerNorm 只装一半）；这里复用本层既有的 `_load_pretrained`
    （名字+shape 匹配、strict=False、打印 loaded/total），形状不符的张量自动排除，不会误装；
    `patch_embed.proj.weight`（3 通道）与本模型（in_chans=48）形状不同，会被自动跳过。
    """
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]                        # 官方 ckpt: {'model': OrderedDict(...)}
    if not isinstance(sd, dict):
        raise TypeError(f"无法识别的权重格式：{type(sd)}")
    mapped = {}
    for k, v in sd.items():
        kr = f"vssm_encoder.{k}"
        for i in range(8):                      # layers.<i>.downsample -> downsamples.<i>
            kr = kr.replace(f"layers.{i}.downsample", f"downsamples.{i}")
        mapped[kr] = v
    return _load_pretrained(net, mapped, "SwinUMamba(vmamba_tiny_e292)", prefixes=("",))


def build_swin_umamba(in_channels: int = 3, num_classes: int = 1, img_size: int = 256,
                      pretrained: str | None = None) -> nn.Module:
    """Swin-UMamba (MICCAI 2024)。需要 mamba_ssm + monai + nnunetv2。

    依赖：`SwinUMamba.py:15` 在**模块级**导入 mamba_ssm 的 selective_scan_fn，所以缺扩展时
    先由 `_require_mamba_ssm` 抛出带 `[adapter]` 前缀的清晰错误（环境问题 vs 适配问题可区分）；
    另外 `SwinUMamba.py:18-20` 还需 nnunetv2（仅类型注解用）与 monai（真正的编解码块）。

    实测缺陷修复：原实现传 `feat_size=[img_size//4, ..., img_size//32]`（4 个元素），而官方
    `SwinUMamba.feat_size` 是**各阶段通道数**（5 个元素），`SwinUMamba.py:516` 要取 `feat_size[4]`
    -> 构建阶段必然 `IndexError: list index out of range`（本地实测，train_comparison.py:454 会记成
    `[FAIL] SwinUMamba/list index out of range`）。现改用官方值 `[48, 96, 192, 384, 768]`
    （SwinUMamba.py:452 默认值 / 官方 `get_swin_umamba_from_plans` 第 674 行），它与 img_size 无关，
    故 img_size 参数保留（不改调用方签名）但不再参与构建。

    预训练（本模型的核心卖点）：官方要求 ImageNet 预训练的 VMamba-Tiny 权重。这里补上最小加载
    入口——`USE_PRETRAINED=0` 时跳过；找不到权重时打印 `[WARN]` 后继续随机初始化（绝不静默假装
    加载成功）；找到就按官方映射装入并打印"装入张量数/模型张量数"。

    输出契约：单通道 raw logits、分辨率与输入一致（H/W 需为 32 的倍数——stem(/2) + patch_embed(/2)
    后还要过 3 次 PatchMerging2D；本项目 `utils/dataset.py` 已把图像补到 32 的倍数）。
    """
    _require_mamba_ssm("SwinUMamba")
    r = repo("swin_umamba")
    _syspath(r)
    from swin_umamba.nnunetv2.nets.SwinUMamba import SwinUMamba  # type: ignore
    net = SwinUMamba(in_chans=in_channels, out_chans=num_classes,
                     feat_size=[48, 96, 192, 384, 768])

    use_pre = os.environ.get("USE_PRETRAINED", "1").strip().lower() \
        not in ("0", "false", "no", "off", "")
    ckpt = _swin_umamba_pretrained_ckpt(pretrained)
    if not use_pre:
        _diag("SwinUMamba 预训练：USE_PRETRAINED=0 -> 跳过加载，编码器保持随机初始化"
              "（用于 '从零训练' 对照）", "swin_umamba_pre")
    elif ckpt is None:
        _diag("SwinUMamba [WARN] 未找到官方要求的 ImageNet 预训练 VMamba-Tiny 权重 "
              "vmamba_tiny_e292.pth -> 编码器随机初始化，指标会明显偏低；"
              "可用 SWIN_UMAMBA_PRETRAINED=<path> 指定，或放到 third_party/Swin-UMamba-main/"
              "Swin-UMamba-main/data/pretrained/vmamba/ 下", "swin_umamba_pre")
    else:
        _diag(f"SwinUMamba 预训练来源：{ckpt}", "swin_umamba_pre_src")
        _load_swin_umamba_pretrained(net, ckpt)
    return ToLogits(net, tag="SwinUMamba")


def build_serp_mamba(in_channels: int = 3, num_classes: int = 1,
                     features=(32, 64, 128, 256, 320)) -> nn.Module:
    """Serp-Mamba (TMI 2025)——高分辨率视网膜血管分割 + 蛇形选择性扫描。

    它与 U-Mamba 共用 nnU-Net 风格的网络定义，位于仓库自带框架
    `Serp_Mamba/U_Mamba_main/umamba/nnunetv2/nets/SerpMamba.py`。
    依赖：nnunetv2 + dynamic_network_architectures + mamba_ssm + einops。
    """
    _require_mamba_ssm("SerpMamba")
    r = repo("serpmamba")
    _syspath(r / "Serp_Mamba" / "U_Mamba_main")
    from umamba.nnunetv2.nets.SerpMamba import SerpMamba  # type: ignore
    net = SerpMamba(**_nnunet_2d_kwargs(in_channels, num_classes, features))
    return ToLogits(net, tag="SerpMamba")


BUILDERS = {
    "cenet": ("CENet", build_cenet),
    "frunet": ("FRUNet", build_frunet),
    "m2unet": ("M2UNet", build_m2unet),
    "erfnet": ("ERFNet", build_erfnet),
    "driu": ("DRIU", build_driu),
    "vmunet": ("VMUNet", build_vmunet),
    "swin_unet": ("SwinUnet", build_swin_unet),
    "transunet": ("TransUNet", build_transunet),
    "umamba": ("UMamba", build_umamba),
    "swin_umamba": ("SwinUMamba", build_swin_umamba),
    "serpmamba": ("SerpMamba", build_serp_mamba),
}

# 展示名 -> 构建键（供 train_comparison.py / test_comparison.py 的 get_model() 使用）
EXT_INDEX = {name: kind for kind, (name, _) in BUILDERS.items()}
EXT_MODEL_NAMES = list(EXT_INDEX)


def load_ckpt_flexible(net: nn.Module, path: str | os.PathLike) -> tuple[int, int]:
    """尽量把官方权重装进去：自动剥掉 'module.'/'model.' 前缀，返回 (missing, unexpected)。"""
    sd = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "model_state_dict", "net"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if not isinstance(sd, dict):
        raise TypeError(f"无法识别的权重格式：{type(sd)}")
    sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    return len(missing), len(unexpected)


def build(kind: str, **kw) -> nn.Module:
    if kind not in BUILDERS:
        raise KeyError(f"未适配的模型：{kind}，可选 {list(BUILDERS)}")
    return BUILDERS[kind][1](**kw)


if __name__ == "__main__":
    size = int(os.environ.get("SMOKE_SIZE", "256"))
    for kind, (name, fn) in BUILDERS.items():
        try:
            m = fn()
            total = sum(p.numel() for p in m.parameters() if p.requires_grad)
            with torch.no_grad():
                y = m(torch.randn(1, 3, size, size))
            print(f"[OK]   {name:<12} {total/1e6:8.3f} M   out={tuple(y.shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name:<12} {type(e).__name__}: {str(e)[:150]}")
