# -*- coding: utf-8 -*-
"""纯 PyTorch 的 `mamba_ssm` 回退（当 CUDA 扩展不可用时自动启用）。

为什么需要：集群上现成的 `selective_scan_cuda.cpython-311-*.so` 是在 glibc >= 2.32 的系统上
编译的，计算节点 glibc 更旧 -> 导入直接报
    ImportError: /libc.so.6: version `GLIBC_2.32' not found
于是 `mamba_ssm` 完全不可用，MambaVesselNet / U-Mamba / Swin-UMamba 三个模型全跑不了。

本模块提供一个**数值等价**的纯 PyTorch 实现，并把伪模块注册进 `sys.modules`：
  * `mamba_ssm.Mamba`                     （mamba_ssm 的 Mamba 模块，参数名/形状对齐）
  * `mamba_ssm.modules.mamba_simple.Mamba`
  * `mamba_ssm.ops.selective_scan_interface.selective_scan_fn / selective_scan_ref`
选择子扫描复用 `baselines_ext.wrappers._selective_scan_fallback`（已与逐点递推逐步比对：
max|Δy| ≈ 3.8e-6，真实 SS2D 形状下为 0；支持分组 B/C）。

**只在真实 mamba_ssm 不可用时才启用**，并在日志里显著提示，便于论文如实注明实现差异。
"""
from __future__ import annotations

import importlib
import importlib.abc
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- 选择子扫描
def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False, **kwargs):
    """与 mamba_ssm.selective_scan_fn 同签名的纯 PyTorch 版本（延迟导入避免循环依赖）。

    形状约定：u, delta `(b, d, l)`；A `(d, n)`；B, C `(b, n, l)` 或分组 `(b, g, n, l)`；
    D `(d,)`；z `(b, d, l)`。返回与 u 同形状（或 (y, last_state)）。
    """
    from baselines_ext.wrappers import _selective_scan_fallback as impl
    if kwargs:
        _warn_once(f"selective_scan_fn 忽略了额外参数：{sorted(kwargs)}")
    return impl(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


selective_scan_ref = selective_scan_fn

_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[mamba-fallback] [WARN] {msg}", flush=True)


# --------------------------------------------------------------------------- Mamba 模块
class Mamba(nn.Module):
    """与 `mamba_ssm.Mamba` 参数化一致的纯 PyTorch 实现（batch_first=True）。

    结构：in_proj -> depthwise causal conv1d -> SiLU -> 选择子扫描（含 D 跳连与 z 门控）-> out_proj
    参数名与官方一致（in_proj/conv1d/x_proj/dt_proj/A_log/D/out_proj），便于与官方 checkpoint 互操作。
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank="auto", dt_min: float = 0.001, dt_max: float = 0.1,
                 dt_init: str = "random", dt_scale: float = 1.0, dt_init_floor: float = 1e-4,
                 conv_bias: bool = True, bias: bool = False, batch_first: bool = True,
                 rmsnorm: bool = False, **kwargs):
        super().__init__()
        if kwargs:
            _warn_once(f"Mamba 忽略了额外参数：{sorted(kwargs)}（回退实现按 Mamba v1 结构）")
        if not batch_first:
            raise NotImplementedError("纯 PyTorch 回退只支持 batch_first=True")

        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.batch_first = batch_first
        self.rmsnorm = rmsnorm

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=self.d_conv,
                                groups=self.d_inner, padding=self.d_conv - 1, bias=conv_bias)
        self.act = nn.SiLU()
        if rmsnorm:
            self.norm = nn.RMSNorm(self.d_inner) if hasattr(nn, "RMSNorm") else nn.Identity()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt_proj 初始化：让 softplus(bias) 落在 [dt_min, dt_max]（与官方一致）
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor, inference_params=None, **kwargs):
        b, l, _d = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)          # (b, 2*d_inner, l)
        x, z = xz.chunk(2, dim=1)
        x = self.conv1d(x)[..., :l]                               # 因果卷积：截掉右侧 padding
        x = self.act(x)
        if self.rmsnorm:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        y = self._ssm(x, z)
        return self.out_proj(y.transpose(1, 2))

    def _ssm(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        A = -torch.exp(self.A_log.float())                        # (d_inner, d_state)
        delta, B, C = torch.split(self.x_proj(x.transpose(1, 2)),
                                  [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj(delta).transpose(1, 2)               # (b, d_inner, l)（softplus 在扫描内做）
        return selective_scan_fn(x, delta, A,
                                 B.transpose(1, 2), C.transpose(1, 2), self.D,
                                 z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)


class Mamba2(Mamba):
    """占位实现：按 Mamba v1 结构近似（SSD 的精确复现代价过高，且本项目用不到 Mamba2）。"""

    def __init__(self, *args, **kwargs):
        _warn_once("Mamba2 回退为 Mamba v1 结构近似（仅当第三方代码显式使用 Mamba2 时才会触发）")
        super().__init__(*args, **kwargs)


def _stub_transformers_if_missing() -> bool:
    """若环境没装 `transformers`，放一个**轻量占位包**。

    原因：mamba_ssm 2.3.x 的 `mamba_ssm/__init__.py` 会连带导入 `MambaLMHeadModel`
    （继承 transformers.PreTrainedModel），于是没装 transformers 时
    `import mamba_ssm` 直接 `ModuleNotFoundError: No module named 'transformers'` ——
    但我们只需要 `Mamba` 模块与 `selective_scan_fn`，完全用不到 LM/transformers。
    这里用一个"任何属性都返回 dummy 类"的占位包顶掉它，使**真实 CUDA 实现**可用；
    真正装了 transformers 的环境不会走这条分支。
    """
    # 幂等：如果已经是我们放的占位包，直接返回（否则 find_spec 会因 __spec__ 缺失而报错）
    existing = sys.modules.get("transformers")
    if existing is not None:
        return getattr(existing, "__mamba_fallback_stub__", False)
    try:
        if importlib.util.find_spec("transformers") is not None:
            return False
    except (ImportError, ValueError):
        pass

    def _dummy(name):
        return type(name, (nn.Module,), {})

    _STUB_PATH = ["<transformers-stub>"]
    stub = types.ModuleType("transformers")
    stub.__path__ = _STUB_PATH
    stub.__file__ = "<transformers-stub-by-mamba_fallback>"
    stub.__version__ = "0.0.0+mamba-fallback-stub"
    stub.__all__ = []
    # 关键：必须给一个合法的 __spec__，否则 importlib.util.find_spec("transformers")
    # 会抛 "ValueError: transformers.__spec__ is None"（实测 SwinUMamba / MambaVesselNet 就死在这）
    stub.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    stub.__loader__ = None
    stub.__mamba_fallback_stub__ = True

    def _getattr(name):          # PEP 562：普通属性名给 dummy 类
        # ⚠️ dunder 绝不能返回 dummy 类：实测 `transformers.__file__.endswith(...)`
        #    会报 "type object '__file__' has no attribute 'endswith'"，
        #    导致 import mamba_ssm 失败并退回慢速回退（进而显存爆掉）。
        if name == "__file__":
            return "<transformers-stub-by-mamba_fallback>"
        if name == "__version__":
            return "0.0.0+mamba-fallback-stub"
        if name in ("__all__",):
            return []
        if name == "__path__":
            return _STUB_PATH
        if name == "__spec__":
            return importlib.machinery.ModuleSpec("transformers", loader=None)
        if name.startswith("__") and name.endswith("__"):
            return None
        return _dummy(name)

    stub.__getattr__ = _getattr

    class _Finder(importlib.abc.MetaPathFinder):
        """把 transformers.* 子模块也映射到 dummy（有些代码会 from transformers.xxx import y）。"""

        def find_spec(self, fullname, path=None, target=None):
            if fullname == "transformers" or not fullname.startswith("transformers."):
                return None
            def _exec(module, _fullname=fullname):
                module.__getattr__ = lambda name: _dummy(name)

            return importlib.machinery.ModuleSpec(fullname, _Loader(_exec))

    class _Loader(importlib.abc.Loader):
        def __init__(self, exec_fn):
            self.exec_fn = exec_fn

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            return m

        def exec_module(self, module):
            self.exec_fn(module)

    sys.modules["transformers"] = stub
    sys.meta_path.append(_Finder())
    print("[mamba-fallback] 未安装 transformers -> 已放入轻量占位包（mamba_ssm 导入图需要，"
          "实际只用 Mamba/selective_scan_fn）", flush=True)
    return True


def patch_triton_compat() -> None:
    """给旧版 triton 补 mamba_ssm 需要的 API。

    实测（服务器 gaic0200，torch 2.6 + 自带 triton）：
        import mamba_ssm -> AttributeError: module 'triton' has no attribute 'set_allocator'
    mamba_ssm 2.3.x 在导入期会调用 `triton.set_allocator(...)`（较新 triton 才有）。
    这里在导入 mamba_ssm **之前**补一个安全的空实现，使真实 CUDA 扩展仍可用；
    其它缺失属性也一并按需补空，避免因版本差异直接放弃官方 kernel。
    """
    try:
        import triton
    except Exception:  # noqa: BLE001
        return
    added = []
    if not hasattr(triton, "set_allocator"):
        def _noop_set_allocator(*a, **k):
            return None

        triton.set_allocator = _noop_set_allocator
        added.append("set_allocator")
    if added:
        print(f"[mamba-fallback] 已为 triton 补兼容垫片：{added}（mamba_ssm 需要）", flush=True)


# --------------------------------------------------------------------------- 安装
def install_mamba_fallback(verbose: bool = True) -> str:
    """返回后端名：`mamba_ssm`（真 CUDA 扩展可用）或 `pytorch-fallback`（本次启用回退）。"""
    patch_triton_compat()      # 旧版 triton 的 API 差异先补齐，再试真扩展
    _stub_transformers_if_missing()   # mamba_ssm 包 __init__ 需要 transformers（仅 LM 头）
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _real  # noqa: F401
        return "mamba_ssm"
    except Exception as e:  # noqa: BLE001
        reason = f"{type(e).__name__}: {str(e)[:160]}"

    import os as _os
    if _os.environ.get("MAMBA_FALLBACK", "1").strip() == "0":
        raise RuntimeError(
            f"[mamba-fallback] MAMBA_FALLBACK=0 已禁用纯 PyTorch 回退，但真实 mamba_ssm 不可用：{reason}")

    pkg = types.ModuleType("mamba_ssm")
    pkg.__version__ = "0.0.0+pytorch-fallback"
    pkg.__path__ = []          # 标记为包
    pkg.Mamba = Mamba
    pkg.Mamba2 = Mamba2

    ops = types.ModuleType("mamba_ssm.ops")
    ops.__path__ = []
    ssi = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    ssi.selective_scan_fn = selective_scan_fn
    ssi.selective_scan_ref = selective_scan_fn
    ssi.mamba_inner_fn = None
    ssi.mamba_inner_ref = None

    mods = types.ModuleType("mamba_ssm.modules")
    mods.__path__ = []
    simple = types.ModuleType("mamba_ssm.modules.mamba_simple")
    simple.Mamba = Mamba
    m2 = types.ModuleType("mamba_ssm.modules.mamba2")
    m2.Mamba2 = Mamba2

    pkg.ops = ops
    ops.selective_scan_interface = ssi
    pkg.modules = mods
    mods.mamba_simple = simple
    mods.mamba2 = m2

    sys.modules.update({
        "mamba_ssm": pkg,
        "mamba_ssm.ops": ops,
        "mamba_ssm.ops.selective_scan_interface": ssi,
        "mamba_ssm.modules": mods,
        "mamba_ssm.modules.mamba_simple": simple,
        "mamba_ssm.modules.mamba2": m2,
    })

    # causal_conv1d 也常被 mamba 系代码引用；缺席时给个纯 PyTorch 版本
    if "causal_conv1d" not in sys.modules:
        try:
            import causal_conv1d  # noqa: F401
        except Exception:  # noqa: BLE001
            cc = types.ModuleType("causal_conv1d")
            cc.__path__ = []

            def causal_conv1d_fn(x, weight, bias=None, activation=None, **kw):
                """x: (b, c, l)（或 (c, l)），weight: (c, 1, k) 的深度卷积权重。"""
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                k = weight.shape[-1]
                y = F.conv1d(x, weight, bias, padding=k - 1, groups=weight.shape[0])[..., :x.shape[-1]]
                return F.silu(y) if activation in ("silu", "swish") else y

            def causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, **kw):
                return causal_conv1d_fn(x, weight, bias, activation), conv_state

            cc.causal_conv1d_fn = causal_conv1d_fn
            cc.causal_conv1d_update = causal_conv1d_update
            ccx = types.ModuleType("causal_conv1d.causal_conv1d_interface")
            ccx.causal_conv1d_fn = causal_conv1d_fn
            ccx.causal_conv1d_update = causal_conv1d_update
            cc.causal_conv1d_interface = ccx
            sys.modules.update({"causal_conv1d": cc,
                                "causal_conv1d.causal_conv1d_interface": ccx})

    if verbose:
        print(f"[mamba-fallback] 真实 mamba_ssm 不可用（{reason}）", flush=True)
        print("[mamba-fallback] 已启用**纯 PyTorch 选择子扫描 + Mamba 模块**回退："
              "数值等价、速度较慢；论文中需注明该实现差异", flush=True)
    _warn_once(f"启用纯 PyTorch mamba 回退（原因：{reason}）")
    return "pytorch-fallback"
