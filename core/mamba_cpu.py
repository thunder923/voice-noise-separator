"""Pure-PyTorch drop-in replacement for mamba_ssm.Mamba (CPU / MPS).

Replicates the parameter layout and the non-fast-path forward of
mamba_ssm.modules.mamba_simple.Mamba exactly, using the official
selective_scan_ref recurrence, so checkpoints trained for mamba_ssm load
and produce the same results without CUDA kernels.

Reference: state-spaces/mamba (Apache-2.0), mamba_simple.py and
ops/selective_scan_interface.py (selective_scan_ref).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False, return_last_state=False):
    """Verbatim port of mamba_ssm.ops.selective_scan_interface.selective_scan_ref."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_fast(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                        delta_softplus=False, block=64):
    """Memory-bounded selective scan, numerically identical to selective_scan_ref.

    The reference materializes (batch, dim, L, dstate) decay tensors for the
    whole sequence, which is gigabytes at audio lengths. Here the decay and
    input terms are precomputed per block of `block` steps while the state
    recurrence itself runs step by step exactly as in the reference. (A
    cumulative-product formulation was tried and rejected: in fp32 the
    cumulative decay underflows to zero under strong per-step decay, which
    destroys the ratios the closed form depends on.)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    B = B.float()
    C = C.float()
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    L = u.shape[2]
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    for s in range(0, L, block):
        e = min(s + block, L)
        dA = torch.exp(torch.einsum('bdl,dn->bdln', delta[:, :, s:e], A))
        dBu = torch.einsum('bdl,bnl,bdl->bdln', delta[:, :, s:e], B[:, :, s:e], u[:, :, s:e])
        for i in range(e - s):
            x = dA[:, :, i] * x + dBu[:, :, i]
            ys.append(torch.einsum('bdn,bn->bd', x, C[:, :, s + i]))
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


class Mamba(nn.Module):
    """Same parameters and math as mamba_ssm's Mamba (slow path only)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False,
                 use_fast_path=True, layer_idx=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2,
                                bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=self.d_inner).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fast(x, dt, A, B, C, self.D.float(), z=z,
                                delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


def install():
    """Register this module as `mamba_ssm` in sys.modules (idempotent)."""
    import sys
    import types
    if "mamba_ssm" in sys.modules:
        return sys.modules["mamba_ssm"]
    mod = types.ModuleType("mamba_ssm")
    mod.Mamba = Mamba
    sys.modules["mamba_ssm"] = mod
    return mod
