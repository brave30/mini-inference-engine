"""Fused residual-add + RMSNorm Triton kernel.

One program per row. The residual add, fp32 variance reduction, normalization and
weight scaling all happen in registers, so the hidden state is read once and written
once (plus the updated residual), versus ~7 separate kernels / HBM round-trips in
PyTorch eager (add, cast, pow, mean, rsqrt, mul, cast, weight-mul).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    X, R, W, Y, stride_x, stride_r, stride_y, N, eps,
    HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        x = x + tl.load(R + row * stride_r + cols, mask=mask, other=0.0)
        # updated residual stream is written back in place (same rounding as HF: add in model dtype)
        tl.store(R + row * stride_r + cols, x, mask=mask)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # HF casts the normalized value to the model dtype before scaling by the weight
    y = (xf * rstd).to(x.dtype) * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float):
    """Returns (normed, residual). If residual is given it is updated in place to x + residual
    and the norm is taken of that sum; otherwise x itself becomes the new residual."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    M, N = x2.shape
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 16)
    if residual is None:
        _fused_add_rmsnorm_kernel[(M,)](
            x2, x2, weight, y, x2.stride(0), x2.stride(0), y.stride(0), N, eps,
            HAS_RESIDUAL=False, BLOCK=BLOCK, num_warps=num_warps)
        return y.view(shape), x
    r2 = residual.view(-1, N)
    _fused_add_rmsnorm_kernel[(M,)](
        x2, r2, weight, y, x2.stride(0), r2.stride(0), y.stride(0), N, eps,
        HAS_RESIDUAL=True, BLOCK=BLOCK, num_warps=num_warps)
    return y.view(shape), residual


def add_rmsnorm_torch(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float):
    """PyTorch eager reference, numerically identical to HF LlamaRMSNorm."""
    if residual is not None:
        x = x + residual
    residual = x
    dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return weight * xf.to(dtype), residual
