import os
import warnings

import torch
import triton
import triton.language as tl

from mini_vllm.utils.context import get_context


_VERIFY_MLP_KERNEL_FAILED = False
_VERIFY_MLP_KERNEL_WARNED = False


def verify_mlp_triton_requested() -> bool:
    if os.environ.get("MINI_VLLM_DISABLE_VERIFY_MLP_TRITON") == "1":
        return False
    return os.environ.get("MINI_VLLM_ENABLE_VERIFY_MLP_TRITON") == "1"


def _verify_mlp_triton_enabled() -> bool:
    return verify_mlp_triton_requested()


def _max_verify_tokens() -> int:
    return int(os.environ.get("MINI_VLLM_VERIFY_MLP_MAX_TOKENS", "16"))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _verify_linear_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[None, :] * stride_bn + k_offsets[:, None] * stride_bk,
            mask=(offs_n[None, :] < N) & (k_offsets[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _should_use_verify_mlp_kernel(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if not _verify_mlp_triton_enabled():
        return False
    if _VERIFY_MLP_KERNEL_FAILED:
        return False
    if not x.is_cuda or x.ndim != 2 or weight.ndim != 2:
        return False
    if x.shape[1] != weight.shape[1]:
        return False
    if x.shape[0] <= 0 or x.shape[0] > _max_verify_tokens():
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if weight.dtype != x.dtype or not weight.is_contiguous():
        return False
    context = get_context()
    return context.is_prefill and context.block_tables is not None and context.max_seqlen_k > context.max_seqlen_q


def triton_verify_mlp_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _verify_linear_kernel[grid](
        x,
        weight,
        y,
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=bias is not None,
    )
    return y


def maybe_triton_verify_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    global _VERIFY_MLP_KERNEL_FAILED, _VERIFY_MLP_KERNEL_WARNED

    if not _should_use_verify_mlp_kernel(x, weight):
        return None

    try:
        return triton_verify_mlp_linear(x, weight, bias)
    except Exception as exc:  # pragma: no cover - depends on local triton/cuda environment
        _VERIFY_MLP_KERNEL_FAILED = True
        if not _VERIFY_MLP_KERNEL_WARNED:
            warnings.warn(
                f"verify MLP Triton kernel unavailable, falling back to torch linear: {exc}",
                RuntimeWarning,
            )
            _VERIFY_MLP_KERNEL_WARNED = True
        return None
