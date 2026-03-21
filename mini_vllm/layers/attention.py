import os
import sys
from pathlib import Path

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from mini_vllm.utils.context import get_context


_HILDA_FP8_KERNELS = None


def _load_hilda_fp8_kernels():
    global _HILDA_FP8_KERNELS
    if _HILDA_FP8_KERNELS is not None:
        return _HILDA_FP8_KERNELS

    candidates = []
    env_path = os.environ.get("HILDA_KERNEL_TRITON_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("/root/hilda-kernel/triton-kernels"))

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            from kernels.paged_attention_fp8 import hilda_paged_attention_fp8_decode
            from kernels.store_kvcache_fp8 import hilda_store_kvcache_fp8, hilda_copy_kvcache_fp8
        except ImportError:
            continue
        _HILDA_FP8_KERNELS = {
            "decode": hilda_paged_attention_fp8_decode,
            "store": hilda_store_kvcache_fp8,
            "copy": hilda_copy_kvcache_fp8,
        }
        return _HILDA_FP8_KERNELS

    raise ImportError(
        "Unable to import hilda FP8 kernels. Set HILDA_KERNEL_TRITON_PATH or make /root/hilda-kernel/triton-kernels available."
    )


def store_kvcache_fp8(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    _load_hilda_fp8_kernels()["store"](key, value, k_cache, v_cache, slot_mapping)


def copy_kvcache_fp8(src_k_cache: torch.Tensor, src_v_cache: torch.Tensor, dst_k_cache: torch.Tensor, dst_v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    _load_hilda_fp8_kernels()["copy"](src_k_cache, src_v_cache, dst_k_cache, dst_v_cache, slot_mapping)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = "auto"
        self.k_cache = self.v_cache = torch.tensor([])
        self.prefill_k_cache = self.prefill_v_cache = torch.tensor([])
        self.decode_k_cache = self.decode_v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        if self.kv_cache_dtype == "fp8" and self.prefill_k_cache.numel() and self.decode_k_cache.numel():
            if context.is_prefill:
                store_kvcache(k, v, self.prefill_k_cache, self.prefill_v_cache, context.slot_mapping)
                k_cache, v_cache = self.prefill_k_cache, self.prefill_v_cache
                if context.block_tables is not None:
                    k, v = k_cache, v_cache
                return flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )

            store_kvcache_fp8(k, v, self.decode_k_cache, self.decode_v_cache, context.slot_mapping)
            return _load_hilda_fp8_kernels()["decode"](
                q,
                self.decode_k_cache,
                self.decode_v_cache,
                context.block_tables,
                context.context_lens,
                self.scale,
            )

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            return flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )

        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        )
