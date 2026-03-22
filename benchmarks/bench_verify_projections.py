"""Benchmark Qwen verify projection kernels under batch=1 speculative shapes.

Measures the projection modules that dominate speculative verify forward:
- attention: qkv_proj, o_proj
- MLP: gate_up_proj, SwiGLU, down_proj

If the local Hilda Triton kernels repo is available, also benchmarks the
standalone Hilda SwiGLU kernel against the repo's current activation path.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_vllm import LLM
from mini_vllm.utils.context import set_context, reset_context


def default_target_model() -> str:
    candidates = [
        "/root/autodl-tmp/Qwen3-8B",
        "/root/autodl-tmp",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark verify projection modules")
    parser.add_argument("--target-model", default=default_target_model())
    parser.add_argument("--draft-model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--verify-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist-port", type=int, default=24530)
    parser.add_argument("--output-dir", default="benchmarks/results")
    return parser.parse_args()


def set_seed(rng_seed: int) -> None:
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)


def bench_cuda(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(iters):
        t0 = perf_counter()
        fn()
        torch.cuda.synchronize()
        samples_ms.append((perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": mean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def try_load_hilda_swiglu():
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
            from kernels.swiglu import hilda_swiglu  # type: ignore
        except ImportError:
            continue
        return hilda_swiglu, candidate_str
    return None, None


def main() -> None:
    args = parse_args()
    os.environ["MINI_VLLM_DIST_PORT"] = str(args.dist_port)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    llm = None
    try:
        llm = LLM(args.target_model, spec_decode_model=args.draft_model, spec_decode_gamma=args.gamma)
        model = llm.model_runner.model.model
        layer = model.layers[0]
        hidden_size = llm.config.hf_config.hidden_size
        verify_tokens = args.verify_batch_size * (args.gamma + 1)
        dtype = llm.config.hf_config.torch_dtype
        device = torch.device("cuda")

        hidden_states = torch.randn(verify_tokens, hidden_size, device=device, dtype=dtype)
        positions = torch.arange(verify_tokens, device=device, dtype=torch.int64)

        qkv_proj = layer.self_attn.qkv_proj
        o_proj = layer.self_attn.o_proj
        gate_up_proj = layer.mlp.gate_up_proj
        down_proj = layer.mlp.down_proj
        act_fn = layer.mlp.act_fn

        set_context(
            True,
            cu_seqlens_q=torch.tensor([0, verify_tokens], device=device, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, verify_tokens + 256], device=device, dtype=torch.int32),
            max_seqlen_q=verify_tokens,
            max_seqlen_k=verify_tokens + 256,
            slot_mapping=torch.arange(verify_tokens, device=device, dtype=torch.int32),
            block_tables=torch.zeros((1, 1), device=device, dtype=torch.int32),
        )

        gate_up_out = gate_up_proj(hidden_states)
        gate_part, up_part = gate_up_out.chunk(2, dim=-1)
        swiglu_out = act_fn(gate_up_out)
        hilda_swiglu, hilda_kernel_path = try_load_hilda_swiglu()

        o_proj_in = torch.randn(
            verify_tokens,
            layer.self_attn.total_num_heads * layer.self_attn.head_dim,
            device=device,
            dtype=dtype,
        )

        benchmarks = {
            "qkv_proj": bench_cuda(lambda: qkv_proj(hidden_states), args.warmup, args.iters),
            "o_proj": bench_cuda(lambda: o_proj(o_proj_in), args.warmup, args.iters),
            "gate_up_proj": bench_cuda(lambda: gate_up_proj(hidden_states), args.warmup, args.iters),
            "silu_and_mul": bench_cuda(lambda: act_fn(gate_up_out), args.warmup, args.iters),
            "down_proj": bench_cuda(lambda: down_proj(swiglu_out), args.warmup, args.iters),
            "mlp_end_to_end": bench_cuda(lambda: down_proj(act_fn(gate_up_proj(hidden_states))), args.warmup, args.iters),
        }

        if hilda_swiglu is not None:
            benchmarks["hilda_swiglu"] = bench_cuda(
                lambda: hilda_swiglu(gate_part, up_part),
                args.warmup,
                args.iters,
            )

        summary = {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "gamma": args.gamma,
            "verify_batch_size": args.verify_batch_size,
            "verify_tokens": verify_tokens,
            "hidden_size": hidden_size,
            "dtype": str(dtype),
            "num_layers": len(model.layers),
            "kernel_repo": hilda_kernel_path,
            "verify_mlp_triton_context": True,
            "results_ms": benchmarks,
            "derived": {
                "mlp_linear_only_ms": benchmarks["gate_up_proj"]["mean_ms"] + benchmarks["down_proj"]["mean_ms"],
                "mlp_activation_share_pct": (
                    benchmarks["silu_and_mul"]["mean_ms"] / benchmarks["mlp_end_to_end"]["mean_ms"] * 100.0
                    if benchmarks["mlp_end_to_end"]["mean_ms"] > 0
                    else 0.0
                ),
                "attn_projection_ms": benchmarks["qkv_proj"]["mean_ms"] + benchmarks["o_proj"]["mean_ms"],
            },
        }

        output_path = output_dir / "verify-projection-bench.json"
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"\nResults saved to {output_path}")
    finally:
        reset_context()
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
