"""Benchmark batch=1 latency for baseline vs speculative decoding."""

import argparse
import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from random import randint, seed
from time import perf_counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_vllm import LLM, SamplingParams


def default_target_model() -> str:
    candidates = [
        "/root/autodl-tmp/Qwen3-8B",
        "/root/autodl-tmp",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch=1 speculative decoding latency benchmark")
    parser.add_argument("--target-model", default=default_target_model())
    parser.add_argument("--draft-model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--output-dir", default="benchmarks/results")
    return parser.parse_args()


def build_prompt(prompt_len: int, rng_seed: int) -> list[int]:
    seed(rng_seed)
    return [randint(0, 10000) for _ in range(prompt_len)]


def warmup(llm: LLM):
    llm.generate([[1, 2, 3, 4]], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=4), use_tqdm=False)


def run_latency_benchmark(llm: LLM, prompt_token_ids: list[int], sampling_params: SamplingParams, label: str) -> dict:
    warmup(llm)

    first_token_time = None
    emitted_tokens = 0
    total_start = perf_counter()
    for event in llm.generate_stream([prompt_token_ids], sampling_params):
        if not event["token_ids"]:
            continue
        emitted_tokens += len(event["token_ids"])
        now = perf_counter()
        if first_token_time is None:
            first_token_time = now
    total_time = perf_counter() - total_start

    ttft_s = (first_token_time - total_start) if first_token_time is not None else None
    tpot_s = None
    if ttft_s is not None and emitted_tokens > 1:
        tpot_s = (total_time - ttft_s) / (emitted_tokens - 1)

    stats = llm.last_generate_stats or {}
    return {
        "label": label,
        "prompt_tokens": len(prompt_token_ids),
        "output_tokens": emitted_tokens,
        "ttft_ms": ttft_s * 1000 if ttft_s is not None else None,
        "tpot_ms": tpot_s * 1000 if tpot_s is not None else None,
        "total_time_ms": total_time * 1000,
        "prefill_time_ms": stats.get("prefill_time_s", 0.0) * 1000,
        "decode_time_ms": stats.get("decode_time_s", 0.0) * 1000,
        "decode_throughput_toks": stats.get("decode_throughput_toks"),
        "spec_acceptance_rate": stats.get("spec_acceptance_rate"),
        "spec_proposed": stats.get("spec_proposed"),
        "spec_accepted": stats.get("spec_accepted"),
    }


def print_comparison(baseline: dict, spec: dict):
    print("\n" + "=" * 88)
    print("BATCH=1 LATENCY COMPARISON: Baseline vs Speculative Decoding")
    print("=" * 88)
    metrics = [
        ("TTFT (ms)", "ttft_ms"),
        ("TPOT (ms)", "tpot_ms"),
        ("Total Time (ms)", "total_time_ms"),
        ("Output Tokens", "output_tokens"),
        ("Decode Throughput (tok/s)", "decode_throughput_toks"),
        ("Spec Acceptance Rate", "spec_acceptance_rate"),
    ]
    header = f"{'Metric':<30} {'Baseline':>16} {'SpecDecode':>16} {'Delta':>12}"
    print(header)
    print("-" * len(header))
    for name, key in metrics:
        bv = baseline.get(key)
        sv = spec.get(key)
        if bv is not None and sv is not None and isinstance(bv, (int, float)) and isinstance(sv, (int, float)):
            delta = ((sv - bv) / bv * 100) if bv not in (0, 0.0) else 0.0
            if key == "spec_acceptance_rate":
                bv_str = f"{bv * 100:.1f}%"
                sv_str = f"{sv * 100:.1f}%"
                delta_str = "n/a"
            else:
                bv_str = f"{bv:.2f}"
                sv_str = f"{sv:.2f}"
                delta_str = f"{delta:+.1f}%"
            print(f"{name:<30} {bv_str:>16} {sv_str:>16} {delta_str:>12}")
        else:
            bv_str = f"{bv:.2f}" if isinstance(bv, (int, float)) else "n/a"
            sv_str = f"{sv:.2f}" if isinstance(sv, (int, float)) else "n/a"
            print(f"{name:<30} {bv_str:>16} {sv_str:>16} {'n/a':>12}")
    print("=" * 88)


def main():
    args = parse_args()
    print(f"Resolved target model path: {args.target_model}")
    print(f"Resolved draft model path: {args.draft_model}")
    prompt_token_ids = build_prompt(args.prompt_len, args.seed)
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.max_output_len)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 60)
    print(f"Running BASELINE batch=1 latency benchmark ({args.target_model})...")
    print("=" * 60)
    llm_base = LLM(args.target_model, enforce_eager=args.enforce_eager)
    baseline_result = run_latency_benchmark(llm_base, prompt_token_ids, sampling_params, "baseline")
    llm_base.exit()
    del llm_base

    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(
        "Running SPEC DECODE batch=1 latency benchmark "
        f"(target={args.target_model}, draft={args.draft_model}, gamma={args.gamma})..."
    )
    print("=" * 60)
    llm_spec = LLM(
        args.target_model,
        spec_decode_model=args.draft_model,
        spec_decode_gamma=args.gamma,
        enforce_eager=args.enforce_eager,
    )
    spec_result = run_latency_benchmark(llm_spec, prompt_token_ids, sampling_params, f"spec-decode-gamma{args.gamma}")
    llm_spec.exit()
    del llm_spec

    print_comparison(baseline_result, spec_result)

    combined = {
        "timestamp": timestamp,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "gamma": args.gamma,
        "prompt_len": args.prompt_len,
        "max_output_len": args.max_output_len,
        "baseline": baseline_result,
        "spec_decode": spec_result,
    }
    json_path = output_dir / f"{timestamp}-spec-decode-batch1.json"
    json_path.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
