"""Benchmark: Baseline vs Speculative Decoding comparison."""
import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from random import randint, seed
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Spec decode benchmark")
    parser.add_argument("--target-model", default="/root/autodl-tmp/Qwen3-8B")
    parser.add_argument("--draft-model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--max-output-len", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--output-dir", default="benchmarks/results")
    return parser.parse_args()


def run_benchmark(llm, prompt_token_ids, sampling_params_list, label):
    """Run benchmark and return stats dict."""
    llm.generate(["Benchmark warmup"], SamplingParams(), use_tqdm=False)

    prompts = prompt_token_ids
    t0 = perf_counter()
    outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)
    total_time = perf_counter() - t0

    stats = llm.last_generate_stats or {}
    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)
    result = {
        "label": label,
        "num_requests": len(prompts),
        "prompt_tokens": sum(len(p) for p in prompt_token_ids),
        "output_tokens": total_output_tokens,
        "total_time_s": total_time,
        "total_throughput_toks": total_output_tokens / total_time if total_time else 0,
        **{k: v for k, v in stats.items() if k not in ("num_requests",)},
    }
    return result


def print_comparison(baseline, spec):
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON: Baseline vs Speculative Decoding")
    print("=" * 90)

    metrics = [
        ("Total Throughput (tok/s)", "total_throughput_toks"),
        ("Prefill Throughput (tok/s)", "prefill_throughput_toks"),
        ("Decode Throughput (tok/s)", "decode_throughput_toks"),
        ("Output Tokens", "output_tokens"),
        ("Prefill Time (s)", "prefill_time_s"),
        ("Decode Time (s)", "decode_time_s"),
        ("Total Time (s)", "total_time_s"),
        ("Spec Acceptance Rate", "spec_acceptance_rate"),
    ]

    header = f"{'Metric':<30} {'Baseline':>15} {'SpecDecode':>15} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    for name, key in metrics:
        bv = baseline.get(key)
        sv = spec.get(key)
        if bv is not None and sv is not None:
            delta = ((sv - bv) / bv * 100) if bv != 0 else 0
            print(f"{name:<30} {bv:>15.2f} {sv:>15.2f} {delta:>+9.1f}%")
        elif sv is not None:
            print(f"{name:<30} {'n/a':>15} {sv:>15.2f} {'':>10}")
        else:
            bv_s = f"{bv:.2f}" if bv is not None else "n/a"
            sv_s = f"{sv:.2f}" if sv is not None else "n/a"
            print(f"{name:<30} {bv_s:>15} {sv_s:>15} {'n/a':>10}")
    print("=" * 90)


def main():
    args = parse_args()
    seed(0)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(50, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params_list = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(50, args.max_output_len))
        for _ in range(args.num_seqs)
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    # === Baseline ===
    print("=" * 60)
    print(f"Running BASELINE benchmark ({args.target_model})...")
    print("=" * 60)
    llm_base = LLM(args.target_model, enforce_eager=args.enforce_eager)
    baseline_result = run_benchmark(llm_base, prompt_token_ids, sampling_params_list, "baseline")
    llm_base.exit()
    del llm_base

    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    # === Speculative Decoding ===
    print("\n" + "=" * 60)
    print(f"Running SPEC DECODE benchmark (target={args.target_model}, draft={args.draft_model}, gamma={args.gamma})...")
    print("=" * 60)
    llm_spec = LLM(
        args.target_model,
        spec_decode_model=args.draft_model,
        spec_decode_gamma=args.gamma,
        enforce_eager=args.enforce_eager,
    )
    spec_result = run_benchmark(llm_spec, prompt_token_ids, sampling_params_list, f"spec-decode-gamma{args.gamma}")
    llm_spec.exit()
    del llm_spec

    # === Print comparison ===
    print_comparison(baseline_result, spec_result)

    # === Save results ===
    combined = {
        "timestamp": timestamp,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "gamma": args.gamma,
        "baseline": baseline_result,
        "spec_decode": spec_result,
    }
    json_path = output_dir / f"{timestamp}-spec-decode-comparison.json"
    json_path.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
