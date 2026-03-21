"""Benchmark: Baseline vs Speculative Decoding comparison."""
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from random import randint, seed
from time import perf_counter

sys.path.insert(0, "/root/hilda-vllm/hilda-vllm")

from mini_vllm import LLM, SamplingParams


def run_benchmark(llm, prompt_token_ids, sampling_params_list, label):
    """Run benchmark and return stats dict."""
    # Warmup
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
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON: Baseline vs Speculative Decoding")
    print("=" * 80)

    metrics = [
        ("Total Throughput (tok/s)", "total_throughput_toks"),
        ("Prefill Throughput (tok/s)", "prefill_throughput_toks"),
        ("Decode Throughput (tok/s)", "decode_throughput_toks"),
        ("Output Tokens", "output_tokens"),
        ("Prefill Time (s)", "prefill_time_s"),
        ("Decode Time (s)", "decode_time_s"),
        ("Total Time (s)", "total_time_s"),
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
        else:
            bv_s = f"{bv:.2f}" if bv is not None else "n/a"
            sv_s = f"{sv:.2f}" if sv is not None else "n/a"
            print(f"{name:<30} {bv_s:>15} {sv_s:>15} {'n/a':>10}")
    print("=" * 80)


def main():
    seed(0)
    num_seqs = 64  # fewer seqs for faster benchmark
    max_input_len = 512
    max_output_len = 256

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(50, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params_list = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(50, max_output_len))
        for _ in range(num_seqs)
    ]

    output_dir = Path("/root/hilda-vllm/hilda-vllm/benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    # === Baseline ===
    print("=" * 60)
    print("Running BASELINE benchmark (Qwen3-0.6B, no spec decode)...")
    print("=" * 60)
    llm_base = LLM("/root/huggingface/Qwen3-0.6B", enforce_eager=True)
    baseline_result = run_benchmark(llm_base, prompt_token_ids, sampling_params_list, "baseline")
    llm_base.exit()
    del llm_base

    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    # === Speculative Decoding ===
    print("\n" + "=" * 60)
    print("Running SPEC DECODE benchmark (target=Qwen3-0.6B, draft=Qwen2.5-0.5B-Instruct, gamma=5)...")
    print("=" * 60)
    llm_spec = LLM(
        "/root/huggingface/Qwen3-0.6B",
        spec_decode_model="/root/huggingface/Qwen2.5-0.5B-Instruct",
        spec_decode_gamma=5,
        enforce_eager=True,
    )
    spec_result = run_benchmark(llm_spec, prompt_token_ids, sampling_params_list, "spec-decode-gamma5")
    llm_spec.exit()
    del llm_spec

    # === Print comparison ===
    print_comparison(baseline_result, spec_result)

    # === Save results ===
    combined = {
        "timestamp": timestamp,
        "baseline": baseline_result,
        "spec_decode": spec_result,
    }
    json_path = output_dir / f"{timestamp}-spec-decode-comparison.json"
    json_path.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
