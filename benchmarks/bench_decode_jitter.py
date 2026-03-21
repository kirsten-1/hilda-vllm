import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from random import randint, seed
from statistics import mean
from time import perf_counter

from mini_vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--engine-name", default="hilda-vllm")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--long-prompt-len", type=int, default=4096)
    parser.add_argument("--num-long-seqs", type=int, default=4)
    parser.add_argument("--short-prompt-len", type=int, default=64)
    parser.add_argument("--num-short-seqs", type=int, default=256)
    parser.add_argument("--short-output-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="auto")
    return parser.parse_args()


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q / 100
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize_latencies(values):
    return {
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values) if values else None,
        "mean_ms": mean(values) if values else None,
    }


def save_results(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = f"{result['engine']}-decode-jitter".replace("/", "-")
    json_path = output_dir / f"{timestamp}-{slug}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved detailed results to {json_path}")


def build_result(args, llm, total_time_s, decode_latencies_ms, mixed_decode_latencies_ms, short_output_tokens):
    decode_stats = summarize_latencies(decode_latencies_ms)
    mixed_stats = summarize_latencies(mixed_decode_latencies_ms)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "scenario": "decode-jitter",
        "engine": args.engine_name,
        "backend": "hilda-vllm",
        "model": os.path.expanduser(args.model),
        "long_prompt_tokens": args.long_prompt_len,
        "num_long_requests": args.num_long_seqs,
        "short_prompt_tokens": args.short_prompt_len,
        "num_short_requests": args.num_short_seqs,
        "short_output_tokens": short_output_tokens,
        "decode_steps": len(decode_latencies_ms),
        "mixed_decode_steps": len(mixed_decode_latencies_ms),
        "decode_latency_p50_ms": decode_stats["p50_ms"],
        "decode_latency_p95_ms": decode_stats["p95_ms"],
        "decode_latency_p99_ms": decode_stats["p99_ms"],
        "decode_latency_max_ms": decode_stats["max_ms"],
        "decode_latency_mean_ms": decode_stats["mean_ms"],
        "mixed_decode_latency_p50_ms": mixed_stats["p50_ms"],
        "mixed_decode_latency_p95_ms": mixed_stats["p95_ms"],
        "mixed_decode_latency_p99_ms": mixed_stats["p99_ms"],
        "mixed_decode_latency_max_ms": mixed_stats["max_ms"],
        "mixed_decode_latency_mean_ms": mixed_stats["mean_ms"],
        "mixed_decode_jitter_ratio": (mixed_stats["p95_ms"] / mixed_stats["p50_ms"]) if mixed_stats["p50_ms"] else None,
        "total_time_s": total_time_s,
    }


def print_summary(result: dict):
    print("Decode Jitter Results:")
    print(f"Engine: {result['engine']}")
    print(f"Mixed decode steps: {result['mixed_decode_steps']}")
    print(f"Mixed decode p50/p95/max (ms): {result['mixed_decode_latency_p50_ms']:.2f} / {result['mixed_decode_latency_p95_ms']:.2f} / {result['mixed_decode_latency_max_ms']:.2f}" if result['mixed_decode_steps'] else "Mixed decode p50/p95/max (ms): n/a")
    print(f"All decode p50/p95/max (ms): {result['decode_latency_p50_ms']:.2f} / {result['decode_latency_p95_ms']:.2f} / {result['decode_latency_max_ms']:.2f}" if result['decode_steps'] else "All decode p50/p95/max (ms): n/a")


def main():
    args = parse_args()
    seed(0)
    llm = LLM(
        os.path.expanduser(args.model),
        enforce_eager=False,
        max_model_len=args.long_prompt_len + 256,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)

    long_prompts = [
        [randint(0, 10000) for _ in range(args.long_prompt_len)]
        for _ in range(args.num_long_seqs)
    ]
    short_prompts = [
        [randint(0, 10000) for _ in range(args.short_prompt_len)]
        for _ in range(args.num_short_seqs)
    ]

    llm.add_request(long_prompts[0], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1))
    for prompt in short_prompts:
        llm.add_request(prompt, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.short_output_len))
    for prompt in long_prompts[1:]:
        llm.add_request(prompt, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1))

    decode_latencies_ms = []
    mixed_decode_latencies_ms = []
    total_start = perf_counter()
    while not llm.is_finished():
        pending_prefill = any(seq.num_prompt_tokens_remaining > 0 for seq in llm.scheduler.waiting)
        step_start = perf_counter()
        _, _, num_decode_tokens = llm.step()
        elapsed_ms = (perf_counter() - step_start) * 1000
        if num_decode_tokens > 0:
            decode_latencies_ms.append(elapsed_ms)
            if pending_prefill:
                mixed_decode_latencies_ms.append(elapsed_ms)
    total_time_s = perf_counter() - total_start
    result = build_result(args, llm, total_time_s, decode_latencies_ms, mixed_decode_latencies_ms, args.num_short_seqs * args.short_output_len)
    print_summary(result)
    save_results(result, Path(args.output_dir))

    from benchmarks.render_readme import build_readme, load_latest_results, load_latest_decode_jitter_results, README_PATH

    README_PATH.write_text(build_readme(load_latest_results(), load_latest_decode_jitter_results()))
    print(f"Updated {README_PATH}")
    llm.exit()


if __name__ == "__main__":
    main()
