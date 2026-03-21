import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from random import randint, seed
from time import perf_counter

from mini_vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--engine-name", default="hilda-vllm")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="auto")
    return parser.parse_args()


def save_results(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = f"{result['engine']}-decode-heavy".replace("/", "-")
    json_path = output_dir / f"{timestamp}-{slug}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved detailed results to {json_path}")


def build_result(args, llm, total_time_s: float):
    stats = llm.last_generate_stats or {}
    decode_tokens = stats.get("decode_tokens", 0)
    prefill_tokens = stats.get("prefill_tokens", 0)
    total_tokens = decode_tokens + prefill_tokens
    decode_share = decode_tokens / total_tokens if total_tokens else None
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "scenario": "decode-heavy",
        "engine": args.engine_name,
        "backend": "hilda-vllm",
        "model": os.path.expanduser(args.model),
        "prompt_tokens_per_request": args.prompt_len,
        "output_tokens_per_request": args.output_len,
        "num_requests": args.num_requests,
        "kv_cache_dtype": args.kv_cache_dtype,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_time_s": stats.get("prefill_time_s"),
        "decode_time_s": stats.get("decode_time_s"),
        "total_time_s": total_time_s,
        "prefill_throughput_toks": stats.get("prefill_throughput_toks"),
        "decode_throughput_toks": stats.get("decode_throughput_toks"),
        "total_throughput_toks": stats.get("total_throughput_toks"),
        "decode_share": decode_share,
    }


def print_summary(result: dict):
    print("Decode-heavy Results:")
    print(f"Engine: {result['engine']}")
    print(f"KV cache dtype: {result['kv_cache_dtype']}")
    print(f"Decode share: {result['decode_share']:.2%}" if result['decode_share'] is not None else "Decode share: n/a")
    print(f"Prefill throughput (tok/s): {result['prefill_throughput_toks']:.2f}" if result['prefill_throughput_toks'] is not None else "Prefill throughput (tok/s): n/a")
    print(f"Decode throughput (tok/s): {result['decode_throughput_toks']:.2f}" if result['decode_throughput_toks'] is not None else "Decode throughput (tok/s): n/a")
    print(f"Total throughput (tok/s): {result['total_throughput_toks']:.2f}" if result['total_throughput_toks'] is not None else "Total throughput (tok/s): n/a")


def main():
    args = parse_args()
    seed(0)
    llm = LLM(
        os.path.expanduser(args.model),
        enforce_eager=False,
        max_model_len=args.prompt_len + args.output_len + 256,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)

    prompts = [[randint(0, 10000) for _ in range(args.prompt_len)] for _ in range(args.num_requests)]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)
        for _ in range(args.num_requests)
    ]

    t0 = perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    total_time_s = perf_counter() - t0
    result = build_result(args, llm, total_time_s)
    print_summary(result)
    save_results(result, Path(args.output_dir))

    from benchmarks.render_readme import build_readme, load_latest_results, load_latest_decode_jitter_results, load_latest_decode_heavy_results, README_PATH

    README_PATH.write_text(
        build_readme(
            load_latest_results(),
            load_latest_decode_jitter_results(),
            load_latest_decode_heavy_results(),
        )
    )
    print(f"Updated {README_PATH}")
    llm.exit()


if __name__ == "__main__":
    main()
