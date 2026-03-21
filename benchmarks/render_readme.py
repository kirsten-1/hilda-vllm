import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
README_PATH = REPO_ROOT / "README.md"
ENGINES = ["vllm", "nano-vllm", "hilda-vllm"]
DECODE_HEAVY_ENGINES = ["hilda-vllm", "hilda-vllm-fp8"]


def _load_latest_by_scenario(scenario: str, allowed_engines: list[str]) -> dict[str, dict]:
    latest = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        if data.get("scenario", "throughput") != scenario:
            continue
        engine = data.get("engine")
        if engine in allowed_engines:
            latest[engine] = data
    return latest


def load_latest_results():
    return _load_latest_by_scenario("throughput", ENGINES)


def load_latest_decode_jitter_results():
    return _load_latest_by_scenario("decode-jitter", ENGINES + ["hilda-vllm-fp8"])


def load_latest_decode_heavy_results():
    return _load_latest_by_scenario("decode-heavy", DECODE_HEAVY_ENGINES)


def fmt_num(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def fmt_pct(value, digits=1):
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def build_readme(results, jitter_results=None, decode_heavy_results=None):
    lines = [
        "# hilda-vllm",
        "",
        "## Benchmark",
        "",
        "Performance Results:",
        "",
        "| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for engine in ENGINES:
        data = results.get(engine)
        if data is None:
            lines.append(f"| {engine} | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {engine} | {fmt_num(data.get('output_tokens'), 0)} | "
            f"{fmt_num(data.get('total_time_s'))} | {fmt_num(data.get('total_throughput_toks'))} |"
        )

    lines.extend([
        "",
        "## FP8 KV Cache",
        "",
        "`--kv-cache-dtype fp8` is experimental and opt-in.",
        "",
        "Recommended usage:",
        "",
        "- Decode-heavy workloads with short prompts and long outputs.",
        "- Latency-first experiments where decode p50/p95 matters more than mixed prefill throughput.",
        "- Longer-context experiments where the smaller KV cache footprint is useful.",
        "",
        "Current caveat:",
        "",
        "- Prompt-heavy or mixed workloads can still regress total throughput versus the default BF16 KV cache path.",
    ])

    if decode_heavy_results:
        lines.extend([
            "",
            "## Decode-heavy Benchmark",
            "",
            "Short-prompt + long-output workload where decode dominates (>90% of generated-step tokens):",
            "",
            "| Inference Engine | KV Cache | Decode Share | Decode Throughput (tokens/s) | Total Throughput (tokens/s) |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for engine in DECODE_HEAVY_ENGINES:
            data = decode_heavy_results.get(engine)
            if data is None:
                lines.append(f"| {engine} | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {engine} | {data.get('kv_cache_dtype', 'auto')} | {fmt_pct(data.get('decode_share'))} | "
                f"{fmt_num(data.get('decode_throughput_toks'))} | {fmt_num(data.get('total_throughput_toks'))} |"
            )

    if jitter_results:
        lines.extend([
            "",
            "## Decode Jitter",
            "",
            "Mixed long-prefill + short-decode benchmark:",
            "",
            "| Inference Engine | Mixed Decode P50 (ms) | Mixed Decode P95 (ms) | Mixed Decode Max (ms) | Mixed Decode Steps |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for engine in ["hilda-vllm", "hilda-vllm-fp8"]:
            data = jitter_results.get(engine)
            if data is None:
                continue
            lines.append(
                f"| {engine} | {fmt_num(data.get('mixed_decode_latency_p50_ms'))} | "
                f"{fmt_num(data.get('mixed_decode_latency_p95_ms'))} | "
                f"{fmt_num(data.get('mixed_decode_latency_max_ms'))} | "
                f"{fmt_num(data.get('mixed_decode_steps'), 0)} |"
            )

    lines.extend([
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=. python benchmarks/bench.py --backend vllm --model /root/huggingface/Qwen3-0.6B --engine-name vllm",
        "PYTHONPATH=. python benchmarks/bench.py --backend nano-vllm --model /root/huggingface/Qwen3-0.6B --engine-name nano-vllm",
        "PYTHONPATH=. python benchmarks/bench.py --backend hilda-vllm --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm",
        "PYTHONPATH=. python benchmarks/bench_decode_jitter.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm",
        "PYTHONPATH=. python benchmarks/bench_decode_jitter.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm-fp8 --kv-cache-dtype fp8",
        "PYTHONPATH=. python benchmarks/bench_decode_heavy.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm",
        "PYTHONPATH=. python benchmarks/bench_decode_heavy.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm-fp8 --kv-cache-dtype fp8",
        "```",
        "",
        "## Next",
        "",
        "Start with Chunked Prefill.",
        "",
        "- Changes stay mostly in scheduler + model_runner.",
        "- It is easy to validate with long-prompt decode latency.",
        "- It is a good base for later work like speculative decoding.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    README_PATH.write_text(
        build_readme(
            load_latest_results(),
            load_latest_decode_jitter_results(),
            load_latest_decode_heavy_results(),
        )
    )
    print(README_PATH)
