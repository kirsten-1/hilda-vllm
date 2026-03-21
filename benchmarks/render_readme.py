import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
README_PATH = REPO_ROOT / "README.md"
ENGINES = ["vllm", "nano-vllm", "hilda-vllm"]


def load_latest_results():
    latest = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        engine = data.get("engine")
        if engine in ENGINES:
            latest[engine] = data
    return latest


def fmt_num(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def build_readme(results):
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
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=. python benchmarks/bench.py --backend vllm --model /root/huggingface/Qwen3-0.6B --engine-name vllm",
        "PYTHONPATH=. python benchmarks/bench.py --backend nano-vllm --model /root/huggingface/Qwen3-0.6B --engine-name nano-vllm",
        "PYTHONPATH=. python benchmarks/bench.py --backend hilda-vllm --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm",
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
    README_PATH.write_text(build_readme(load_latest_results()))
    print(README_PATH)
