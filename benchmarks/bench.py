import argparse
import csv
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from random import randint, seed
from time import perf_counter


BACKENDS = {
    "hilda-vllm": "mini_vllm",
    "nano-vllm": "nanovllm",
    "vllm": "vllm",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="hilda-vllm")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--engine-name", default=None)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="auto")
    return parser.parse_args()


def format_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def print_summary(result: dict):
    headers = [
        "Inference Engine",
        "Prompt Tokens",
        "Output Tokens",
        "Prefill (tok/s)",
        "Decode (tok/s)",
        "Total Throughput (tok/s)",
        "Prefill Time (s)",
        "Decode Time (s)",
        "Total Time (s)",
    ]
    row = [
        format_metric(result["engine"]),
        format_metric(result["prompt_tokens"]),
        format_metric(result["output_tokens"]),
        format_metric(result["prefill_throughput_toks"]),
        format_metric(result["decode_throughput_toks"]),
        format_metric(result["total_throughput_toks"]),
        format_metric(result["prefill_time_s"]),
        format_metric(result["decode_time_s"]),
        format_metric(result["total_time_s"]),
    ]
    widths = [max(len(header), len(value)) for header, value in zip(headers, row)]

    def render(values):
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    print("Performance Results:")
    print(render(headers))
    print(render(["-" * width for width in widths]))
    print(render(row))

    diagnostic_items = [
        ("Avg Decode Active BS", "avg_decode_active_bs"),
        ("Avg Decode Graph BS", "avg_decode_graph_bs"),
        ("Avg Decode Padded Slots", "avg_decode_padded_slots"),
        ("Decode Steps", "decode_steps"),
        ("Max Decode Active BS", "decode_active_bs_max"),
        ("Max Running", "running_max"),
        ("Max Waiting", "waiting_max"),
        ("Prefill Alloc Blocked", "prefill_allocation_blocked_steps"),
        ("Prefill No Work", "prefill_no_schedulable_steps"),
        ("Min Free KV Blocks", "min_free_blocks_observed"),
        ("KV Used Blocks", "kv_used_blocks"),
        ("KV Total Blocks", "kv_num_blocks"),
    ]
    available_items = [(label, key) for label, key in diagnostic_items if key in result]
    if available_items:
        print("\nDiagnostics:")
        label_width = max(len(label) for label, _ in available_items)
        for label, key in available_items:
            print(f"{label.ljust(label_width)} : {format_metric(result[key])}")


def save_results(result: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = result["engine"].replace("/", "-")
    json_path = output_dir / f"{timestamp}-{slug}.json"
    csv_path = output_dir / "performance_results.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    fieldnames = [
        "timestamp",
        "engine",
        "backend",
        "model",
        "num_requests",
        "prompt_tokens",
        "output_tokens",
        "prefill_tokens",
        "decode_tokens",
        "prefill_time_s",
        "decode_time_s",
        "total_time_s",
        "prefill_throughput_toks",
        "decode_throughput_toks",
        "total_throughput_toks",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: result.get(name) for name in fieldnames})
    print(f"Saved detailed results to {json_path}")
    print(f"Appended summary row to {csv_path}")


def load_backend(backend: str):
    module = importlib.import_module(BACKENDS[backend])
    return module.LLM, module.SamplingParams


def prepare_prompts(prompt_token_ids: list[list[int]], backend: str):
    if backend == "vllm":
        return [{"prompt_token_ids": prompt} for prompt in prompt_token_ids]
    return prompt_token_ids


def count_output_tokens(outputs, backend: str):
    if backend == "vllm":
        return sum(len(completion.token_ids) for request in outputs for completion in request.outputs)
    return sum(len(output["token_ids"]) for output in outputs)


def build_result(args, llm, outputs, prompt_token_ids, total_time_s: float):
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "engine": args.engine_name or args.backend,
        "backend": args.backend,
        "model": os.path.expanduser(args.model),
        "num_requests": len(prompt_token_ids),
        "prompt_tokens": sum(len(prompt) for prompt in prompt_token_ids),
        "output_tokens": count_output_tokens(outputs, args.backend),
        "prefill_tokens": None,
        "decode_tokens": None,
        "prefill_time_s": None,
        "decode_time_s": None,
        "total_time_s": total_time_s,
        "prefill_throughput_toks": None,
        "decode_throughput_toks": None,
        "total_throughput_toks": count_output_tokens(outputs, args.backend) / total_time_s if total_time_s else 0.0,
    }
    if getattr(llm, "last_generate_stats", None):
        result.update(llm.last_generate_stats)
    return result


def main():
    args = parse_args()
    seed(0)
    num_seqs = 512
    max_input_len = 1024
    max_ouput_len = 1024

    LLM, SamplingParams = load_backend(args.backend)
    path = os.path.expanduser(args.model)
    llm_kwargs = {"enforce_eager": False, "max_model_len": 4096}
    if args.backend == "hilda-vllm":
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    llm = LLM(path, **llm_kwargs)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    llm.generate(["Benchmark: "], SamplingParams())
    prompts = prepare_prompts(prompt_token_ids, args.backend)
    t0 = perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    total_time_s = perf_counter() - t0
    result = build_result(args, llm, outputs, prompt_token_ids, total_time_s)
    print_summary(result)
    save_results(result, Path(args.output_dir))

    from benchmarks.render_readme import (
        README_PATH,
        build_readme,
        load_latest_decode_heavy_results,
        load_latest_decode_jitter_results,
        load_latest_results,
    )

    README_PATH.write_text(
        build_readme(
            load_latest_results(),
            load_latest_decode_jitter_results(),
            load_latest_decode_heavy_results(),
        )
    )
    print(f"Updated {README_PATH}")


if __name__ == "__main__":
    main()
