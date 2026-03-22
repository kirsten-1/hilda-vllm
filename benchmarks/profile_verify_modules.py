"""Profile speculative verify forward at the per-module level.

Captures a single verify invocation, wraps Qwen decoder projection modules with
record_function labels, exports a chrome trace, and summarizes kernel time by
module type and layer.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from random import randint, seed

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_vllm import LLM, SamplingParams
from mini_vllm.engine.model_runner import ModelRunner


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
    parser = argparse.ArgumentParser(description="Profile verify forward by Qwen projection module")
    parser.add_argument("--target-model", default=default_target_model())
    parser.add_argument("--draft-model", default="/root/huggingface/Qwen3-0.6B")
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--max-output-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist-port", type=int, default=24520)
    parser.add_argument("--output-dir", default="benchmarks/results")
    return parser.parse_args()


def build_prompt(prompt_len: int, rng_seed: int) -> list[int]:
    seed(rng_seed)
    return [randint(0, 10000) for _ in range(prompt_len)]


def set_seed(rng_seed: int) -> None:
    seed(rng_seed)
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)


def aggregate_trace(trace_path: Path, num_layers: int) -> dict:
    trace = json.loads(trace_path.read_text())["traceEvents"]

    module_spans = []
    for ev in trace:
        name = ev.get("name", "")
        if ev.get("ph") == "X" and isinstance(name, str) and name.startswith("verify.module.layer"):
            module_spans.append({"name": name, "ts": ev["ts"], "end": ev["ts"] + ev["dur"]})

    kernels = []
    for ev in trace:
        if ev.get("ph") == "X" and ev.get("cat") == "kernel":
            kernels.append({
                "name": ev.get("name", ""),
                "ts": ev["ts"],
                "end": ev["ts"] + ev["dur"],
                "dur": ev["dur"],
            })

    module_kernel = defaultdict(float)
    module_gemm = defaultdict(float)
    module_flash = defaultdict(float)
    module_store = defaultdict(float)
    for span in module_spans:
        start = span["ts"]
        end = span["end"]
        for ker in kernels:
            if ker["ts"] >= start and ker["end"] <= end:
                module_kernel[span["name"]] += ker["dur"]
                lname = ker["name"].lower()
                if "cutlass::kernel2" in lname or "gemm" in lname:
                    module_gemm[span["name"]] += ker["dur"]
                elif "flash_fwd" in lname or "flash_attn" in lname:
                    module_flash[span["name"]] += ker["dur"]
                elif "store_kvcache" in lname:
                    module_store[span["name"]] += ker["dur"]

    by_type = defaultdict(lambda: defaultdict(float))
    for name, dur in module_kernel.items():
        mod_type = name.split(".")[-1]
        by_type[mod_type]["kernel_us"] += dur
        by_type[mod_type]["gemm_us"] += module_gemm.get(name, 0.0)
        by_type[mod_type]["flash_us"] += module_flash.get(name, 0.0)
        by_type[mod_type]["store_us"] += module_store.get(name, 0.0)

    per_layer = []
    for li in range(num_layers):
        row = {"layer": li}
        for kind in ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "attn", "mlp"]:
            key = f"verify.module.layer{li:02d}.{kind}"
            row[kind + "_kernel_us"] = round(module_kernel.get(key, 0.0), 3)
            row[kind + "_gemm_us"] = round(module_gemm.get(key, 0.0), 3)
        per_layer.append(row)

    return {
        "by_module_type_us": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in by_type.items()},
        "top_modules_by_kernel_us": sorted([
            {
                "name": name,
                "kernel_us": round(module_kernel[name], 3),
                "gemm_us": round(module_gemm.get(name, 0.0), 3),
                "flash_us": round(module_flash.get(name, 0.0), 3),
                "store_us": round(module_store.get(name, 0.0), 3),
            }
            for name in module_kernel
        ], key=lambda x: x["kernel_us"], reverse=True)[:24],
        "per_layer_us": per_layer,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "verify-module-profiler-trace.json"
    summary_path = output_dir / "verify-module-profiler-summary.json"

    state = {"captured": False, "profile": None}
    orig_run_verify = ModelRunner.run_verify

    @torch.inference_mode()
    def instrumented_run_verify(self, seqs, draft_tokens, draft_probs):
        if state["captured"]:
            return orig_run_verify(self, seqs, draft_tokens, draft_probs)

        wrapped_modules = []
        for li, layer in enumerate(self.model.model.layers):
            targets = {
                f"verify.module.layer{li:02d}.qkv_proj": layer.self_attn.qkv_proj,
                f"verify.module.layer{li:02d}.o_proj": layer.self_attn.o_proj,
                f"verify.module.layer{li:02d}.gate_up_proj": layer.mlp.gate_up_proj,
                f"verify.module.layer{li:02d}.down_proj": layer.mlp.down_proj,
                f"verify.module.layer{li:02d}.attn": layer.self_attn.attn,
                f"verify.module.layer{li:02d}.mlp": layer.mlp,
            }
            for label, module in targets.items():
                orig_forward = module.forward

                def make_wrapped(orig, name):
                    def wrapped(*f_args, **f_kwargs):
                        with record_function(name):
                            return orig(*f_args, **f_kwargs)
                    return wrapped

                module.forward = make_wrapped(orig_forward, label)
                wrapped_modules.append((module, orig_forward))

        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                result = orig_run_verify(self, seqs, draft_tokens, draft_probs)
            state["captured"] = True
            state["profile"] = prof
            return result
        finally:
            for module, orig_forward in wrapped_modules:
                module.forward = orig_forward

    ModelRunner.run_verify = instrumented_run_verify

    os.environ["MINI_VLLM_DIST_PORT"] = str(args.dist_port)
    prompt = build_prompt(args.prompt_len, args.seed)
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.max_output_len)
    llm = None
    num_layers = 0
    try:
        set_seed(args.seed)
        llm = LLM(args.target_model, spec_decode_model=args.draft_model, spec_decode_gamma=args.gamma)
        num_layers = len(llm.model_runner.model.model.layers)
        llm.generate([[1, 2, 3, 4]], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=4), use_tqdm=False)
        set_seed(args.seed)
        llm.add_request(prompt, sp)
        while not llm.is_finished() and not state["captured"]:
            llm.step()
        prof = state["profile"]
        if prof is None:
            raise RuntimeError("verify module profile not captured")
        prof.export_chrome_trace(str(trace_path))
    finally:
        ModelRunner.run_verify = orig_run_verify
        if llm is not None:
            llm.exit()

    summary = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "gamma": args.gamma,
        "seed": args.seed,
        "num_layers": num_layers,
        **aggregate_trace(trace_path, num_layers),
        "artifacts": {"trace": str(trace_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
