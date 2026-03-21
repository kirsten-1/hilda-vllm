"""Smoke test for speculative decoding."""
import sys
sys.path.insert(0, "/root/hilda-vllm/hilda-vllm")

from mini_vllm import LLM, SamplingParams

print("=== Smoke Test: Speculative Decoding ===")
print()

# Test 1: Baseline (no spec decode)
print("[1] Loading baseline engine (Qwen3-0.6B, no spec decode)...")
try:
    llm_base = LLM("/root/huggingface/Qwen3-0.6B", enforce_eager=True)
    prompts = ["Hello, how are you?", "What is 2+2?", "Tell me a joke."]
    sp = SamplingParams(temperature=1.0, max_tokens=32, ignore_eos=True)
    outputs_base = llm_base.generate(prompts, sp, use_tqdm=False)
    print("Baseline outputs:")
    for i, out in enumerate(outputs_base):
        print(f"  [{i}] {out['text'][:80]}...")
    print("Baseline OK\n")
    llm_base.exit()
    del llm_base
except Exception as e:
    print(f"Baseline FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

import torch, gc
gc.collect()
torch.cuda.empty_cache()

# Test 2: Speculative decoding
print("[2] Loading spec decode engine (target=Qwen3-0.6B, draft=Qwen2.5-0.5B-Instruct)...")
try:
    llm_spec = LLM(
        "/root/huggingface/Qwen3-0.6B",
        spec_decode_model="/root/huggingface/Qwen2.5-0.5B-Instruct",
        spec_decode_gamma=5,
        enforce_eager=True,
    )
    outputs_spec = llm_spec.generate(prompts, sp, use_tqdm=False)
    print("Spec decode outputs:")
    for i, out in enumerate(outputs_spec):
        print(f"  [{i}] {out['text'][:80]}...")
    print("Spec decode OK\n")
    llm_spec.exit()
    del llm_spec
except Exception as e:
    print(f"Spec decode FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("=== All smoke tests passed ===")
