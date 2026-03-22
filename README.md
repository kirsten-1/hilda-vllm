# hilda-vllm

A minimal high-performance LLM inference engine built from scratch, inspired by [nano-vllm](https://github.com/niconielsen32/nano-vllm).

## Features

- PagedAttention with hash-based prefix caching
- Chunked prefill with adaptive chunk sizing
- Persistent batching with dense slot compaction
- FP8 KV cache (BF16 prefill + FP8 decode)
- Speculative decoding with adaptive gamma
- Top-k / Top-p (nucleus) sampling
- OpenAI-compatible API server with SSE streaming
- CUDA graph capture/replay
- Tensor parallelism

## Benchmark

Benchmark device: NVIDIA GeForce RTX 5090 (Blackwell), `torch 2.10.0+cu128`.

Model: Qwen3-0.6B, 256 requests, random prompts (50-512 tokens), max output 1024 tokens.

| Inference Engine | Output Tokens | Time (s) | Throughput (tok/s) |
| --- | ---: | ---: | ---: |
| vllm | 133,966 | 12.79 | 10,474 |
| nano-vllm | 133,966 | 13.75 | 9,741 |
| hilda-vllm | 133,966 | 12.98 | 10,318 |
| hilda-vllm + persistent batching | 133,966 | 14.21 | 9,425 |

> Persistent batching adds ~6.5% overhead from per-step tensor reconstruction for padded slots. The trade-off enables stable decode slot management needed for speculative decoding and preemption.

## FP8 KV Cache

`--kv-cache-dtype fp8` is experimental and opt-in.

Recommended usage:

- Decode-heavy workloads with short prompts and long outputs.
- Latency-first experiments where decode p50/p95 matters more than mixed prefill throughput.
- Longer-context experiments where the smaller KV cache footprint is useful.

Current caveat:

- Prompt-heavy or mixed workloads can still regress total throughput versus the default BF16 KV cache path.

## Decode-heavy Benchmark

Short-prompt + long-output workload where decode dominates (>90% of generated-step tokens):

| Inference Engine | KV Cache | Decode Share | Decode Throughput (tok/s) | Total Throughput (tok/s) |
| --- | --- | ---: | ---: | ---: |
| hilda-vllm | auto | 94.1% | 14,593 | 14,589 |
| hilda-vllm-fp8 | fp8 | 94.1% | 14,828 | 14,825 |

## Decode Jitter

Mixed long-prefill + short-decode benchmark:

| Inference Engine | Mixed Decode P50 (ms) | Mixed Decode P95 (ms) | Mixed Decode Max (ms) | Mixed Decode Steps |
| --- | ---: | ---: | ---: | ---: |
| hilda-vllm | 5.72 | 124.04 | 196.16 | 18 |
| hilda-vllm-fp8 | 4.82 | 123.33 | 199.45 | 18 |

## Speculative Decoding

Target: Qwen3-8B, Draft: Qwen3-0.6B, gamma=3, 8 requests.

| Mode | Output Tokens | Time (s) | Throughput (tok/s) | Acceptance Rate |
| --- | ---: | ---: | ---: | ---: |
| Baseline (Qwen3-8B) | 435 | 1.99 | 218.89 | - |
| Spec Decode (gamma=3) | 435 | 3.52 | 123.56 | 33.7% |

> Speculative decoding is a latency optimization for low-batch scenarios (batch=1). At batch=8 the draft/verify overhead exceeds the savings from accepted tokens. The 33.7% acceptance rate reflects the vocabulary distribution gap between Qwen3-8B and Qwen3-0.6B.
>
> Batch=1 is the intended use case. On Qwen3-8B + Qwen3-0.6B with gamma=3 and seed=0, baseline decode measured TTFT 31.32 ms / TPOT 10.81 ms / 91.22 tok/s, while speculative decoding measured TTFT 53.51 ms / TPOT 9.14 ms / 105.41 tok/s at 65.5% acceptance.
>
> Performance is highly acceptance-rate sensitive. Low-acceptance prompts still regress, so speculative decoding remains experimental rather than a default production path.
>
> The verify-MLP Triton skinny-GEMM experiment is disabled by default. On the same 8B workload it regressed verify forward time versus the default CUTLASS/cuBLAS path.

## Commands

```bash
# Main throughput benchmark
PYTHONPATH=. python benchmarks/bench.py --backend vllm --model /root/huggingface/Qwen3-0.6B --engine-name vllm
PYTHONPATH=. python benchmarks/bench.py --backend nano-vllm --model /root/huggingface/Qwen3-0.6B --engine-name nano-vllm
PYTHONPATH=. python benchmarks/bench.py --backend hilda-vllm --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm

# Decode jitter
PYTHONPATH=. python benchmarks/bench_decode_jitter.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm
PYTHONPATH=. python benchmarks/bench_decode_jitter.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm-fp8 --kv-cache-dtype fp8

# Decode-heavy
PYTHONPATH=. python benchmarks/bench_decode_heavy.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm
PYTHONPATH=. python benchmarks/bench_decode_heavy.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm-fp8 --kv-cache-dtype fp8

# Speculative decoding
PYTHONPATH=. python benchmarks/bench_spec_decode.py --target-model /root/autodl-tmp/Qwen3-8B --draft-model /root/huggingface/Qwen3-0.6B --gamma 3

# Batch=1 speculative decoding latency
PYTHONPATH=. python benchmarks/bench_spec_decode_batch1.py --target-model /root/autodl-tmp --draft-model /root/huggingface/Qwen3-0.6B --gamma 3 --seed 0

# Re-enable the experimental verify-MLP Triton path for kernel experiments
MINI_VLLM_ENABLE_VERIFY_MLP_TRITON=1 PYTHONPATH=. python benchmarks/bench_spec_decode_batch1.py --target-model /root/autodl-tmp --draft-model /root/huggingface/Qwen3-0.6B --gamma 3 --seed 0
```

## OpenAI API Server

Run a local OpenAI-compatible server on top of the local model:

```bash
python -m mini_vllm.server --model /root/huggingface/Qwen3-0.6B --port 8000

# Non-streaming
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "你好"}]
  }'

# SSE streaming
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "用一句话自我介绍"}],
    "stream": true
  }'

# With speculative decoding
python -m mini_vllm.server --model /root/autodl-tmp/Qwen3-8B --spec-decode-model /root/huggingface/Qwen3-0.6B --spec-decode-gamma 3 --port 8000

# Concurrent server benchmark (non-streaming)
PYTHONPATH=. python benchmarks/bench_server_concurrent.py --url http://127.0.0.1:8000/v1/chat/completions --model Qwen3-0.6B --num-requests 32 --max-tokens 256

# Concurrent server benchmark (streaming + TTFT)
PYTHONPATH=. python benchmarks/bench_server_concurrent.py --url http://127.0.0.1:8000/v1/chat/completions --model Qwen3-0.6B --num-requests 32 --max-tokens 256 --stream --tokenizer /root/huggingface/Qwen3-0.6B
```

`stream=true` produces OpenAI-compatible SSE with real step-level incremental chunks. The first chunk arrives after prompt prefill finishes. When continuous batching is available, streamed and non-streamed requests share the same background engine loop and can be merged into the same decode steps.

## Next

- Publish representative server concurrency results for both non-streaming throughput and streaming TTFT.
- Improve speculative decoding heuristics for low-acceptance prompts instead of relying on a single fixed gamma.
- Revisit verify forward kernels only with a new implementation that beats the default GEMM path in 8B end-to-end tests.
