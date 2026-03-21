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
```

`stream=true` produces OpenAI-compatible SSE with real step-level incremental chunks. The first chunk arrives after prompt prefill finishes. Each streaming request currently holds the engine exclusively.

## Next

- Reduce streaming exclusivity so streamed requests do not monopolize the engine.
- Batch=1 latency benchmark for speculative decoding to demonstrate the target use case.
- Pre-allocated tensor buffers for persistent batching to close the remaining ~6.5% throughput gap.
