# hilda-vllm

## Benchmark

Performance Results:

Benchmark device: NVIDIA GeForce RTX 5090 (Blackwell), running with `torch 2.10.0+cu128`.

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
| --- | ---: | ---: | ---: |
| vllm | 133,966 | 12.79 | 10473.98 |
| nano-vllm | 133,966 | 13.75 | 9740.91 |
| hilda-vllm | 133,966 | 12.98 | 10317.81 |

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

| Inference Engine | KV Cache | Decode Share | Decode Throughput (tokens/s) | Total Throughput (tokens/s) |
| --- | --- | ---: | ---: | ---: |
| hilda-vllm | auto | 94.1% | 14592.54 | 14589.25 |
| hilda-vllm-fp8 | fp8 | 94.1% | 14827.95 | 14824.62 |

## Decode Jitter

Mixed long-prefill + short-decode benchmark:

| Inference Engine | Mixed Decode P50 (ms) | Mixed Decode P95 (ms) | Mixed Decode Max (ms) | Mixed Decode Steps |
| --- | ---: | ---: | ---: | ---: |
| hilda-vllm | 5.72 | 124.04 | 196.16 | 18 |
| hilda-vllm-fp8 | 4.82 | 123.33 | 199.45 | 18 |

## Commands

```bash
PYTHONPATH=. python benchmarks/bench.py --backend vllm --model /root/huggingface/Qwen3-0.6B --engine-name vllm
PYTHONPATH=. python benchmarks/bench.py --backend nano-vllm --model /root/huggingface/Qwen3-0.6B --engine-name nano-vllm
PYTHONPATH=. python benchmarks/bench.py --backend hilda-vllm --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm
PYTHONPATH=. python benchmarks/bench_decode_jitter.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm
PYTHONPATH=. python benchmarks/bench_decode_jitter.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm-fp8 --kv-cache-dtype fp8
PYTHONPATH=. python benchmarks/bench_decode_heavy.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm
PYTHONPATH=. python benchmarks/bench_decode_heavy.py --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm-fp8 --kv-cache-dtype fp8
```

## OpenAI API Server

Run a local OpenAI-compatible server on top of the local model:

```bash
python -m mini_vllm.server --model /root/huggingface/Qwen3-0.6B --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "你好"}]
  }'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "用一句话自我介绍"}],
    "stream": true
  }'
```

Current `stream=true` output is OpenAI-compatible SSE with real step-level incremental chunks. The first chunk still arrives after prompt prefill finishes, and each streaming request still holds the engine exclusively.

## Next

- Improve speculative decoding so draft/verify overhead can beat baseline on realistic model pairs.
- Reduce streaming exclusivity so streamed requests do not monopolize the engine.
