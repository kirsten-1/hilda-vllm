# hilda-vllm

## Benchmark

Performance Results:

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

## Next

Start with Chunked Prefill.

- Changes stay mostly in scheduler + model_runner.
- It is easy to validate with long-prompt decode latency.
- It is a good base for later work like speculative decoding.
