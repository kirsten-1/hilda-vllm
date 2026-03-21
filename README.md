# hilda-vllm

## Benchmark

Performance Results:

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
| --- | ---: | ---: | ---: |
| vllm | 133,966 | 12.79 | 10473.98 |
| nano-vllm | 133,966 | 13.75 | 9740.91 |
| hilda-vllm | 133,966 | 13.09 | 10231.27 |

## Commands

```bash
PYTHONPATH=. python benchmarks/bench.py --backend vllm --model /root/huggingface/Qwen3-0.6B --engine-name vllm
PYTHONPATH=. python benchmarks/bench.py --backend nano-vllm --model /root/huggingface/Qwen3-0.6B --engine-name nano-vllm
PYTHONPATH=. python benchmarks/bench.py --backend hilda-vllm --model /root/huggingface/Qwen3-0.6B --engine-name hilda-vllm
```

## Next

Start with Chunked Prefill.

- Changes stay mostly in scheduler + model_runner.
- It is easy to validate with long-prompt decode latency.
- It is a good base for later work like speculative decoding.
