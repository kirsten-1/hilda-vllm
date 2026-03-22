import argparse
import asyncio
from statistics import mean
from time import perf_counter

import aiohttp


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark concurrent OpenAI-compatible server throughput")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


async def send_request(session: aiohttp.ClientSession, url: str, payload: dict) -> tuple[dict, float]:
    t0 = perf_counter()
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        body = await response.json()
    return body, perf_counter() - t0


async def run_benchmark(args) -> dict:
    prompts = [f"Count from 1 to {index} in short form." for index in range(1, args.num_requests + 1)]
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            send_request(
                session,
                args.url,
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                },
            )
            for prompt in prompts
        ]
        t0 = perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = perf_counter() - t0

    payloads = [payload for payload, _ in results]
    latencies = [elapsed for _, elapsed in results]
    total_completion_tokens = sum(payload["usage"]["completion_tokens"] for payload in payloads)
    total_prompt_tokens = sum(payload["usage"]["prompt_tokens"] for payload in payloads)
    return {
        "num_requests": args.num_requests,
        "total_time_s": total_time,
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "throughput_toks": total_completion_tokens / total_time if total_time else 0.0,
        "mean_latency_s": mean(latencies) if latencies else 0.0,
        "max_latency_s": max(latencies) if latencies else 0.0,
    }


def print_summary(result: dict):
    print("Concurrent Server Benchmark:")
    print(f"Requests: {result['num_requests']}")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Completion tokens: {result['completion_tokens']}")
    print(f"Total time (s): {result['total_time_s']:.2f}")
    print(f"Completion throughput (tok/s): {result['throughput_toks']:.2f}")
    print(f"Mean request latency (s): {result['mean_latency_s']:.2f}")
    print(f"Max request latency (s): {result['max_latency_s']:.2f}")


def main():
    result = asyncio.run(run_benchmark(parse_args()))
    print_summary(result)


if __name__ == "__main__":
    main()
