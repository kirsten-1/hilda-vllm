import argparse
import asyncio
import json
from statistics import mean
from time import perf_counter
from typing import Any

import aiohttp


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark concurrent OpenAI-compatible server throughput")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--stream", action="store_true", help="Use SSE streaming and report TTFT")
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer path for estimating completion tokens in streaming mode",
    )
    return parser.parse_args()


async def _iter_sse_events(response: aiohttp.ClientResponse):
    buffer = ""
    async for chunk in response.content.iter_any():
        buffer += chunk.decode("utf-8")
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = [line[6:] for line in raw_event.splitlines() if line.startswith("data: ")]
            if not data_lines:
                continue
            payload = "\n".join(data_lines)
            if payload == "[DONE]":
                yield payload
                continue
            yield json.loads(payload)


async def send_request_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    t0 = perf_counter()
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        body = await response.json()
    return body, perf_counter() - t0


async def send_request_stream(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    t0 = perf_counter()
    first_token_s = None
    text_parts: list[str] = []
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        async for event in _iter_sse_events(response):
            if event == "[DONE]":
                break
            choice = event["choices"][0]
            delta_text = choice.get("delta", {}).get("content")
            if delta_text is None:
                delta_text = choice.get("text", "")
            if delta_text:
                if first_token_s is None:
                    first_token_s = perf_counter() - t0
                text_parts.append(delta_text)
    latency_s = perf_counter() - t0
    return {
        "text": "".join(text_parts),
        "ttft_s": first_token_s if first_token_s is not None else latency_s,
    }, latency_s


def _estimate_stream_tokens(texts: list[str], tokenizer_path: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)


async def run_benchmark(args) -> dict[str, Any]:
    prompts = [f"Count from 1 to {index} in short form." for index in range(1, args.num_requests + 1)]
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for prompt in prompts:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
            }
            if args.stream:
                payload["stream"] = True
                tasks.append(send_request_stream(session, args.url, payload))
            else:
                tasks.append(send_request_json(session, args.url, payload))
        t0 = perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = perf_counter() - t0

    payloads = [payload for payload, _ in results]
    latencies = [elapsed for _, elapsed in results]
    result: dict[str, Any] = {
        "mode": "stream" if args.stream else "non-stream",
        "num_requests": args.num_requests,
        "total_time_s": total_time,
        "mean_latency_s": mean(latencies) if latencies else 0.0,
        "max_latency_s": max(latencies) if latencies else 0.0,
    }

    if args.stream:
        ttfts = [payload["ttft_s"] for payload in payloads]
        texts = [payload["text"] for payload in payloads]
        result.update(
            {
                "output_chars": sum(len(text) for text in texts),
                "mean_ttft_s": mean(ttfts) if ttfts else 0.0,
                "max_ttft_s": max(ttfts) if ttfts else 0.0,
            }
        )
        if args.tokenizer:
            completion_tokens = _estimate_stream_tokens(texts, args.tokenizer)
            result["completion_tokens"] = completion_tokens
            result["throughput_toks"] = completion_tokens / total_time if total_time else 0.0
        return result

    total_completion_tokens = sum(payload["usage"]["completion_tokens"] for payload in payloads)
    total_prompt_tokens = sum(payload["usage"]["prompt_tokens"] for payload in payloads)
    result.update(
        {
            "completion_tokens": total_completion_tokens,
            "prompt_tokens": total_prompt_tokens,
            "throughput_toks": total_completion_tokens / total_time if total_time else 0.0,
        }
    )
    return result


def print_summary(result: dict[str, Any]):
    print("Concurrent Server Benchmark:")
    print(f"Mode: {result['mode']}")
    print(f"Requests: {result['num_requests']}")
    print(f"Total time (s): {result['total_time_s']:.2f}")
    print(f"Mean request latency (s): {result['mean_latency_s']:.2f}")
    print(f"Max request latency (s): {result['max_latency_s']:.2f}")
    if result["mode"] == "stream":
        print(f"Mean TTFT (s): {result['mean_ttft_s']:.2f}")
        print(f"Max TTFT (s): {result['max_ttft_s']:.2f}")
        print(f"Output chars: {result['output_chars']}")
        if "completion_tokens" in result:
            print(f"Estimated completion tokens: {result['completion_tokens']}")
            print(f"Estimated completion throughput (tok/s): {result['throughput_toks']:.2f}")
        else:
            print("Estimated completion tokens: unavailable (pass --tokenizer to enable)")
    else:
        print(f"Prompt tokens: {result['prompt_tokens']}")
        print(f"Completion tokens: {result['completion_tokens']}")
        print(f"Completion throughput (tok/s): {result['throughput_toks']:.2f}")


def main():
    result = asyncio.run(run_benchmark(parse_args()))
    print_summary(result)


if __name__ == "__main__":
    main()
