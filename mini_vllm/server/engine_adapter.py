import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
import threading
from typing import Any

from mini_vllm import LLM, SamplingParams
from mini_vllm.server.protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelList,
    UsageInfo,
)


class UnsupportedRequestError(ValueError):
    pass


class EngineAdapter:
    def __init__(
        self,
        model: str | None = None,
        *,
        llm: LLM | None = None,
        model_name: str | None = None,
        **llm_kwargs: Any,
    ):
        if llm is None and model is None:
            raise ValueError("Either model or llm must be provided")
        self.llm = llm or LLM(model, **llm_kwargs)
        self.model_name = model_name or (Path(model).name if model else "mini_vllm")
        self._lock = asyncio.Lock()

    async def list_models(self) -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=self.model_name,
                    created=int(time.time()),
                )
            ]
        )

    async def create_completion(self, request: CompletionRequest) -> CompletionResponse:
        self._validate_request(request.model, request.n)
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        outputs = await self._generate(
            prompts,
            self._sampling_params(request.temperature, request.max_tokens, request.top_k, request.top_p),
        )
        usage = self._build_usage(prompts, outputs)
        choices = [
            CompletionChoice(
                index=i,
                text=output["text"],
                finish_reason=self._finish_reason(output["token_ids"], request.max_tokens),
            )
            for i, output in enumerate(outputs)
        ]
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=self.model_name,
            choices=choices,
            usage=usage,
        )

    async def stream_completion(self, request: CompletionRequest) -> AsyncIterator[dict[str, Any]]:
        self._validate_request(request.model, request.n)
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        sampling_params = self._sampling_params(request.temperature, request.max_tokens, request.top_k, request.top_p)
        created = int(time.time())
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        states = [self._new_text_state() for _ in prompts]
        async for event in self._generate_stream(prompts, sampling_params):
            request_index = event["request_index"]
            delta_text = self._append_text_delta(states[request_index], event["token_ids"])
            if delta_text:
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": self.model_name,
                    "choices": [{"index": request_index, "text": delta_text, "finish_reason": None}],
                }
            if event["is_finished"]:
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": request_index,
                            "text": "",
                            "finish_reason": self._finish_reason(states[request_index]["token_ids"], request.max_tokens),
                        }
                    ],
                }

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._validate_request(request.model, request.n)
        prompt = self._build_chat_prompt(request.messages)
        output = (
            await self._generate(
                [prompt],
                self._sampling_params(request.temperature, request.max_tokens, request.top_k, request.top_p),
            )
        )[0]
        usage = self._build_usage([prompt], [output])
        choice = ChatCompletionChoice(
            index=0,
            message=ChatCompletionResponseMessage(content=output["text"]),
            finish_reason=self._finish_reason(output["token_ids"], request.max_tokens),
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=self.model_name,
            choices=[choice],
            usage=usage,
        )

    async def stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncIterator[dict[str, Any]]:
        self._validate_request(request.model, request.n)
        prompt = self._build_chat_prompt(request.messages)
        sampling_params = self._sampling_params(request.temperature, request.max_tokens, request.top_k, request.top_p)
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        state = self._new_text_state()
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        async for event in self._generate_stream([prompt], sampling_params):
            delta_text = self._append_text_delta(state, event["token_ids"])
            if delta_text:
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                }
            if event["is_finished"]:
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": self._finish_reason(state["token_ids"], request.max_tokens),
                        }
                    ],
                }

    def close(self):
        if hasattr(self.llm, "exit"):
            self.llm.exit()

    async def _generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(
                self.llm.generate,
                prompts,
                sampling_params,
                False,
            )

    async def _generate_stream(self, prompts: list[str], sampling_params: SamplingParams) -> AsyncIterator[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def push(item):
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker():
            try:
                for event in self.llm.generate_stream(prompts, sampling_params):
                    push(event)
            except Exception as exc:  # pragma: no cover - surfaced in async consumer
                push(exc)
            finally:
                push(sentinel)

        async with self._lock:
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                await asyncio.to_thread(thread.join)

    def _build_chat_prompt(self, messages) -> str:
        rendered_messages = [
            {
                "role": message.role,
                "content": self._flatten_message_content(message.content),
            }
            for message in messages
        ]
        tokenizer = self.llm.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_lines = [f"{message['role']}: {message['content']}" for message in rendered_messages]
        prompt_lines.append("assistant:")
        return "\n".join(prompt_lines)

    @staticmethod
    def _flatten_message_content(content: str | list[Any]) -> str:
        if isinstance(content, str):
            return content
        return "".join(part.text for part in content)

    def _build_usage(self, prompts: list[str], outputs: list[dict[str, Any]]) -> UsageInfo:
        prompt_tokens = sum(len(self.llm.tokenizer.encode(prompt)) for prompt in prompts)
        completion_tokens = sum(len(output["token_ids"]) for output in outputs)
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _new_text_state(self) -> dict[str, Any]:
        return {"token_ids": [], "text": ""}

    def _append_text_delta(self, state: dict[str, Any], token_ids: list[int]) -> str:
        state["token_ids"].extend(token_ids)
        full_text = self.llm.tokenizer.decode(state["token_ids"])
        previous_text = state["text"]
        delta_text = full_text[len(previous_text):] if full_text.startswith(previous_text) else full_text
        state["text"] = full_text
        return delta_text

    @staticmethod
    def _finish_reason(token_ids: list[int], max_tokens: int) -> str:
        return "length" if len(token_ids) >= max_tokens else "stop"

    def _validate_request(self, model: str, n: int):
        if model != self.model_name:
            raise UnsupportedRequestError(f"Unknown model: {model}")
        if n != 1:
            raise UnsupportedRequestError("Only n=1 is supported")

    @staticmethod
    def _sampling_params(temperature: float, max_tokens: int, top_k: int, top_p: float) -> SamplingParams:
        # mini_vllm currently rejects exact zero temperature, but many OpenAI clients send it.
        safe_temperature = max(temperature, 1e-5)
        return SamplingParams(
            temperature=safe_temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
        )
