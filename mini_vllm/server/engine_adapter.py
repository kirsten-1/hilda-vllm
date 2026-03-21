import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
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
            self._sampling_params(request.temperature, request.max_tokens),
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
        outputs = await self._generate(
            prompts,
            self._sampling_params(request.temperature, request.max_tokens),
        )
        created = int(time.time())
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        for i, output in enumerate(outputs):
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_name,
                "choices": [{"index": i, "text": output["text"], "finish_reason": None}],
            }
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": i,
                        "text": "",
                        "finish_reason": self._finish_reason(output["token_ids"], request.max_tokens),
                    }
                ],
            }

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._validate_request(request.model, request.n)
        prompt = self._build_chat_prompt(request.messages)
        output = (
            await self._generate(
                [prompt],
                self._sampling_params(request.temperature, request.max_tokens),
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
        output = (
            await self._generate(
                [prompt],
                self._sampling_params(request.temperature, request.max_tokens),
            )
        )[0]
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        if output["text"]:
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [{"index": 0, "delta": {"content": output["text"]}, "finish_reason": None}],
            }
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": self._finish_reason(output["token_ids"], request.max_tokens),
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

    @staticmethod
    def _finish_reason(token_ids: list[int], max_tokens: int) -> str:
        return "length" if len(token_ids) >= max_tokens else "stop"

    def _validate_request(self, model: str, n: int):
        if model != self.model_name:
            raise UnsupportedRequestError(f"Unknown model: {model}")
        if n != 1:
            raise UnsupportedRequestError("Only n=1 is supported")

    @staticmethod
    def _sampling_params(temperature: float, max_tokens: int) -> SamplingParams:
        # mini_vllm currently rejects exact zero temperature, but many OpenAI clients send it.
        safe_temperature = max(temperature, 1e-5)
        return SamplingParams(temperature=safe_temperature, max_tokens=max_tokens)
