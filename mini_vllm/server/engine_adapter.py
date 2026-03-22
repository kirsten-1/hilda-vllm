import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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


_STREAM_SENTINEL = object()


@dataclass
class _QueuedRequest:
    loop: asyncio.AbstractEventLoop
    prompts: list[str]
    sampling_params: SamplingParams
    stream: bool
    future: asyncio.Future | None = None
    queue: asyncio.Queue | None = None
    outputs: dict[int, dict[str, Any]] = field(default_factory=dict)
    remaining: int = 0


@dataclass
class _ActiveSequence:
    request: _QueuedRequest
    seq: Any
    request_index: int
    emitted_tokens: int = 0
    finish_event_sent: bool = False


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
        self._supports_continuous_batching = all(hasattr(self.llm, attr) for attr in ("add_request", "step"))
        self._closed = False
        self._engine_error: Exception | None = None

        if self._supports_continuous_batching:
            self._worker_condition = threading.Condition()
            self._pending_requests: list[_QueuedRequest] = []
            self._active_sequences: dict[int, _ActiveSequence] = {}
            self._stop_worker = False
            self._worker = threading.Thread(target=self._engine_loop, daemon=True)
            self._worker.start()
        else:
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
        if self._closed:
            return
        self._closed = True
        if self._supports_continuous_batching:
            with self._worker_condition:
                self._stop_worker = True
                self._worker_condition.notify_all()
            self._worker.join()
        if hasattr(self.llm, "exit"):
            self.llm.exit()

    async def _generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[dict[str, Any]]:
        if not self._supports_continuous_batching:
            return await self._generate_legacy(prompts, sampling_params)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = _QueuedRequest(
            loop=loop,
            prompts=list(prompts),
            sampling_params=sampling_params,
            stream=False,
            future=future,
            remaining=len(prompts),
        )
        self._submit_request(request)
        return await future

    async def _generate_legacy(self, prompts: list[str], sampling_params: SamplingParams) -> list[dict[str, Any]]:
        async with self._lock:
            return await asyncio.to_thread(
                self.llm.generate,
                prompts,
                sampling_params,
                False,
            )

    async def _generate_stream(self, prompts: list[str], sampling_params: SamplingParams) -> AsyncIterator[dict[str, Any]]:
        if not self._supports_continuous_batching:
            async for item in self._generate_stream_legacy(prompts, sampling_params):
                yield item
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        request = _QueuedRequest(
            loop=loop,
            prompts=list(prompts),
            sampling_params=sampling_params,
            stream=True,
            queue=queue,
            remaining=len(prompts),
        )
        self._submit_request(request)
        while True:
            item = await queue.get()
            if item is _STREAM_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    async def _generate_stream_legacy(self, prompts: list[str], sampling_params: SamplingParams) -> AsyncIterator[dict[str, Any]]:
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

    def _submit_request(self, request: _QueuedRequest):
        if self._engine_error is not None:
            raise RuntimeError("Engine loop is unavailable") from self._engine_error
        with self._worker_condition:
            if self._closed:
                raise RuntimeError("Engine adapter is closed")
            self._pending_requests.append(request)
            self._worker_condition.notify_all()

    def _engine_loop(self):
        while True:
            with self._worker_condition:
                while not self._pending_requests and not self._active_sequences and not self._stop_worker:
                    self._worker_condition.wait()
                if self._stop_worker and not self._pending_requests and not self._active_sequences:
                    return
                pending_requests = self._pending_requests
                self._pending_requests = []

            for request in pending_requests:
                self._activate_request(request)

            if not self._active_sequences:
                continue

            try:
                self.llm.step()
            except Exception as exc:  # pragma: no cover - threaded engine failures are surfaced asynchronously
                self._engine_error = exc
                self._fail_active_requests(exc)
                continue

            self._publish_step_updates()

    def _activate_request(self, request: _QueuedRequest):
        try:
            new_entries = []
            for request_index, prompt in enumerate(request.prompts):
                seq = self.llm.add_request(prompt, request.sampling_params)
                new_entries.append(_ActiveSequence(request=request, seq=seq, request_index=request_index))
            for entry in new_entries:
                self._active_sequences[entry.seq.seq_id] = entry
        except Exception as exc:
            self._fail_request(request, exc)

    def _publish_step_updates(self):
        completed_requests: list[_QueuedRequest] = []
        for seq_id, entry in list(self._active_sequences.items()):
            completion_token_ids = list(entry.seq.completion_token_ids)
            new_token_ids = completion_token_ids[entry.emitted_tokens:]
            if entry.request.stream and (new_token_ids or (entry.seq.is_finished and not entry.finish_event_sent)):
                self._push_stream_item(
                    entry.request,
                    {
                        "request_index": entry.request_index,
                        "seq_id": entry.seq.seq_id,
                        "token_ids": list(new_token_ids),
                        "completion_token_ids": completion_token_ids,
                        "is_finished": entry.seq.is_finished,
                    },
                )
                entry.finish_event_sent = entry.seq.is_finished
            entry.emitted_tokens = len(completion_token_ids)

            if not entry.seq.is_finished:
                continue

            entry.request.outputs[entry.request_index] = {
                "text": self.llm.tokenizer.decode(completion_token_ids),
                "token_ids": completion_token_ids,
            }
            entry.request.remaining -= 1
            del self._active_sequences[seq_id]
            if entry.request.remaining == 0:
                completed_requests.append(entry.request)

        for request in completed_requests:
            self._complete_request(request)

    def _fail_active_requests(self, exc: Exception):
        requests = []
        seen = set()
        for entry in self._active_sequences.values():
            request_id = id(entry.request)
            if request_id in seen:
                continue
            seen.add(request_id)
            requests.append(entry.request)
        self._active_sequences.clear()
        for request in requests:
            self._fail_request(request, exc)

    def _complete_request(self, request: _QueuedRequest):
        if request.stream:
            self._push_stream_item(request, _STREAM_SENTINEL)
            return

        outputs = [request.outputs[i] for i in range(len(request.prompts))]
        request.loop.call_soon_threadsafe(self._resolve_future, request.future, outputs)

    def _fail_request(self, request: _QueuedRequest, exc: Exception):
        if request.stream:
            self._push_stream_item(request, exc)
            self._push_stream_item(request, _STREAM_SENTINEL)
            return
        request.loop.call_soon_threadsafe(self._reject_future, request.future, exc)

    @staticmethod
    def _resolve_future(future: asyncio.Future | None, value: Any):
        if future is None or future.done() or future.cancelled():
            return
        future.set_result(value)

    @staticmethod
    def _reject_future(future: asyncio.Future | None, exc: Exception):
        if future is None or future.done() or future.cancelled():
            return
        future.set_exception(exc)

    @staticmethod
    def _push_stream_item(request: _QueuedRequest, item: Any):
        if request.queue is None:
            return
        request.loop.call_soon_threadsafe(request.queue.put_nowait, item)

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
