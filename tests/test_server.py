import json

from fastapi.testclient import TestClient

from mini_vllm.server.app import create_app
from mini_vllm.server.engine_adapter import EngineAdapter


class FakeTokenizer:
    chat_template = "fake"

    def encode(self, text):
        return [ord(ch) for ch in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        parts = [f"{message['role']}: {message['content']}" for message in messages]
        parts.append("assistant:")
        return "\n".join(parts)


class FakeLLM:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [
            {
                "text": f"reply:{index}",
                "token_ids": [10 + index, 20 + index],
            }
            for index, _prompt in enumerate(prompts)
        ]

    def exit(self):
        return None


def make_client():
    adapter = EngineAdapter(llm=FakeLLM(), model_name="Qwen3-0.6B")
    return TestClient(create_app(adapter))


def _collect_sse_lines(response):
    lines = []
    for line in response.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        lines.append(line)
    return lines


def test_models_endpoint_returns_served_model():
    client = make_client()

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "Qwen3-0.6B"


def test_completions_endpoint_returns_openai_shape():
    client = make_client()

    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen3-0.6B",
            "prompt": ["hi", "yo"],
            "max_tokens": 4,
            "temperature": 0.7,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "text_completion"
    assert [choice["text"] for choice in payload["choices"]] == ["reply:0", "reply:1"]
    assert payload["usage"]["completion_tokens"] == 4
    assert payload["usage"]["prompt_tokens"] == 4


def test_chat_completions_endpoint_uses_chat_prompt_and_returns_message():
    client = make_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen3-0.6B",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == "reply:0"


def test_chat_completions_stream_returns_sse_chunks():
    client = make_client()

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "Qwen3-0.6B",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    ) as response:
        lines = _collect_sse_lines(response)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0][6:])
    second = json.loads(lines[1][6:])
    third = json.loads(lines[2][6:])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert second["choices"][0]["delta"]["content"] == "reply:0"
    assert third["choices"][0]["finish_reason"] == "stop"


def test_streaming_unknown_model_returns_bad_request():
    client = make_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "other-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["detail"]["message"] == "Unknown model: other-model"
