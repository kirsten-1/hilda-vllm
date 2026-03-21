from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "hilda-vllm"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str | list[TextContentPart]


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    prompt: str | list[str]
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    n: int = 1


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length"]


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    n: int = 1


class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
