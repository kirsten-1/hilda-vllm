import argparse
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from mini_vllm.server.engine_adapter import EngineAdapter, UnsupportedRequestError
from mini_vllm.server.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorDetail,
    ErrorResponse,
)


def create_app(adapter: EngineAdapter) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.adapter = adapter
        try:
            yield
        finally:
            adapter.close()

    app = FastAPI(title="hilda-vllm OpenAI-Compatible Server", lifespan=lifespan)
    app.state.adapter = adapter

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        return await request.app.state.adapter.list_models()

    @app.post("/v1/completions")
    async def create_completion(payload: CompletionRequest, request: Request):
        try:
            return await request.app.state.adapter.create_completion(payload)
        except UnsupportedRequestError as exc:
            raise _bad_request(str(exc)) from exc

    @app.post("/v1/chat/completions")
    async def create_chat_completion(payload: ChatCompletionRequest, request: Request):
        try:
            return await request.app.state.adapter.create_chat_completion(payload)
        except UnsupportedRequestError as exc:
            raise _bad_request(str(exc)) from exc

    return app


def _bad_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=ErrorResponse(error=ErrorDetail(message=message)).model_dump()["error"],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = EngineAdapter(
        args.model,
        model_name=args.served_model_name,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    uvicorn.run(
        create_app(adapter),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
