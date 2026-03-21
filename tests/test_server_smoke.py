import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

MODEL_PATH = Path('/root/huggingface/Qwen3-0.6B')
REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _wait_for_health(port: int, timeout_s: float = 90.0):
    deadline = time.time() + timeout_s
    url = f'http://127.0.0.1:{port}/health'
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                payload = json.loads(response.read().decode('utf-8'))
            if payload.get('status') == 'ok':
                return
        except URLError:
            time.sleep(1)
    raise TimeoutError(f'server did not become healthy within {timeout_s} seconds')


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode('utf-8'))
        return response.status, body


def _post_stream(url: str, payload: dict) -> tuple[int, list[str]]:
    request = Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(request, timeout=120) as response:
        lines = [line.decode('utf-8').strip() for line in response if line.strip()]
        return response.status, lines


def main():
    if not MODEL_PATH.is_dir():
        print(f'skip: model directory not found: {MODEL_PATH}')
        return 0

    port = _free_port()
    cmd = [
        sys.executable,
        '-m',
        'mini_vllm.server',
        '--model',
        str(MODEL_PATH),
        '--served-model-name',
        'Qwen3-0.6B',
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
        '--max-model-len',
        '512',
        '--max-num-batched-tokens',
        '1024',
        '--enforce-eager',
    ]
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _wait_for_health(port)
    try:
        with urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10) as response:
            models = json.loads(response.read().decode('utf-8'))
            assert response.status == 200
            assert models['data'][0]['id'] == 'Qwen3-0.6B'

        status, chat = _post_json(
            f'http://127.0.0.1:{port}/v1/chat/completions',
            {
                'model': 'Qwen3-0.6B',
                'messages': [{'role': 'user', 'content': 'Say hi in one short sentence.'}],
                'max_tokens': 8,
                'temperature': 0.1,
            },
        )
        assert status == 200
        assert chat['object'] == 'chat.completion'
        assert chat['choices'][0]['message']['content']

        status, lines = _post_stream(
            f'http://127.0.0.1:{port}/v1/chat/completions',
            {
                'model': 'Qwen3-0.6B',
                'messages': [{'role': 'user', 'content': 'Reply with one word.'}],
                'max_tokens': 4,
                'temperature': 0.1,
                'stream': True,
            },
        )
        assert status == 200
        assert lines[-1] == 'data: [DONE]'
        first_chunk = json.loads(lines[0][6:])
        assert first_chunk['object'] == 'chat.completion.chunk'
        print('server smoke test passed')
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        output = ''
        if process.stdout is not None:
            output = process.stdout.read()
        if process.returncode not in (0, -15):
            print(output)


if __name__ == '__main__':
    raise SystemExit(main())
