import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional

from capture_core import RequestCapture, initialize_capture_root
from protocol_converter import (
    ConversionError,
    OpenAIStreamConverter,
    anthropic_error,
    anthropic_error_bytes,
    anthropic_message_to_sse,
    anthropic_to_openai_request,
    compact_json,
    openai_error_to_anthropic,
    openai_to_anthropic_message,
    sse_event,
)


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
REQUEST_HEADERS_TO_DROP = HOP_BY_HOP_HEADERS | {
    "accept",
    "accept-encoding",
    "authorization",
    "content-encoding",
    "content-length",
    "content-type",
    "host",
    "user-agent",
    "x-api-key",
    "x-app",
}
RESPONSE_HEADERS_TO_DROP = HOP_BY_HOP_HEADERS | {
    "content-encoding",
    "content-length",
    "content-type",
}


@dataclass(frozen=True)
class Settings:
    listen_host: str
    listen_port: int
    upstream_url: str
    log_dir: Path
    timeout_seconds: float
    upstream_model: Optional[str] = None
    upstream_api_key: Optional[str] = None
    reasoning_mode: str = "preserve"
    reasoning_field: str = "reasoning_content"
    token_limit_field: str = "max_tokens"
    include_stream_usage: bool = True
    map_reasoning_effort: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)


def filtered_headers(
    headers: Iterable[tuple[str, str]], drop: set[str]
) -> list[tuple[str, str]]:
    return [(key, value) for key, value in headers if key.lower() not in drop]


def request_header_items(request: Any) -> list[tuple[str, str]]:
    return [
        (key.decode("latin-1"), value.decode("latin-1"))
        for key, value in request.headers.raw
    ]


def compact_json_bytes(value: Any) -> bytes:
    return compact_json(value).encode("utf-8")


def apply_raw_response_headers(response: Any, headers: list[tuple[str, str]]) -> Any:
    response.raw_headers = [
        (key.encode("latin-1"), value.encode("latin-1")) for key, value in headers
    ]
    return response


def _is_client_only_header(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("anthropic-")
        or lowered.startswith("x-stainless-")
        or lowered.startswith("x-claude-")
    )


def build_openai_headers(
    incoming_headers: Iterable[tuple[str, str]],
    settings: Settings,
    *,
    stream: bool,
) -> list[tuple[str, str]]:
    headers = [
        (key, value)
        for key, value in incoming_headers
        if key.lower() not in REQUEST_HEADERS_TO_DROP
        and not _is_client_only_header(key)
    ]
    headers.extend(
        [
            ("content-type", "application/json"),
            ("accept", "text/event-stream" if stream else "application/json"),
            ("user-agent", "openai-full-capture-proxy/1"),
        ]
    )
    if settings.upstream_api_key:
        headers.append(("authorization", f"Bearer {settings.upstream_api_key}"))
    for key, value in settings.extra_headers.items():
        lowered = key.lower()
        headers = [(name, item) for name, item in headers if name.lower() != lowered]
        headers.append((key, value))
    return headers


def converted_response_headers(
    upstream_headers: Iterable[tuple[str, str]],
    *,
    content_type: str,
    content_length: Optional[int] = None,
) -> list[tuple[str, str]]:
    headers = filtered_headers(upstream_headers, RESPONSE_HEADERS_TO_DROP)
    headers.append(("content-type", content_type))
    if content_length is not None:
        headers.append(("content-length", str(content_length)))
    if content_type.startswith("text/event-stream") and not any(
        key.lower() == "cache-control" for key, _ in headers
    ):
        headers.append(("cache-control", "no-cache"))
    return headers


def create_app(settings: Settings, upstream_transport: Any = None) -> Any:
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse

    capture_root = settings.log_dir / "raw"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        initialize_capture_root(capture_root)
        timeout = httpx.Timeout(settings.timeout_seconds, read=None)
        app.state.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            http2=False,
            transport=upstream_transport,
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(title="OpenAI full capture proxy", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "protocol": "anthropic-to-openai-chat-completions",
            "upstream_url": settings.upstream_url,
            "upstream_model": settings.upstream_model,
            "capture_root": str(capture_root),
        }

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"status": "ok"}

    @app.head("/")
    async def root_head() -> Response:
        return Response(status_code=200)

    def json_response(
        capture: RequestCapture,
        *,
        body: bytes,
        status_code: int,
        upstream_headers: Iterable[tuple[str, str]] = (),
        source: str = "proxy",
        transport_error: Optional[str] = None,
    ) -> Response:
        response_headers = converted_response_headers(
            upstream_headers,
            content_type="application/json",
            content_length=len(body),
        )
        capture.start_response(
            status_code=status_code,
            headers=response_headers,
            is_sse=False,
            source=source,
        )
        capture.append_response(body)
        capture.finalize(transport_error=transport_error)
        response = Response(content=body, status_code=status_code)
        return apply_raw_response_headers(response, response_headers)

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def anthropic_proxy(path: str, request: Request) -> Response:
        raw_body = await request.body()
        incoming_headers = request_header_items(request)
        capture = RequestCapture(capture_root)
        capture.start_request(
            method=request.method,
            path=request.url.path,
            query=request.url.query,
            url=str(request.url),
            headers=incoming_headers,
            raw_body=raw_body,
            upstream_url=settings.upstream_url,
            client_host=request.client.host if request.client else None,
            client_port=request.client.port if request.client else None,
        )

        is_messages = (
            request.method.upper() == "POST"
            and request.url.path.rstrip("/") == "/v1/messages"
        )
        if not is_messages:
            body = anthropic_error_bytes(
                "not_found_error",
                f"Unsupported endpoint: {request.method} {request.url.path}",
            )
            return json_response(capture, body=body, status_code=404)

        try:
            anthropic_request = json.loads(raw_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            body = anthropic_error_bytes(
                "invalid_request_error", f"Request body is not valid JSON: {exc}"
            )
            return json_response(capture, body=body, status_code=400)
        if not isinstance(anthropic_request, dict):
            body = anthropic_error_bytes(
                "invalid_request_error", "Request body must be a JSON object"
            )
            return json_response(capture, body=body, status_code=400)

        try:
            openai_request = anthropic_to_openai_request(
                anthropic_request,
                model_override=settings.upstream_model,
                reasoning_mode=settings.reasoning_mode,
                reasoning_field=settings.reasoning_field,
                token_limit_field=settings.token_limit_field,
                include_stream_usage=settings.include_stream_usage,
                map_reasoning_effort=settings.map_reasoning_effort,
                extra_body=settings.extra_body,
            )
        except ConversionError as exc:
            body = anthropic_error_bytes("invalid_request_error", str(exc))
            return json_response(capture, body=body, status_code=400)

        wants_stream = bool(openai_request.get("stream"))
        upstream_headers = build_openai_headers(
            incoming_headers, settings, stream=wants_stream
        )
        upstream_request = app.state.client.build_request(
            "POST",
            settings.upstream_url,
            headers=upstream_headers,
            content=compact_json_bytes(openai_request),
        )

        try:
            upstream_response = await app.state.client.send(
                upstream_request, stream=True
            )
        except httpx.HTTPError as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            body = anthropic_error_bytes("api_error", error_text)
            return json_response(
                capture,
                body=body,
                status_code=502,
                transport_error=error_text,
            )

        response_header_items = list(upstream_response.headers.multi_items())
        if upstream_response.status_code < 200 or upstream_response.status_code >= 300:
            try:
                upstream_error_body = await upstream_response.aread()
            except httpx.HTTPError as exc:
                upstream_error_body = str(exc).encode("utf-8", errors="replace")
            finally:
                await upstream_response.aclose()
            body = openai_error_to_anthropic(
                upstream_error_body, upstream_response.status_code
            )
            return json_response(
                capture,
                body=body,
                status_code=upstream_response.status_code,
                upstream_headers=response_header_items,
                source="openai_upstream",
            )

        requested_model = str(openai_request.get("model") or "unknown")
        if not wants_stream:
            try:
                upstream_body = await upstream_response.aread()
                openai_response = json.loads(upstream_body)
                if not isinstance(openai_response, dict):
                    raise ConversionError("OpenAI response must be a JSON object")
                anthropic_message = openai_to_anthropic_message(
                    openai_response,
                    requested_model=requested_model,
                    reasoning_mode=settings.reasoning_mode,
                    reasoning_field=settings.reasoning_field,
                )
                body = compact_json_bytes(anthropic_message)
            except (UnicodeDecodeError, json.JSONDecodeError, ConversionError) as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                body = anthropic_error_bytes("api_error", error_text)
                return json_response(
                    capture,
                    body=body,
                    status_code=502,
                    upstream_headers=response_header_items,
                    transport_error=error_text,
                )
            except httpx.HTTPError as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                body = anthropic_error_bytes("api_error", error_text)
                return json_response(
                    capture,
                    body=body,
                    status_code=502,
                    upstream_headers=response_header_items,
                    transport_error=error_text,
                )
            finally:
                await upstream_response.aclose()
            return json_response(
                capture,
                body=body,
                status_code=upstream_response.status_code,
                upstream_headers=response_header_items,
                source="openai_converted",
            )

        upstream_content_type = upstream_response.headers.get("content-type", "")
        if (
            "application/json" in upstream_content_type.lower()
            and "text/event-stream" not in upstream_content_type.lower()
        ):
            try:
                upstream_body = await upstream_response.aread()
                openai_response = json.loads(upstream_body)
                if not isinstance(openai_response, dict):
                    raise ConversionError("OpenAI response must be a JSON object")
                anthropic_message = openai_to_anthropic_message(
                    openai_response,
                    requested_model=requested_model,
                    reasoning_mode=settings.reasoning_mode,
                    reasoning_field=settings.reasoning_field,
                )
                body = b"".join(anthropic_message_to_sse(anthropic_message))
            except (UnicodeDecodeError, json.JSONDecodeError, ConversionError) as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                body = anthropic_error_bytes("api_error", error_text)
                return json_response(
                    capture,
                    body=body,
                    status_code=502,
                    upstream_headers=response_header_items,
                    transport_error=error_text,
                )
            finally:
                await upstream_response.aclose()
            client_headers = converted_response_headers(
                response_header_items,
                content_type="text/event-stream; charset=utf-8",
                content_length=len(body),
            )
            capture.start_response(
                status_code=200,
                headers=client_headers,
                is_sse=True,
                source="openai_converted",
            )
            capture.append_response(body)
            capture.finalize()
            response = Response(content=body, status_code=200)
            return apply_raw_response_headers(response, client_headers)

        client_headers = converted_response_headers(
            response_header_items,
            content_type="text/event-stream; charset=utf-8",
        )
        capture.start_response(
            status_code=upstream_response.status_code,
            headers=client_headers,
            is_sse=True,
            source="openai_converted",
        )
        converter = OpenAIStreamConverter(
            requested_model=requested_model,
            reasoning_mode=settings.reasoning_mode,
            reasoning_field=settings.reasoning_field,
        )

        async def relay() -> AsyncIterator[bytes]:
            transport_error: Optional[str] = None
            client_disconnected = False
            try:
                async for chunk in upstream_response.aiter_bytes():
                    for converted_chunk in converter.feed(chunk):
                        capture.append_response(converted_chunk)
                        yield converted_chunk
                for converted_chunk in converter.finish():
                    capture.append_response(converted_chunk)
                    yield converted_chunk
            except asyncio.CancelledError as exc:
                transport_error = f"{type(exc).__name__}: {exc}"
                client_disconnected = True
                raise
            except (httpx.HTTPError, ConversionError) as exc:
                transport_error = f"{type(exc).__name__}: {exc}"
                error_chunk = sse_event(
                    "error", anthropic_error("api_error", transport_error)
                )
                capture.append_response(error_chunk)
                yield error_chunk
            except BaseException as exc:
                transport_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                await upstream_response.aclose()
                capture.finalize(
                    transport_error=transport_error,
                    client_disconnected=client_disconnected,
                )

        response = StreamingResponse(
            relay(),
            status_code=upstream_response.status_code,
        )
        return apply_raw_response_headers(response, client_headers)

    return app


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _env_json_object(name: str) -> dict[str, Any]:
    raw = os.getenv(name)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _string_headers(value: dict[str, Any], name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(f"{name} keys and values must be strings")
        result[key] = item
    return result


def parse_args(argv: Optional[list[str]] = None) -> Settings:
    default_url = os.getenv("OPENAI_UPSTREAM_URL") or os.getenv("UPSTREAM_URL")
    parser = argparse.ArgumentParser(
        description=(
            "接收 Anthropic Messages API，转换为 OpenAI Chat Completions，"
            "并完整保存 Anthropic 请求和响应"
        )
    )
    parser.add_argument(
        "--listen-host",
        default=os.getenv("PROXY_LISTEN_HOST", "0.0.0.0"),
        help="代理监听地址，默认读取 PROXY_LISTEN_HOST",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=int(os.getenv("PROXY_LISTEN_PORT", "30303")),
        help="代理监听端口，默认读取 PROXY_LISTEN_PORT",
    )
    parser.add_argument(
        "--upstream-url",
        default=default_url,
        required=default_url is None,
        help=(
            "OpenAI Chat Completions 完整 URL，例如 "
            "http://127.0.0.1:8000/v1/chat/completions"
        ),
    )
    parser.add_argument(
        "--upstream-model",
        default=os.getenv("OPENAI_MODEL"),
        help="固定上游模型名；未设置时保留 Anthropic 请求中的 model",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.getenv("CAPTURE_LOG_DIR", "capture_logs")),
        help="原始采集目录，默认读取 CAPTURE_LOG_DIR",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "300")),
        help="连接/写入超时；流式读取不设置总超时",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["preserve", "drop"],
        default=os.getenv("OPENAI_REASONING_MODE", "preserve"),
        help="preserve 在 reasoning_content 与 Anthropic thinking 间转换",
    )
    parser.add_argument(
        "--reasoning-field",
        default=os.getenv("OPENAI_REASONING_FIELD", "reasoning_content"),
        help="OpenAI 兼容服务使用的推理字段名",
    )
    parser.add_argument(
        "--token-limit-field",
        choices=["max_tokens", "max_completion_tokens"],
        default=os.getenv("OPENAI_TOKEN_LIMIT_FIELD", "max_tokens"),
        help="上游接受的输出 token 限制字段",
    )
    parser.add_argument(
        "--no-stream-usage",
        action="store_true",
        help="不发送 stream_options.include_usage",
    )
    parser.add_argument(
        "--no-reasoning-effort",
        action="store_true",
        help="不把 output_config.effort 转为 reasoning_effort",
    )
    args = parser.parse_args(argv)

    include_stream_usage = _env_bool("OPENAI_STREAM_INCLUDE_USAGE", True)
    map_reasoning_effort = _env_bool("OPENAI_MAP_REASONING_EFFORT", True)
    if args.no_stream_usage:
        include_stream_usage = False
    if args.no_reasoning_effort:
        map_reasoning_effort = False

    return Settings(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_url=args.upstream_url,
        log_dir=args.log_dir,
        timeout_seconds=args.timeout_seconds,
        upstream_model=args.upstream_model,
        upstream_api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_mode=args.reasoning_mode,
        reasoning_field=args.reasoning_field,
        token_limit_field=args.token_limit_field,
        include_stream_usage=include_stream_usage,
        map_reasoning_effort=map_reasoning_effort,
        extra_body=_env_json_object("OPENAI_EXTRA_BODY_JSON"),
        extra_headers=_string_headers(
            _env_json_object("OPENAI_EXTRA_HEADERS_JSON"),
            "OPENAI_EXTRA_HEADERS_JSON",
        ),
    )


def main() -> None:
    import uvicorn

    settings = parse_args()
    uvicorn.run(
        create_app(settings),
        host=settings.listen_host,
        port=settings.listen_port,
        workers=1,
    )


if __name__ == "__main__":
    main()
