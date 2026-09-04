from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional


LOGGER = logging.getLogger(__name__)


class ConversionError(ValueError):
    """Raised when one protocol cannot be represented safely in the other."""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def anthropic_error(error_type: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }


def anthropic_error_bytes(error_type: str, message: str) -> bytes:
    return compact_json(anthropic_error(error_type, message)).encode("utf-8")


def openai_error_to_anthropic(raw_body: bytes, status_code: int) -> bytes:
    message: Optional[str] = None
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
        elif isinstance(payload.get("message"), str):
            message = payload["message"]
    if not message:
        message = raw_body.decode("utf-8", errors="replace").strip()
    if not message:
        message = f"OpenAI upstream returned HTTP {status_code}"
    message = message[:4000]

    if status_code == 400:
        error_type = "invalid_request_error"
    elif status_code == 401:
        error_type = "authentication_error"
    elif status_code == 403:
        error_type = "permission_error"
    elif status_code == 404:
        error_type = "not_found_error"
    elif status_code == 429:
        error_type = "rate_limit_error"
    elif status_code in {529}:
        error_type = "overloaded_error"
    else:
        error_type = "api_error"
    return anthropic_error_bytes(error_type, message)


def _require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversionError(f"{description} must be an object")
    return value


def _require_list(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConversionError(f"{description} must be an array")
    return value


def _join_text_blocks(value: Any, description: str, separator: str = "") -> str:
    if isinstance(value, str):
        return value
    blocks = _require_list(value, description)
    texts: list[str] = []
    for index, raw_block in enumerate(blocks):
        block = _require_object(raw_block, f"{description}[{index}]")
        if block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise ConversionError(
                f"{description}[{index}] contains unsupported block type "
                f"{block.get('type')!r}"
            )
        texts.append(block["text"])
    return separator.join(texts)


def _openai_image_part(block: dict[str, Any], description: str) -> dict[str, Any]:
    source = _require_object(block.get("source"), f"{description}.source")
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise ConversionError(f"{description}.source has invalid base64 data")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url" and isinstance(source.get("url"), str):
        url = source["url"]
    else:
        raise ConversionError(
            f"{description}.source type {source_type!r} is not supported"
        )
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_text(block: dict[str, Any], description: str) -> str:
    content = block.get("content", "")
    if isinstance(content, str):
        text = content
    else:
        text = _join_text_blocks(content, f"{description}.content", separator="\n")
    if block.get("is_error") is True:
        return "[tool_error]\n" + text
    return text


def _assistant_message(
    content: Any,
    *,
    description: str,
    reasoning_mode: str,
    reasoning_field: str,
) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    blocks = _require_list(content, description)
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for index, raw_block in enumerate(blocks):
        block = _require_object(raw_block, f"{description}[{index}]")
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "thinking" and isinstance(block.get("thinking"), str):
            if reasoning_mode == "preserve":
                reasoning_parts.append(block["thinking"])
        elif block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            tool_input = block.get("input")
            if not isinstance(tool_id, str) or not tool_id:
                raise ConversionError(f"{description}[{index}].id must be a string")
            if not isinstance(name, str) or not name:
                raise ConversionError(f"{description}[{index}].name must be a string")
            if not isinstance(tool_input, dict):
                raise ConversionError(f"{description}[{index}].input must be an object")
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": compact_json(tool_input),
                    },
                }
            )
        else:
            raise ConversionError(
                f"{description}[{index}] contains unsupported block type {block_type!r}"
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if reasoning_parts:
        message[reasoning_field] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _user_messages(content: Any, description: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    blocks = _require_list(content, description)
    tool_messages: list[dict[str, Any]] = []
    content_parts: list[dict[str, Any]] = []
    for index, raw_block in enumerate(blocks):
        block = _require_object(raw_block, f"{description}[{index}]")
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            content_parts.append({"type": "text", "text": block["text"]})
        elif block_type == "image":
            content_parts.append(_openai_image_part(block, f"{description}[{index}]"))
        elif block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise ConversionError(
                    f"{description}[{index}].tool_use_id must be a string"
                )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": _tool_result_text(block, f"{description}[{index}]"),
                }
            )
        else:
            raise ConversionError(
                f"{description}[{index}] contains unsupported block type {block_type!r}"
            )

    result = list(tool_messages)
    if content_parts:
        if all(part.get("type") == "text" for part in content_parts):
            user_content: Any = "".join(part["text"] for part in content_parts)
        else:
            user_content = content_parts
        result.append({"role": "user", "content": user_content})
    elif not tool_messages:
        result.append({"role": "user", "content": ""})
    return result


def _convert_tools(value: Any) -> list[dict[str, Any]]:
    tools = _require_list(value, "tools")
    converted: list[dict[str, Any]] = []
    for index, raw_tool in enumerate(tools):
        tool = _require_object(raw_tool, f"tools[{index}]")
        name = tool.get("name")
        parameters = tool.get("input_schema")
        if not isinstance(name, str) or not name:
            raise ConversionError(f"tools[{index}].name must be a string")
        if not isinstance(parameters, dict):
            raise ConversionError(f"tools[{index}].input_schema must be an object")
        function: dict[str, Any] = {"name": name, "parameters": parameters}
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        converted.append({"type": "function", "function": function})
    return converted


def _convert_tool_choice(value: Any) -> tuple[Any, Optional[bool]]:
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise ConversionError(f"unsupported tool_choice {value!r}")
        return value, None
    choice = _require_object(value, "tool_choice")
    choice_type = choice.get("type")
    if choice_type == "auto":
        converted: Any = "auto"
    elif choice_type == "any":
        converted = "required"
    elif choice_type == "none":
        converted = "none"
    elif choice_type == "tool" and isinstance(choice.get("name"), str):
        converted = {
            "type": "function",
            "function": {"name": choice["name"]},
        }
    else:
        raise ConversionError(f"unsupported tool_choice type {choice_type!r}")
    disable_parallel = choice.get("disable_parallel_tool_use")
    parallel = False if disable_parallel is True else None
    return converted, parallel


def anthropic_to_openai_request(
    request: dict[str, Any],
    *,
    model_override: Optional[str] = None,
    reasoning_mode: str = "preserve",
    reasoning_field: str = "reasoning_content",
    token_limit_field: str = "max_tokens",
    include_stream_usage: bool = True,
    map_reasoning_effort: bool = True,
    extra_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if reasoning_mode not in {"preserve", "drop"}:
        raise ConversionError(f"unsupported reasoning mode {reasoning_mode!r}")
    if token_limit_field not in {"max_tokens", "max_completion_tokens"}:
        raise ConversionError(f"unsupported token limit field {token_limit_field!r}")
    if not isinstance(request, dict):
        raise ConversionError("request body must be a JSON object")

    source_model = request.get("model")
    model = model_override or source_model
    if not isinstance(model, str) or not model:
        raise ConversionError("model must be a non-empty string")

    messages: list[dict[str, Any]] = []
    system = request.get("system")
    if system is not None:
        system_text = _join_text_blocks(system, "system", separator="\n\n")
        if system_text:
            messages.append({"role": "system", "content": system_text})

    source_messages = _require_list(request.get("messages", []), "messages")
    for index, raw_message in enumerate(source_messages):
        message = _require_object(raw_message, f"messages[{index}]")
        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant":
            messages.append(
                _assistant_message(
                    content,
                    description=f"messages[{index}].content",
                    reasoning_mode=reasoning_mode,
                    reasoning_field=reasoning_field,
                )
            )
        elif role == "user":
            messages.extend(_user_messages(content, f"messages[{index}].content"))
        else:
            raise ConversionError(f"messages[{index}] has unsupported role {role!r}")

    converted: dict[str, Any] = {"model": model, "messages": messages}
    if "max_tokens" in request:
        max_tokens = request["max_tokens"]
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens <= 0
        ):
            raise ConversionError("max_tokens must be a positive integer")
        converted[token_limit_field] = max_tokens

    for source_name, target_name in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop_sequences", "stop"),
        ("service_tier", "service_tier"),
    ):
        if source_name in request:
            converted[target_name] = request[source_name]

    stream = bool(request.get("stream", False))
    converted["stream"] = stream
    if stream and include_stream_usage:
        converted["stream_options"] = {"include_usage": True}

    if request.get("tools") is not None:
        tools = _convert_tools(request["tools"])
        if tools:
            converted["tools"] = tools

    if request.get("tool_choice") is not None:
        tool_choice, parallel = _convert_tool_choice(request["tool_choice"])
        converted["tool_choice"] = tool_choice
        if parallel is not None:
            converted["parallel_tool_calls"] = parallel

    output_config = request.get("output_config")
    if map_reasoning_effort and isinstance(output_config, dict):
        effort = output_config.get("effort")
        if isinstance(effort, str) and effort:
            converted["reasoning_effort"] = effort

    ignored = {
        "cache_control",
        "context_management",
        "metadata",
        "thinking",
    }
    known = {
        "model",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "top_p",
        "stop_sequences",
        "stream",
        "tools",
        "tool_choice",
        "output_config",
        "service_tier",
    } | ignored
    unknown = sorted(set(request) - known)
    if unknown:
        LOGGER.warning("Ignoring unsupported Anthropic request fields: %s", unknown)

    if extra_body:
        reserved = {
            "model",
            "messages",
            "tools",
            "tool_choice",
            "stream",
            "stream_options",
            "max_tokens",
            "max_completion_tokens",
        }
        conflicts = sorted(set(extra_body) & reserved)
        if conflicts:
            raise ConversionError(
                f"OPENAI_EXTRA_BODY_JSON cannot override reserved fields: {conflicts}"
            )
        converted.update(extra_body)
    return converted


def _message_id(value: Any) -> str:
    if isinstance(value, str) and value:
        return value if value.startswith("msg_") else "msg_" + value
    return "msg_" + uuid.uuid4().hex


def _tool_id(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return "toolu_" + uuid.uuid4().hex


def _reasoning_signature(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "openai_proxy_" + digest


def _tool_input(value: Any, description: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    if not isinstance(value, str):
        raise ConversionError(f"{description} must be a JSON object string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"{description} is invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConversionError(f"{description} must decode to an object")
    return parsed


def map_finish_reason(value: Any, *, has_tools: bool = False) -> str:
    if has_tools or value in {"tool_calls", "function_call"}:
        return "tool_use"
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "refusal",
        "end_turn": "end_turn",
        "max_tokens": "max_tokens",
        "refusal": "refusal",
    }
    return mapping.get(value, "end_turn")


def openai_to_anthropic_message(
    response: dict[str, Any],
    *,
    requested_model: Optional[str] = None,
    reasoning_mode: str = "preserve",
    reasoning_field: str = "reasoning_content",
) -> dict[str, Any]:
    if reasoning_mode not in {"preserve", "drop"}:
        raise ConversionError(f"unsupported reasoning mode {reasoning_mode!r}")
    choices = _require_list(response.get("choices"), "OpenAI response.choices")
    if not choices:
        raise ConversionError("OpenAI response.choices is empty")
    choice = _require_object(choices[0], "OpenAI response.choices[0]")
    message = _require_object(
        choice.get("message"), "OpenAI response.choices[0].message"
    )

    content: list[dict[str, Any]] = []
    reasoning = message.get(reasoning_field)
    if reasoning_mode == "preserve" and isinstance(reasoning, str) and reasoning:
        content.append(
            {
                "type": "thinking",
                "thinking": reasoning,
                "signature": _reasoning_signature(reasoning),
            }
        )

    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(message.get("refusal"), str) and message["refusal"]:
        content.append({"type": "text", "text": message["refusal"]})
    elif text is not None and text != "":
        raise ConversionError(
            "OpenAI response message.content must be a string or null"
        )

    tool_calls = message.get("tool_calls") or []
    for index, raw_call in enumerate(_require_list(tool_calls, "tool_calls")):
        call = _require_object(raw_call, f"tool_calls[{index}]")
        function = _require_object(
            call.get("function"), f"tool_calls[{index}].function"
        )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ConversionError(f"tool_calls[{index}].function.name must be a string")
        content.append(
            {
                "type": "tool_use",
                "id": _tool_id(call.get("id")),
                "name": name,
                "input": _tool_input(
                    function.get("arguments"),
                    f"tool_calls[{index}].function.arguments",
                ),
            }
        )

    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    if not isinstance(input_tokens, int):
        input_tokens = 0
    if not isinstance(output_tokens, int):
        output_tokens = 0

    model = response.get("model") or requested_model or "unknown"
    return {
        "id": _message_id(response.get("id")),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": str(model),
        "stop_reason": map_finish_reason(
            choice.get("finish_reason"), has_tools=bool(tool_calls)
        ),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def sse_event(event_type: str, payload: dict[str, Any]) -> bytes:
    return (f"event: {event_type}\ndata: {compact_json(payload)}\n\n").encode("utf-8")


def anthropic_message_to_sse(message: dict[str, Any]) -> list[bytes]:
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    start_message = {
        "id": message["id"],
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": message.get("model", "unknown"),
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": 0,
        },
    }
    chunks = [
        sse_event(
            "message_start",
            {"type": "message_start", "message": start_message},
        )
    ]
    for index, block in enumerate(message.get("content", [])):
        block_type = block.get("type")
        if block_type == "text":
            initial = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block.get("text", "")}]
        elif block_type == "thinking":
            initial = {"type": "thinking", "thinking": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block.get("thinking", "")},
                {"type": "signature_delta", "signature": block.get("signature", "")},
            ]
        elif block_type == "tool_use":
            initial = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": {},
            }
            deltas = [
                {
                    "type": "input_json_delta",
                    "partial_json": compact_json(block.get("input", {})),
                }
            ]
        else:
            raise ConversionError(
                f"unsupported Anthropic response block {block_type!r}"
            )
        chunks.append(
            sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": initial,
                },
            )
        )
        for delta in deltas:
            chunks.append(
                sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": delta,
                    },
                )
            )
        chunks.append(
            sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )
        )
    chunks.extend(
        [
            sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": message.get("stop_reason", "end_turn"),
                        "stop_sequence": message.get("stop_sequence"),
                    },
                    "usage": {"output_tokens": usage.get("output_tokens", 0)},
                },
            ),
            sse_event("message_stop", {"type": "message_stop"}),
        ]
    )
    return chunks


class _OpenAISSEDecoder:
    def __init__(self) -> None:
        self._buffer = b""
        self._data_lines: list[bytes] = []

    def feed(self, chunk: bytes) -> list[Any]:
        self._buffer += chunk
        records: list[Any] = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            record = self._consume_line(line.rstrip(b"\r"))
            if record is not None:
                records.append(record)
        return records

    def finish(self) -> list[Any]:
        records: list[Any] = []
        if self._buffer:
            record = self._consume_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""
            if record is not None:
                records.append(record)
        if self._data_lines:
            record = self._dispatch()
            if record is not None:
                records.append(record)
        return records

    def _consume_line(self, line: bytes) -> Any:
        if not line:
            return self._dispatch()
        if line.startswith(b":"):
            return None
        field, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            self._data_lines.append(value)
        return None

    def _dispatch(self) -> Any:
        if not self._data_lines:
            return None
        raw = b"\n".join(self._data_lines)
        self._data_lines = []
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConversionError(f"OpenAI SSE contains invalid UTF-8: {exc}") from exc
        if text.strip() == "[DONE]":
            return _STREAM_DONE
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"OpenAI SSE contains invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConversionError("OpenAI SSE data must be a JSON object")
        return payload


_STREAM_DONE = object()


@dataclass
class _ToolCallState:
    tool_id: Optional[str] = None
    name: str = ""
    arguments: str = ""


class OpenAIStreamConverter:
    """Incrementally converts OpenAI Chat Completions SSE to Anthropic SSE."""

    def __init__(
        self,
        *,
        requested_model: Optional[str] = None,
        reasoning_mode: str = "preserve",
        reasoning_field: str = "reasoning_content",
    ) -> None:
        if reasoning_mode not in {"preserve", "drop"}:
            raise ConversionError(f"unsupported reasoning mode {reasoning_mode!r}")
        self.requested_model = requested_model
        self.reasoning_mode = reasoning_mode
        self.reasoning_field = reasoning_field
        self._decoder = _OpenAISSEDecoder()
        self._started = False
        self._finalized = False
        self._message_id: Optional[str] = None
        self._model: Optional[str] = None
        self._finish_reason: Any = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._active_type: Optional[str] = None
        self._active_index: Optional[int] = None
        self._active_reasoning = ""
        self._next_block_index = 0
        self._tool_calls: dict[int, _ToolCallState] = {}

    def feed(self, chunk: bytes) -> list[bytes]:
        output: list[bytes] = []
        for record in self._decoder.feed(chunk):
            output.extend(self._consume_record(record))
        return output

    def finish(self) -> list[bytes]:
        output: list[bytes] = []
        for record in self._decoder.finish():
            output.extend(self._consume_record(record))
        if self._finalized:
            return output
        if self._finish_reason is None:
            raise ConversionError("OpenAI stream ended before finish_reason")
        output.extend(self._finish_message())
        return output

    def _consume_record(self, record: Any) -> list[bytes]:
        if record is _STREAM_DONE:
            if self._finalized:
                return []
            if self._finish_reason is None:
                raise ConversionError("OpenAI stream sent [DONE] before finish_reason")
            return self._finish_message()
        if self._finalized:
            raise ConversionError("OpenAI stream sent data after [DONE]")
        return self._consume_payload(record)

    def _consume_payload(self, payload: dict[str, Any]) -> list[bytes]:
        if isinstance(payload.get("error"), dict):
            message = payload["error"].get("message") or compact_json(payload["error"])
            raise ConversionError(f"OpenAI stream error: {message}")

        usage = payload.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int):
                self._input_tokens = prompt_tokens
            if isinstance(completion_tokens, int):
                self._output_tokens = completion_tokens

        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise ConversionError("OpenAI stream chunk is missing choices")
        if not choices:
            return []
        choice = _require_object(choices[0], "OpenAI stream choices[0]")
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}

        output = self._ensure_started(payload)
        reasoning = delta.get(self.reasoning_field)
        if (
            self.reasoning_mode == "preserve"
            and isinstance(reasoning, str)
            and reasoning
        ):
            output.extend(self._append_content("thinking", reasoning))

        text = delta.get("content")
        if isinstance(text, str) and text:
            output.extend(self._append_content("text", text))

        raw_calls = delta.get("tool_calls")
        if raw_calls is not None:
            for raw_call in _require_list(raw_calls, "OpenAI stream tool_calls"):
                call = _require_object(raw_call, "OpenAI stream tool_call")
                index = call.get("index", 0)
                if not isinstance(index, int) or index < 0:
                    raise ConversionError(
                        "OpenAI tool call index must be a non-negative integer"
                    )
                state = self._tool_calls.setdefault(index, _ToolCallState())
                if isinstance(call.get("id"), str) and call["id"]:
                    state.tool_id = call["id"]
                function = call.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                    if isinstance(name, str) and name:
                        if not state.name:
                            state.name = name
                        elif name != state.name:
                            state.name += name
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        state.arguments += arguments

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            self._finish_reason = finish_reason
        return output

    def _ensure_started(self, payload: dict[str, Any]) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        self._message_id = _message_id(payload.get("id"))
        model = payload.get("model") or self.requested_model or "unknown"
        self._model = str(model)
        message = {
            "id": self._message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": self._model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": self._input_tokens,
                "output_tokens": 0,
            },
        }
        return [
            sse_event(
                "message_start",
                {"type": "message_start", "message": message},
            )
        ]

    def _append_content(self, block_type: str, value: str) -> list[bytes]:
        output: list[bytes] = []
        if self._active_type != block_type:
            output.extend(self._close_active_block())
            self._active_type = block_type
            self._active_index = self._next_block_index
            self._next_block_index += 1
            if block_type == "thinking":
                initial = {"type": "thinking", "thinking": ""}
                self._active_reasoning = ""
            else:
                initial = {"type": "text", "text": ""}
            output.append(
                sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self._active_index,
                        "content_block": initial,
                    },
                )
            )

        if block_type == "thinking":
            delta = {"type": "thinking_delta", "thinking": value}
            self._active_reasoning += value
        else:
            delta = {"type": "text_delta", "text": value}
        output.append(
            sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._active_index,
                    "delta": delta,
                },
            )
        )
        return output

    def _close_active_block(self) -> list[bytes]:
        if self._active_type is None or self._active_index is None:
            return []
        output: list[bytes] = []
        if self._active_type == "thinking":
            output.append(
                sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._active_index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": _reasoning_signature(self._active_reasoning),
                        },
                    },
                )
            )
        output.append(
            sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._active_index},
            )
        )
        self._active_type = None
        self._active_index = None
        self._active_reasoning = ""
        return output

    def _finish_message(self) -> list[bytes]:
        if not self._started:
            raise ConversionError("OpenAI stream did not contain a completion choice")
        output = self._close_active_block()
        for tool_index in sorted(self._tool_calls):
            state = self._tool_calls[tool_index]
            if not state.name:
                raise ConversionError(
                    f"OpenAI tool call {tool_index} is missing function.name"
                )
            tool_input = _tool_input(
                state.arguments,
                f"OpenAI tool call {tool_index} function.arguments",
            )
            block_index = self._next_block_index
            self._next_block_index += 1
            output.extend(
                [
                    sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": _tool_id(state.tool_id),
                                "name": state.name,
                                "input": {},
                            },
                        },
                    ),
                    sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": compact_json(tool_input),
                            },
                        },
                    ),
                    sse_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": block_index},
                    ),
                ]
            )

        output.extend(
            [
                sse_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": map_finish_reason(
                                self._finish_reason,
                                has_tools=bool(self._tool_calls),
                            ),
                            "stop_sequence": None,
                        },
                        "usage": {"output_tokens": self._output_tokens},
                    },
                ),
                sse_event("message_stop", {"type": "message_stop"}),
            ]
        )
        self._finalized = True
        return output
