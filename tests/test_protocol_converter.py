from __future__ import annotations

import json
import unittest

from capture_core import AnthropicMessageAggregator, SSEDecoder
from protocol_converter import (
    ConversionError,
    OpenAIStreamConverter,
    anthropic_message_to_sse,
    anthropic_to_openai_request,
    openai_error_to_anthropic,
    openai_to_anthropic_message,
)


def openai_sse(payload: dict) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n".encode("utf-8")


def aggregate_anthropic_stream(raw: bytes) -> AnthropicMessageAggregator:
    aggregator = AnthropicMessageAggregator()
    decoder = SSEDecoder(aggregator)
    for offset in range(0, len(raw), 7):
        decoder.feed(raw[offset : offset + 7])
    decoder.finish()
    return aggregator


class RequestConversionTests(unittest.TestCase):
    def test_converts_reasoning_tools_results_and_parameters(self):
        request = {
            "model": "client-model",
            "system": [
                {
                    "type": "text",
                    "text": "first",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "second"},
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "run"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "plan", "signature": "sig"},
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Bash",
                            "input": {"command": "pwd"},
                        },
                        {
                            "type": "tool_use",
                            "id": "call_2",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "/workspace",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_2",
                            "content": [{"type": "text", "text": "missing"}],
                            "is_error": True,
                        },
                        {"type": "text", "text": "continue"},
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "max_tokens": 4096,
            "temperature": 1,
            "stop_sequences": ["STOP"],
            "stream": True,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "metadata": {"user_id": "session"},
        }

        converted = anthropic_to_openai_request(
            request,
            model_override="target-model",
            extra_body={"enable_thinking": True},
        )

        self.assertEqual(converted["model"], "target-model")
        self.assertEqual(converted["messages"][0]["content"], "first\n\nsecond")
        assistant = converted["messages"][2]
        self.assertEqual(assistant["reasoning_content"], "plan")
        self.assertEqual(len(assistant["tool_calls"]), 2)
        self.assertEqual(
            json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
            {"command": "pwd"},
        )
        self.assertEqual(converted["messages"][3]["role"], "tool")
        self.assertEqual(converted["messages"][4]["role"], "tool")
        self.assertTrue(converted["messages"][4]["content"].startswith("[tool_error]"))
        self.assertEqual(
            converted["messages"][5], {"role": "user", "content": "continue"}
        )
        self.assertEqual(
            converted["tools"][0]["function"]["description"], "Run a command"
        )
        self.assertEqual(converted["tool_choice"], "auto")
        self.assertFalse(converted["parallel_tool_calls"])
        self.assertEqual(converted["max_tokens"], 4096)
        self.assertEqual(converted["stop"], ["STOP"])
        self.assertEqual(converted["reasoning_effort"], "medium")
        self.assertTrue(converted["stream_options"]["include_usage"])
        self.assertTrue(converted["enable_thinking"])

    def test_converts_base64_image(self):
        converted = anthropic_to_openai_request(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "inspect"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AAAA",
                                },
                            },
                        ],
                    }
                ],
            }
        )
        image = converted["messages"][0]["content"][1]
        self.assertEqual(image["type"], "image_url")
        self.assertEqual(image["image_url"]["url"], "data:image/png;base64,AAAA")

    def test_rejects_unsupported_content_and_reserved_extra_body(self):
        with self.assertRaisesRegex(ConversionError, "document"):
            anthropic_to_openai_request(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": [{"type": "document"}]}],
                }
            )
        with self.assertRaisesRegex(ConversionError, "reserved fields"):
            anthropic_to_openai_request(
                {"model": "m", "messages": []},
                extra_body={"model": "override"},
            )


class ResponseConversionTests(unittest.TestCase):
    def test_converts_non_stream_reasoning_multiple_tools_and_usage(self):
        converted = openai_to_anthropic_message(
            {
                "id": "chatcmpl-1",
                "model": "target-model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "inspect first",
                            "content": "using tools",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Bash",
                                        "arguments": '{"command":"pwd"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                },
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        )
        self.assertEqual(converted["id"], "msg_chatcmpl-1")
        self.assertEqual(
            [block["type"] for block in converted["content"]],
            ["thinking", "text", "tool_use", "tool_use"],
        )
        self.assertTrue(
            converted["content"][0]["signature"].startswith("openai_proxy_")
        )
        self.assertEqual(converted["content"][2]["input"], {"command": "pwd"})
        self.assertEqual(converted["stop_reason"], "tool_use")
        self.assertEqual(converted["usage"], {"input_tokens": 11, "output_tokens": 7})

    def test_stream_survives_arbitrary_chunks_and_keeps_tool_indexes(self):
        payloads = [
            {
                "id": "chatcmpl-stream",
                "model": "target-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "reasoning_content": "plan"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-stream",
                "model": "target-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "done"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-stream",
                "model": "target-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "Bash",
                                        "arguments": '{"command":',
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "function": {"name": "Read", "arguments": "{"},
                                },
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-stream",
                "model": "target-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"pwd"}'}},
                                {"index": 1, "function": {"arguments": "}"}},
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
            {
                "id": "chatcmpl-stream",
                "model": "target-model",
                "choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5},
            },
        ]
        raw = b"".join(openai_sse(payload) for payload in payloads)
        raw += b"data: [DONE]\n\n"

        converter = OpenAIStreamConverter(requested_model="target-model")
        output: list[bytes] = []
        offsets = [1, 2, 5, 3, 11]
        position = 0
        sequence = 0
        while position < len(raw):
            size = offsets[sequence % len(offsets)]
            output.extend(converter.feed(raw[position : position + size]))
            position += size
            sequence += 1
        output.extend(converter.finish())

        aggregator = aggregate_anthropic_stream(b"".join(output))
        self.assertTrue(aggregator.complete)
        self.assertEqual(aggregator.message_id, "msg_chatcmpl-stream")
        self.assertEqual(
            [block["type"] for block in aggregator.message["content"]],
            ["thinking", "text", "tool_use", "tool_use"],
        )
        self.assertEqual(aggregator.message["content"][2]["input"], {"command": "pwd"})
        self.assertEqual(aggregator.message["content"][3]["input"], {})
        self.assertEqual(aggregator.message["stop_reason"], "tool_use")
        self.assertEqual(aggregator.message["usage"]["output_tokens"], 5)

    def test_non_stream_message_can_be_rendered_as_valid_anthropic_sse(self):
        message = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Bash",
                    "input": {"command": "pwd"},
                },
            ],
            "model": "m",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
        aggregator = aggregate_anthropic_stream(
            b"".join(anthropic_message_to_sse(message))
        )
        self.assertTrue(aggregator.complete)
        self.assertEqual(aggregator.message["content"][1]["input"], {"command": "pwd"})

    def test_maps_openai_error_without_leaking_non_json_shape(self):
        converted = json.loads(
            openai_error_to_anthropic(
                b'{"error":{"message":"bad key","type":"auth"}}', 401
            )
        )
        self.assertEqual(converted["type"], "error")
        self.assertEqual(converted["error"]["type"], "authentication_error")
        self.assertEqual(converted["error"]["message"], "bad key")


if __name__ == "__main__":
    unittest.main()
