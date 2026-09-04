# 05 采集与关联机制

[上一步：导出 ShareGPT](README_04_导出_ShareGPT.md) · [返回总览](README.md) · [下一篇：测试与运维](README_06_测试与运维.md)

## 协议转换边界

`request.body` 始终保存 Claude Code 发来的原始 Anthropic JSON。转换器随后生成 OpenAI Chat Completions 请求；上游 API key 和转换后的 OpenAI body 默认不落盘。

上游响应先转换为 Anthropic 格式，再写入 `response.body` 并返回客户端。因此：

- 非流式 `response.body` 是转换后的 Anthropic Message JSON。
- 流式 `response.body` 是转换后的 Anthropic SSE。
- `sse_events.jsonl` 和 `response.json.message` 仍由原有 Anthropic 聚合器生成。
- 原有 finalize 和 ShareGPT 导出器无需理解 OpenAI 协议。

## 请求转换

- system 文本块合并为 OpenAI system message。
- assistant `tool_use` 转为 `tool_calls`。
- user `tool_result` 转为独立 tool message；错误结果带有 `[tool_error]` 标记。
- 工具 `input_schema` 转为 function `parameters`，保留描述。
- `output_config.effort` 默认转为 `reasoning_effort`。
- `thinking` 历史默认通过配置的推理字段传递，signature 不转发。
- `cache_control`、`metadata` 和 `context_management` 没有直接等价字段，当前不会发给上游。

图片支持 Anthropic base64 和 URL source。文档、server tool、citation 等没有可靠 Chat Completions 等价结构的 block 会直接返回 400。

## 流式响应转换

转换器增量解析 OpenAI SSE，可以处理 JSON 行跨任意网络 chunk 的情况。thinking/text 会立即转发；工具 arguments 先按 OpenAI tool index 聚合，在结束时为每个工具生成独立且连续的 Anthropic content block：

```text
message_start
thinking/text content_block_start + delta + stop
tool 0 content_block_start + input_json_delta + stop
tool 1 content_block_start + input_json_delta + stop
message_delta
message_stop
```

这避免了多个工具复用 index、delta 先于 start 等非法 Anthropic SSE。OpenAI 流缺少 `finish_reason`、工具 JSON 不合法或上游中途断开时，capture 会标记为 partial，并向客户端发送 Anthropic error event（客户端已断开时除外）。

## 关联规则

主要关联键保持不变：

```text
转换后 Anthropic response.message.id
                  ==
Harbor/Claude Code session 中 assistant.message.id
```

OpenAI response id 会增加 `msg_` 前缀后作为 Anthropic message id；相同 id 同时写入 capture 并返回 Claude Code，所以仍可精确关联主 agent、subagent 和 round。

相邻时间、客户端 IP 和 prompt 内容不参与关联。无法精确关联的数据进入 `unmatched`，冲突进入 `conflicts`。

