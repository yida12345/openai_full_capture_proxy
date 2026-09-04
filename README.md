# OpenAI Full Capture Proxy

接收 Anthropic Messages API 请求，将其转换为 OpenAI Chat Completions 请求发给上游，再把 OpenAI JSON 或 SSE 响应转换回 Anthropic 格式。代理完整保存客户端看到的 Anthropic 请求、响应、SSE 事件和聚合后的 Message。

本项目从 `anthropic_full_capture_proxy` 独立复制而来。离线整理与 ShareGPT 导出格式保持不变，可以继续通过转换后响应的 `message.id` 与 Harbor/Claude Code session 精确关联。

## 数据流

```text
Harbor / Claude Code
        │ Anthropic POST /v1/messages
        ▼
proxy.py
  ├─ 保存原始 Anthropic 请求
  ├─ protocol_converter.py 转为 OpenAI Chat Completions
  ├─ POST 配置的 /v1/chat/completions URL
  ├─ OpenAI JSON/SSE 转回 Anthropic JSON/SSE
  └─ 保存并返回转换后的 Anthropic 响应
        │
        ▼
capture_logs/<run>/raw/{inflight,completed}
        │
        ▼
finalize.py / finalize-harbor.py / finalize-node.py
        │
        ▼
export_sharegpt.py / export_sharegpt_recovered.py
```

采集边界是 Claude Code 可见的 Anthropic 协议。默认不额外保存包含上游凭据的 OpenAI 原始报文。

## 已支持的转换

- system 文本与 user/assistant 文本
- Anthropic `tool_use`、`tool_result` 与 OpenAI function tool calls
- 多个并行工具调用和分片 arguments
- base64/URL 图片输入
- 流式与非流式响应
- `reasoning_content` 与 Anthropic thinking block
- token usage 和常见停止原因
- OpenAI HTTP 错误到 Anthropic error 的转换

不支持的内容块会返回明确的 `invalid_request_error`，不会静默丢弃。当前不实现 `/v1/messages/count_tokens`，该端点会返回 404。

## 按运行顺序阅读

1. [安装并启动代理](README_01_安装并启动代理.md)
2. [运行 Harbor](README_02_运行_Harbor.md)
3. [整理最终数据集](README_03_整理最终数据集.md)
4. [导出 ShareGPT SFT 数据](README_04_导出_ShareGPT.md)
5. [采集与关联机制](README_05_采集与关联机制.md)
6. [测试与运维](README_06_测试与运维.md)

