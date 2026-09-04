# 06 测试与运维

[上一步：采集与关联机制](README_05_采集与关联机制.md) · [返回总览](README.md)

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖：

- Anthropic system、文本、图片和参数转换
- thinking/reasoning 双向转换与代理侧 signature
- 单个及多个并行工具调用、tool result 和错误结果
- OpenAI SSE 跨任意网络 chunk 增量解析
- 非流式 OpenAI JSON 转 Anthropic Message
- 流式、非流式、上游认证错误和 header 改写的代理集成
- capture 文件隔离、SSE 聚合和 message id 关联
- Harbor 主/子 agent 后处理与 ShareGPT 导出

真实上游上线前还应使用目标模型做一次流式工具调用 smoke test。OpenAI 兼容实现可能不支持 `reasoning_effort`、`stream_options.include_usage` 或 `reasoning_content`；按目标服务调整环境变量，不要在转换器里硬编码供应商参数。

## 运维注意事项

- 每次 Harbor run 使用独立 `--log-dir`。
- 原始 `raw/` 是客户端侧 Anthropic 事实源，生成数据集后也不要删除。
- 定期检查 `finalization_report.json` 中的 `unmatched`、`conflicts` 和 `inflight`。
- 检查 `state.json`；目录位于 `completed/` 只代表采集已收尾，不代表模型响应完整。
- system、messages、工具参数和工具结果可能包含内部路径或敏感数据，应限制日志目录权限。
- 不要把上游 API key 写入源码、命令行或 `OPENAI_EXTRA_HEADERS_JSON` 示例文件。
- 代理保存 HTTP 应用层可见内容，不是 TLS、TCP 或 HTTP/2 帧级抓包。

