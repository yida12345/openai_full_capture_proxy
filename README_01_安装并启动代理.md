# 01 安装并启动代理

[返回总览](README.md) · [下一步：运行 Harbor](README_02_运行_Harbor.md)

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

也可以使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell 激活命令为：

```powershell
.venv\Scripts\Activate.ps1
```

## 2. 配置上游

`--upstream-url` 必须是完整的 OpenAI Chat Completions URL，而不是 base URL：

```text
http://127.0.0.1:8000/v1/chat/completions
```

推荐通过环境变量配置：

```bash
export PROXY_LISTEN_HOST=0.0.0.0
export PROXY_LISTEN_PORT=30303
export OPENAI_UPSTREAM_URL=http://127.0.0.1:8000/v1/chat/completions
export OPENAI_MODEL=openai/GLM-5.1
export OPENAI_API_KEY=sk-upstream-key
export CAPTURE_LOG_DIR=./capture_logs/example_run
export UPSTREAM_TIMEOUT_SECONDS=300
python proxy.py
```

`OPENAI_API_KEY` 为空时不发送 Authorization header，适合无认证的本地 vLLM/SGLang 服务。密钥不会写入 capture。

也可以使用命令行参数；密钥仍只从环境变量读取，避免出现在进程参数中：

```bash
python proxy.py \
  --listen-host 0.0.0.0 \
  --listen-port 30303 \
  --upstream-url http://127.0.0.1:8000/v1/chat/completions \
  --upstream-model openai/GLM-5.1 \
  --log-dir ./capture_logs/example_run \
  --timeout-seconds 300
```

程序不会自动加载 `config.example.env`，需要由 shell、Docker 或任务调度器注入。

## 3. OpenAI 兼容开关

不同 OpenAI 兼容服务对扩展字段的支持并不一致：

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `OPENAI_REASONING_MODE` | `preserve` | 在 `reasoning_content` 与 Anthropic thinking 间转换；设为 `drop` 可禁用 |
| `OPENAI_REASONING_FIELD` | `reasoning_content` | 上游使用的推理字段名 |
| `OPENAI_TOKEN_LIMIT_FIELD` | `max_tokens` | 可改为 `max_completion_tokens` |
| `OPENAI_STREAM_INCLUDE_USAGE` | `true` | 请求流末尾 usage chunk；不支持时设为 `false` |
| `OPENAI_MAP_REASONING_EFFORT` | `true` | 将 `output_config.effort` 转为 `reasoning_effort` |
| `OPENAI_EXTRA_BODY_JSON` | `{}` | 加入上游请求的服务专用字段 |
| `OPENAI_EXTRA_HEADERS_JSON` | `{}` | 加入上游请求的自定义 header |

例如部分本地模型需要显式启用 thinking：

```bash
export OPENAI_EXTRA_BODY_JSON='{"enable_thinking":true,"chat_template_kwargs":{"enable_thinking":true}}'
```

额外 body 不能覆盖 `model`、`messages`、`tools`、`stream` 和 token 限制等核心转换字段。

## 4. 健康检查

```bash
curl http://127.0.0.1:30303/healthz
```

`GET /` 和 `HEAD /` 也返回 200，供 Harbor 做存活检查。

## 5. 配置 Harbor / Claude Code

让 Claude Code 继续按 Anthropic API 使用代理：

```bash
export ANTHROPIC_BASE_URL=http://代理IP:30303
export ANTHROPIC_API_KEY=client-placeholder
```

入站 `X-Api-Key`/`Authorization` 只用于兼容 Anthropic 客户端，不会转发给 OpenAI 上游。上游认证只读取 `OPENAI_API_KEY`。

一个代理进程可并发服务多个 Claude Code。每个请求仍使用独立 capture 目录和 SSE 状态，不按 IP、时间或 prompt 推断 task。
