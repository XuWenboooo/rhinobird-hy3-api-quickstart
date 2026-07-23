# Hy3 API Quickstart

> **5 分钟跑通第一次调用，半小时上手主要能力**

Hy3 是腾讯混元团队研发的 295B MoE（混合专家）大模型，提供 OpenAI 兼容的 API 接口。本文档帮助你快速接入并开始使用。

---

## 目录

- [1. 基础信息](#1-基础信息)
- [2. 环境准备](#2-环境准备)
- [3. 5 分钟快速开始](#3-5-分钟快速开始)
- [4. 核心参数说明](#4-核心参数说明)
- [5. 进阶能力](#5-进阶能力)
- [6. 常见报错与排查](#6-常见报错与排查)
- [7. 更多示例](#7-更多示例)

---

## 1. 基础信息

### 1.1 API 端点

Hy3 支持以下接入方式：

| 接入方式 | Base URL | 适用场景 |
|----------|----------|----------|
| **TokenHub（推荐）** | `https://tokenhub.tencentmaas.com/v1` | 云 API，开箱即用 |
| **腾讯云 LKEAP** | `https://api.lkeap.cloud.tencent.com/plan/v3` | 腾讯云用户 |
| **自部署 (vLLM/SGLang)** | `http://127.0.0.1:8000/v1` | 私有化部署 |

### 1.2 可用模型

| 模型名 | 说明 |
|--------|------|
| `hy3` | Hy3 正式版（推荐） |
| `hy3-preview` | Hy3 Preview 版 |
| `hunyuan/hy3` | EdgeOne 平台上的名称 |

### 1.3 API Key

- **TokenHub**：在 [TokenHub 控制台](https://tokenhub.tencentmaas.com) 创建 API Key
- **腾讯云**：在 [腾讯云控制台](https://console.cloud.tencent.com) 获取 SecretId/SecretKey
- **自部署**：本地部署时无需认证，`api_key` 传 `"EMPTY"` 即可

### 1.4 速率限制与定价

| 项目 | 说明 |
|------|------|
| **上下文窗口** | 256K tokens |
| **最大输出** | 32K tokens（正式版）/ 128K tokens（Preview） |
| **输入价格** | ¥1.00 / 百万 tokens |
| **输出价格** | ¥4.00 / 百万 tokens |
| **缓存命中** | ¥0.25 / 百万 tokens |
| **并发限制** | 视具体平台套餐而定，免费版通常为 5～10 QPS |

### 1.5 模型规格

| 属性 | 值 |
|------|-----|
| 架构 | Mixture-of-Experts (MoE) |
| 总参数 | 295B |
| 激活参数 | 21B |
| MTP 层参数 | 3.8B |
| 上下文长度 | 256K |
| 专家数量 | 192 个，top-8 激活 |

---

## 2. 环境准备

### 安装 Python SDK

```bash
pip install openai
```

### 设置环境变量

```bash
# Linux / macOS
export HY3_API_KEY="your-api-key-here"
export HY3_BASE_URL="https://tokenhub.tencentmaas.com/v1"

# Windows PowerShell
$env:HY3_API_KEY = "your-api-key-here"
$env:HY3_BASE_URL = "https://tokenhub.tencentmaas.com/v1"
```

---

## 3. 5 分钟快速开始

### 3.1 最简调用 — Python (OpenAI SDK)

```python
import os
from openai import OpenAI

# 1. 创建客户端
client = OpenAI(
    api_key=os.environ.get("HY3_API_KEY", "your-api-key"),
    base_url=os.environ.get("HY3_BASE_URL", "https://tokenhub.tencentmaas.com/v1"),
)

# 2. 发起对话请求
response = client.chat.completions.create(
    model="hy3",
    messages=[
        {"role": "user", "content": "你好！请用一句话介绍你自己。"},
    ],
    temperature=0.9,
    max_tokens=512,
)

# 3. 打印结果
print(response.choices[0].message.content)
```

**运行：**
```bash
python quickstart_demo.py
```

### 3.2 最简调用 — curl

```bash
curl https://tokenhub.tencentmaas.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $HY3_API_KEY" \
  -d '{
    "model": "hy3",
    "messages": [
      {"role": "user", "content": "你好！请用一句话介绍你自己。"}
    ],
    "temperature": 0.9,
    "max_tokens": 512
  }'
```

**预期输出（示例）：**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1720000000,
  "model": "hy3",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！我是腾讯混元 Hy3，一个 295B 参数的混合专家大模型，擅长推理、编程和智能体任务。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 35,
    "total_tokens": 55
  }
}
```

### 3.3 自部署方式调用

如果你已通过 vLLM 或 SGLang 自行部署 Hy3：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="hy3",
    messages=[
        {"role": "user", "content": "你好！请简单介绍一下你自己。"},
    ],
    temperature=0.9,
    top_p=1.0,
    # 自部署时通过 extra_body 传入 reasoning_effort
    extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
)
print(response.choices[0].message.content)
```

---

## 4. 核心参数说明

### 4.1 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 0.9 | 采样温度。0～2，越高越随机，越低越确定。推荐 0.9 |
| `top_p` | float | 1.0 | 核采样。只从累积概率达到 top_p 的 token 中采样。推荐 1.0 |
| `max_tokens` | int | — | 最大输出 token 数。不传则由模型自动决定 |
| `stop` | str / list | — | 停止词。遇到该内容时停止生成 |
| `stream` | bool | false | 是否启用流式输出 |

### 4.2 推理模式 (reasoning_effort)

Hy3 支持「快慢思考融合」——可根据任务复杂度切换推理深度。

| 值 | 模式 | 适用场景 |
|----|------|----------|
| `"no_think"` / `"low"` | 快思考（直接回复） | 日常对话、简单问答 |
| `"medium"` | 中等推理 | 一般分析任务 |
| `"high"` | 慢思考（深度思维链） | 数学、编程、复杂推理 |

**云 API 调用方式：**
```python
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "证明√2是无理数"}],
    reasoning_effort="high",  # 开启深度思考
)
```

**自部署 (vLLM) 调用方式：**
```python
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "证明√2是无理数"}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
)
```

> **注意**：开启深度思考后，响应中会包含 `reasoning_content` 字段（思考过程），`content` 字段为最终答案。

### 4.3 工具调用 (Tools / Function Calling)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto",  # auto / none / required
)
```

`tool_choice` 选项：
| 值 | 说明 |
|----|------|
| `"auto"` | 模型自动决定是否调用工具（默认） |
| `"none"` | 不调用工具，直接回复 |
| `"required"` | 强制调用工具 |
| `{"type": "function", "function": {"name": "xxx"}}` | 强制调用指定工具 |

---

## 5. 进阶能力

### 5.1 流式输出 (Streaming)

```python
stream = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "写一首关于人工智能的诗"}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### 5.2 多轮对话

```python
messages = [
    {"role": "system", "content": "你是一个专业的 Python 编程助手。"},
    {"role": "user", "content": "Python 中如何反转一个列表？"},
    {"role": "assistant", "content": "可以使用 `list.reverse()` 方法或切片 `list[::-1]`。"},
    {"role": "user", "content": "这两种方式有什么区别？"},
]

response = client.chat.completions.create(
    model="hy3",
    messages=messages,
)
```

### 5.3 结构化输出

Hy3 支持 `response_format` 参数强制 JSON 输出：

```python
response = client.chat.completions.create(
    model="hy3",
    messages=[{"role": "user", "content": "列出 3 种排序算法的名称和复杂度"}],
    response_format={"type": "json_object"},
)
```

### 5.4 获取 Token 用量

每次请求的响应中均包含 `usage` 字段：

```python
print(f"输入 tokens: {response.usage.prompt_tokens}")
print(f"输出 tokens: {response.usage.completion_tokens}")
print(f"总计 tokens: {response.usage.total_tokens}")
```

---

## 6. 常见报错与排查

### 6.1 认证错误 (401 Unauthorized)

```
Error: 401 - Invalid API Key
```

**原因**：API Key 无效或未传入。
**解决**：
- 检查 `Authorization: Bearer <key>` 头部是否正确
- 确认 API Key 未过期（TokenHub 控制台可查看）
- 检查环境变量是否正确设置

### 6.2 限流错误 (429 Too Many Requests)

```
Error: 429 - Rate limit exceeded
```

**原因**：超出 API 速率限制。
**解决**：
- 降低请求频率，实现指数退避重试（参见 [06_error_handling.py](examples/06_error_handling.py)）
- 升级套餐以获得更高并发配额

### 6.3 超时错误

```
Error: Request timed out / ReadTimeout
```

**原因**：请求时间过长，服务器无响应。
**解决**：
- 增大 `timeout` 参数：`OpenAI(..., timeout=60.0)`
- 减小 `max_tokens` 以缩短生成时间
- 启用 `stream=True` 获取增量输出

### 6.4 模型不可用 (404 Not Found)

```
Error: 404 - Model not found
```

**原因**：模型名错误或未开通。
**解决**：
- 确认模型名拼写：应为 `hy3` 或 `hy3-preview`
- TokenHub 需在控制台先开通对应模型

### 6.5 内容过滤 (400 Bad Request)

```
Error: 400 - Content filtered
```

**原因**：输入或输出触发安全审核。
**解决**：
- 修改 prompt 措辞
- 确认内容不违反使用条款

### 6.6 连接错误

```
Error: ConnectionError / Connection refused
```

**原因**：网络不通或 Base URL 不正确。
**解决**：
- 确认 `base_url` 拼写正确（注意 `/v1` 后缀）
- 检查防火墙/代理设置
- 自部署时确认服务已启动：`curl http://127.0.0.1:8000/v1/models`

---

## 7. 更多示例

完整的可运行示例请查看 [examples/](examples/) 目录：

| # | 示例 | 文件 | 说明 |
|---|------|------|------|
| 1 | Basic Chat | [`01_basic_chat.py`](examples/01_basic_chat.py) | 单轮 & 多轮对话 |
| 2 | Streaming | [`02_streaming.py`](examples/02_streaming.py) | 流式请求 + 逐 chunk 解析 |
| 3 | Latency Comparison | [`03_latency_compare.py`](examples/03_latency_compare.py) | 流式 vs 非流式：首 token 时延 & 总耗时 |
| 4 | Tool Calling | [`04_tool_calling.py`](examples/04_tool_calling.py) | 单次调用 + 多轮工具循环 |
| 5 | Reasoning Mode | [`05_reasoning_mode.py`](examples/05_reasoning_mode.py) | 深度思考 开/关 对比 |
| 6 | Error Handling | [`06_error_handling.py`](examples/06_error_handling.py) | 超时/限流/网络错误的重试与退避 |

所有示例可直接运行：

```bash
export HY3_API_KEY="your-key"
cd examples
python 01_basic_chat.py
```

---

## 附录：vLLM 部署快速参考

```bash
# 1. 启动 vLLM 服务
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
vllm serve tencent/Hy3 \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --port 8000 \
  --served-model-name hy3

# 2. 验证服务
curl http://127.0.0.1:8000/v1/models
```

## 附录：SGLang 部署快速参考

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3 \
  --tp-size 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --speculative-algorithm EAGLE \
  --port 8000 \
  --served-model-name hy3
```
