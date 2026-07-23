# Hy3 API Examples

本目录包含 6 个可直接运行的 Python 示例，覆盖 Hy3 API 的主要能力。

## 示例列表

| # | 文件 | 说明 | 关键能力 |
|---|------|------|----------|
| 1 | `01_basic_chat.py` | 单轮 & 多轮对话 | `chat.completions.create`, system prompt, JSON 输出 |
| 2 | `02_streaming.py` | 流式请求 + 逐 chunk 解析 | `stream=True`, delta 解析, reasoning_content |
| 3 | `03_latency_compare.py` | 流式 vs 非流式延迟对比 | TTFT, 总耗时, 用户体验分析 |
| 4 | `04_tool_calling.py` | 工具调用与多轮循环 | tools 定义, tool_choice, 多轮工具循环 |
| 5 | `05_reasoning_mode.py` | 思考模式 开/关 对比 | reasoning_effort, 快慢思考, reasoning_content |
| 6 | `06_error_handling.py` | 错误处理与指数退避重试 | 重试装饰器, 超时处理, 生产级模板 |

## 运行方式

```bash
# 1. 设置 API Key
export HY3_API_KEY="your-api-key-here"

# 2. (可选) 自定义 Base URL
export HY3_BASE_URL="https://tokenhub.tencentmaas.com/v1"

# 3. 运行任意示例
python 01_basic_chat.py
```

## 环境要求

- Python 3.8+
- `openai` 包：`pip install openai`

## 输出说明

每个示例运行后会打印：
1. 📤 **完整请求** — 请求参数和配置
2. 📥 **完整 Response 解析** — 响应的关键字段
3. 💬 **示例输出** — 模型的实际回复（截取关键部分）

## 注意事项

- 示例中使用了模拟数据（如天气查询、快递查询结果），确保在没有外部依赖的情况下也能跑通
- 请替换 `your-api-key-here` 为真实的 API Key，或通过环境变量 `HY3_API_KEY` 传入
- 不同 Hy3 API 平台的参数传递方式可能略有差异，请参考 `quickstart.md`
