"""
Example 1: Basic Chat — 单轮 & 多轮对话
===========================================

本示例演示 Hy3 API 最基本的对话能力：
  - 单轮对话：发送一个问题，获取回复
  - 多轮对话：在上下文中进行多次交互
  - System prompt：设定 AI 的角色和行为

运行方式：
    export HY3_API_KEY="your-api-key"
    python 01_basic_chat.py
"""

import os
import json
from openai import OpenAI

# ============================================================
# 配置
# ============================================================

API_KEY = os.environ.get("HY3_API_KEY", "your-api-key-here")
BASE_URL = os.environ.get("HY3_BASE_URL", "https://tokenhub.tencentmaas.com/v1")
MODEL = "hy3"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def print_section(title: str):
    """打印分隔标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# 1. 单轮对话 — 最简调用
# ============================================================

print_section("1. 单轮对话 (Single-Turn Chat)")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "你好！请用一句话介绍你自己。"},
    ],
    temperature=0.9,
    max_tokens=256,
)

print("📤 请求:")
print(f"   model: {MODEL}")
print(f"   messages: [{{role: 'user', content: '你好！请用一句话介绍你自己。'}}]")
print(f"   temperature: 0.9")
print(f"   max_tokens: 256")
print()

print("📥 完整 Response 对象:")
# 简化打印 response 的关键字段
print(f"   id: {response.id}")
print(f"   model: {response.model}")
print(f"   finish_reason: {response.choices[0].finish_reason}")
print(f"   content: {response.choices[0].message.content}")
print(f"   usage: prompt_tokens={response.usage.prompt_tokens}, "
      f"completion_tokens={response.usage.completion_tokens}, "
      f"total_tokens={response.usage.total_tokens}")

print()
print("💬 示例输出:")
print(f"   {response.choices[0].message.content}")


# ============================================================
# 2. 多轮对话 — 带上下文
# ============================================================

print_section("2. 多轮对话 (Multi-Turn Chat)")

messages = [
    {
        "role": "system",
        "content": "你是一个专业的 Python 编程助手。请用简洁清晰的中文回答问题，"
                   "必要时给出代码示例。",
    },
    {"role": "user", "content": "Python 中如何反转一个列表？"},
    {
        "role": "assistant",
        "content": "有两种常用方式：\n"
                   "1. 使用 `.reverse()` 方法原地反转：`my_list.reverse()`\n"
                   "2. 使用切片创建新列表：`reversed_list = my_list[::-1]`\n\n"
                   "你想了解它们的区别吗？",
    },
    {"role": "user", "content": "是的，请详细说明这两种方式的区别。"},
]

print("📤 请求 (多轮消息):")
for i, msg in enumerate(messages):
    role = msg["role"]
    content_preview = msg["content"][:80].replace("\n", " ")
    print(f"   [{i}] {role}: {content_preview}...")
print()

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.9,
    max_tokens=512,
)

print("📥 完整 Response 对象:")
print(f"   id: {response.id}")
print(f"   model: {response.model}")
print(f"   finish_reason: {response.choices[0].finish_reason}")
print(f"   usage: prompt_tokens={response.usage.prompt_tokens}, "
      f"completion_tokens={response.usage.completion_tokens}, "
      f"total_tokens={response.usage.total_tokens}")

print()
print("💬 示例输出:")
print(f"   {response.choices[0].message.content}")


# ============================================================
# 3. 带结构化输出的单轮对话
# ============================================================

print_section("3. 结构化 JSON 输出")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "列出 3 种常见的排序算法，用 JSON 格式返回，"
                       "包含 name（名称）和 complexity（时间复杂度）字段。",
        },
    ],
    temperature=0.3,  # 较低温度以增强格式确定性
    max_tokens=512,
    response_format={"type": "json_object"},
)

print("📤 请求:")
print(f"   model: {MODEL}")
print(f"   response_format: {{type: 'json_object'}}")
print(f"   temperature: 0.3")
print()

print("📥 完整 Response 对象:")
print(f"   finish_reason: {response.choices[0].finish_reason}")
print(f"   content (raw JSON string):")
print(f"   {response.choices[0].message.content}")

# 尝试解析 JSON
try:
    parsed = json.loads(response.choices[0].message.content)
    print()
    print("💬 解析后的示例输出:")
    print(f"   {json.dumps(parsed, ensure_ascii=False, indent=2)}")
except json.JSONDecodeError:
    print("   (JSON 解析失败，模型返回了非严格 JSON 格式)")

print()
print("✅ Example 1 完成！")
