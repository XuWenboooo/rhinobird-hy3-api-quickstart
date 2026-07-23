"""
Example 2: Streaming — 流式请求 + 逐 chunk 解析
================================================

本示例演示 Hy3 API 的流式输出能力：
  - 启用 streaming 模式
  - 逐 chunk 解析 delta 内容
  - 区分 reasoning_content 和普通 content
  - 计算流式传输的统计信息

运行方式：
    export HY3_API_KEY="your-api-key"
    python 02_streaming.py
"""

import os
import time
from openai import OpenAI

# ============================================================
# 配置
# ============================================================

API_KEY = os.environ.get("HY3_API_KEY", "your-api-key-here")
BASE_URL = os.environ.get("HY3_BASE_URL", "https://tokenhub.tencentmaas.com/v1")
MODEL = "hy3"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# 1. 基本流式请求
# ============================================================

print_section("1. 基本流式请求 (Basic Streaming)")

prompt = "请用三句话介绍人工智能的发展历史。"

print("📤 请求:")
print(f"   model: {MODEL}")
print(f"   stream: True")
print(f"   prompt: \"{prompt}\"")
print()

print("📥 流式输出 (实时打印):")
print("-" * 40)

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.9,
    max_tokens=512,
    stream=True,
)

# 收集完整内容
full_content = ""
chunk_count = 0
start_time = time.time()

for chunk in stream:
    chunk_count += 1
    delta = chunk.choices[0].delta

    if delta.content:
        full_content += delta.content
        print(delta.content, end="", flush=True)

end_time = time.time()

print("\n" + "-" * 40)
print()
print(f"📊 统计:")
print(f"   总 chunks: {chunk_count}")
print(f"   总耗时: {end_time - start_time:.2f}s")
print(f"   输出字符数: {len(full_content)}")


# ============================================================
# 2. 流式请求 — 进阶：逐 chunk 解析所有字段
# ============================================================

print_section("2. 逐 Chunk 解析 (Per-Chunk Parsing)")

prompt = "请比较 Python 和 JavaScript 的主要区别。"

print("📤 请求:")
print(f"   prompt: \"{prompt}\"")

print()
print("📥 逐 chunk 详情 (前 10 个 chunks):")
print(f"   {'#':<5} {'delta_content':<30} {'finish_reason':<15}")
print(f"   {'-'*5} {'-'*30} {'-'*15}")

stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.9,
    max_tokens=256,
    stream=True,
)

chunk_count = 0
full_text = ""

for chunk in stream:
    chunk_count += 1
    delta = chunk.choices[0].delta
    finish_reason = chunk.choices[0].finish_reason or "null"
    content = delta.content or ""

    full_text += content

    if chunk_count <= 10:
        # 截断显示以避免过长
        display = content[:25].replace("\n", "\\n") + ("..." if len(content) > 25 else "")
        print(f"   {chunk_count:<5} {display:<30} {finish_reason:<15}")

if chunk_count > 10:
    print(f"   ... (省略 {chunk_count - 10} 个 chunks)")

print()
print(f"📊 统计:")
print(f"   总 chunks: {chunk_count}")
print(f"   完整输出长度: {len(full_text)} 字符")

print()
print("💬 完整输出:")
print(full_text[:300] + ("..." if len(full_text) > 300 else ""))


# ============================================================
# 3. 流式请求 — 带 reasoning (深度思考时可用)
# ============================================================

print_section("3. 流式请求 — 捕获推理过程 (Reasoning Content)")

prompt = "计算 15 * 37 + 28 * 43 的结果。请一步一步算。"

print("📤 请求:")
print(f"   prompt: \"{prompt}\"")
print(f"   reasoning_effort: \"high\" (尝试启用深度思考)")
print()
print("📥 流式输出 (区分 reasoning 和 answer):")
print("-" * 40)

try:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=512,
        stream=True,
        # 注意: 不同平台 reasoning_effort 传递方式不同
        # TokenHub: 直接传 reasoning_effort="high"
        # 自部署: 通过 extra_body
        reasoning_effort="high",
    )

    reasoning_text = ""
    answer_text = ""

    for chunk in stream:
        delta = chunk.choices[0].delta

        # reasoning_content: 模型的思考过程
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_text += delta.reasoning_content
            print(f"[思考] {delta.reasoning_content}", end="", flush=True)

        # content: 模型最终输出
        if delta.content:
            answer_text += delta.content
            print(delta.content, end="", flush=True)

    print("\n" + "-" * 40)
    print()
    if reasoning_text:
        print(f"📊 思考过程 ({len(reasoning_text)} 字符)")
        print(f"📊 最终答案 ({len(answer_text)} 字符)")
    else:
        print("ℹ️  当前请求未触发深度思考（reasoning_content 为空）。")
        print("   这可能是 API 平台或模型版本的差异，请以实际平台文档为准。")

except Exception as e:
    print(f"\n⚠️  错误: {e}")
    print("   提示: 部分平台可能不支持 reasoning_effort 顶层参数，")
    print("   请参考 quickstart.md 了解不同平台的传参方式。")


print()
print("✅ Example 2 完成！")
