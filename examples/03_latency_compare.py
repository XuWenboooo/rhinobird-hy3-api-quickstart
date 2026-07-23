"""
Example 3: Non-Streaming vs Streaming — 首 Token 时延 & 总耗时对比
==================================================================

本示例对比 Hy3 API 的两种输出模式：
  - 非流式 (non-streaming)：等待完整响应后一次性返回
  - 流式 (streaming)：逐步返回生成内容

关键指标：
  - TTFT (Time To First Token)：首个 token 的等待时间
  - 总耗时：从发起请求到收到完整回复的时间
  - 用户体验差异：流式可提前看到部分内容

运行方式：
    export HY3_API_KEY="your-api-key"
    python 03_latency_compare.py
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

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0)


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# 测试配置
# ============================================================

# 使用相同 prompt 和参数，确保对比公平
PROMPTS = [
    "请用 200 字左右介绍机器学习的主要分类。",
    "请列出 Python 中 10 个常用的标准库并简要说明其用途。",
    "请写一段约 150 字的文字，解释什么是深度学习。",
]

# 限制输出长度以保持测试可重复
MAX_TOKENS = 300


# ============================================================
# 1. 非流式请求 (Non-Streaming)
# ============================================================

print_section("1. 非流式请求 (Non-Streaming)")

non_streaming_results = []

for i, prompt in enumerate(PROMPTS):
    print(f"📤 请求 #{i+1}: \"{prompt[:60]}...\"")

    start_time = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    end_time = time.time()

    total_time = end_time - start_time
    content = response.choices[0].message.content
    token_count = response.usage.completion_tokens

    non_streaming_results.append({
        "prompt": prompt,
        "total_time": total_time,
        "output_chars": len(content),
        "output_tokens": token_count,
    })

    print(f"   ✅ 总耗时: {total_time:.2f}s | "
          f"输出 tokens: {token_count} | "
          f"字符数: {len(content)}")
    print()


# ============================================================
# 2. 流式请求 (Streaming)
# ============================================================

print_section("2. 流式请求 (Streaming)")

streaming_results = []

for i, prompt in enumerate(PROMPTS):
    print(f"📤 请求 #{i+1}: \"{prompt[:60]}...\"")

    ttft = None          # Time To First Token
    start_time = time.time()
    full_content = ""
    chunk_count = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},  # 请求在最后 chunk 包含 usage
    )

    for chunk in stream:
        chunk_count += 1
        delta = chunk.choices[0].delta

        if delta.content:
            # 记录首 token 到达时间
            if ttft is None:
                ttft = time.time() - start_time
            full_content += delta.content

    end_time = time.time()
    total_time = end_time - start_time

    streaming_results.append({
        "prompt": prompt,
        "ttft": ttft,
        "total_time": total_time,
        "chunk_count": chunk_count,
        "output_chars": len(full_content),
    })

    if ttft is not None:
        print(f"   ✅ TTFT: {ttft:.3f}s | "
              f"总耗时: {total_time:.2f}s | "
              f"chunks: {chunk_count} | "
              f"字符数: {len(full_content)}")
    else:
        print(f"   ⚠️  未收到有效内容")
    print()


# ============================================================
# 3. 对比分析
# ============================================================

print_section("3. 对比分析 (Comparison)")

print(f"{'指标':<30} {'非流式':>12} {'流式':>12} {'差异':>12}")
print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")

for i in range(len(PROMPTS)):
    ns = non_streaming_results[i]
    ss = streaming_results[i]

    print(f"Prompt #{i+1}:")
    print(f"  {'总耗时':<28} {ns['total_time']:>11.2f}s {ss['total_time']:>11.2f}s "
          f"{ns['total_time'] - ss['total_time']:>+11.2f}s")

    if ss['ttft'] is not None:
        # 用户感知延迟：流式为首 token 时间；非流式为总耗时
        print(f"  {'用户感知延迟':<26} {ns['total_time']:>11.2f}s {ss['ttft']:>11.3f}s "
              f"{ns['total_time'] - ss['ttft']:>+11.2f}s ★")
    print()

print("★ 用户感知延迟 = 非流式的总耗时 vs 流式的首 token 时间")
print("   流式模式下，用户几乎立即看到响应开始生成，体验显著更好。")
print()

# 汇总统计
avg_ns_time = sum(r["total_time"] for r in non_streaming_results) / len(non_streaming_results)
avg_s_time = sum(r["total_time"] for r in streaming_results) / len(streaming_results)
avg_ttft = sum(r["ttft"] for r in streaming_results if r["ttft"]) / len(streaming_results)

print(f"📊 汇总统计:")
print(f"   非流式平均总耗时:     {avg_ns_time:.2f}s")
print(f"   流式平均总耗时:       {avg_s_time:.2f}s")
print(f"   流式平均首 Token 延迟: {avg_ttft:.3f}s")
print()
print(f"💡 结论:")
print(f"   - 流式模式的首 token 延迟约 {avg_ttft:.3f}s，用户可快速看到响应")
print(f"   - 非流式模式下用户需等待约 {avg_ns_time:.2f}s 才能看到任何内容")
print(f"   - 总耗时方面两者相近（差距主要来自网络传输模式的不同）")

print()
print("✅ Example 3 完成！")
