"""
Example 5: Reasoning Mode — 思考过程 开/关 对比
================================================

本示例演示 Hy3 的「快慢思考融合」能力：
  - 快思考 (no_think / low)：直接输出答案，适合日常对话
  - 慢思考 (high / medium)：先推理再回答，适合复杂任务
  - 对比同一问题在不同推理模式下的输出差异
  - 展示如何获取 reasoning_content（思考过程）

运行方式：
    export HY3_API_KEY="your-api-key"
    python 05_reasoning_mode.py
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
# 测试问题集
# ============================================================

# 简单问题 — 不需要深度思考
SIMPLE_QUESTION = "请用一句话介绍腾讯公司。"

# 中等难度 — 需要一定推理
MEDIUM_QUESTION = "一个长方形的长是宽的 2 倍，周长是 36 厘米。求这个长方形的面积。"

# 困难问题 — 需要深度推理
HARD_QUESTION = "甲、乙、丙三人中只有一个人说了真话。甲说：'乙在说谎'。乙说：'丙在说谎'。丙说：'甲和乙都在说谎'。请问谁说了真话？请逐步推理。"


def run_comparison(question: str, label: str):
    """对比不同 reasoning_effort 模式下的输出"""
    print(f"\n{'─' * 56}")
    print(f"📝 问题 ({label}): {question[:60]}...")
    print(f"{'─' * 56}")

    modes = [
        ("no_think", "快思考 — 直接回答"),
        ("high", "慢思考 — 深度推理"),
    ]

    results = {}

    for effort, description in modes:
        print(f"\n🔹 {description} (reasoning_effort='{effort}')")
        print(f"{'─' * 40}")

        start_time = time.time()

        try:
            # 使用流式以分别捕获 reasoning_content 和 content
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": question}],
                temperature=0.9,
                max_tokens=1024,
                stream=True,
                reasoning_effort=effort,
            )

            reasoning_text = ""
            answer_text = ""
            chunk_count = 0

            for chunk in stream:
                chunk_count += 1
                delta = chunk.choices[0].delta

                # 捕获推理过程
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_text += delta.reasoning_content

                # 捕获最终回答
                if delta.content:
                    answer_text += delta.content

            end_time = time.time()

            # 打印结果
            if reasoning_text:
                print(f"   🧠 思考过程 ({len(reasoning_text)} 字符):")
                # 缩进显示思考内容的前 300 字符
                for line in reasoning_text[:300].split("\n"):
                    print(f"      {line}")
                if len(reasoning_text) > 300:
                    print(f"      ... (共 {len(reasoning_text)} 字符)")

            print(f"\n   💬 最终回答 ({len(answer_text)} 字符):")
            for line in answer_text[:400].split("\n"):
                print(f"      {line}")
            if len(answer_text) > 400:
                print(f"      ... (共 {len(answer_text)} 字符)")

            print(f"\n   ⏱️  总耗时: {end_time - start_time:.2f}s | "
                  f"chunks: {chunk_count} | "
                  f"思考: {len(reasoning_text)}字符 | "
                  f"回答: {len(answer_text)}字符")

            results[effort] = {
                "reasoning_chars": len(reasoning_text),
                "answer_chars": len(answer_text),
                "time": end_time - start_time,
                "chunks": chunk_count,
            }

        except Exception as e:
            print(f"   ⚠️ 错误: {e}")
            print(f"   提示: 部分平台可能不支持顶层 reasoning_effort 参数。")
            print(f"   自部署 (vLLM) 请通过 extra_body={{'chat_template_kwargs': {{'reasoning_effort': '{effort}'}}}} 传递。")
            results[effort] = None

    # 对比总结
    if all(v is not None for v in results.values()):
        print(f"\n   📊 对比总结:")
        print(f"   {'模式':<15} {'思考字符':>8} {'回答字符':>8} {'耗时':>8}")
        for effort, desc in modes:
            r = results[effort]
            print(f"   {effort:<15} {r['reasoning_chars']:>8} {r['answer_chars']:>8} {r['time']:>7.2f}s")

        # 分析差异
        no_think = results["no_think"]
        high = results["high"]
        if high["reasoning_chars"] > no_think["reasoning_chars"]:
            print(f"\n   💡 深度思考模式产生了 {high['reasoning_chars'] - no_think['reasoning_chars']} "
                  f"字符的额外推理内容。")
            print(f"   深度思考模式耗时约是快思考的 {high['time'] / no_think['time']:.1f} 倍，")
            print(f"   但答案质量通常更高（尤其是复杂推理题）。")

    return results


# ============================================================
# 1. 简单问题对比
# ============================================================

print_section("1. 简单问题 — 快思考 vs 慢思考")
run_comparison(SIMPLE_QUESTION, "简单")


# ============================================================
# 2. 困难问题对比
# ============================================================

print_section("2. 困难问题 — 快思考 vs 慢思考（重点对比）")
run_comparison(HARD_QUESTION, "困难")


# ============================================================
# 3. 推理模式使用建议
# ============================================================

print_section("3. 使用建议")

print("""
┌─────────────────┬──────────────────────┬──────────────────────┐
│ 推理模式         │ 适用场景              │ 特点                  │
├─────────────────┼──────────────────────┼──────────────────────┤
│ no_think / low  │ 日常对话、简单问答     │ 速度快，成本低         │
│                 │ 信息检索、翻译         │ 直接返回答案           │
├─────────────────┼──────────────────────┼──────────────────────┤
│ medium          │ 一般分析、综合         │ 适度推理，平衡速度质量  │
│                 │ 中等难度任务           │                       │
├─────────────────┼──────────────────────┼──────────────────────┤
│ high            │ 数学证明、复杂编程     │ 深度思维链，推理更准确  │
│                 │ 逻辑推理、代码调试     │ 耗时较长，成本较高      │
└─────────────────┴──────────────────────┴──────────────────────┘

💡 最佳实践:
  - 对于明确简单的任务，使用 "no_think" 节省成本和延迟
  - 对于需要推理的任务，先用 "high" 尝试
  - 可以通过比较两次调用的结果来判断是否需要深度思考
  - reasoning_content 可用于展示模型的推理过程，增强可解释性

⚠️ 自部署 (vLLM) 用户注意:
   reasoning_effort 通过 extra_body 传递:
   extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}}

   SGLang 用户请参考最新 SGLang cookbook。
""")

print()
print("✅ Example 5 完成！")
