"""
Example 4: Tool Calling — 单次调用 + 多轮工具循环
===================================================

本示例演示 Hy3 API 的工具调用 (Function Calling) 能力：
  - 定义工具 (Tool Schema)
  - 单次工具调用：模型返回 tool_calls
  - 多轮工具循环：模型 → 调用工具 → 回传结果 → 模型整合回答
  - tool_choice 的用法 (auto / none / required)

运行方式：
    export HY3_API_KEY="your-api-key"
    python 04_tool_calling.py
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
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# 定义模拟工具
# ============================================================

# 模拟的天气查询函数
def get_weather(city: str) -> dict:
    """模拟天气查询（实际项目中可替换为真实 API 调用）"""
    weather_data = {
        "北京": {"temperature": "32°C", "weather": "晴", "humidity": "45%"},
        "上海": {"temperature": "28°C", "weather": "多云转小雨", "humidity": "70%"},
        "深圳": {"temperature": "33°C", "weather": "雷阵雨", "humidity": "85%"},
        "杭州": {"temperature": "30°C", "weather": "阴", "humidity": "60%"},
    }
    result = weather_data.get(city, {"temperature": "未知", "weather": "无数据", "humidity": "未知"})
    result["city"] = city
    return result


# 模拟的快递查询函数
def query_express(tracking_number: str) -> dict:
    """模拟快递查询（实际项目中可替换为真实 API 调用）"""
    return {
        "tracking_number": tracking_number,
        "status": "运输中",
        "location": "广州分拣中心",
        "estimated_delivery": "2026-07-25",
        "carrier": "顺丰速运",
    }


# ============================================================
# 定义工具 Schema
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气信息，包括温度、天气状况和湿度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海、深圳",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_express",
            "description": "根据快递单号查询包裹的物流状态和预计送达时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {
                        "type": "string",
                        "description": "快递单号",
                    },
                },
                "required": ["tracking_number"],
            },
        },
    },
]

# 工具名 → 实现函数的映射
TOOL_HANDLERS = {
    "get_weather": get_weather,
    "query_express": query_express,
}


# ============================================================
# 1. 单次工具调用
# ============================================================

print_section("1. 单次工具调用 (Single Tool Call)")

messages = [
    {"role": "user", "content": "北京今天天气怎么样？"},
]

print("📤 请求:")
print(f"   model: {MODEL}")
print(f"   messages: [{{role: 'user', content: '北京今天天气怎么样？'}}]")
print(f"   tools: [get_weather, query_express]")
print(f"   tool_choice: 'auto'")
print()

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
    temperature=0.9,
)

assistant_msg = response.choices[0].message

print("📥 完整 Response 对象:")
print(f"   finish_reason: {response.choices[0].finish_reason}")
print(f"   has_tool_calls: {assistant_msg.tool_calls is not None}")

if assistant_msg.tool_calls:
    for i, tc in enumerate(assistant_msg.tool_calls):
        print(f"\n   Tool Call #{i+1}:")
        print(f"     id: {tc.id}")
        print(f"     function.name: {tc.function.name}")
        print(f"     function.arguments: {tc.function.arguments}")

        # 执行工具调用
        args = json.loads(tc.function.arguments)
        result = TOOL_HANDLERS[tc.function.name](**args)
        print(f"\n💬 模拟工具执行结果:")
        print(f"   {json.dumps(result, ensure_ascii=False)}")

        print(f"\n💬 示例输出（将工具结果返回模型后）:")
        # 构建完整的工具调用消息
        messages.append(assistant_msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.9,
            max_tokens=256,
        )
        print(f"   {final_response.choices[0].message.content}")
else:
    print("   (模型选择直接回复，未触发工具调用)")
    print(f"   content: {assistant_msg.content}")


# ============================================================
# 2. 多轮工具循环 (Multi-Turn Tool Loop)
# ============================================================

print_section("2. 多轮工具循环 (Multi-Turn Tool Loop)")

messages = [
    {
        "role": "user",
        "content": "请帮我做三件事：1) 查一下深圳的天气；2) 查快递单号 SF1234567890 的物流状态；"
                   "3) 根据天气和物流情况，给我一个简短的出行建议。",
    },
]

MAX_TOOL_ROUNDS = 5  # 防止无限循环

print("📤 请求（含多个工具需求）:")
print(f"   user: {messages[0]['content'][:80]}...")
print()

for turn in range(MAX_TOOL_ROUNDS):
    print(f"--- 第 {turn + 1} 轮 ---")

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.9,
    )

    assistant_msg = response.choices[0].message

    # 如果模型不再调用工具，输出最终回复
    if not assistant_msg.tool_calls:
        print(f"✅ 模型最终回复:")
        print(f"   {assistant_msg.content}")
        break

    # 处理所有工具调用
    messages.append(assistant_msg)
    for tc in assistant_msg.tool_calls:
        print(f"   🔧 调用工具: {tc.function.name}({tc.function.arguments})")
        args = json.loads(tc.function.arguments)
        result = TOOL_HANDLERS[tc.function.name](**args)
        print(f"   📦 工具返回: {json.dumps(result, ensure_ascii=False)}")

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result, ensure_ascii=False),
        })
    print()

if turn + 1 >= MAX_TOOL_ROUNDS:
    print("⚠️  达到最大工具轮次限制，中断循环。")


# ============================================================
# 3. tool_choice 选项说明
# ============================================================

print_section("3. tool_choice 选项对比")

test_messages = [
    {"role": "user", "content": "今天心情不错，给我推荐一部电影吧。"},
]

# 3a. tool_choice="none" — 不使用工具
print("3a. tool_choice='none' — 不使用工具，直接回复")
response = client.chat.completions.create(
    model=MODEL,
    messages=test_messages,
    tools=TOOLS,
    tool_choice="none",  # 模型不能调用工具
    temperature=0.9,
    max_tokens=128,
)
msg = response.choices[0].message
print(f"   has_tool_calls: {msg.tool_calls is not None}")
print(f"   content: {msg.content[:100]}...")
print()

# 3b. tool_choice="required" — 强制调用工具
print("3b. tool_choice='required' — 强制调用工具")
print("   (即使问题不需要工具，模型也必须调用)")
response = client.chat.completions.create(
    model=MODEL,
    messages=test_messages,
    tools=TOOLS,
    tool_choice="required",
    temperature=0.9,
)
msg = response.choices[0].message
print(f"   has_tool_calls: {msg.tool_calls is not None}")
if msg.tool_calls:
    print(f"   仍然调用了: {msg.tool_calls[0].function.name}")
print()

print("✅ Example 4 完成！")
