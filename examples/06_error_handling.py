"""
Example 6: Error Handling & Retry — 超时/限流/网络错误的重试与退避
=================================================================

本示例演示 Hy3 API 调用的健壮错误处理：
  - 常见错误类型识别与分类
  - 指数退避重试策略
  - 超时处理
  - 限流 (429) 的智能等待
  - 生产级别的重试装饰器

运行方式：
    export HY3_API_KEY="your-api-key"
    python 06_error_handling.py
"""

import os
import time
import random
import functools
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
    InternalServerError,
    BadRequestError,
    AuthenticationError,
)

# ============================================================
# 配置
# ============================================================

API_KEY = os.environ.get("HY3_API_KEY", "your-api-key-here")
BASE_URL = os.environ.get("HY3_BASE_URL", "https://tokenhub.tencentmaas.com/v1")
MODEL = "hy3"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=30.0)


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# 1. 错误类型识别
# ============================================================

print_section("1. 常见错误类型")

ERROR_TABLE = """
┌──────────────────────────┬──────────┬──────────────────────────────────┐
│ 错误类型                  │ HTTP 码   │ 处理策略                           │
├──────────────────────────┼──────────┼──────────────────────────────────┤
│ AuthenticationError      │ 401      │ 检查 API Key，不重试                │
│ BadRequestError          │ 400      │ 检查请求参数，不重试                 │
│ RateLimitError           │ 429      │ 等待后重试（指数退避）               │
│ InternalServerError      │ 500      │ 指数退避重试                        │
│ APITimeoutError          │ —        │ 指数退避重试，增大 timeout           │
│ APIConnectionError       │ —        │ 指数退避重试，检查网络               │
│ APIError (其他)          │ 其他      │ 根据具体情况决定                     │
└──────────────────────────┴──────────┴──────────────────────────────────┘
"""
print(ERROR_TABLE)


# ============================================================
# 2. 生产级重试装饰器
# ============================================================

print_section("2. 生产级重试装饰器 (Retry with Exponential Backoff)")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_errors: tuple = (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    ),
):
    """
    指数退避重试装饰器。

    参数:
        max_retries: 最大重试次数
        base_delay: 基础等待时间（秒）
        max_delay: 最大等待时间（秒）
        backoff_factor: 退避因子
        jitter: 是否添加随机抖动
        retryable_errors: 可重试的错误类型
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_error = e

                    if attempt == max_retries:
                        print(f"   ❌ 已达最大重试次数 ({max_retries})，放弃。")
                        raise

                    # 计算延迟
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)  # 50%-100% 抖动

                    error_type = type(e).__name__
                    print(f"   ⚠️  第 {attempt + 1}/{max_retries} 次重试 — "
                          f"{error_type}: {str(e)[:80]} — "
                          f"等待 {delay:.1f}s...")
                    time.sleep(delay)

                except (AuthenticationError, BadRequestError) as e:
                    # 不可重试的错误
                    print(f"   ❌ 不可重试错误: {type(e).__name__}: {e}")
                    raise

            # 理论上不会走到这里
            raise last_error

        return wrapper
    return decorator


# ============================================================
# 3. 使用重试装饰器发起请求
# ============================================================

print_section("3. 使用重试装饰器 — 正常请求示例")

@retry_with_backoff(max_retries=3, base_delay=1.0)
def chat_with_retry(messages, **kwargs):
    """带自动重试的对话请求"""
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **kwargs,
    )


try:
    response = chat_with_retry(
        messages=[{"role": "user", "content": "用一句话问候世界。"}],
        temperature=0.9,
        max_tokens=64,
    )
    print(f"✅ 请求成功:")
    print(f"   {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ 请求失败: {type(e).__name__}: {e}")


# ============================================================
# 4. 模拟超时处理
# ============================================================

print_section("4. 超时处理")

# 创建一个带短超时的客户端来演示超时处理
short_timeout_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=0.001,  # 极短超时，几乎必定触发
)

print("📤 发送请求（使用极短超时以触发超时错误）...")

try:
    response = short_timeout_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "你好"}],
        max_tokens=10,
    )
    print("✅ 请求成功（超时未触发）")
except APITimeoutError as e:
    print(f"⏰ 超时错误捕获: {type(e).__name__}")
    print(f"   错误信息: {e}")
    print()
    print(f"💡 解决方案:")
    print(f"   1. 增大 timeout 参数: OpenAI(timeout=60.0)")
    print(f"   2. 减小 max_tokens 以缩短生成时间")
    print(f"   3. 使用 streaming 模式获取增量输出")
except Exception as e:
    print(f"   其他错误: {type(e).__name__}: {e}")


# ============================================================
# 5. 错误处理最佳实践
# ============================================================

print_section("5. 生产环境推荐模式")

print("""
以下是一个生产级 API 调用的推荐模板：

```python
import time
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

client = OpenAI(
    api_key="your-key",
    base_url="https://tokenhub.tencentmaas.com/v1",
    timeout=60.0,       # 设置合理超时
    max_retries=2,      # SDK 内置重试（仅对网络错误）
)

def safe_chat(messages, max_retries=3, **kwargs):
    \"\"\"带完整错误处理的生产级对话函数\"\"\"
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="hy3",
                messages=messages,
                **kwargs,
            )
            return response

        except RateLimitError as e:
            # 429 — 等待后重试
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"限流，等待 {wait:.1f}s 后重试...")
            time.sleep(wait)

        except (APITimeoutError, APIConnectionError) as e:
            # 网络/超时 — 指数退避
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"网络错误，等待 {wait}s 后重试...")
            time.sleep(wait)

        except Exception as e:
            # 未知错误 — 记录并抛出
            print(f"未知错误: {type(e).__name__}: {e}")
            raise

    return None
```

注意事项:
  1. 对于 401/400 等客户端错误，不应重试（请求本身有问题）
  2. 对于 429/500/超时，指数退避重试是合理策略
  3. 生产环境应记录每次重试的元数据（时间戳、错误类型、延迟等）
  4. 可使用 tenacity 库（pip install tenacity）获得更完善的重试管理
  5. 设置合理的 max_retries 和 max_delay 避免雪崩效应
""")

print()
print("✅ Example 6 完成！")
print()
print("=" * 60)
print("  🎉 所有 6 个示例已完成！")
print("  返回 quickstart.md 查看完整文档。")
print("=" * 60)
