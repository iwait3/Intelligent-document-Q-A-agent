#!/usr/bin/env python3
"""详细的连接调试脚本"""

import os
import sys
import json
import time
import requests
import config
from langchain_openai import ChatOpenAI

def test_api_direct():
    """直接测试 OpenRouter API"""
    print("=== 直接 API 测试 ===")

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "SmartDoc"
    }

    data = {
        "model": config.CHAT_MODEL,
        "messages": [
            {"role": "user", "content": "Hello, just testing connection"}
        ],
        "max_tokens": 50
    }

    try:
        print(f"发送请求到: {config.OPENROUTER_BASE_URL}/chat/completions")
        print(f"使用模型: {config.CHAT_MODEL}")

        response = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ API 调用成功")
            print(f"响应: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ API 调用失败: {response.text}")

    except requests.exceptions.Timeout:
        print("❌ 请求超时")
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误")
    except Exception as e:
        print(f"❌ 其他错误: {type(e).__name__}: {e}")

def test_env_vars():
    """检查所有环境变量"""
    print("\n=== 环境变量检查 ===")
    print(f"OPENROUTER_API_KEY: {'已设置' if config.OPENROUTER_API_KEY else '未设置'}")
    if config.OPENROUTER_API_KEY:
        print(f"API Key 长度: {len(config.OPENROUTER_API_KEY)}")
        print(f"API Key 前缀: {config.OPENROUTER_API_KEY[:10]}...")

    print(f"CHAT_MODEL: {config.CHAT_MODEL}")
    print(f"OPENROUTER_BASE_URL: {config.OPENROUTER_BASE_URL}")
    print(f"LOCAL_EMBEDDING_MODEL: {config.LOCAL_EMBEDDING_MODEL}")

def test_llm_with_retry():
    """测试 LLM 连接（带重试）"""
    print("\n=== LLM 连接测试（带重试）===")

    llm = ChatOpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        model=config.CHAT_MODEL,
        temperature=0.3,
        timeout=30,
        request_timeout=30,
        max_retries=2
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Just say 'test ok'"}
    ]

    for i in range(3):  # 尝试3次
        try:
            print(f"尝试第 {i+1} 次...")
            start_time = time.time()
            response = llm.invoke(messages)
            elapsed = time.time() - start_time
            print(f"✅ 成功! 耗时: {elapsed:.2f}秒")
            print(f"响应: {response.content}")
            return True
        except Exception as e:
            print(f"❌ 第 {i+1} 次失败: {type(e).__name__}: {e}")
            if i < 2:
                time.sleep(2)  # 等待2秒再重试

    return False

if __name__ == "__main__":
    test_env_vars()
    test_api_direct()
    test_llm_with_retry()