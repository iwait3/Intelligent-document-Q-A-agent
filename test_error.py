#!/usr/bin/env python3
"""测试脚本，复现 'Cannot send a request, as the client has been closed' 错误"""

import os
import json
import config
from langchain_openai import ChatOpenAI
import sys

def test_llm_connection():
    """测试 LLM 连接"""
    print("=== 测试 LLM 连接 ===")

    # 创建 LLM 实例
    llm = ChatOpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        model=config.CHAT_MODEL,
        temperature=0.3,
    )

    print(f"模型配置: {config.CHAT_MODEL}")
    print(f"API Key: {'已设置' if config.OPENROUTER_API_KEY else '未设置'}")

    # 测试消息
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Hello, please respond with 'Test successful'"}
    ]

    try:
        # 调用 LLM
        response = llm.invoke(messages)
        print("Success:", response.content)
        return True
    except Exception as e:
        print("Error:", type(e).__name__, str(e))
        return False

if __name__ == "__main__":
    # 检查环境变量
    print("=== 检查环境变量 ===")
    print("OPENROUTER_API_KEY:", "已设置" if config.OPENROUTER_API_KEY else "未设置")
    print("CHAT_MODEL:", config.CHAT_MODEL)
    print("OPENROUTER_BASE_URL:", config.OPENROUTER_BASE_URL)

    # 测试连接
    success = test_llm_connection()

    if not success:
        print("\n=== 可能的原因 ===")
        print("1. API Key 未正确设置或已过期")
        print("2. 网络连接问题")
        print("3. OpenRouter 服务不可用")
        print("4. 模型名称不正确或已被移除")