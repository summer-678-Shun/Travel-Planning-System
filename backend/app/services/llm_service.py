"""LLM服务模块：基于 LangChain ChatOpenAI。"""

import os
from langchain_openai import ChatOpenAI
from ..config import get_settings

# 全局LLM实例
_llm_instance = None


def get_llm() -> ChatOpenAI:
    """
    获取LLM实例(单例模式)

    Returns:
        LangChain ChatOpenAI实例
    """
    global _llm_instance

    if _llm_instance is None:
        settings = get_settings()

        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or settings.openai_base_url
        model = os.getenv("LLM_MODEL_ID") or os.getenv("OPENAI_MODEL") or settings.openai_model

        if not api_key:
            raise ValueError("LLM_API_KEY或OPENAI_API_KEY未配置,请在.env文件中设置")

        _llm_instance = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
        )

        print("✅ LangChain LLM服务初始化成功")
        print("   提供商: OpenAI兼容接口")
        print(f"   模型: {model}")
        print(f"   Base URL: {base_url}")

    return _llm_instance


def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_instance
    _llm_instance = None
