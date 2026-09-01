"""统一的环境配置读取（缺配置抛清晰错误，兼容 BASE_URL/OPENAI_BASE_URL）。"""
from __future__ import annotations

import os


def load_env(override: bool = True) -> dict:
    """读取 .env 并返回规范化的模型配置字典。

    dotenv 在此处延迟导入，保证 import harness 本身只依赖标准库，
    便于 CI 在未安装第三方依赖时对 harness 做单元测试。
    """
    from dotenv import load_dotenv  # 延迟导入，见 docstring

    load_dotenv(override=override)
    model_id = os.getenv("MODEL_ID")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
    fallback_model_id = os.getenv("FALLBACK_MODEL_ID")
    if not model_id:
        raise RuntimeError("Missing MODEL_ID in .env")
    return {
        "model_id": model_id,
        "api_key": api_key,
        "base_url": base_url,
        "fallback_model_id": fallback_model_id,
    }
