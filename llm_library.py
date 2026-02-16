from __future__ import annotations

from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "google": {
        "provider": "google",
        "base_url": None,
        "lc_type": ChatGoogleGenerativeAI,
        "api_key_env": "GOOGLE_API_KEY",
    },
    "together": {
        "provider": "together",
        "base_url": "https://api.together.xyz/v1",
        "lc_type": ChatOpenAI,
        "api_key_env": "TOGETHER_API_KEY",
    },
    "deepinfra": {
        "provider": "deepinfra",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "lc_type": ChatOpenAI,
        "api_key_env": "DEEPINFRA_API_KEY",
    },
    "moonshot": {
        "provider": "moonshot",
        "base_url": "https://api.moonshot.ai/v1",
        "lc_type": ChatOpenAI,
        "api_key_env": "MOONSHOT_API_KEY",
    },
}

LLM_MODELS: Dict[str, Dict[str, str]] = {
    "GEMINI_2_5_FLASH_LITE": {
        "name": "gemini-2.5-flash-lite",
        "provider": "google",
    },
    "GEMINI_2_5_FLASH": {
        "name": "gemini-2.5-flash",
        "provider": "google",
    },
    "KIMI_K2_5_TOGETHER": {
        "name": "moonshotai/Kimi-K2.5",
        "provider": "together",
    },
    "KIMI_K2_INSTRUCT_0905_TOGETHER": {
        "name": "moonshotai/Kimi-K2-Instruct-0905",
        "provider": "together",
    },
    "KIMI_K2_INSTRUCT_0905_DEEPINFRA": {
        "name": "moonshotai/Kimi-K2-Instruct-0905",
        "provider": "deepinfra",
    },
    "QWEN3_CODER_480B_A35B_DEEPINFRA": {
        "name": "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
        "provider": "deepinfra",
    },
    "QWEN3_NEXT_80B_A3B_DEEPINFRA": {
        "name": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "provider": "deepinfra",
    },
    "KIMI_K2_TURBO_PREVIEW": {
        "name": "kimi-k2-turbo-preview",
        "provider": "moonshot",
    },
    "KIMI_K2_5": {
        "name": "kimi-k2.5",
        "provider": "moonshot",
    },
    "KIMI_K2_THINKING_TURBO": {
        "name": "kimi-k2-thinking-turbo",
        "provider": "moonshot",
    },
}
