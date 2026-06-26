from __future__ import annotations

import os
import uuid
from typing import Any


def create_llm(config: Any):
    """Create ChatOpenAI from config.ini.

    운영환경에서 이미 llm = ChatOpenAI(...)를 만들었다면 이 함수를 쓰지 않고
    run_one_cycle(df=df, config=config, llm=llm)처럼 직접 넘기면 됩니다.
    """
    create_from_config = config.getboolean("llm", "create_from_config", fallback=True)
    if not create_from_config:
        raise ValueError("llm.create_from_config=false 입니다. run_one_cycle(..., llm=llm)로 직접 주입하세요.")

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage  # noqa: F401
    except ImportError as exc:
        raise ImportError("pip install langchain-openai langchain-core 이 필요합니다.") from exc

    os.environ["OPENAI_API_KEY"] = "api_key"

    base_url = _optional(config, "llm", "base_url")
    model_name = _model_name(config)
    temperature = config.getfloat("llm", "temperature", fallback=0.0)
    max_retries = config.getint("llm", "max_retries", fallback=2)

    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "max_retries": max_retries,
    }

    if base_url:
        kwargs["base_url"] = base_url
        kwargs["openai_proxy"] = base_url

    default_headers = _build_default_headers(config)
    if default_headers:
        kwargs["default_headers"] = default_headers

    return ChatOpenAI(**kwargs)


def _build_default_headers(config: Any) -> dict[str, str]:
    headers = {
        "Content-Type": config.get("llm", "content_type", fallback="application/json").strip() or "application/json",
        "x-dep-ticket": _optional(config, "llm", "x_dep_ticket"),
        "Send-System_Name": _optional(config, "llm", "send_system_name"),
        "User_Id": _optional(config, "llm", "user_id"),
        "User_Type": _optional(config, "llm", "user_type"),
        "Prompt-Msg_Id": _uuid_or_value(config, "llm", "prompt_msg_id"),
        "Completion-Msg-Id": _uuid_or_value(config, "llm", "completion_msg_id"),
    }
    return {key: value for key, value in headers.items() if value}


def _model_name(config: Any) -> str:
    model_name = _optional(config, "llm", "model_name")
    if model_name:
        return model_name
    old_model_key = _optional(config, "llm", "model")
    return old_model_key or "gpt-4.1-mini"


def _optional(config: Any, section: str, option: str) -> str:
    value = config.get(section, option, fallback="").strip()
    return "" if value.lower() in {"", "none", "null"} else value


def _uuid_or_value(config: Any, section: str, option: str) -> str:
    value = config.get(section, option, fallback="auto").strip()
    if value.lower() in {"", "auto", "uuid", "uuid4"}:
        return str(uuid.uuid4())
    return value
