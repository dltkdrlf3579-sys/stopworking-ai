from __future__ import annotations

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
    except ImportError as exc:
        raise ImportError("pip install langchain-openai 이 필요합니다.") from exc

    model = config.get("llm", "model", fallback="gpt-4.1-mini")
    temperature = config.getfloat("llm", "temperature", fallback=0.0)
    max_retries = config.getint("llm", "max_retries", fallback=2)

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_retries": max_retries,
    }

    optional_kwargs = {
        "base_url": _optional(config, "llm", "base_url"),
        "openai_proxy": _optional(config, "llm", "openai_proxy"),
        "api_key": _optional(config, "llm", "api_key"),
    }
    kwargs.update({key: value for key, value in optional_kwargs.items() if value})

    default_headers = _build_default_headers(config)
    if default_headers:
        kwargs["default_headers"] = default_headers

    return ChatOpenAI(**kwargs)


def _build_default_headers(config: Any) -> dict[str, str]:
    headers = {
        "Content-Type": config.get("llm", "content_type", fallback="application/json").strip() or "application/json",
        "X-dep_ticket": _optional(config, "llm", "x_dep_ticket"),
        "Send-System-Name": _optional(config, "llm", "send_system_name"),
        "User-Id": _optional(config, "llm", "user_id"),
        "User-Type": _optional(config, "llm", "user_type"),
        "Prompt-MSG-ID": _uuid_or_value(config, "llm", "prompt_msg_id"),
        "Completion-Msg-Id": _uuid_or_value(config, "llm", "completion_msg_id"),
    }
    return {key: value for key, value in headers.items() if value}


def _optional(config: Any, section: str, option: str) -> str:
    value = config.get(section, option, fallback="").strip()
    return "" if value.lower() in {"", "none", "null"} else value


def _uuid_or_value(config: Any, section: str, option: str) -> str:
    value = config.get(section, option, fallback="auto").strip()
    if value.lower() in {"", "auto", "uuid", "uuid4"}:
        return str(uuid.uuid4())
    return value
