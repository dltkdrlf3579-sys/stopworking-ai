from __future__ import annotations

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

    return ChatOpenAI(model=model, temperature=temperature, max_retries=max_retries)

