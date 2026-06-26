from __future__ import annotations

import json
import re
from typing import Any


def invoke_text(llm: Any, system: str, user: str) -> str:
    """Call LangChain-like llm and return content as text."""
    if hasattr(llm, "invoke"):
        try:
            response = llm.invoke([("system", system), ("human", user)])
        except TypeError:
            response = llm.invoke(f"{system}\n\n{user}")
    elif callable(llm):
        response = llm(f"{system}\n\n{user}")
    else:
        raise TypeError("llm은 .invoke(...)를 지원하거나 callable이어야 합니다.")

    return str(getattr(response, "content", response))


def invoke_json(llm: Any, system: str, user: str) -> dict:
    text = invoke_text(llm, system, user)
    return parse_json_object(text)


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM JSON 파싱 실패: {exc}\n원문 일부: {text[:1000]}") from exc

    if not isinstance(obj, dict):
        raise ValueError("LLM 응답 JSON이 object가 아닙니다.")
    return obj

