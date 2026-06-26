from __future__ import annotations

from collections import deque
import json
import re
import threading
import time
from typing import Any


class LlmRuntimeConfig:
    def __init__(self) -> None:
        self.calls_per_minute = 25
        self.retry_wait_seconds = 300
        self.max_attempts = 20
        self.rate_limiter = RateLimiter(self.calls_per_minute, 60)


class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: int) -> None:
        self.max_calls = max(1, max_calls)
        self.window_seconds = max(1, window_seconds)
        self._lock = threading.Lock()
        self._calls: deque[float] = deque()

    def update(self, max_calls: int, window_seconds: int = 60) -> None:
        with self._lock:
            self.max_calls = max(1, max_calls)
            self.window_seconds = max(1, window_seconds)
            self._calls.clear()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                sleep_seconds = max(0.1, self.window_seconds - (now - self._calls[0]) + 0.1)

            print(f"[llm-rate-limit] waiting {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)


_RUNTIME = LlmRuntimeConfig()


def configure_llm_runtime(calls_per_minute: int = 25, retry_wait_seconds: int = 300, max_attempts: int = 20) -> None:
    _RUNTIME.calls_per_minute = max(1, int(calls_per_minute))
    _RUNTIME.retry_wait_seconds = max(1, int(retry_wait_seconds))
    _RUNTIME.max_attempts = max(1, int(max_attempts))
    _RUNTIME.rate_limiter.update(_RUNTIME.calls_per_minute, 60)


def invoke_text(llm: Any, system: str, user: str) -> str:
    """Call LangChain-like llm and return content as text."""
    last_exc: Exception | None = None
    for attempt in range(1, _RUNTIME.max_attempts + 1):
        _RUNTIME.rate_limiter.wait()
        try:
            return _invoke_text_once(llm, system, user)
        except Exception as exc:
            last_exc = exc
            if not is_retryable_llm_error(exc) or attempt >= _RUNTIME.max_attempts:
                raise

            print(
                f"[llm-retry] attempt={attempt}/{_RUNTIME.max_attempts} "
                f"wait={_RUNTIME.retry_wait_seconds}s error={exc}",
                flush=True,
            )
            time.sleep(_RUNTIME.retry_wait_seconds)

    raise RuntimeError("LLM 호출 재시도 한도를 초과했습니다.") from last_exc


def _invoke_text_once(llm: Any, system: str, user: str) -> str:
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


def is_retryable_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if response_status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    text = str(exc).lower()
    retry_markers = [
        "too many request",
        "too many requests",
        "rate limit",
        "ratelimit",
        "429",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "server error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(marker in text for marker in retry_markers)


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
