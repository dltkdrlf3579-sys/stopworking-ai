from __future__ import annotations

from typing import Any

from .config import getbool
from .html_extract import extract_phenomenon


def row_to_case(row: Any, config: Any) -> dict:
    cols = config["columns"]
    row_id = safe_get(row, cols.get("id", "출원번호"))

    extracted = extract_phenomenon(
        safe_get(row, cols.get("phenomenon", "현상")),
        save_images=getbool(config, "artifacts", "save_images", False),
        image_dir=config.get("artifacts", "image_dir", fallback="artifacts/images"),
        row_id=str(row_id),
    )

    return {
        "id": row_id,
        "title": safe_get(row, cols.get("title", "제목")),
        "major": safe_get(row, cols.get("major", "대분류")),
        "middle": safe_get(row, cols.get("middle", "중분류")),
        "phenomenon_text": extracted.text,
        "image_count": extracted.image_count,
        "image_paths": extracted.image_paths,
        "action": safe_get(row, cols.get("action", "조치")),
        "label": normalize_label(safe_get(row, cols.get("label", "정답_판단결과"))),
    }


def safe_get(row: Any, key: str) -> str:
    try:
        value = row[key]
    except Exception:
        value = getattr(row, key, "")
    if value is None:
        return ""
    return str(value)


def normalize_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if "진" in text:
        return "진성"
    if "가" in text:
        return "가성"
    return text

