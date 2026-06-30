from __future__ import annotations

from typing import Any

from .config import getbool
from .html_extract import extract_phenomenon


def row_to_case(row: Any, config: Any) -> dict:
    cols = config["columns"]
    row_id = safe_get(row, cols.get("id", "출원번호"))
    max_phenomenon_chars = config.getint("preprocess", "max_phenomenon_chars", fallback=8000)
    max_action_chars = config.getint("preprocess", "max_action_chars", fallback=2000)
    max_title_chars = config.getint("preprocess", "max_title_chars", fallback=300)
    max_total_images = config.getint("preprocess", "max_total_images", fallback=5)
    max_phenomenon_images = config.getint("preprocess", "max_phenomenon_images", fallback=3)
    max_action_images = config.getint("preprocess", "max_action_images", fallback=2)
    max_phenomenon_images, max_action_images = normalize_image_limits(
        max_total_images,
        max_phenomenon_images,
        max_action_images,
    )

    phenomenon = extract_phenomenon(
        safe_get(row, cols.get("phenomenon", "현상")),
        save_images=getbool(config, "artifacts", "save_images", False),
        image_dir=config.get("artifacts", "image_dir", fallback="artifacts/images"),
        row_id=f"{row_id}_phenomenon",
        max_chars=max_phenomenon_chars,
        max_images=max_phenomenon_images,
    )
    action = extract_phenomenon(
        safe_get(row, cols.get("action", "조치")),
        save_images=getbool(config, "artifacts", "save_images", False),
        image_dir=config.get("artifacts", "image_dir", fallback="artifacts/images"),
        row_id=f"{row_id}_action",
        max_chars=max_action_chars,
        max_images=max_action_images,
    )
    image_paths = phenomenon.image_paths + action.image_paths
    image_data_urls = phenomenon.image_data_urls + action.image_data_urls
    image_count = phenomenon.image_count + action.image_count
    original_image_count = phenomenon.original_image_count + action.original_image_count
    omitted_image_count = phenomenon.omitted_image_count + action.omitted_image_count

    return {
        "id": row_id,
        "title": truncate_plain_text(safe_get(row, cols.get("title", "제목")), max_title_chars),
        "major": safe_get(row, cols.get("major", "대분류")),
        "middle": safe_get(row, cols.get("middle", "중분류")),
        "phenomenon_text": phenomenon.text,
        "image_count": image_count,
        "original_image_count": original_image_count,
        "omitted_image_count": omitted_image_count,
        "phenomenon_image_count": phenomenon.image_count,
        "action_image_count": action.image_count,
        "image_paths": image_paths[:max_total_images],
        "image_data_urls": image_data_urls[:max_total_images],
        "phenomenon_truncated": phenomenon.truncated,
        "action_truncated": action.truncated,
        "action": action.text,
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


def truncate_plain_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[길이제한으로 이하 생략]"


def normalize_image_limits(max_total: int, phenomenon_limit: int, action_limit: int) -> tuple[int, int]:
    max_total = max(0, max_total)
    phenomenon_limit = max(0, phenomenon_limit)
    action_limit = max(0, action_limit)

    if phenomenon_limit + action_limit <= max_total:
        return phenomenon_limit, action_limit

    phenomenon_limit = min(phenomenon_limit, max_total)
    action_limit = max(0, max_total - phenomenon_limit)
    return phenomenon_limit, action_limit


def normalize_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if "진" in text:
        return "진성"
    if "가" in text:
        return "가성"
    return text
