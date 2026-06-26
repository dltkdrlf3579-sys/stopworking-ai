from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATA_IMAGE_RE = re.compile(
    r"data:image/(?P<ext>png|jpeg|jpg|gif|webp);base64,(?P<data>[A-Za-z0-9+/=\s]+)",
    re.IGNORECASE,
)

DATA_URI_RE = re.compile(
    r"data:[^,\s\"'<>)]*,[A-Za-z0-9+/=\s%_-]{128,}",
    re.IGNORECASE,
)

BARE_BASE64_RE = re.compile(
    r"base64\s*,\s*[A-Za-z0-9+/=\s]{512,}",
    re.IGNORECASE,
)


@dataclass
class ExtractedPhenomenon:
    text: str
    image_count: int
    original_image_count: int
    omitted_image_count: int
    image_paths: list[str]
    truncated: bool = False


def extract_phenomenon(
    raw_html_or_text: object,
    save_images: bool = False,
    image_dir: str | None = None,
    row_id: str = "",
    max_chars: int | None = None,
    max_images: int | None = None,
) -> ExtractedPhenomenon:
    raw = "" if raw_html_or_text is None else str(raw_html_or_text)
    image_paths: list[str] = []

    matches = list(DATA_IMAGE_RE.finditer(raw))
    selected_matches = limit_matches(matches, max_images)
    if save_images and selected_matches:
        if not image_dir:
            raise ValueError("save_images=True이면 image_dir가 필요합니다.")
        out_dir = Path(image_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_paths = _write_images(selected_matches, out_dir, row_id=row_id)

    text_without_assets = remove_embedded_assets(raw)
    text = html_to_text(text_without_assets)
    text, truncated = truncate_text(text, max_chars)
    return ExtractedPhenomenon(
        text=text,
        image_count=len(selected_matches),
        original_image_count=len(matches),
        omitted_image_count=max(0, len(matches) - len(selected_matches)),
        image_paths=image_paths,
        truncated=truncated,
    )


def html_to_text(raw: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
    except Exception:
        raw = re.sub(r"<(script|style|noscript|svg|canvas)\b[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", raw)

    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_embedded_assets(raw: str) -> str:
    text = DATA_IMAGE_RE.sub(" [이미지첨부] ", raw)
    text = DATA_URI_RE.sub(" [임베디드데이터제거] ", text)
    text = BARE_BASE64_RE.sub(" base64,[임베디드데이터제거] ", text)
    return text


def truncate_text(text: str, max_chars: int | None) -> tuple[str, bool]:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + " ...[길이제한으로 이하 생략]", True


def limit_matches(matches: list[re.Match[str]], max_images: int | None) -> list[re.Match[str]]:
    if max_images is None or max_images < 0:
        return matches
    return matches[:max_images]


def _write_images(matches: Iterable[re.Match[str]], out_dir: Path, row_id: str = "") -> list[str]:
    paths: list[str] = []
    safe_row_id = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", row_id or "row")

    for idx, match in enumerate(matches, start=1):
        ext = match.group("ext").lower().replace("jpeg", "jpg")
        payload = re.sub(r"\s+", "", match.group("data"))
        digest = hashlib.sha1(payload[:2048].encode("ascii", errors="ignore")).hexdigest()[:10]
        path = out_dir / f"{safe_row_id}_{idx}_{digest}.{ext}"
        try:
            path.write_bytes(base64.b64decode(payload, validate=False))
            paths.append(str(path))
        except Exception:
            continue

    return paths
