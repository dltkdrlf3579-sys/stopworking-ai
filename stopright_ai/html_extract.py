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


@dataclass
class ExtractedPhenomenon:
    text: str
    image_count: int
    image_paths: list[str]


def extract_phenomenon(raw_html_or_text: object, save_images: bool = False, image_dir: str | None = None, row_id: str = "") -> ExtractedPhenomenon:
    raw = "" if raw_html_or_text is None else str(raw_html_or_text)
    image_paths: list[str] = []

    matches = list(DATA_IMAGE_RE.finditer(raw))
    if save_images and matches:
        if not image_dir:
            raise ValueError("save_images=True이면 image_dir가 필요합니다.")
        out_dir = Path(image_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_paths = _write_images(matches, out_dir, row_id=row_id)

    text_without_images = DATA_IMAGE_RE.sub(" [이미지첨부] ", raw)
    text = html_to_text(text_without_images)
    return ExtractedPhenomenon(text=text, image_count=len(matches), image_paths=image_paths)


def html_to_text(raw: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        text = soup.get_text(" ", strip=True)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw)

    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

