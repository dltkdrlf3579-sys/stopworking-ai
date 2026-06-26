from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def make_run_dir(config: Any) -> Path:
    root = Path(config.get("artifacts", "output_dir", fallback="outputs"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_predictions(path: Path, pred_df: pd.DataFrame) -> None:
    safe_df = pred_df.copy()
    for col in safe_df.columns:
        if safe_df[col].map(lambda v: isinstance(v, (dict, list))).any():
            safe_df[col] = safe_df[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
    safe_df.to_csv(path, index=False, encoding="utf-8-sig")


def append_hard_cases(config: Any, pred_df: pd.DataFrame) -> None:
    hard_path = Path(config.get("artifacts", "hard_cases_path", fallback="artifacts/hard_cases.csv"))
    hard_path.parent.mkdir(parents=True, exist_ok=True)

    hard = pred_df[(pred_df["correct"] == False) | (pred_df["confidence"] < 60) | (pred_df["review_needed"] == True)].copy()
    if hard.empty:
        return

    cols = [c for c in ["id", "label", "pred", "confidence", "reason", "major", "middle", "title"] if c in hard.columns]
    hard = hard[cols].drop_duplicates()

    if hard_path.exists():
        old = pd.read_csv(hard_path)
        hard = pd.concat([old, hard], ignore_index=True).drop_duplicates()

    hard.to_csv(hard_path, index=False, encoding="utf-8-sig")

