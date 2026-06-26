from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "출원번호",
    "제목",
    "대분류",
    "중분류",
    "현상",
    "조치",
    "정답_판단결과",
]


def load_df(config: Any) -> pd.DataFrame:
    """Load df for local tests or company runtime.

    운영환경에서는 config.ini의 [data] mode를 adapter로 두고
    load_df_from_company_db() 안에 사내 executeSQL(query)를 연결하세요.
    """
    mode = config.get("data", "mode", fallback="adapter").strip().lower()
    path = config.get("data", "path", fallback="").strip()

    if mode == "csv":
        df = pd.read_csv(path)
    elif mode == "parquet":
        df = pd.read_parquet(path)
    elif mode == "adapter":
        df = load_df_from_company_db(config)
    else:
        raise ValueError(f"Unsupported data.mode: {mode}")

    validate_df(df)
    return df


def load_df_from_company_db(config: Any) -> pd.DataFrame:
    """사내 DB 연결 지점.

    아래 TODO만 운영환경에 맞게 바꾸면 됩니다.

    예시:
        from your_company_db import executeSQL

        query = read_query(config)
        df = executeSQL(query)
        return df
    """
    query = read_query(config)

    # TODO: 사내 환경에서 아래 3줄을 실제 모듈명으로 교체하세요.
    # from your_company_db import executeSQL
    # df = executeSQL(query)
    # return df

    raise NotImplementedError(
        "data.py의 load_df_from_company_db()에 사내 executeSQL(query)를 연결하세요. "
        f"현재 query 길이: {len(query)}"
    )


def read_query(config: Any) -> str:
    query_file = config.get("data", "query_file", fallback="queries/main.sql").strip()
    path = Path(query_file)
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path.resolve()}")
    return path.read_text(encoding="utf-8")


def validate_df(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"df에 필수 컬럼이 없습니다: {missing}")

