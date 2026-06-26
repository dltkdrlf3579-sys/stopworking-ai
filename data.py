from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

try:
    from IQADB_CONNECT310 import *  # noqa: F401,F403
except ImportError:
    pass


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

    config.ini의 [data] module_folder에 IQADB_CONNECT310.py가 있는 폴더를 넣으면
    sys.path에 추가한 뒤 executeSQL(query)를 호출합니다.
    """
    add_module_folder_to_path(config)
    query = read_query(config)
    execute_sql = get_execute_sql()
    return execute_sql(query)


def add_module_folder_to_path(config: Any) -> None:
    module_folder = config.get("data", "module_folder", fallback="").strip()
    if not module_folder:
        return

    abs_module_folder = os.path.abspath(module_folder)
    if abs_module_folder not in sys.path:
        sys.path.insert(0, abs_module_folder)


def get_execute_sql():
    execute_sql = globals().get("executeSQL")
    if callable(execute_sql):
        return execute_sql

    try:
        module = importlib.import_module("IQADB_CONNECT310")
    except ImportError as exc:
        raise ImportError(
            "IQADB_CONNECT310을 import하지 못했습니다. "
            "config.ini의 [data] module_folder에 IQADB_CONNECT310.py가 있는 폴더를 넣으세요."
        ) from exc

    execute_sql = getattr(module, "executeSQL", None)
    if not callable(execute_sql):
        raise AttributeError("IQADB_CONNECT310 모듈에서 executeSQL 함수를 찾지 못했습니다.")

    return execute_sql


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
