from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
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
    print(f"[data] load mode={mode}", flush=True)

    if mode == "csv":
        print(f"[data] reading csv: {path}", flush=True)
        df = pd.read_csv(path)
    elif mode == "parquet":
        print(f"[data] reading parquet: {path}", flush=True)
        df = pd.read_parquet(path)
    elif mode == "adapter":
        df = load_df_from_company_db(config)
    else:
        raise ValueError(f"Unsupported data.mode: {mode}")

    validate_df(df)
    return df


def load_df_from_company_db(config: Any) -> pd.DataFrame:
    """사내 DB 연결 지점.

    사용자가 작성한 execute_SQL(query)를 가져와 그대로 호출합니다.
    """
    add_module_folder_to_path(config)
    query = read_query(config)
    print(f"[data] query loaded: chars={len(query)}", flush=True)
    execute_SQL = load_execute_sql(config)
    print("[data] executing SQL", flush=True)
    df = execute_SQL(query)
    print(f"[data] SQL completed: rows={len(df)}, cols={len(df.columns)}", flush=True)
    return df


def add_module_folder_to_path(config: Any) -> None:
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    module_folder = clean_config_path(config.get("data", "module_folder", fallback=""))
    if not module_folder:
        return

    abs_module_folder = os.path.abspath(module_folder)
    if abs_module_folder not in sys.path:
        sys.path.insert(0, abs_module_folder)


def load_execute_sql(config: Any):
    adapter_module_name = config.get("data", "sql_adapter_module", fallback="sql_adapter").strip() or "sql_adapter"

    try:
        adapter_module = import_adapter_module(adapter_module_name)
    except ImportError as exc:
        raise ImportError(
            f"{adapter_module_name} import 중 실패했습니다. "
            f"원본 에러: {exc!r}. "
            f"module_folder={config.get('data', 'module_folder', fallback='')!r}. "
            "sql_adapter.py 안의 IQADB_CONNECT310 import 또는 그 하위 의존성도 확인하세요."
        ) from exc

    execute_SQL = getattr(adapter_module, "execute_SQL", None)
    if not callable(execute_SQL):
        raise AttributeError(f"{adapter_module_name}.py 안에서 execute_SQL(query) 함수를 찾지 못했습니다.")

    return execute_SQL


def import_adapter_module(adapter_module_name: str):
    cleaned = clean_config_path(adapter_module_name)

    if cleaned.endswith(".py") or "\\" in cleaned or "/" in cleaned:
        adapter_path = Path(cleaned)
        if not adapter_path.is_absolute():
            adapter_path = Path(__file__).resolve().parent / adapter_path
        if not adapter_path.exists():
            raise ImportError(f"adapter file not found: {adapter_path}")

        spec = importlib.util.spec_from_file_location(adapter_path.stem, adapter_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load adapter spec: {adapter_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    if cleaned.endswith(".py"):
        cleaned = cleaned[:-3]
    return importlib.import_module(cleaned)


def clean_config_path(value: str) -> str:
    return value.strip().strip('"').strip("'")


def read_query(config: Any) -> str:
    query_file = clean_config_path(config.get("data", "query_file", fallback="queries/main.sql"))
    path = Path(query_file)
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path.resolve()}")
    return path.read_text(encoding="utf-8")


def validate_df(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"df에 필수 컬럼이 없습니다: {missing}")
