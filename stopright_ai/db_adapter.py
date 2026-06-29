from __future__ import annotations

from typing import Any, Callable

from data import add_module_folder_to_path, import_adapter_module


def load_execute_sql(config: Any) -> Callable[[str], Any]:
    module = load_adapter_module(config)
    execute_sql = getattr(module, "execute_SQL", None)
    if not callable(execute_sql):
        raise AttributeError("sql_adapter.py 안에서 execute_SQL(query) 함수를 찾지 못했습니다.")
    return execute_sql


def load_execute_dml(config: Any) -> Callable[[str], None]:
    module = load_adapter_module(config)
    execute_dml = getattr(module, "execute_DML", None)
    if not callable(execute_dml):
        raise AttributeError("sql_adapter.py 안에서 execute_DML(sql) 함수를 찾지 못했습니다.")
    return execute_dml


def load_adapter_module(config: Any):
    add_module_folder_to_path(config)
    adapter_module_name = config.get("data", "sql_adapter_module", fallback="sql_adapter").strip() or "sql_adapter"
    return import_adapter_module(adapter_module_name)
