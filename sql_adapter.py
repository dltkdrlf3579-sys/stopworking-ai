from __future__ import annotations

import pandas as pd

from IQADB_CONNECT310 import *  # noqa: F401,F403


def execute_SQL(query: str) -> pd.DataFrame:
    """네가 기존에 쓰던 execute_SQL 함수를 여기에 그대로 넣으면 됩니다.

    IQADB_CONNECT310은 위에서 import 되어 있습니다.
    반드시 query 문자열을 받아 pandas.DataFrame을 반환하게만 맞추면 됩니다.

    예:
        # 기존 네 execute_SQL 내용
        ...
    """
    raise NotImplementedError("sql_adapter.py에 네가 만든 execute_SQL(query) 함수 내용을 붙여 넣으세요.")
