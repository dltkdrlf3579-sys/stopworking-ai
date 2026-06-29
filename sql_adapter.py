from __future__ import annotations

import traceback

import pandas as pd

from IQADB_CONNECT310 import *  # noqa: F401,F403


def execute_SQL(query: str) -> pd.DataFrame:
    conn = iqadb1()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            data = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
            df = pd.DataFrame(data, columns=col_names)
    except Exception as e:
        print(f"오류발생 : {e}", flush=True)
        traceback.print_exc()
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


def execute_DML(sql: str) -> None:
    conn = iqadb1()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        if hasattr(conn, "commit"):
            conn.commit()
    except Exception as e:
        print(f"DML 오류발생 : {e}", flush=True)
        traceback.print_exc()
        if hasattr(conn, "rollback"):
            conn.rollback()
        raise
    finally:
        conn.close()
