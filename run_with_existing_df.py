"""이미 df와 llm을 만든 노트북/운영 스크립트에서 쓰는 진입점 예시.

사용 예:
    from run_with_existing_df import run_once_with_objects
    result = run_once_with_objects(df, llm)
"""

from __future__ import annotations

from stopright_ai.config import load_config
from stopright_ai.loop import run_one_cycle


def run_once_with_objects(df, llm, config_path: str = "config.ini"):
    config = load_config(config_path)
    return run_one_cycle(df=df, config=config, llm=llm)

