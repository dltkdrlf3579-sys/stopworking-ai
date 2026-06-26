from __future__ import annotations

from stopright_ai.config import load_config
from stopright_ai.loop import run_improvement_loop

from data import load_df


def main() -> None:
    config = load_config("config.ini")
    df = load_df(config)
    run_improvement_loop(df=df, config=config)


if __name__ == "__main__":
    main()

