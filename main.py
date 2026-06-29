from __future__ import annotations

from stopright_ai.config import load_config
from stopright_ai.logging import install_timestamped_print
from stopright_ai.loop import run_improvement_loop

from data import load_df


def main() -> None:
    install_timestamped_print()
    print("[main] loading config.ini")
    config = load_config("config.ini")
    print("[main] loading dataframe")
    df = load_df(config)
    print(f"[main] dataframe loaded: rows={len(df)}, cols={len(df.columns)}")
    print("[main] starting improvement loop")
    run_improvement_loop(df=df, config=config)


if __name__ == "__main__":
    main()
