from __future__ import annotations

from stopright_ai.config import load_config
from stopright_ai.loop import run_improvement_loop

from data import load_df


def main() -> None:
    print("[main] loading config.ini", flush=True)
    config = load_config("config.ini")
    print("[main] loading dataframe", flush=True)
    df = load_df(config)
    print(f"[main] dataframe loaded: rows={len(df)}, cols={len(df.columns)}", flush=True)
    print("[main] starting improvement loop", flush=True)
    run_improvement_loop(df=df, config=config)


if __name__ == "__main__":
    main()
