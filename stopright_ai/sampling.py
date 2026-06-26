from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_eval_df(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    batch_size = config.getint("runtime", "batch_size", fallback=2000)
    seed = config.getint("sampling", "random_seed", fallback=42)
    hard_path = Path(config.get("artifacts", "hard_cases_path", fallback="artifacts/hard_cases.csv"))

    base = stratified_sample(df, min(batch_size, len(df)), seed=seed)

    if hard_path.exists():
        try:
            hard = pd.read_csv(hard_path)
            common_cols = [c for c in base.columns if c in hard.columns]
            if common_cols:
                base = pd.concat([base, hard[common_cols]], ignore_index=True)
                base = base.drop_duplicates()
        except Exception:
            pass

    return base.head(batch_size).reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    strat_cols = [col for col in ["정답_판단결과", "대분류", "중분류"] if col in df.columns]
    if not strat_cols:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    # pandas groupby sample 비율 방식. 소수 그룹도 최대한 보존한다.
    frac = n / len(df)
    sampled = (
        df.groupby(strat_cols, group_keys=False, dropna=False)
        .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed))
        .reset_index(drop=True)
    )

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        rest = df.drop(sampled.index, errors="ignore")
        if len(rest):
            add = rest.sample(n=min(n - len(sampled), len(rest)), random_state=seed)
            sampled = pd.concat([sampled, add], ignore_index=True)

    return sampled.reset_index(drop=True)

