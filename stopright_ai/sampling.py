from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_eval_df(df: pd.DataFrame, config: Any, cycle: int = 1) -> pd.DataFrame:
    train_df, _ = build_train_validation_dfs(df, config, cycle=cycle)
    return train_df


def build_train_validation_dfs(df: pd.DataFrame, config: Any, cycle: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch_size = config.getint("runtime", "batch_size", fallback=2000)
    validation_size = config.getint("runtime", "validation_size", fallback=batch_size)
    base_seed = config.getint("sampling", "random_seed", fallback=42)
    vary_seed = config.getboolean("sampling", "vary_seed_by_cycle", fallback=True)
    seed = base_seed + cycle - 1 if vary_seed else base_seed

    train_df = build_sample_df(df, config, sample_size=batch_size, seed=seed, include_hard=True)
    validation_pool = drop_rows_by_ids(df, train_df, id_col=get_id_col(config))
    if validation_pool.empty:
        validation_pool = df
    validation_df = build_sample_df(
        validation_pool,
        config,
        sample_size=min(validation_size, len(validation_pool)),
        seed=seed + 100_000,
        include_hard=False,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)


def build_sample_df(
    df: pd.DataFrame,
    config: Any,
    sample_size: int,
    seed: int,
    include_hard: bool = False,
) -> pd.DataFrame:
    sample_size = min(max(0, sample_size), len(df))
    if sample_size <= 0:
        return df.head(0).copy()

    id_col = get_id_col(config)
    hard_path = Path(config.get("artifacts", "hard_cases_path", fallback="artifacts/hard_cases.csv"))
    hard_size = config.getint("sampling", "hard_size", fallback=0) if include_hard else 0
    hard_rows = load_hard_rows_from_source(df, hard_path, id_col=id_col, limit=min(hard_size, sample_size), seed=seed)

    base_target = max(0, sample_size - len(hard_rows))
    pool = drop_rows_by_ids(df, hard_rows, id_col=id_col)
    base = stratified_sample(pool, min(base_target, len(pool)), seed=seed) if base_target else df.head(0).copy()
    sample = concat_and_dedupe([hard_rows, base], id_col=id_col)

    if len(sample) < sample_size:
        rest = drop_rows_by_ids(df, sample, id_col=id_col)
        if not rest.empty:
            add = stratified_sample(rest, min(sample_size - len(sample), len(rest)), seed=seed + 1)
            sample = concat_and_dedupe([sample, add], id_col=id_col)

    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=seed)
    return sample.reset_index(drop=True)


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
    )

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        rest = df.drop(sampled.index, errors="ignore")
        if len(rest):
            add = rest.sample(n=min(n - len(sampled), len(rest)), random_state=seed)
            sampled = pd.concat([sampled, add], ignore_index=True)

    return sampled.reset_index(drop=True)


def load_hard_rows_from_source(df: pd.DataFrame, hard_path: Path, id_col: str, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0 or not hard_path.exists() or id_col not in df.columns:
        return df.head(0).copy()

    try:
        hard = pd.read_csv(hard_path)
    except Exception:
        return df.head(0).copy()

    if "id" not in hard.columns:
        return df.head(0).copy()

    if "exclude_from_metrics" in hard.columns:
        hard = hard[~hard["exclude_from_metrics"].map(normalize_bool)]
    if hard.empty:
        return df.head(0).copy()

    hard_ids = hard["id"].dropna().astype(str).str.strip().drop_duplicates()
    if hard_ids.empty:
        return df.head(0).copy()
    if len(hard_ids) > limit:
        hard_ids = hard_ids.sample(n=limit, random_state=seed)

    source = df[df[id_col].astype(str).str.strip().isin(set(hard_ids))].copy()
    if len(source) > limit:
        source = source.sample(n=limit, random_state=seed)
    return source.reset_index(drop=True)


def drop_rows_by_ids(df: pd.DataFrame, rows: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if rows.empty or id_col not in df.columns or id_col not in rows.columns:
        return df.copy()
    ids = set(rows[id_col].dropna().astype(str).str.strip())
    if not ids:
        return df.copy()
    return df[~df[id_col].astype(str).str.strip().isin(ids)].copy()


def concat_and_dedupe(parts: list[pd.DataFrame], id_col: str) -> pd.DataFrame:
    non_empty = [part for part in parts if not part.empty]
    if not non_empty:
        return pd.DataFrame()
    merged = pd.concat(non_empty, ignore_index=True)
    if id_col in merged.columns:
        merged = merged.drop_duplicates(subset=[id_col])
    else:
        merged = merged.drop_duplicates()
    return merged


def get_id_col(config: Any) -> str:
    try:
        return config["columns"].get("id", "출원번호")
    except Exception:
        return "출원번호"


def normalize_bool(value: object) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, bool) and missing:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)
