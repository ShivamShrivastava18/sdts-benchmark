"""Splits and the size ladder.

* ``split(dataset_id, df)``: stratified train/val/test at 60/20/20, seeded
  by a hash of the dataset id, so the split is identical across every run
  forever. Numeric NaNs in all three splits are imputed with the medians of
  the *training* split only (no leakage). Categorical missing values are
  already the level ``"missing"`` from the loader.
* ``ladder_subsample(train_df, n, seed)``: stratified subsample of the
  training split preserving class proportions. Validation and test splits
  are never touched by the ladder; only the generator's training data
  shrinks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from sdts.data import loaders
from sdts.data.schema import Schema, get_schema

LADDER: tuple[int, ...] = (200, 500, 1000, 2000, 5000, 10000, 20000)
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.6, 0.2, 0.2


def split_seed(dataset_id: str) -> int:
    """Deterministic 32-bit seed derived from the dataset id."""
    return int.from_bytes(hashlib.sha256(dataset_id.encode()).digest()[:4], "big")


@dataclass
class Splits:
    dataset: str
    schema: Schema
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    medians: dict[str, float]

    @property
    def max_rung(self) -> int | None:
        rungs = [n for n in LADDER if n <= len(self.train)]
        return max(rungs) if rungs else None

    @property
    def rungs(self) -> list[int]:
        return [n for n in LADDER if n <= len(self.train)]


def split(dataset_id: str, df: pd.DataFrame | None = None, schema: Schema | None = None) -> Splits:
    schema = schema or get_schema(dataset_id)
    if df is None:
        df = loaders.load(dataset_id)
    schema.validate(df)
    seed = split_seed(dataset_id)
    y = df[schema.target]
    rest, test = train_test_split(df, test_size=TEST_FRAC, stratify=y, random_state=seed)
    val_frac_of_rest = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train, val = train_test_split(rest, test_size=val_frac_of_rest,
                                  stratify=rest[schema.target], random_state=seed)
    medians = {c: float(train[c].median()) for c in schema.numeric}
    train, val, test = (_impute(part, medians) for part in (train, val, test))
    return Splits(dataset_id, schema, train, val, test, medians)


def _impute(part: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    part = part.copy()
    for c, m in medians.items():
        part[c] = part[c].fillna(m)
    return part


def ladder_subsample(train_df: pd.DataFrame, n: int, seed: int, target: str) -> pd.DataFrame:
    """Stratified subsample of ``n`` rows from the training split."""
    if n > len(train_df):
        raise ValueError(f"requested n={n} but the training split has {len(train_df)} rows")
    if n == len(train_df):
        return train_df.copy()
    sub, _ = train_test_split(train_df, train_size=n, stratify=train_df[target], random_state=seed)
    return sub
