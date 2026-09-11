"""Common feature preprocessing for every metric: standardised numeric
columns and one-hot categorical columns with the level set fixed by the
schema, so real and synthetic frames always map to the same matrix.
``Preprocessor.fit`` is called on whichever frame a metric declares as its
reference (documented per metric)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdts.data.schema import Schema


class Preprocessor:
    def __init__(self, schema: Schema, include_target: bool = False) -> None:
        self.schema = schema
        self.include_target = include_target
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        num = df[self.schema.numeric].to_numpy(dtype=float)
        self._mean = num.mean(axis=0) if num.size else np.zeros(0)
        std = num.std(axis=0) if num.size else np.zeros(0)
        self._std = np.where(std > 0, std, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        s = self.schema
        parts = []
        if s.numeric:
            num = df[s.numeric].to_numpy(dtype=float)
            parts.append((num - self._mean) / self._std)
        for c in s.categorical:
            codes = pd.Categorical(df[c], categories=s.levels[c]).codes
            oh = np.zeros((len(df), len(s.levels[c])))
            oh[np.arange(len(df)), codes] = 1.0
            parts.append(oh)
        if self.include_target:
            parts.append(df[s.target].to_numpy(dtype=float)[:, None])
        return np.hstack(parts) if parts else np.zeros((len(df), 0))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
