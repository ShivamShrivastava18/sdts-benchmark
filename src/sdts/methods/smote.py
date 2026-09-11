"""SMOTE-style interpolation baselines, following the two variants in
arXiv 2406.12945 (Margeloiu et al.):

* ``smote``: conditioned on the target. Each synthetic row is an
  interpolation between a real training row and one of its k nearest
  neighbours *of the same class*. Class counts in the output match the
  training class proportions.
* ``ucsmote``: unconditional. The real target is treated as an ordinary
  categorical column and every row carries a dummy class of 0, so the
  interpolation runs across the whole training set.

Interpolation (SMOTE-NC semantics): numeric columns are linearly
interpolated with lambda ~ U(0, 1); categorical columns take the majority
level among the base row and its k neighbours (ties broken by the base
row). Distances use standardised numeric columns plus one-hot categorical
columns, computed on the fitted frame. k = 5 by default; k is capped at
(class size - 1) so tiny classes still work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sdts.methods.base import Method, register, seed_everything


class _Smote(Method):
    conditional: bool = True

    @classmethod
    def default_hparams(cls) -> dict[str, Any]:
        return {"k_neighbors": 5}

    def fit(self, df: pd.DataFrame) -> None:
        seed_everything(self.seed)
        s = self.schema
        self._df = df.reset_index(drop=True)
        self._num_cols = list(s.numeric)
        self._cat_cols = list(s.categorical) + ([] if self.conditional else [s.target])
        if self.conditional:
            self._y = self._df[s.target].to_numpy()
        else:
            self._y = np.zeros(len(self._df), dtype=int)

        parts = []
        if self._num_cols:
            self._scaler = StandardScaler().fit(self._df[self._num_cols])
            parts.append(np.nan_to_num(self._scaler.transform(self._df[self._num_cols])))
        if self._cat_cols:
            self._ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            parts.append(self._ohe.fit_transform(self._df[self._cat_cols].astype(str)))
        self._X = np.hstack(parts) if parts else np.zeros((len(self._df), 1))

        self._neighbors: dict[int, np.ndarray] = {}
        self._class_idx: dict[int, np.ndarray] = {}
        for cls in np.unique(self._y):
            idx = np.flatnonzero(self._y == cls)
            k = min(int(self.hparams["k_neighbors"]), len(idx) - 1)
            self._class_idx[cls] = idx
            if k <= 0:
                self._neighbors[cls] = np.zeros((len(idx), 0), dtype=int)
                continue
            nn = NearestNeighbors(n_neighbors=k + 1).fit(self._X[idx])
            _, nbr = nn.kneighbors(self._X[idx])
            self._neighbors[cls] = idx[nbr[:, 1:]]  # drop self
        self._fitted = True

    def _class_counts(self, n: int) -> dict[int, int]:
        classes, counts = np.unique(self._y, return_counts=True)
        raw = counts / counts.sum() * n
        alloc = np.floor(raw).astype(int)
        rem = n - alloc.sum()
        order = np.argsort(-(raw - alloc))
        for i in order[:rem]:
            alloc[i] += 1
        return {int(c): int(a) for c, a in zip(classes, alloc)}

    def sample(self, n: int) -> pd.DataFrame:
        rng = self.rng()
        num = self._df[self._num_cols].to_numpy(dtype=float) if self._num_cols else None
        cat = self._df[self._cat_cols].astype(object).to_numpy() if self._cat_cols else None
        rows_num, rows_cat, rows_y = [], [], []
        for cls, m in self._class_counts(n).items():
            if m == 0:
                continue
            idx = self._class_idx[cls]
            nbrs = self._neighbors[cls]
            base_pos = rng.integers(0, len(idx), size=m)
            base = idx[base_pos]
            if nbrs.shape[1] == 0:  # singleton class: copy the row
                partner = base
                group = base[:, None]
            else:
                partner = nbrs[base_pos, rng.integers(0, nbrs.shape[1], size=m)]
                group = np.hstack([base[:, None], nbrs[base_pos]])
            lam = rng.random((m, 1))
            if num is not None:
                rows_num.append(num[base] + lam * (num[partner] - num[base]))
            if cat is not None:
                rows_cat.append(_majority(cat, group))
            rows_y.append(np.full(m, cls))
        out = pd.DataFrame(index=range(n))
        if num is not None:
            out[self._num_cols] = np.vstack(rows_num)
        if cat is not None:
            out[self._cat_cols] = np.vstack(rows_cat)
        if self.conditional:
            out[self.schema.target] = np.concatenate(rows_y)
        perm = rng.permutation(n)
        return self.finalize(out.iloc[perm])


def _majority(cat: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Per row, per column: most frequent level among ``group`` rows; ties -> base (first)."""
    m, g = group.shape
    out = np.empty((m, cat.shape[1]), dtype=object)
    for j in range(cat.shape[1]):
        vals = cat[group, j]  # m x g
        for i in range(m):
            levels, counts = np.unique(vals[i].astype(str), return_counts=True)
            best = counts.max()
            winners = set(levels[counts == best])
            out[i, j] = vals[i, 0] if str(vals[i, 0]) in winners else levels[counts == best][0]
    return out


@register("smote")
class Smote(_Smote):
    conditional = True


@register("ucsmote")
class UCSmote(_Smote):
    conditional = False
