"""Fidelity: how distinguishable the synthetic frame is from fresh real data.

The real reference for every fidelity metric is an equal-sized, seeded,
stratified sample of the **held-out test split** (``reference_sample``),
never the training rows: copying the training data must not score as
perfect fidelity.

* ``c2st_auc`` (primary fidelity): a HistGradientBoosting discriminator
  separating real from synthetic, 5-fold stratified CV, out-of-fold AUC.
  0.5 means indistinguishable.
* ``ks_mean``: mean per-column two-sample Kolmogorov-Smirnov statistic
  over numeric columns.
* ``tvd_mean``: mean per-column total variation distance over categorical
  columns (target included).
* ``corr_dist``: Frobenius norm of the difference between association
  matrices of real and synthetic, with Pearson for numeric pairs,
  Cramer's V (bias-corrected) for categorical pairs, and the correlation
  ratio (eta) for mixed pairs. Target included as categorical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sdts.data.schema import Schema
from sdts.eval._prep import Preprocessor


def reference_sample(schema: Schema, test: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(test))
    if n == len(test):
        return test
    sub, _ = train_test_split(test, train_size=n, stratify=test[schema.target], random_state=seed)
    return sub


def c2st_auc(schema: Schema, real: pd.DataFrame, synth: pd.DataFrame, seed: int) -> float:
    prep = Preprocessor(schema, include_target=True).fit(real)
    X = np.vstack([prep.transform(real), prep.transform(synth)])
    y = np.r_[np.ones(len(real)), np.zeros(len(synth))]
    clf = HistGradientBoostingClassifier(random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def ks_mean(schema: Schema, real: pd.DataFrame, synth: pd.DataFrame) -> float:
    if not schema.numeric:
        return float("nan")
    return float(np.mean([ks_2samp(real[c], synth[c]).statistic for c in schema.numeric]))


def tvd_mean(schema: Schema, real: pd.DataFrame, synth: pd.DataFrame) -> float:
    cols = list(schema.categorical) + [schema.target]
    vals = []
    for c in cols:
        p = real[c].astype(str).value_counts(normalize=True)
        q = synth[c].astype(str).value_counts(normalize=True)
        idx = p.index.union(q.index)
        vals.append(0.5 * float(np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum()))
    return float(np.mean(vals))


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a.astype(str), b.astype(str)).to_numpy(dtype=float)
    n = ct.sum()
    if n == 0 or min(ct.shape) < 2:
        return 0.0
    row, col = ct.sum(1, keepdims=True), ct.sum(0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((ct - expected) ** 2 / np.where(expected > 0, expected, np.nan))
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rc = r - (r - 1) ** 2 / max(n - 1, 1)
    kc = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def _corr_ratio(cat: pd.Series, num: pd.Series) -> float:
    x = num.to_numpy(dtype=float)
    if x.std() == 0:
        return 0.0
    groups = pd.Series(x).groupby(cat.astype(str).to_numpy())
    ss_between = float(sum(len(g) * (g.mean() - x.mean()) ** 2 for _, g in groups))
    ss_total = float(((x - x.mean()) ** 2).sum())
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def association_matrix(schema: Schema, df: pd.DataFrame) -> np.ndarray:
    cats = list(schema.categorical) + [schema.target]
    cols = list(schema.numeric) + cats
    is_cat = {c: c in cats for c in cols}
    k = len(cols)
    m = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = cols[i], cols[j]
            if not is_cat[a] and not is_cat[b]:
                x, y = df[a].to_numpy(float), df[b].to_numpy(float)
                v = 0.0 if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])
            elif is_cat[a] and is_cat[b]:
                v = _cramers_v(df[a], df[b])
            elif is_cat[a]:
                v = _corr_ratio(df[a], df[b])
            else:
                v = _corr_ratio(df[b], df[a])
            m[i, j] = m[j, i] = v
    return m


def corr_dist(schema: Schema, real: pd.DataFrame, synth: pd.DataFrame) -> float:
    return float(np.linalg.norm(association_matrix(schema, real) - association_matrix(schema, synth)))


def fidelity(schema: Schema, real_ref: pd.DataFrame, synth: pd.DataFrame, seed: int) -> dict[str, float]:
    return {
        "c2st_auc": c2st_auc(schema, real_ref, synth, seed),
        "ks_mean": ks_mean(schema, real_ref, synth),
        "tvd_mean": tvd_mean(schema, real_ref, synth),
        "corr_dist": corr_dist(schema, real_ref, synth),
    }
