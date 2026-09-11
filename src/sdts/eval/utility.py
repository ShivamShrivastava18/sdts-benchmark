"""Utility: train-on-synthetic, test-on-real (TSTR).

Three fixed downstream classifiers (logistic regression, random forest,
XGBoost; constants in ``configs/methods.yaml`` and ``CLASSIFIERS`` below,
never tuned) are trained on the synthetic frame and evaluated on the full
real test split. The **primary metric is TSTR AUROC averaged over the
three classifiers** (``PRIMARY_METRIC``). Accuracy and F1 are recorded as
secondary. TRTR (train on the real n-row subsample) is the ceiling and
the majority-class AUROC (0.5) the floor.

Features are standardised numeric plus one-hot categorical, fitted on the
classifier's own training frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from sdts.data.schema import Schema
from sdts.eval._prep import Preprocessor

PRIMARY_METRIC = "tstr_auroc_mean"
CLASSIFIER_NAMES = ("lr", "rf", "xgb")


def make_classifier(name: str, seed: int):
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1,
                                      n_jobs=1, random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=1.0,
                             colsample_bytree=1.0, n_jobs=1, random_state=seed,
                             verbosity=0, tree_method="hist")
    raise KeyError(name)


def _scores(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, proba))
    return {"auroc": auroc, "acc": float(accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, zero_division=0))}


def train_test(schema: Schema, train: pd.DataFrame, test: pd.DataFrame, seed: int
               ) -> dict[str, dict[str, float]]:
    """Fit each classifier on ``train`` and score on ``test``."""
    prep = Preprocessor(schema).fit(train)
    Xtr, Xte = prep.transform(train), prep.transform(test)
    ytr, yte = train[schema.target].to_numpy(), test[schema.target].to_numpy()
    out: dict[str, dict[str, float]] = {}
    for name in CLASSIFIER_NAMES:
        if len(np.unique(ytr)) < 2:  # degenerate synthetic data: constant prediction
            proba = np.full(len(yte), float(ytr[0]) if len(ytr) else 0.0)
        else:
            clf = make_classifier(name, seed).fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:, 1]
        out[name] = _scores(yte, proba)
    return out


def utility(schema: Schema, real_train: pd.DataFrame, synthetic: pd.DataFrame,
            test: pd.DataFrame, seed: int) -> tuple[dict[str, float], dict[str, Any]]:
    """Return (flat metrics, per-classifier dict)."""
    tstr = train_test(schema, synthetic, test, seed)
    trtr = train_test(schema, real_train, test, seed)
    per_clf = {c: {**{f"tstr_{k}": v for k, v in tstr[c].items()},
                   **{f"trtr_{k}": v for k, v in trtr[c].items()}} for c in CLASSIFIER_NAMES}
    metrics: dict[str, float] = {}
    for prefix, res in (("tstr", tstr), ("trtr", trtr)):
        for k in ("auroc", "acc", "f1"):
            metrics[f"{prefix}_{k}_mean"] = float(np.nanmean([res[c][k] for c in CLASSIFIER_NAMES]))
    metrics["majority_auroc"] = 0.5
    return metrics, per_clf
