"""Privacy: distance-based leakage measures.

All distances are Euclidean on a common preprocessing: numeric columns
standardised with the mean and std of the **real training subsample**,
categorical columns one-hot with schema levels, target appended as 0/1
(see ``sdts.eval._prep.Preprocessor``).

* ``dcr_median``: median over synthetic rows of the distance to the
  closest real training record.
* ``dcr_rate``: proportion of synthetic rows whose closest real training
  record is nearer than their closest held-out real record, following
  arXiv 2406.12945 (Kindji et al., Section 3.2, Figure 3). The held-out
  set is an equal-sized, seeded, stratified sample of the test split.
  1/2 means no memorisation; a copy of the training set scores 1.
* ``nndr_median``: median over synthetic rows of the ratio between the
  distances to the nearest and second-nearest real training record.
  Small values mean synthetic rows sit on top of single real rows.
* ``mia_auc``: distance-threshold membership inference. Members are the
  real training rows, non-members an equal-sized held-out sample; the
  attacker scores each record by minus its distance to the nearest
  synthetic row. Reported as ROC AUC; 0.5 means no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from sdts.data.schema import Schema
from sdts.eval._prep import Preprocessor


def _nn_dist(query: np.ndarray, ref: np.ndarray, k: int = 1) -> np.ndarray:
    k = min(k, len(ref))
    d, _ = NearestNeighbors(n_neighbors=k).fit(ref).kneighbors(query)
    return d


def privacy(schema: Schema, real_train: pd.DataFrame, holdout: pd.DataFrame,
            synth: pd.DataFrame) -> dict[str, float]:
    prep = Preprocessor(schema, include_target=True).fit(real_train)
    Xtr, Xho, Xs = prep.transform(real_train), prep.transform(holdout), prep.transform(synth)

    d_train = _nn_dist(Xs, Xtr, k=2)
    d_hold = _nn_dist(Xs, Xho, k=1)[:, 0]
    dcr = d_train[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        nndr = np.where(d_train[:, 1] > 0, dcr / d_train[:, 1], 1.0) if d_train.shape[1] > 1 \
            else np.ones_like(dcr)

    # membership inference: score = -distance to nearest synthetic row
    d_mem = _nn_dist(Xtr, Xs)[:, 0]
    d_non = _nn_dist(Xho, Xs)[:, 0]
    y = np.r_[np.ones(len(d_mem)), np.zeros(len(d_non))]
    score = -np.r_[d_mem, d_non]

    return {
        "dcr_median": float(np.median(dcr)),
        "dcr_rate": float(np.mean(dcr < d_hold)),
        "nndr_median": float(np.median(nndr)),
        "mia_auc": float(roc_auc_score(y, score)),
    }
