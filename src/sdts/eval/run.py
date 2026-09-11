"""``evaluate``: every metric for one cell, from the frames the runner
already holds. Returns (metrics, per_classifier).

Inputs:
    real_train  the n-row real training subsample the generator was fitted on
    synth       the n-row synthetic frame
    test        the full real test split (utility is scored on all of it)
The fidelity and privacy references are equal-sized stratified samples of
the test split, seeded by ``seed``; nothing from the validation split is
used.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sdts.data.schema import Schema
from sdts.eval.fidelity import fidelity, reference_sample
from sdts.eval.privacy import privacy
from sdts.eval.utility import PRIMARY_METRIC, utility


def evaluate(schema: Schema, real_train: pd.DataFrame, synth: pd.DataFrame,
             test: pd.DataFrame, seed: int) -> tuple[dict[str, float], dict[str, Any]]:
    schema.validate(synth)
    schema.validate(real_train)
    metrics, per_clf = utility(schema, real_train, synth, test, seed)
    ref = reference_sample(schema, test, len(synth), seed)
    metrics.update(fidelity(schema, ref, synth, seed))
    holdout = reference_sample(schema, test, len(real_train), seed)
    metrics.update(privacy(schema, real_train, holdout, synth))
    assert PRIMARY_METRIC in metrics
    return metrics, per_clf
