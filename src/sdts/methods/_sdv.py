"""Shared glue for the SDV synthesizers (GaussianCopula, CTGAN, TVAE).

Metadata is built from the schema, never detected from data, so the
types SDV sees are the types every metric sees. Categorical columns are
handed to SDV as strings and the target as a categorical; ``finalize``
casts the sample back.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sdv.metadata import Metadata

from sdts.data.schema import Schema
from sdts.methods.base import Method, seed_everything

TABLE = "table"


def metadata_from_schema(schema: Schema) -> Metadata:
    columns: dict[str, dict[str, str]] = {}
    for c in schema.columns:
        if c == schema.target or c in schema.categorical:
            columns[c] = {"sdtype": "categorical"}
        else:
            columns[c] = {"sdtype": "numerical", "computer_representation": "Float"}
    return Metadata.load_from_dict({"tables": {TABLE: {"columns": columns}}})


def to_sdv_frame(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    out = df.copy()
    for c in schema.categorical:
        out[c] = out[c].astype(str)
    out[schema.target] = out[schema.target].astype(str)
    return out


class SDVMethod(Method):
    synthesizer_cls: type
    fit_kwargs: dict[str, Any] = {}

    def make_synthesizer(self, metadata: Metadata):
        return self.synthesizer_cls(metadata, **self.hparams)

    def fit(self, df: pd.DataFrame) -> None:
        seed_everything(self.seed)
        self._synth = self.make_synthesizer(metadata_from_schema(self.schema))
        self._synth.fit(to_sdv_frame(df, self.schema))
        self._fitted = True

    def sample(self, n: int) -> pd.DataFrame:
        seed_everything(self.seed)
        # SDV otherwise pins sampling to its own FIXED_RNG_SEED, which would
        # make every seed produce the same sample from the same fit.
        self._synth._set_random_state(self.seed)
        out = self._synth.sample(num_rows=n)
        return self.finalize(out)
