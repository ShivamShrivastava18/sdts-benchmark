"""Dataset schema: column order, types, categorical levels, target.

Inferred once per dataset from the cleaned frame and cached to
``configs/schemas/<id>.json`` (committed). Every method and every metric
casts through ``Schema.cast`` so they cannot disagree about types.

Inference rule (overrides in ``configs/datasets.yaml`` win):

* object/bool columns are categorical;
* numeric columns with at most two distinct non-null values are categorical
  (binary flags coded 0/1);
* everything else is numeric;
* ``categorical:`` and ``numeric:`` lists in the registry entry override.

Canonical dtypes: numeric -> float64, categorical -> pandas ``category``
with the level set fixed at inference time (levels are strings; missing
categoricals carry the literal level ``"missing"``), target -> int64 in
{0, 1}.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdts.data import loaders

SCHEMA_DIR = loaders.REPO_ROOT / "configs" / "schemas"
MAX_BINARY_LEVELS = 2


@dataclass
class Schema:
    dataset: str
    target: str
    task: str
    columns: list[str]                    # full order, target included
    numeric: list[str]
    categorical: list[str]
    levels: dict[str, list[str]] = field(default_factory=dict)
    spec_hash: str = ""
    n_rows: int = 0

    # ----------------------------------------------------------------- helpers
    @property
    def features(self) -> list[str]:
        return [c for c in self.columns if c != self.target]

    def cast(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [c for c in self.columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{self.dataset}: frame lacks columns {missing_cols}")
        out = pd.DataFrame(index=df.index)
        for c in self.columns:
            if c == self.target:
                out[c] = pd.to_numeric(df[c]).astype("int64")
            elif c in self.categorical:
                s = df[c]
                if isinstance(s.dtype, pd.CategoricalDtype):
                    s = s.astype(object)
                s = s.where(s.notna(), loaders.MISSING_LEVEL).map(_level_str)
                unknown = set(s.unique()) - set(self.levels[c])
                if unknown:
                    raise ValueError(f"{self.dataset}.{c}: levels not in schema: {sorted(unknown)[:5]}")
                out[c] = pd.Categorical(s, categories=self.levels[c])
            else:
                out[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
        return out

    def validate(self, df: pd.DataFrame) -> None:
        """Raise unless ``df`` has exactly the canonical columns, order and dtypes."""
        if list(df.columns) != self.columns:
            raise ValueError(f"columns/order differ: {list(df.columns)} != {self.columns}")
        for c in self.columns:
            if c == self.target:
                if df[c].dtype != np.dtype("int64"):
                    raise ValueError(f"{c}: target dtype {df[c].dtype} != int64")
            elif c in self.categorical:
                if not isinstance(df[c].dtype, pd.CategoricalDtype):
                    raise ValueError(f"{c}: expected category dtype, got {df[c].dtype}")
                if list(df[c].cat.categories) != self.levels[c]:
                    raise ValueError(f"{c}: categorical levels differ from schema")
            elif df[c].dtype != np.dtype("float64"):
                raise ValueError(f"{c}: expected float64, got {df[c].dtype}")

    # ----------------------------------------------------------------- io
    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> "Schema":
        return cls(**json.loads(path.read_text()))


def _level_str(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (np.integer,)):
        return str(int(v))
    return str(v)


def infer(dataset_id: str, df: pd.DataFrame, spec: dict[str, Any]) -> Schema:
    target = spec["target"]
    force_cat = set(spec.get("categorical") or [])
    force_num = set(spec.get("numeric") or [])
    bad = (force_cat | force_num) - set(df.columns)
    if bad:
        raise ValueError(f"{dataset_id}: type overrides name unknown columns {sorted(bad)}")
    categorical: list[str] = []
    numeric: list[str] = []
    for c in df.columns:
        if c == target:
            continue
        if c in force_cat:
            categorical.append(c)
        elif c in force_num:
            numeric.append(c)
        elif df[c].dtype == object or df[c].dtype == bool:
            categorical.append(c)
        elif df[c].nunique(dropna=True) <= MAX_BINARY_LEVELS:
            categorical.append(c)
        else:
            numeric.append(c)
    levels: dict[str, list[str]] = {}
    for c in categorical:
        s = df[c].where(df[c].notna(), loaders.MISSING_LEVEL).map(_level_str)
        levels[c] = sorted(s.unique().tolist())
    columns = [c for c in df.columns if c != target] + [target]
    return Schema(
        dataset=dataset_id, target=target, task=spec.get("task", "binary"),
        columns=columns, numeric=numeric, categorical=categorical, levels=levels,
        spec_hash=loaders.spec_hash(spec), n_rows=int(len(df)),
    )


def get_schema(dataset_id: str, schema_dir: Path = SCHEMA_DIR) -> Schema:
    """Load the cached schema, inferring and writing it if absent or stale."""
    spec = loaders.registry()[dataset_id]
    path = schema_dir / f"{dataset_id}.json"
    if path.exists():
        schema = Schema.from_json(path)
        if schema.spec_hash == loaders.spec_hash(spec):
            return schema
    df = loaders.load_raw_clean(dataset_id)
    schema = infer(dataset_id, df, spec)
    schema.to_json(path)
    return schema
