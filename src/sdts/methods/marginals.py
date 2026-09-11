"""Independent empirical marginals: the floor.

Every column is sampled independently from its own empirical training
distribution (with replacement). Zero correlation is preserved by
construction, so any method that loses to this one has learned nothing
about the joint. The target is treated as just another column.
"""

from __future__ import annotations

import pandas as pd

from sdts.methods.base import Method, register, seed_everything


@register("marginals")
class Marginals(Method):
    def fit(self, df: pd.DataFrame) -> None:
        seed_everything(self.seed)
        self._columns = {c: df[c].to_numpy(copy=True) for c in df.columns}
        self._fitted = True

    def sample(self, n: int) -> pd.DataFrame:
        rng = self.rng()
        out = {c: rng.choice(v, size=n, replace=True) for c, v in self._columns.items()}
        return self.finalize(pd.DataFrame(out))
