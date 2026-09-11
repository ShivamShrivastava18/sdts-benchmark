"""Real rows: the ceiling.

Returns a disjoint (without-replacement) sample of the real rows it was
fitted on. Utility cannot be higher than real data of the same size, and
privacy cannot be worse, so this pins both ends of every metric. When
``n`` exceeds the number of fitted rows it samples with replacement and
records ``with_replacement: true`` in ``extra_result_fields`` so the
result file carries it.
"""

from __future__ import annotations

import pandas as pd

from sdts.methods.base import Method, register, seed_everything


@register("real_subsample")
class RealSubsample(Method):
    def fit(self, df: pd.DataFrame) -> None:
        seed_everything(self.seed)
        self._df = df.reset_index(drop=True)
        self._fitted = True

    def sample(self, n: int) -> pd.DataFrame:
        rng = self.rng()
        replace = n > len(self._df)
        self.extra_result_fields["with_replacement"] = bool(replace)
        idx = rng.choice(len(self._df), size=n, replace=replace)
        return self.finalize(self._df.iloc[idx])
