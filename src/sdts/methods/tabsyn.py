"""TabSyn (Zhang et al., ICLR 2024): registered but disabled.

Stretch goal, not required for the claim. The runner refuses to schedule
disabled methods and grid.yaml must list it as inactive. fit/sample raise
so that it can never silently produce a number."""

from __future__ import annotations

import pandas as pd

from sdts.methods.base import Method, register


@register("tabsyn")
class TabSyn(Method):
    enabled = False

    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError("tabsyn is a disabled stub; see configs/methods.yaml")

    def sample(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("tabsyn is a disabled stub; see configs/methods.yaml")
