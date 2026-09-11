"""SDV GaussianCopulaSynthesizer with SDV defaults (beta marginals,
Gaussian copula over the probability-integral transforms)."""

from __future__ import annotations

from typing import Any

from sdv.single_table import GaussianCopulaSynthesizer

from sdts.methods._sdv import SDVMethod
from sdts.methods.base import register


@register("copula")
class Copula(SDVMethod):
    synthesizer_cls = GaussianCopulaSynthesizer

    @classmethod
    def default_hparams(cls) -> dict[str, Any]:
        return {"default_distribution": "beta", "enforce_min_max_values": True}
