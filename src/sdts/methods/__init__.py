"""Generators. Every method, including the trivial baselines, implements
base.Method. Importing this package registers every method in
``base.REGISTRY``; ``configs/methods.yaml`` records implementation
choices and which are active."""

from sdts.methods import (  # noqa: F401  (registration side effects)
    base,
    copula,
    ctgan,
    marginals,
    real_subsample,
    smote,
    tabddpm,
    tabsyn,
    tvae,
)
from sdts.methods.base import REGISTRY, Method, get_method, register

__all__ = ["REGISTRY", "Method", "get_method", "register"]
