"""The Method interface. Every generator, including the trivial baselines,
implements exactly this and nothing in the runner special-cases a method.

    class Method:
        name: str
        def __init__(self, schema: Schema, hparams: dict, seed: int): ...
        def fit(self, df: pd.DataFrame) -> None: ...
        def sample(self, n: int) -> pd.DataFrame: ...

``sample`` returns the same columns, in the same order, with the same
dtypes as the frame passed to ``fit``; ``finalize`` enforces that by
casting through the schema, so a method only has to produce the right
values. ``seed_everything`` is called at the start of ``fit`` and
``sample`` so that output is a deterministic function of (data, hparams,
seed).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd

# Load xgboost's OpenMP runtime BEFORE torch initialises its own. torch,
# scikit-learn and xgboost each bundle a separate libomp; on macOS, importing
# xgboost after torch has initialised OpenMP segfaults inside xgboost.
import xgboost  # noqa: F401  (import-order guard, must precede torch)

from sdts.data.schema import Schema

REGISTRY: dict[str, type["Method"]] = {}


def configure_torch_threads() -> None:
    """Pin torch to a single CPU thread.

    torch, scikit-learn and xgboost each ship their own OpenMP runtime. On
    macOS, running a torch parallel region in the same process as a
    scikit-learn or xgboost one deadlocks (torch after xgboost) or
    segfaults (scikit-learn after torch); reproduced 2026-09-07 with three
    lines of each. With one thread torch never opens an OpenMP team, and
    the problem disappears in both directions. GPU execution is unaffected.
    """
    try:
        import torch

        torch.set_num_threads(1)  # initialises torch's libomp; xgboost is already loaded
    except ImportError:  # pragma: no cover
        pass


configure_torch_threads()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


class Method(ABC):
    """Fit on a training frame, sample a frame with identical columns,
    order and dtypes."""

    name: str = "base"
    #: set to False for registered-but-disabled stubs (TabSyn)
    enabled: bool = True
    #: filled by real_subsample when it has to sample with replacement
    extra_result_fields: dict[str, Any]

    def __init__(self, schema: Schema, hparams: dict | None, seed: int) -> None:
        self.schema = schema
        self.hparams = dict(self.default_hparams())
        self.hparams.update(hparams or {})
        self.seed = int(seed)
        self.extra_result_fields = {}
        self._fitted = False

    @classmethod
    def default_hparams(cls) -> dict[str, Any]:
        return {}

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame: ...

    # ------------------------------------------------------------ helpers
    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast a raw sample to the canonical columns, order and dtypes."""
        out = self.schema.cast(df).reset_index(drop=True)
        self.schema.validate(out)
        return out

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed + salt)


def register(name: str) -> Callable[[type[Method]], type[Method]]:
    """Class decorator adding a Method subclass to REGISTRY under ``name``."""

    def deco(cls: type[Method]) -> type[Method]:
        if name in REGISTRY and REGISTRY[name] is not cls:
            raise ValueError(f"method {name!r} registered twice")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def get_method(name: str) -> type[Method]:
    import sdts.methods  # noqa: F401  ensure registration

    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(REGISTRY)}") from None
