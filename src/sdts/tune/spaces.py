"""Hyper-parameter search spaces, one per method, frozen by
PREREGISTRATION.md Section 3.

Source: Kindji, Rojas-Barahona, Fromont, Urvoy, "Tabular Data Generation
Models: An In-Depth Survey and Performance Benchmarks with Extensive
Tuning", arXiv 2406.12945 (v4), Appendix A, *reduced* columns:
Table A.6 (TVAE), A.7 (CTGAN), A.9 (TabDDPM), A.10 (SMOTE / ucSMOTE).
Compute-bound adaptations, declared in the preregistration before any
tuned cell ran, are marked ADAPTED below.

``qLogUniform(low, high, q)`` is sampled log-uniformly and rounded to the
nearest multiple of q, as in the source.

A space is a ``Space`` with ``sampler`` ("tpe" or "grid"), ``n_trials``
and ``suggest(trial) -> hparams``. Methods with no hyper-parameters
(marginals, real_subsample, copula) return ``None`` and skip tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import optuna

TUNING_TRIALS = 20


@dataclass(frozen=True)
class Space:
    method: str
    sampler: str                       # "tpe" | "grid"
    n_trials: int
    suggest: Callable[[optuna.Trial], dict[str, Any]]
    grid: dict[str, list[Any]] | None = None
    source: str = ""


def _qlog(trial: optuna.Trial, name: str, low: float, high: float, q: float) -> float:
    v = trial.suggest_float(name, low, high, log=True)
    return float(round(round(v / q) * q, 12))


# --------------------------------------------------------------------- CTGAN
def _ctgan(trial: optuna.Trial) -> dict[str, Any]:
    g_depth = trial.suggest_categorical("generator_depth", [3, 4])
    d_depth = trial.suggest_categorical("discriminator_depth", [2, 3])
    return {
        "discriminator_lr": _qlog(trial, "discriminator_lr", 4e-4, 2.1e-3, 5e-5),
        "generator_lr": _qlog(trial, "generator_lr", 5e-5, 1.3e-3, 5e-5),
        "batch_size": trial.suggest_categorical("batch_size", [100, 500, 1000]),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 128]),
        "generator_dim": tuple([128] * g_depth),           # generator dim fixed at 128 (reduced)
        "discriminator_dim": tuple([256] * d_depth),       # discriminator dim fixed at 256 (reduced)
        "generator_decay": _qlog(trial, "generator_decay", 1e-6, 6.4e-6, 1e-7),
        "discriminator_decay": _qlog(trial, "discriminator_decay", 1e-6, 8e-6, 1e-6),
        "log_frequency": trial.suggest_categorical("log_frequency", [False, True]),
        "epochs": 300,                                     # ADAPTED: SDV default (source: 400)
    }


# ---------------------------------------------------------------------- TVAE
def _tvae(trial: optuna.Trial) -> dict[str, Any]:
    enc = trial.suggest_categorical("encoder_dim", [256, 512])
    dec = trial.suggest_categorical("decoder_dim", [256, 512])
    dec_depth = trial.suggest_categorical("decoder_depth", [2, 4])
    return {
        # learning rate: not exposed by SDV 1.38.3, library default (ADAPTED)
        "batch_size": 100,
        "embedding_dim": trial.suggest_categorical("embedding_dim", [16, 32, 64]),
        "compress_dims": (enc, enc),                       # encoder depth fixed at 2 (reduced)
        "decompress_dims": tuple([dec] * dec_depth),
        "loss_factor": trial.suggest_categorical("loss_factor", [3, 2]),
        "l2scale": _qlog(trial, "l2scale", 1e-5, 6.3e-5, 1e-6),
        "epochs": 300,                                     # ADAPTED: SDV default (source: 400)
    }


# ------------------------------------------------------------------- TabDDPM
def _tabddpm(trial: optuna.Trial) -> dict[str, Any]:
    n_layers = trial.suggest_categorical("n_layers", [2, 4])           # ADAPTED: source {2, 4, 6}
    first = trial.suggest_categorical("first_dim", [128, 256, 512])    # ADAPTED: source {256, 512, 1024}
    middle = trial.suggest_categorical("middle_dim", [128, 256, 512])  # ADAPTED: source {512, 1024}
    last = trial.suggest_categorical("last_dim", [128, 256, 512])      # ADAPTED: source {256, 512, 1024}
    layers = [first] + [middle] * (n_layers - 2) + [last]
    return {
        "d_layers": layers,
        "batch_size": trial.suggest_categorical("batch_size", [256, 1024]),  # ADAPTED: source {4096}
        "lr": _qlog(trial, "lr", 3.5e-4, 9.2e-4, 1e-5),
        "dropout": 0.0,
        "num_timesteps": 1000,
        "steps": 10000,                                    # ADAPTED: source 20000
    }


# --------------------------------------------------------------------- SMOTE
SMOTE_K = list(range(2, 21))


def _smote(trial: optuna.Trial) -> dict[str, Any]:
    return {"k_neighbors": trial.suggest_categorical("k_neighbors", SMOTE_K)}


SPACES: dict[str, Space | None] = {
    "marginals": None,
    "real_subsample": None,
    "copula": None,
    "ctgan": Space("ctgan", "tpe", TUNING_TRIALS, _ctgan, source="Kindji et al. 2406.12945 Table A.7 (reduced)"),
    "tvae": Space("tvae", "tpe", TUNING_TRIALS, _tvae, source="Kindji et al. 2406.12945 Table A.6 (reduced)"),
    "tabddpm": Space("tabddpm", "tpe", TUNING_TRIALS, _tabddpm, source="Kindji et al. 2406.12945 Table A.9 (reduced, adapted)"),
    "smote": Space("smote", "grid", len(SMOTE_K), _smote, grid={"k_neighbors": SMOTE_K},
                   source="Kindji et al. 2406.12945 Table A.10"),
    "ucsmote": Space("ucsmote", "grid", len(SMOTE_K), _smote, grid={"k_neighbors": SMOTE_K},
                     source="Kindji et al. 2406.12945 Table A.10"),
}


def get_space(method: str) -> Space | None:
    if method not in SPACES:
        raise KeyError(f"no search-space entry for {method!r}; every active method must be listed")
    return SPACES[method]
