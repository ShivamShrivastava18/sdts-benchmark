"""Tuning driver: Optuna TPE, a fixed number of trials, objective = the
primary metric on the **validation** split.

    python -m sdts.tune.run_tuning --dataset adult --n 1000 --method ctgan

Per (dataset, n, method) one study is run with the fixed tuning seed
(``TUNING_SEED``). Each trial fits and samples in the same isolated
worker as ``run_cell`` (same per-cell timeout), then scores the synthetic
frame with ``sdts.eval.utility.train_test`` against the validation split.
The study is written as JSON to ``results/tuning/<dataset>__n<n>__<method>__budget<B>.json``
with every trial (params, effective hparams, value, state, seconds), the
best trial and its hparams, the sampler and seeds, and the git SHA.

Failed or timed-out trials are recorded with value -inf and told to
Optuna as FAIL, so they can never be the best trial.

Leakage guard: the objective only ever sees ``TuningSplits``, which
exposes ``train`` and ``val`` and raises ``TestSplitAccess`` on any access
to ``test``, so a tuning code path that reached for it would fail loudly
rather than leak.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

from sdts.data import loaders, splits
from sdts.data.schema import Schema, get_schema
from sdts.eval.utility import CLASSIFIER_NAMES, train_test
from sdts.runner.run_cell import GRID_YAML, REPO_ROOT, code_sha, git_sha
from sdts.tune.spaces import Space, get_space

TUNING_SEED = 0
TUNING_DIR = REPO_ROOT / "results" / "tuning"
optuna.logging.set_verbosity(optuna.logging.WARNING)


class TestSplitAccess(RuntimeError):
    """Raised when tuning code touches the test split."""


class TuningSplits:
    """The only view of the data the tuning objective receives."""

    def __init__(self, sp: splits.Splits) -> None:
        self.dataset = sp.dataset
        self.schema: Schema = sp.schema
        self.train: pd.DataFrame = sp.train
        self.val: pd.DataFrame = sp.val

    @property
    def test(self) -> pd.DataFrame:
        raise TestSplitAccess("the test split must never be read during tuning")

    def __getattr__(self, name: str) -> Any:
        if "test" in name:
            raise TestSplitAccess(f"the test split must never be read during tuning ({name})")
        raise AttributeError(name)


def study_id(dataset: str, n: int | str, method: str, budget: int) -> str:
    return f"{dataset}__n{n}__{method}__budget{budget}"


def _fit_sample(dataset: str, n: int | str, method: str, seed: int, hparams: dict[str, Any],
                timeout: float) -> tuple[str, pd.DataFrame | None, dict[str, Any], str | None]:
    """Run the worker once. Returns (status, synthetic frame, effective hparams, error)."""
    work = Path(tempfile.mkdtemp(prefix=f"sdts_tune_{method}_"))
    try:
        spec = {"dataset": dataset, "n": n, "method": method, "seed": seed,
                "hparams": hparams, "out_dir": str(work)}
        (work / "spec.json").write_text(json.dumps(spec, default=str))
        try:
            proc = subprocess.run([sys.executable, "-m", "sdts.runner.worker", str(work / "spec.json")],
                                  cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
                                  env={**os.environ, "PYTHONHASHSEED": str(seed)})
        except subprocess.TimeoutExpired:
            return "timeout", None, hparams, f"worker exceeded {timeout:.0f}s"
        rep = work / "worker.json"
        if not rep.exists():
            return "failed", None, hparams, f"worker exited {proc.returncode}\n{proc.stderr[-2000:]}"
        report = json.loads(rep.read_text())
        if report["status"] != "ok":
            return report["status"], None, hparams, report["error"]
        return "ok", pd.read_parquet(work / "synthetic.parquet"), report["hparams"], None
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


def validation_score(ts: TuningSplits, synth: pd.DataFrame, seed: int) -> tuple[float, dict[str, float]]:
    synth = ts.schema.cast(synth)
    res = train_test(ts.schema, synth, ts.val, seed)
    aurocs = {c: res[c]["auroc"] for c in CLASSIFIER_NAMES}
    return float(np.nanmean(list(aurocs.values()))), aurocs


def make_sampler(space: Space, seed: int) -> optuna.samplers.BaseSampler:
    """TPE for tpe spaces. Grid spaces are enumerated in ``tune`` without a
    sampler; this returns a GridSampler only for convenience in tests."""
    if space.sampler == "grid":
        return optuna.samplers.GridSampler(space.grid, seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def tune(dataset: str, n: int | str, method: str, budget: int | None = None, tuning_seed: int = TUNING_SEED,
         timeout: float | None = None, out_dir: Path = TUNING_DIR, n_trials: int | None = None) -> Path:
    space = get_space(method)
    if timeout is None:
        timeout = float(yaml.safe_load(GRID_YAML.read_text())["budget"]["max_cell_seconds"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = get_schema(dataset)
    if space is None:
        budget = 0
        sid = study_id(dataset, n, method, budget)
        doc = {"study_id": sid, "dataset": dataset, "n": n, "method": method, "tuning_budget": 0,
               "tuning_skipped": "no hyperparameters", "best_hparams": {}, "trials": [],
               "env": {"git_sha": git_sha(), "code_sha": code_sha()}, "finished_at": _now()}
        path = out_dir / f"{sid}.json"
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
        return path

    budget = space.n_trials if budget is None else budget
    n_trials = budget if n_trials is None else n_trials
    sid = study_id(dataset, n, method, budget)
    path = out_dir / f"{sid}.json"
    ts = TuningSplits(splits.split(dataset, schema=schema))
    started = _now()
    t_start = time.perf_counter()
    trials: list[dict[str, Any]] = []
    if space.sampler == "grid":
        # enumerate the grid deterministically; Optuna's GridSampler cannot be
        # driven in ask-and-tell mode (it calls Study.stop when exhausted)
        keys = list(space.grid)
        points = [dict(zip(keys, vals)) for vals in itertools.product(*(space.grid[k] for k in keys))]
        n_trials = min(n_trials, len(points))
        study = None
    else:
        study = optuna.create_study(direction="maximize", sampler=make_sampler(space, tuning_seed))

    for i in range(n_trials):
        trial = study.ask() if study is not None else optuna.trial.FixedTrial(points[i])
        hp = space.suggest(trial)
        t0 = time.perf_counter()
        status, synth, eff, err = _fit_sample(dataset, n, method, tuning_seed, hp, timeout)
        rec: dict[str, Any] = {"number": i, "params": trial.params, "hparams": eff, "status": status,
                               "value": -math.inf, "per_classifier": None, "seconds": None, "error": err}
        if status == "ok":
            try:
                value, per = validation_score(ts, synth, tuning_seed)
                rec.update({"value": value, "per_classifier": per})
                if study is not None:
                    study.tell(trial, value)
            except Exception as exc:  # noqa: BLE001
                rec.update({"status": "failed", "error": f"validation scoring failed: {exc!r}"})
                if study is not None:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
        elif study is not None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        rec["seconds"] = time.perf_counter() - t0
        trials.append(rec)
        print(f"[{sid}] trial {i + 1}/{n_trials} {status} value={rec['value']:.4f} {rec['seconds']:.0f}s", flush=True)

    ok = [t for t in trials if t["status"] == "ok"]
    best = max(ok, key=lambda t: t["value"]) if ok else None
    doc = {
        "study_id": sid, "dataset": dataset, "n": n, "method": method, "tuning_budget": budget,
        "n_trials": n_trials, "sampler": space.sampler, "tuning_seed": tuning_seed,
        "objective": "tstr_auroc_mean on the validation split", "space_source": space.source,
        "best_trial": best["number"] if best else None, "best_value": best["value"] if best else None,
        "best_hparams": best["hparams"] if best else None,
        "n_ok": len(ok), "n_failed": len(trials) - len(ok),
        "trials": trials, "env": {"git_sha": git_sha(), "code_sha": code_sha()},
        "started_at": started, "finished_at": _now(), "total_seconds": time.perf_counter() - t_start,
    }
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return path


def best_hparams(dataset: str, n: int | str, method: str, budget: int, out_dir: Path = TUNING_DIR) -> dict[str, Any] | None:
    """Winning configuration from a committed study, or None if the study has no ok trial."""
    space = get_space(method)
    if space is None:
        return {}
    path = Path(out_dir) / f"{study_id(dataset, n, method, budget)}.json"
    doc = json.loads(path.read_text())
    return doc["best_hparams"]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.tune.run_tuning", description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", required=True, help="integer rung or 'full'")
    ap.add_argument("--method", required=True)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None, help="override the trial count (tests only)")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=TUNING_DIR)
    args = ap.parse_args(argv)
    n: int | str = "full" if args.n == "full" else int(args.n)
    path = tune(args.dataset, n, args.method, args.budget, timeout=args.timeout, out_dir=args.out_dir,
                n_trials=args.trials)
    doc = json.loads(path.read_text())
    print(f"best trial {doc.get('best_trial')} value={doc.get('best_value')} hparams={doc.get('best_hparams')}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
