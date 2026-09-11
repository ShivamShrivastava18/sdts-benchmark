"""One cell -> one JSON result file.

    python -m sdts.runner.run_cell --dataset adult --n 1000 --method ctgan --seed 3
    python -m sdts.runner.run_cell --smoke

Steps: seed, load, split, subsample, fit, sample (all in a worker
subprocess with a timeout, see ``sdts.runner.worker``), then evaluate in
this process and write the result. The result file is written for every
terminal status: ok, failed, timeout, oom. ``error`` carries the
traceback when status is not ok.

``--smoke`` runs pima, n=200, ctgan, seed 0, default hparams into
``runs/smoke/`` and must finish in under 120 seconds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sdts.data import loaders, splits
from sdts.data.schema import get_schema

REPO_ROOT = loaders.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "results"
GRID_YAML = REPO_ROOT / "configs" / "grid.yaml"
METHODS_YAML = REPO_ROOT / "configs" / "methods.yaml"
TERMINAL = ("ok", "failed", "timeout", "oom")
GIT_DIRTY_IGNORE = ("runs/BUDGET.md",)


def cell_id(dataset: str, n: int | str, method: str, seed: int, budget: int) -> str:
    return f"{dataset}__n{n}__{method}__seed{seed}__budget{budget}"


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=REPO_ROOT, capture_output=True, text=True).stdout
        # runs/BUDGET.md is tracked and appended by run_grid itself; it never
        # changes what a cell computes, so it does not count as dirty.
        dirty = [ln for ln in status.splitlines() if ln.strip() and not ln.endswith(GIT_DIRTY_IGNORE)]
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return ""


# Everything that runs inside a cell or a tuning trial. src/sdts/analysis is
# excluded on purpose: it only reads result files, never produces them, so it
# can evolve while a grid runs without splitting the result set.
CODE_PATHS = ("src/sdts/__init__.py", "src/sdts/doctor.py", "src/sdts/data", "src/sdts/methods",
              "src/sdts/eval", "src/sdts/runner", "src/sdts/tune", "configs", "third_party",
              "pyproject.toml")


def code_sha() -> str:
    """Content hash of the cell-producing code at HEAD (CODE_PATHS).
    Unchanged by commits that only add results or analysis code, so a grid
    that is committed in batches keeps one code_sha."""
    import hashlib

    try:
        out = subprocess.check_output(["git", "ls-tree", "HEAD", *CODE_PATHS], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL, text=True)
        return hashlib.sha256(out.encode()).hexdigest()[:16]
    except Exception:
        return ""


def env_info() -> dict[str, str]:
    import importlib.metadata as md

    methods_cfg = yaml.safe_load(METHODS_YAML.read_text())
    gpu = ""
    report = REPO_ROOT / "runs" / "doctor_report.json"
    if report.exists():
        try:
            for c in json.loads(report.read_text())["checks"]:
                if c["name"] == "gpu":
                    gpu = c.get("detail", "")
        except Exception:
            gpu = ""
    return {
        "git_sha": git_sha(),
        "code_sha": code_sha(),
        "python": sys.version.split()[0],
        "torch": md.version("torch"),
        "sdv": md.version("sdv"),
        "tabddpm_impl": methods_cfg["methods"]["tabddpm"]["tabddpm_impl"],
        "gpu": gpu,
        "platform": platform.platform(),
    }


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_cell(dataset: str, n: int | str, method: str, seed: int, budget: int = 0,
             hparams: dict[str, Any] | None = None, timeout: float | None = None,
             results_dir: Path = RESULTS_DIR, keep_synthetic: bool = False) -> Path:
    """Run one cell and write ``<results_dir>/<cell_id>.json``. Returns the path."""
    if timeout is None:
        timeout = float(yaml.safe_load(GRID_YAML.read_text())["budget"]["max_cell_seconds"])
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cid = cell_id(dataset, n, method, seed, budget)
    out_path = results_dir / f"{cid}.json"
    started = _now()
    t0 = time.perf_counter()
    result: dict[str, Any] = {
        "cell_id": cid, "dataset": dataset, "n_train": None, "method": method,
        "seed": seed, "tuning_budget": budget, "hparams": dict(hparams or {}),
        "status": "failed", "metrics": {}, "per_classifier": {},
        "cost": {"fit_seconds": None, "sample_seconds": None, "peak_rss_mb": None,
                 "eval_seconds": None, "total_seconds": None},
        "env": env_info(), "extra": {},
        "started_at": started, "finished_at": None, "error": None,
    }

    work = Path(tempfile.mkdtemp(prefix=f"sdts_{cid}_"))
    try:
        spec = {"dataset": dataset, "n": n, "method": method, "seed": seed,
                "hparams": hparams or {}, "out_dir": str(work)}
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec))
        proc_error = None
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "sdts.runner.worker", str(spec_path)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
            )
        except subprocess.TimeoutExpired as exc:
            result["status"] = "timeout"
            result["error"] = (f"worker exceeded {timeout:.0f}s\n"
                               f"stderr tail:\n{(exc.stderr or '')[-2000:] if isinstance(exc.stderr, str) else ''}")
            proc = None
        if proc is not None:
            report_path = work / "worker.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                result["status"] = report["status"]
                result["error"] = report["error"]
                result["n_train"] = report.get("n_train")
                result["env"]["torch_threads"] = report.get("torch_threads")
                if report["status"] == "ok":
                    result["hparams"] = report["hparams"]
                    result["cost"].update(report["cost"])
                    result["extra"] = report.get("extra", {})
            else:
                # killed by a signal (SIGKILL on memory pressure shows as -9) or crashed before reporting
                result["status"] = "oom" if proc.returncode in (-9, 137) else "failed"
                result["error"] = (f"worker exited {proc.returncode} without a report\n"
                                   f"stderr tail:\n{proc.stderr[-4000:]}")

        if result["status"] == "ok":
            try:
                t_eval = time.perf_counter()
                from sdts.eval.run import evaluate  # torch-free path

                schema = get_schema(dataset)
                sp = splits.split(dataset)
                train = pd.read_parquet(work / "train.parquet")
                synth = pd.read_parquet(work / "synthetic.parquet")
                train, synth = schema.cast(train), schema.cast(synth)
                metrics, per_clf = evaluate(schema, train, synth, sp.test, seed)
                result["metrics"], result["per_classifier"] = metrics, per_clf
                result["cost"]["eval_seconds"] = time.perf_counter() - t_eval
                if keep_synthetic:
                    shutil.copy(work / "synthetic.parquet", results_dir / f"{cid}.synthetic.parquet")
            except Exception:  # noqa: BLE001
                result["status"] = "failed"
                result["error"] = "evaluation failed:\n" + traceback.format_exc()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    result["finished_at"] = _now()
    result["cost"]["total_seconds"] = time.perf_counter() - t0
    assert result["status"] in TERMINAL
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return out_path


def is_valid_result(path: Path) -> bool:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return d.get("status") in TERMINAL and "cell_id" in d and d.get("finished_at")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.runner.run_cell", description=__doc__)
    ap.add_argument("--dataset")
    ap.add_argument("--n", help="integer rung or 'full'")
    ap.add_argument("--method")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0, help="tuning budget recorded in the cell id")
    ap.add_argument("--hparams", default="{}", help="JSON dict of hyper-parameter overrides")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--keep-synthetic", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="pima, n=200, ctgan, seed 0 into runs/smoke/")
    args = ap.parse_args(argv)

    if args.smoke:
        grid = yaml.safe_load(GRID_YAML.read_text())["grids"]["smoke"]
        args.dataset, args.n, args.method = grid["datasets"][0], grid["n"][0], grid["methods"][0]
        args.seed, args.budget = grid["seeds"][0], grid["tuning_budget"]
        args.results_dir = REPO_ROOT / grid["results_dir"]
        args.timeout = args.timeout or 110.0
    if not (args.dataset and args.n and args.method):
        ap.error("--dataset, --n and --method are required (or --smoke)")
    n: int | str = "full" if str(args.n) == "full" else int(args.n)

    t0 = time.perf_counter()
    path = run_cell(args.dataset, n, args.method, args.seed, args.budget,
                    json.loads(args.hparams), args.timeout, args.results_dir, args.keep_synthetic)
    d = json.loads(path.read_text())
    elapsed = time.perf_counter() - t0
    primary = d["metrics"].get("tstr_auroc_mean")
    print(f"{d['cell_id']}: status={d['status']} tstr_auroc_mean={primary} "
          f"fit={d['cost']['fit_seconds']} eval={d['cost']['eval_seconds']} total={elapsed:.1f}s")
    print(f"wrote {path}")
    if d["status"] != "ok":
        print(d["error"], file=sys.stderr)
    if args.smoke and elapsed > 120:
        print(f"SMOKE FAILED: {elapsed:.0f}s > 120s", file=sys.stderr)
        return 2
    return 0 if d["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
