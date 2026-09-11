"""Resumable grid driver. The only entry point for multi-cell work.

    python -m sdts.runner.run_grid --grid stage0
    python -m sdts.runner.run_grid --grid stage1 --confirm-full

* expands a grid from ``configs/grid.yaml`` into cells (dataset order,
  smallest n first, then method, then seed), capping integer rungs at each
  dataset's training-split size and mapping ``full`` to the whole split;
* skips any cell whose result file is valid (terminal status) unless
  ``--retry-failed`` is given, in which case failed/timeout/oom cells rerun;
* enforces ``budget.max_cell_seconds`` per cell via ``run_cell``;
* writes ``runs/<grid_id>/manifest.json`` listing every planned cell and
  its state, updated after every cell;
* appends to ``runs/BUDGET.md`` every 50 completed cells: cells done, wall
  clock consumed, projected total;
* refuses to run a grid marked ``full: true`` without ``--confirm-full``;
* for a grid with ``tuning_budget > 0``, runs (or reuses) the tuning study
  for each (dataset, n, method) before its seed cells and passes the
  winning configuration to every seed. Methods without a search space run
  with budget 0 in their cell id. A study with no successful trial makes
  every seed cell a recorded failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from sdts.data import loaders, splits
from sdts.runner.run_cell import RESULTS_DIR, cell_id, is_valid_result, run_cell
from sdts.tune import run_tuning

REPO_ROOT = loaders.REPO_ROOT
GRID_YAML = REPO_ROOT / "configs" / "grid.yaml"
RUNS_DIR = REPO_ROOT / "runs"
BUDGET_MD = RUNS_DIR / "BUDGET.md"
BUDGET_EVERY = 50


def load_grid_config(path: Path = GRID_YAML) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def expand(grid_name: str, cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the ordered list of cell specs for a grid."""
    cfg = cfg or load_grid_config()
    g = cfg["grids"][grid_name]
    methods = cfg["methods_active"] if g["methods"] == "all_active" else list(g["methods"])
    seeds = cfg["seeds"] if g["seeds"] == "all" else list(g["seeds"])
    budget = int(g.get("tuning_budget", 0))
    cells: list[dict[str, Any]] = []
    for d in g["datasets"]:
        n_spec = g["n"]
        if isinstance(n_spec, dict):
            n_spec = n_spec[d]
        if n_spec == "ladder":
            n_spec = list(cfg["ladder"])
        train_rows = len(splits.split(d).train)
        ns: list[int | str] = []
        for n in n_spec:
            if n == "full":
                ns.append("full")
            elif int(n) <= train_rows:
                ns.append(int(n))
        for n in ns:
            for m in methods:
                b = budget if run_tuning.get_space(m) is not None else 0
                for s in seeds:
                    cells.append({"dataset": d, "n": n, "method": m, "seed": int(s), "budget": b,
                                  "cell_id": cell_id(d, n, m, s, b)})
    return cells


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Manifest:
    def __init__(self, path: Path, grid: str, cells: list[dict[str, Any]]) -> None:
        self.path = path
        self.data: dict[str, Any] = {
            "grid": grid, "created_at": _now(), "updated_at": _now(),
            "cells": {c["cell_id"]: {**c, "state": "pending", "status": None} for c in cells},
        }
        if path.exists():
            try:
                old = json.loads(path.read_text())
                for cid, rec in old.get("cells", {}).items():
                    if cid in self.data["cells"]:
                        self.data["cells"][cid].update({k: rec.get(k) for k in ("state", "status", "seconds")})
                self.data["created_at"] = old.get("created_at", self.data["created_at"])
            except Exception:
                pass

    def set(self, cid: str, **fields: Any) -> None:
        self.data["cells"][cid].update(fields)
        self.data["updated_at"] = _now()
        self.data["summary"] = self.summary()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.data["cells"].values():
            key = rec["state"] if rec["state"] != "done" else f"done_{rec['status']}"
            out[key] = out.get(key, 0) + 1
        return out


def append_budget(grid: str, done: int, total: int, elapsed_s: float, new_cells: int, new_seconds: float) -> None:
    per_cell = new_seconds / new_cells if new_cells else float("nan")
    projected = elapsed_s + per_cell * (total - done)
    line = (f"- {_now()} grid={grid}: {done}/{total} cells done, "
            f"{elapsed_s / 3600:.2f} h wall clock this run, "
            f"{per_cell:.0f} s/cell over the last {new_cells}, "
            f"projected total {projected / 3600:.1f} h\n")
    BUDGET_MD.parent.mkdir(parents=True, exist_ok=True)
    with BUDGET_MD.open("a") as fh:
        fh.write(line)


def _tuned_hparams(c: dict[str, Any], timeout: float, tuning_dir: Path,
                   n_trials: int | None) -> tuple[dict[str, Any] | None, str | None]:
    """Study for this cell's (dataset, n, method): reuse if present, else run it."""
    sid = run_tuning.study_id(c["dataset"], c["n"], c["method"], c["budget"])
    path = Path(tuning_dir) / f"{sid}.json"
    if not path.exists():
        print(f"tuning {sid} ...", flush=True)
        run_tuning.tune(c["dataset"], c["n"], c["method"], c["budget"], timeout=timeout,
                        out_dir=tuning_dir, n_trials=n_trials)
    doc = json.loads(path.read_text())
    if doc.get("best_hparams") is None:
        return None, f"tuning study {sid} has no successful trial ({doc.get('n_failed')} failed)"
    return doc["best_hparams"], None


def _write_untunable(c: dict[str, Any], error: str, results_dir: Path) -> Path:
    from sdts.runner.run_cell import env_info

    now = _now()
    doc = {"cell_id": c["cell_id"], "dataset": c["dataset"], "n_train": None, "method": c["method"],
           "seed": c["seed"], "tuning_budget": c["budget"], "hparams": {}, "status": "failed",
           "metrics": {}, "per_classifier": {}, "cost": {}, "env": env_info(), "extra": {},
           "started_at": now, "finished_at": now, "error": error}
    path = Path(results_dir) / f"{c['cell_id']}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return path


def run(grid: str, results_dir: Path = RESULTS_DIR, confirm_full: bool = False,
        retry_failed: bool = False, limit: int | None = None, timeout: float | None = None,
        cfg: dict[str, Any] | None = None, runs_dir: Path = RUNS_DIR,
        tuning_dir: Path = run_tuning.TUNING_DIR) -> dict[str, int]:
    cfg = cfg or load_grid_config()
    g = cfg["grids"][grid]
    if g.get("full") and not confirm_full:
        raise SystemExit(f"grid {grid!r} is a full grid; pass --confirm-full to run it")
    if timeout is None:
        timeout = float(cfg["budget"]["max_cell_seconds"])
    results_dir = Path(g.get("results_dir", results_dir))
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    cells = expand(grid, cfg)
    manifest = Manifest(runs_dir / grid / "manifest.json", grid, cells)

    t_start = time.perf_counter()
    ran = skipped = 0
    since_budget_cells, since_budget_seconds = 0, 0.0
    for c in cells:
        cid = c["cell_id"]
        path = results_dir / f"{cid}.json"
        if is_valid_result(path):
            status = json.loads(path.read_text())["status"]
            if status == "ok" or not retry_failed:
                manifest.set(cid, state="skipped", status=status)
                skipped += 1
                continue
        if limit is not None and ran >= limit:
            break
        manifest.set(cid, state="running", started_at=_now())
        t0 = time.perf_counter()
        hparams, tune_error = None, None
        if c["budget"] > 0:
            hparams, tune_error = _tuned_hparams(c, timeout, tuning_dir, g.get("tuning_trials"))
        if tune_error:
            out = _write_untunable(c, tune_error, results_dir)
        else:
            out = run_cell(c["dataset"], c["n"], c["method"], c["seed"], c["budget"], hparams=hparams,
                           timeout=timeout, results_dir=results_dir)
        secs = time.perf_counter() - t0
        status = json.loads(out.read_text())["status"]
        manifest.set(cid, state="done", status=status, seconds=round(secs, 1))
        ran += 1
        since_budget_cells += 1
        since_budget_seconds += secs
        print(f"[{ran + skipped}/{len(cells)}] {cid} {status} {secs:.0f}s", flush=True)
        if since_budget_cells >= BUDGET_EVERY:
            done = sum(1 for r in manifest.data["cells"].values() if r["state"] in ("done", "skipped"))
            append_budget(grid, done, len(cells), time.perf_counter() - t_start,
                          since_budget_cells, since_budget_seconds)
            since_budget_cells, since_budget_seconds = 0, 0.0
    return {"planned": len(cells), "ran": ran, "skipped": skipped, **manifest.summary()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.runner.run_grid", description=__doc__)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--confirm-full", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many new cells")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="print the planned cells and exit")
    args = ap.parse_args(argv)
    if args.dry_run:
        cells = expand(args.grid)
        for c in cells:
            print(c["cell_id"])
        print(f"{len(cells)} cells", file=sys.stderr)
        return 0
    summary = run(args.grid, args.results_dir, args.confirm_full, args.retry_failed,
                  args.limit, args.timeout)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
