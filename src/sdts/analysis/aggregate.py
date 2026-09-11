"""results/*.json -> one long-format table plus a per-cell wide table and a
per-(dataset, n, method) summary with seed mean and std.

    python -m sdts.analysis.aggregate            # writes results/aggregate/*
    python -m sdts.analysis.aggregate --results-dir runs/smoke --out /tmp/x

Outputs (parquet and CSV, committed):
  cells.*    one row per result file: ids, status, every metric flattened,
             per-classifier metrics as <clf>_<metric>, cost, env.git_sha
  long.*     one row per (cell, metric_name, value)
  summary.*  one row per (dataset, n_train, method, tuning_budget):
             n_seeds, n_ok, mean and std over ok seeds for every metric,
             mean cost. Cells with fewer than 2 ok seeds have std NaN.
Nothing here is typed by hand; every number traces to a result JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sdts.data import loaders

REPO_ROOT = loaders.REPO_ROOT
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "aggregate"

ID_COLS = ["cell_id", "dataset", "n_train", "method", "seed", "tuning_budget", "status"]


def load_cells(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        d = json.loads(path.read_text())
        row = {k: d.get(k) for k in ID_COLS}
        row["n_requested"] = d["cell_id"].split("__")[1][1:]
        row.update({f"metric_{k}": v for k, v in (d.get("metrics") or {}).items()})
        for clf, m in (d.get("per_classifier") or {}).items():
            row.update({f"{clf}_{k}": v for k, v in m.items()})
        row.update({f"cost_{k}": v for k, v in (d.get("cost") or {}).items()})
        row["git_sha"] = (d.get("env") or {}).get("git_sha")
        row["code_sha"] = (d.get("env") or {}).get("code_sha")
        row["with_replacement"] = (d.get("extra") or {}).get("with_replacement")
        row["error_head"] = (d.get("error") or "").strip().splitlines()[-1][:200] if d.get("error") else None
        row["file"] = path.name
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=ID_COLS)
    df = pd.DataFrame(rows)
    df["n_train"] = pd.to_numeric(df["n_train"], errors="coerce").astype("Int64")
    return df


def to_long(cells: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in cells.columns if c.startswith(("metric_", "lr_", "rf_", "xgb_", "cost_"))]
    long = cells.melt(id_vars=ID_COLS, value_vars=value_cols, var_name="metric", value_name="value")
    return long.dropna(subset=["value"]).reset_index(drop=True)


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "n_train", "method", "tuning_budget"]
    metric_cols = [c for c in cells.columns if c.startswith(("metric_", "lr_", "rf_", "xgb_"))]
    cost_cols = [c for c in cells.columns if c.startswith("cost_")]
    ok = cells[cells["status"] == "ok"]
    counts = cells.groupby(keys, dropna=False).agg(n_seeds=("seed", "size"),
                                                   n_ok=("status", lambda s: int((s == "ok").sum())))
    statuses = cells.groupby(keys, dropna=False)["status"].apply(lambda s: ",".join(sorted(set(s)))).rename("statuses")
    parts = [counts, statuses]
    if len(ok):
        mean = ok.groupby(keys, dropna=False)[metric_cols].mean().add_suffix("_mean")
        std = ok.groupby(keys, dropna=False)[metric_cols].std(ddof=1).add_suffix("_std")
        cost = ok.groupby(keys, dropna=False)[cost_cols].mean().add_suffix("_mean")
        parts += [mean, std, cost]
    return pd.concat(parts, axis=1).reset_index()


def check_mixed_sha(cells: pd.DataFrame) -> list[str]:
    """Return the code SHAs present (hash of src/configs/third_party/pyproject at
    HEAD); more than one means the result set mixes code versions. Results-only
    commits change git_sha but not code_sha. Files without code_sha (before
    2026-09-07 phase 8) fall back to git_sha with any '-dirty' suffix kept."""
    if not len(cells):
        return []
    key = cells["code_sha"].where(cells["code_sha"].notna() & (cells["code_sha"] != ""), cells["git_sha"])
    return sorted(key.dropna().unique().tolist())


def write_all(results_dir: Path = RESULTS_DIR, out_dir: Path = OUT_DIR) -> dict[str, int]:
    cells = load_cells(results_dir)
    long = to_long(cells) if len(cells) else pd.DataFrame(columns=ID_COLS + ["metric", "value"])
    summary = summarise(cells) if len(cells) else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("cells", cells), ("long", long), ("summary", summary)):
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        df.to_csv(out_dir / f"{name}.csv", index=False)
    shas = check_mixed_sha(cells)
    return {"cells": len(cells), "ok": int((cells["status"] == "ok").sum()) if len(cells) else 0,
            "n_shas": len(shas), "n_git_shas": int(cells["git_sha"].nunique()) if len(cells) else 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.analysis.aggregate", description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    info = write_all(args.results_dir, args.out)
    print(json.dumps(info))
    if info["n_shas"] > 1:
        print("WARNING: results from more than one code SHA", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
