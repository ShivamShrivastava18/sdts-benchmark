"""The five preregistered primary analyses (PREREGISTRATION.md Section 4),
computed from results/aggregate/summary.parquet and written to
results/analysis/ as CSV and JSON.

    python -m sdts.analysis.primary                # tuned rows (budget 20 + untunable at 0)
    python -m sdts.analysis.primary --budget 0     # pilot / default-hparam rows only

Shared definitions (verbatim from the preregistration):
  mean       over ok seeds (summary.parquet, *_mean columns)
  pair std   sqrt((s1^2 + s2^2) / 2) of two methods' seed stds
  baselines  marginals, copula, smote, ucsmote; deep ctgan, tvae, tabddpm;
             real_subsample never competes
  ranking    the 7 competing methods by mean primary metric; a method with
             no ok seed ranks last and is counted
  tau        scipy.stats.kendalltau (tau-b)

1. crossover      per (ladder dataset, deep method): smallest n where
                  deep mean - best baseline mean > pair std; else none
2. headline       per rung: fraction of ladder datasets where the best
                  baseline beats every deep model; any margin, and margin
                  > pair std of the two
3. stability      per dataset and adjacent rung pair: tau over 7 methods
                  and over the 3 deep methods
4. effect_size    per (dataset, n): (top1 - top2) / pair std; fraction < 1
5. subsample      per small dataset: tau between its ranking at true size
                  and each ladder dataset's ranking at the nearest rung
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from sdts.analysis.aggregate import OUT_DIR
from sdts.data import loaders
from sdts.data.splits import LADDER
from sdts.tune.spaces import SPACES

ANALYSIS_DIR = loaders.REPO_ROOT / "results" / "analysis"
PRIMARY = "metric_tstr_auroc_mean"
BASELINES = ["marginals", "copula", "smote", "ucsmote"]
DEEP = ["ctgan", "tvae", "tabddpm"]
COMPETING = BASELINES + DEEP
REFERENCE = "real_subsample"
TUNED_BUDGET = 20


# ----------------------------------------------------------------- helpers
def pair_std(s1: float, s2: float) -> float:
    s1 = 0.0 if s1 is None or np.isnan(s1) else s1
    s2 = 0.0 if s2 is None or np.isnan(s2) else s2
    return float(np.sqrt((s1 ** 2 + s2 ** 2) / 2))


def arms() -> tuple[list[str], list[str]]:
    reg = loaders.registry()
    return ([d for d, v in reg.items() if v.get("arm") == "ladder"],
            [d for d, v in reg.items() if v.get("arm") == "small"])


def select_budget(summary: pd.DataFrame, budget: int = TUNED_BUDGET) -> pd.DataFrame:
    """Rows for the requested tuning budget. Methods without a search space
    only ever have budget 0 and are kept at 0. With budget 0, everything at 0."""
    df = summary[summary["n_ok"] > 0].copy()
    if budget == 0:
        return df[df["tuning_budget"] == 0]
    untunable = [m for m, s in SPACES.items() if s is None]
    keep = (df["tuning_budget"] == budget) | (df["method"].isin(untunable) & (df["tuning_budget"] == 0))
    return df[keep]


def _cell(df: pd.DataFrame, dataset: str, n: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_train"] == n)].set_index("method")


def _ranking(cell: pd.DataFrame, methods: list[str]) -> tuple[list[float], int]:
    """Scores for ``methods`` in fixed order; missing -> -inf (ranks last). Returns (scores, n_missing)."""
    scores, missing = [], 0
    for m in methods:
        if m in cell.index and not np.isnan(cell.loc[m, f"{PRIMARY}_mean"]):
            scores.append(float(cell.loc[m, f"{PRIMARY}_mean"]))
        else:
            scores.append(-np.inf)
            missing += 1
    return scores, missing


def _tau(a: list[float], b: list[float]) -> float:
    a2 = [x if np.isfinite(x) else -1.0 for x in a]
    b2 = [x if np.isfinite(x) else -1.0 for x in b]
    t = kendalltau(a2, b2).statistic
    return float(t) if t is not None and not np.isnan(t) else float("nan")


def _best(cell: pd.DataFrame, methods: list[str]) -> tuple[str | None, float, float]:
    present = [m for m in methods if m in cell.index and not np.isnan(cell.loc[m, f"{PRIMARY}_mean"])]
    if not present:
        return None, float("nan"), float("nan")
    m = max(present, key=lambda x: cell.loc[x, f"{PRIMARY}_mean"])
    return m, float(cell.loc[m, f"{PRIMARY}_mean"]), float(cell.loc[m, f"{PRIMARY}_std"])


# ------------------------------------------------------------- 1 crossover
def crossover(df: pd.DataFrame, ladder: list[str]) -> pd.DataFrame:
    rows = []
    for d in ladder:
        rungs = sorted(int(n) for n in df[df["dataset"] == d]["n_train"].unique())
        if not rungs:
            continue
        for m in DEEP:
            found = None
            margins = []
            for n in rungs:
                cell = _cell(df, d, n)
                bb, bmean, bstd = _best(cell, BASELINES)
                if bb is None or m not in cell.index:
                    margins.append({"n": n, "margin": None, "pair_std": None})
                    continue
                dm, ds = float(cell.loc[m, f"{PRIMARY}_mean"]), float(cell.loc[m, f"{PRIMARY}_std"])
                ps = pair_std(bstd, ds)
                margins.append({"n": n, "best_baseline": bb, "margin": round(dm - bmean, 4), "pair_std": round(ps, 4)})
                if found is None and dm - bmean > ps:
                    found = n
            rows.append({"dataset": d, "method": m, "crossover_n": found, "n_max": rungs[-1],
                         "crossover": found if found is not None else f"none up to {rungs[-1]}",
                         "margins": json.dumps(margins)})
    return pd.DataFrame(rows)


# -------------------------------------------------------------- 2 headline
def headline(df: pd.DataFrame, ladder: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = []
    for d in ladder:
        for n in sorted(int(x) for x in df[df["dataset"] == d]["n_train"].unique()):
            cell = _cell(df, d, n)
            bb, bmean, bstd = _best(cell, BASELINES)
            bd, dmean, dstd = _best(cell, DEEP)
            if bb is None:
                continue
            if bd is None:  # no deep model produced a result: baseline wins by default, counted
                cells.append({"dataset": d, "n_train": n, "best_baseline": bb, "best_deep": None,
                              "margin": None, "pair_std": None, "baseline_wins": True,
                              "baseline_wins_beyond_noise": True, "deep_missing": True})
                continue
            ps = pair_std(bstd, dstd)
            cells.append({"dataset": d, "n_train": n, "best_baseline": bb, "best_deep": bd,
                          "best_baseline_auroc": round(bmean, 4), "best_deep_auroc": round(dmean, 4),
                          "margin": round(bmean - dmean, 4), "pair_std": round(ps, 4),
                          "baseline_wins": bmean > dmean, "baseline_wins_beyond_noise": bmean - dmean > ps,
                          "deep_missing": False})
    cdf = pd.DataFrame(cells)
    if cdf.empty:
        return cdf, pd.DataFrame()
    per_n = (cdf.groupby("n_train").agg(n_datasets=("dataset", "size"),
                                         frac_baseline_wins=("baseline_wins", "mean"),
                                         frac_baseline_wins_beyond_noise=("baseline_wins_beyond_noise", "mean"))
             .reset_index())
    return cdf, per_n


# ------------------------------------------------------------- 3 stability
def stability(df: pd.DataFrame, ladder: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for d in ladder:
        rungs = sorted(int(n) for n in df[df["dataset"] == d]["n_train"].unique())
        for lo, hi in zip(rungs, rungs[1:]):
            a, ma = _ranking(_cell(df, d, lo), COMPETING)
            b, mb = _ranking(_cell(df, d, hi), COMPETING)
            da, _ = _ranking(_cell(df, d, lo), DEEP)
            db, _ = _ranking(_cell(df, d, hi), DEEP)
            rows.append({"dataset": d, "n_lo": lo, "n_hi": hi, "tau_all7": round(_tau(a, b), 4),
                         "tau_deep3": round(_tau(da, db), 4), "missing_methods": ma + mb})
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return rdf, pd.DataFrame()
    per_pair = rdf.groupby(["n_lo", "n_hi"]).agg(n_datasets=("dataset", "size"), tau_all7_mean=("tau_all7", "mean"),
                                                  tau_deep3_mean=("tau_deep3", "mean")).reset_index()
    return rdf, per_pair


# ------------------------------------------------------------ 4 effect size
def effect_size(df: pd.DataFrame, datasets: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for d in datasets:
        for n in sorted(int(x) for x in df[df["dataset"] == d]["n_train"].unique()):
            cell = _cell(df, d, n)
            present = [m for m in COMPETING if m in cell.index]
            if len(present) < 2:
                continue
            ranked = sorted(present, key=lambda m: -cell.loc[m, f"{PRIMARY}_mean"])
            t1, t2 = ranked[0], ranked[1]
            gap = float(cell.loc[t1, f"{PRIMARY}_mean"] - cell.loc[t2, f"{PRIMARY}_mean"])
            ps = pair_std(cell.loc[t1, f"{PRIMARY}_std"], cell.loc[t2, f"{PRIMARY}_std"])
            ratio = gap / ps if ps > 0 else float("inf")
            rows.append({"dataset": d, "n_train": n, "top1": t1, "top2": t2, "gap": round(gap, 4),
                         "pair_std": round(ps, 4), "ratio": round(ratio, 3) if np.isfinite(ratio) else None,
                         "inside_noise": bool(ratio < 1.0)})
    edf = pd.DataFrame(rows)
    summ = {"n_cells": int(len(edf)), "fraction_inside_noise": float(edf["inside_noise"].mean()) if len(edf) else None}
    return edf, summ


# ------------------------------------------------------------- 5 subsample
def subsample_validity(df: pd.DataFrame, ladder: list[str], small: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for s in small:
        sub = df[df["dataset"] == s]
        if sub.empty:
            continue
        n_true = int(sub["n_train"].iloc[0])
        rung = min(LADDER, key=lambda r: (abs(r - n_true), r))
        a, ma = _ranking(_cell(df, s, n_true), COMPETING)
        for d in ladder:
            cell = _cell(df, d, rung)
            if cell.empty:
                continue
            b, mb = _ranking(cell, COMPETING)
            rows.append({"small_dataset": s, "n_true": n_true, "matched_rung": rung, "ladder_dataset": d,
                         "tau": round(_tau(a, b), 4), "missing_methods": ma + mb})
    sdf = pd.DataFrame(rows)
    if sdf.empty:
        return sdf, pd.DataFrame()
    per_small = sdf.groupby(["small_dataset", "matched_rung"]).agg(n_ladder=("ladder_dataset", "size"),
                                                                    tau_mean=("tau", "mean")).reset_index()
    return sdf, per_small


# ------------------------------------------------------------------ driver
def run_all(summary: pd.DataFrame, budget: int = TUNED_BUDGET, out_dir: Path = ANALYSIS_DIR) -> dict[str, Any]:
    df = select_budget(summary, budget)
    ladder, small = arms()
    out_dir.mkdir(parents=True, exist_ok=True)
    cx = crossover(df, ladder)
    hl_cells, hl = headline(df, ladder)
    st, st_pairs = stability(df, ladder)
    es, es_summ = effect_size(df, ladder + small)
    sv, sv_small = subsample_validity(df, ladder, small)
    tables = {"crossover": cx, "headline_cells": hl_cells, "headline": hl, "stability": st,
              "stability_pairs": st_pairs, "effect_size": es, "subsample_validity": sv,
              "subsample_validity_summary": sv_small}
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    summary_doc = {
        "tuning_budget": budget, "n_rows_used": int(len(df)),
        "crossover": {f"{r.dataset}/{r.method}": r.crossover for r in cx.itertuples()} if len(cx) else {},
        "headline": hl.to_dict(orient="records") if len(hl) else [],
        "stability_pairs": st_pairs.to_dict(orient="records") if len(st_pairs) else [],
        "effect_size": es_summ,
        "subsample_validity": sv_small.to_dict(orient="records") if len(sv_small) else [],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_doc, indent=2, default=str) + "\n")
    return summary_doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.analysis.primary", description=__doc__)
    ap.add_argument("--budget", type=int, default=TUNED_BUDGET)
    ap.add_argument("--summary", type=Path, default=OUT_DIR / "summary.parquet")
    ap.add_argument("--out", type=Path, default=ANALYSIS_DIR)
    args = ap.parse_args(argv)
    doc = run_all(pd.read_parquet(args.summary), args.budget, args.out)
    print(json.dumps({k: v for k, v in doc.items() if k != "crossover"}, indent=1, default=str)[:2000])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
