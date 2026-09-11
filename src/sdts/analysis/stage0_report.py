"""Stage 0 pilot report and gate, computed from results/aggregate/*.

    python -m sdts.analysis.stage0_report        # writes runs/STAGE0_REPORT.md

Answers, with numbers, the four PLAN.md Phase 5 questions and states which
gate outcome occurred. Definitions:

* baselines = marginals, copula, smote, ucsmote; deep = ctgan, tvae,
  tabddpm; real_subsample is the reference ceiling and takes part in
  neither group.
* "baseline beats every deep model" in a (dataset, n) cell: the best
  baseline's mean TSTR AUROC exceeds the best deep model's mean.
* pair std: pooled seed std of the two compared methods,
  sqrt((s_baseline^2 + s_deep^2) / 2), the convention of
  PREREGISTRATION.md analysis 4. The gap is "outside the noise" when
  |gap| > pair std. "Error bars overlap" when the +-1 std intervals of
  the two methods intersect.
* sensitivity: the pooled std over all methods in the cell is also
  reported; it is dominated by the noisiest method (marginals).
* Gate, fixed before the pilot ran: green if the best
  baseline wins every cell at n <= 1000; inverted if a deep model wins
  every such cell; ambiguous if the direction is mixed across cells with
  overlapping error bars. Cells where the winner's margin is inside the
  pair std are listed so the reader can weigh the strength of the
  outcome; they do not change it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sdts.analysis.aggregate import OUT_DIR
from sdts.analysis.figures import BASELINES, DEEP, PRIMARY
from sdts.data import loaders

REPORT = loaders.REPO_ROOT / "runs" / "STAGE0_REPORT.md"
SMALL_N = 1000


def pilot_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the cells of the ``stage0`` pilot grid.

    ``results/`` now also holds Stage 1 and Stage 2, so the same (dataset,
    n, method) can appear at two tuning budgets. The pilot is defined by
    configs/grid.yaml, so it is read from there rather than hardcoded; this
    restores the report's original input and changes no definition (D004).
    """
    from sdts.data import splits
    from sdts.runner.run_grid import expand

    wanted: set[tuple[str, int, int]] = set()
    for c in expand("stage0"):
        n = len(splits.split(c["dataset"]).train) if c["n"] == "full" else int(c["n"])
        wanted.add((c["dataset"], n, int(c["budget"])))
    keep = [(d, int(n), int(b)) in wanted
            for d, n, b in zip(df["dataset"], df["n_train"], df["tuning_budget"])]
    return df[keep]


def cell_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, n), sub in summary.groupby(["dataset", "n_train"]):
        sub = sub[sub["n_ok"] > 0].set_index("method")
        base = sub.loc[sub.index.intersection(BASELINES)]
        deep = sub.loc[sub.index.intersection(DEEP)]
        if base.empty or deep.empty:
            continue
        bb, bd = base[f"{PRIMARY}_mean"].idxmax(), deep[f"{PRIMARY}_mean"].idxmax()
        pooled_all = float(np.sqrt(np.nanmean(sub[f"{PRIMARY}_std"] ** 2)))
        sb, sd = float(base.loc[bb, f"{PRIMARY}_std"]), float(deep.loc[bd, f"{PRIMARY}_std"])
        pair = float(np.sqrt((sb ** 2 + sd ** 2) / 2))
        mb, md_ = float(base.loc[bb, f"{PRIMARY}_mean"]), float(deep.loc[bd, f"{PRIMARY}_mean"])
        gap = mb - md_
        overlap = (mb - sb <= md_ + sd) and (md_ - sd <= mb + sb)
        ranked = sub[f"{PRIMARY}_mean"].drop(index="real_subsample", errors="ignore").sort_values(ascending=False)
        top2 = float(ranked.iloc[0] - ranked.iloc[1]) if len(ranked) > 1 else float("nan")
        top2_std = float(np.sqrt(np.nanmean(sub.loc[list(ranked.index[:2]), f"{PRIMARY}_std"] ** 2))) if len(ranked) > 1 else float("nan")
        rows.append({"dataset": d, "n_train": int(n), "best_baseline": bb,
                     "best_baseline_auroc": round(mb, 4), "best_baseline_std": round(sb, 4),
                     "best_deep": bd, "best_deep_auroc": round(md_, 4), "best_deep_std": round(sd, 4),
                     "gap_baseline_minus_deep": round(gap, 4), "pair_std": round(pair, 4),
                     "baseline_wins": gap > 0, "gap_outside_pair_std": abs(gap) > pair,
                     "error_bars_overlap": bool(overlap),
                     "pooled_std_all_methods": round(pooled_all, 4),
                     "gap_outside_all_methods_std": abs(gap) > pooled_all,
                     "top1": ranked.index[0], "top2": ranked.index[1] if len(ranked) > 1 else None,
                     "top2_gap": round(top2, 4), "top2_pair_std": round(top2_std, 4),
                     "top2_gap_inside_noise": top2 < top2_std,
                     "real_subsample_auroc": round(float(sub.loc["real_subsample", f"{PRIMARY}_mean"]), 4)
                     if "real_subsample" in sub.index else None})
    return pd.DataFrame(rows)


def gate(cells: pd.DataFrame) -> str:
    small = cells[cells["n_train"] <= SMALL_N]
    if small.empty:
        return "ambiguous (no cells at n <= %d)" % SMALL_N
    if small["baseline_wins"].all():
        return "green"
    if (~small["baseline_wins"]).all():
        return "inverted"
    return "ambiguous"


def build(summary: pd.DataFrame, cells_df: pd.DataFrame) -> str:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=loaders.REPO_ROOT, text=True).strip()
    ct = cell_table(summary)
    md = [f"# Stage 0 pilot report\n", f"Generated by `python -m sdts.analysis.stage0_report` at commit `{sha}` "
          f"from {len(cells_df)} result files ({int((cells_df.status == 'ok').sum())} ok). "
          "Every number below traces to `results/*.json` via `results/aggregate/`.\n"]
    md += ["## 1. Does any baseline beat every deep model on TSTR AUROC?\n",
           "Per (dataset, n): best baseline vs best deep model, mean over seeds. "
           "pair_std is the pooled seed std of those two methods.\n",
           ct[["dataset", "n_train", "best_baseline", "best_baseline_auroc", "best_baseline_std", "best_deep",
               "best_deep_auroc", "best_deep_std", "gap_baseline_minus_deep", "pair_std", "baseline_wins",
               "gap_outside_pair_std", "error_bars_overlap", "real_subsample_auroc"]].to_markdown(index=False), ""]
    small = ct[ct["n_train"] <= SMALL_N]
    md += [f"At n <= {SMALL_N}: baseline wins in {int(small['baseline_wins'].sum())} of {len(small)} cells; "
           f"gap outside the pair std in {int(small['gap_outside_pair_std'].sum())} of {len(small)}; "
           f"+-1 std error bars overlap in {int(small['error_bars_overlap'].sum())} of {len(small)}.\n",
           "Sensitivity (stricter, not the gate criterion): pooled std over all eight methods in the cell, "
           "dominated by marginals:\n",
           ct[["dataset", "n_train", "gap_baseline_minus_deep", "pooled_std_all_methods",
               "gap_outside_all_methods_std"]].to_markdown(index=False), ""]
    md += ["## 2. Seed noise and top-two gaps\n",
           "Per-method seed std of TSTR AUROC, per cell:\n"]
    piv = summary.pivot_table(index=["dataset", "n_train"], columns="method", values=f"{PRIMARY}_std")
    md += [piv.round(4).to_markdown(), "",
           f"Cells whose top-two gap (excluding real_subsample) is smaller than the pair std of those two methods: "
           f"{int(ct['top2_gap_inside_noise'].sum())} of {len(ct)}.\n",
           ct[["dataset", "n_train", "top1", "top2", "top2_gap", "top2_pair_std", "top2_gap_inside_noise"]]
           .to_markdown(index=False), ""]
    md += ["## 3. Failures and timeouts\n"]
    bad = cells_df[cells_df["status"] != "ok"]
    if bad.empty:
        md += ["None. Every cell finished with status ok.\n"]
    else:
        md += [bad.groupby(["dataset", "n_train", "method", "status"]).size().rename("cells").reset_index()
               .to_markdown(index=False), "", "First error line per failed cell:", ""]
        md += [f"- `{r.cell_id}`: {r.error_head}" for r in bad.itertuples()] + [""]
    md += ["## 4. Wall clock per cell per method\n",
           "Mean seconds over ok seeds (fit, sample, eval, total):\n"]
    cost = (summary.groupby(["dataset", "n_train", "method"])[["cost_fit_seconds_mean", "cost_sample_seconds_mean",
                                                                "cost_eval_seconds_mean", "cost_total_seconds_mean"]]
            .mean().round(1))
    md += [cost.to_markdown(), ""]
    per_method = cells_df[cells_df.status == "ok"].groupby("method")["cost_total_seconds"].agg(["mean", "max"]).round(1)
    md += ["Per method over all Stage 0 cells (seconds, mean and max):", "", per_method.to_markdown(), ""]
    g = gate(ct)
    strong = int((small["baseline_wins"] & small["gap_outside_pair_std"]).sum())
    md += ["## Gate\n", f"**Outcome: {g}.**\n",
           "Rule, fixed before the pilot ran: green if the best baseline beats every deep "
           f"model in every cell at n <= {SMALL_N}; inverted if a deep model wins every such cell; ambiguous if "
           "the direction is mixed across cells with overlapping error bars. The outcome is reported, not chosen.\n",
           f"Strength: in {strong} of {len(small)} cells at n <= {SMALL_N} the winning margin exceeds the pair std; "
           f"in {int(small['error_bars_overlap'].sum())} the +-1 std error bars overlap. "
           "The stricter all-methods pooled std is reported above as a sensitivity check and would call "
           f"{int((small['gap_baseline_minus_deep'].abs() > small['pooled_std_all_methods']).sum())} of {len(small)} "
           "cells decided.\n"]
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.analysis.stage0_report", description=__doc__)
    ap.add_argument("--aggregate-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    summary = pilot_rows(pd.read_parquet(args.aggregate_dir / "summary.parquet"))
    cells_df = pilot_rows(pd.read_parquet(args.aggregate_dir / "cells.parquet"))
    text = build(summary, cells_df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text.split("## Gate")[1].strip().splitlines()[0])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
