"""Tables, LaTeX and markdown, from results/aggregate/summary.parquet.

    python -m sdts.analysis.tables            # results/tables/main.{tex,md}, appendix_cells.{tex,md}

main      one row per (dataset, n): primary metric as mean +- std per
          method, real_subsample as the reference column, best competing
          method in bold
appendix  one row per (dataset, n, method): every metric mean +- std,
          seeds ok, statuses, fit seconds
Rows follow the preregistered budget rule (analysis.primary.select_budget).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sdts.analysis import primary as pr
from sdts.analysis.aggregate import OUT_DIR
from sdts.data import loaders

TABLE_DIR = loaders.REPO_ROOT / "results" / "tables"
METHODS = [pr.REFERENCE] + pr.BASELINES + pr.DEEP
APPENDIX_METRICS = ["tstr_auroc_mean", "tstr_acc_mean", "tstr_f1_mean", "trtr_auroc_mean", "c2st_auc", "ks_mean",
                    "tvd_mean", "corr_dist", "dcr_median", "dcr_rate", "nndr_median", "mia_auc"]


def _fmt(m: float, s: float, digits: int = 3) -> str:
    if m is None or np.isnan(m):
        return "--"
    return f"{m:.{digits}f} $\\pm$ {0 if np.isnan(s) else s:.{digits}f}"


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, n), g in df.groupby(["dataset", "n_train"]):
        g = g.set_index("method")
        row = {"dataset": d, "n": int(n)}
        competing = [m for m in pr.COMPETING if m in g.index]
        best = max(competing, key=lambda m: g.loc[m, f"{pr.PRIMARY}_mean"]) if competing else None
        for m in METHODS:
            if m in g.index:
                cell = _fmt(g.loc[m, f"{pr.PRIMARY}_mean"], g.loc[m, f"{pr.PRIMARY}_std"])
                row[m] = f"\\textbf{{{cell}}}" if m == best else cell
            else:
                row[m] = "--"
        rows.append(row)
    return pd.DataFrame(rows)


def appendix_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.sort_values(["dataset", "n_train", "method"]).itertuples():
        row = {"dataset": r.dataset, "n": int(r.n_train), "method": r.method, "budget": int(r.tuning_budget),
               "seeds_ok": f"{int(r.n_ok)}/{int(r.n_seeds)}", "statuses": r.statuses}
        for k in APPENDIX_METRICS:
            row[k] = _fmt(getattr(r, f"metric_{k}_mean", np.nan), getattr(r, f"metric_{k}_std", np.nan))
        row["fit_s"] = f"{getattr(r, 'cost_fit_seconds_mean', np.nan):.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _escape_latex(df: pd.DataFrame) -> pd.DataFrame:
    r"""Escape underscores in names and cells for LaTeX.

    Method and dataset identifiers contain underscores (real_subsample,
    credit_default), which LaTeX reads as subscripts. The formatted cells
    already carry intentional markup ($\pm$, \textbf{}), so this escapes
    only underscores and leaves the rest alone.
    """
    out = df.copy()
    out.columns = [str(c).replace("_", r"\_") for c in out.columns]
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: str(v).replace("_", r"\_") if isinstance(v, str) else v)
    return out


def write(df_main: pd.DataFrame, df_app: pd.DataFrame, out: Path = TABLE_DIR) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, t, caption in (
        ("main", df_main, "TSTR AUROC (mean $\\pm$ std over 5 seeds) per dataset and training size. "
                          "Bold: best competing method (real\\_subsample is the reference ceiling)."),
        ("appendix_cells", df_app, "Every metric per (dataset, n, method), mean $\\pm$ std over ok seeds."),
    ):
        tex = _escape_latex(t).to_latex(index=False, escape=False, longtable=(name != "main"),
                                        caption=caption, label=f"tab:{name}",
                                        column_format="l" * len(t.columns))
        (out / f"{name}.tex").write_text(tex)
        md = t.replace({r"\\textbf\{(.*)\}": r"**\1**", r" \$\\pm\$ ": " ± "}, regex=True).to_markdown(index=False)
        (out / f"{name}.md").write_text(md + "\n")
        paths += [out / f"{name}.tex", out / f"{name}.md"]
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.analysis.tables", description=__doc__)
    ap.add_argument("--budget", type=int, default=pr.TUNED_BUDGET)
    ap.add_argument("--summary", type=Path, default=OUT_DIR / "summary.parquet")
    ap.add_argument("--out", type=Path, default=TABLE_DIR)
    args = ap.parse_args(argv)
    df = pr.select_budget(pd.read_parquet(args.summary), args.budget)
    for p in write(main_table(df), appendix_table(df), args.out):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
