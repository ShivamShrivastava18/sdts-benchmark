"""EXPLORATORY: per-classifier breakdown of TSTR AUROC (declared exploratory
in PREREGISTRATION.md Section 6).

    python -m sdts.analysis.exploratory.per_classifier
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sdts.analysis import primary as pr  # noqa: E402
from sdts.analysis.aggregate import OUT_DIR  # noqa: E402
from sdts.analysis.exploratory import EXPLORATORY_FIG_DIR, exploratory_name, watermark  # noqa: E402
from sdts.analysis.figures import ORDER, STYLE  # noqa: E402


def make(summary_path: Path = OUT_DIR / "summary.parquet", budget: int = pr.TUNED_BUDGET,
         out: Path = EXPLORATORY_FIG_DIR) -> list[Path]:
    df = pr.select_budget(pd.read_parquet(summary_path), budget)
    clfs = ["lr", "rf", "xgb"]
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 3, figsize=(12, 3 * len(datasets)), squeeze=False, sharey=True)
    for i, d in enumerate(datasets):
        for j, c in enumerate(clfs):
            ax = axes[i, j]
            sub = df[df["dataset"] == d]
            for m in ORDER:
                s = sub[sub["method"] == m].sort_values("n_train")
                if s.empty or f"{c}_tstr_auroc_mean" not in s:
                    continue
                ax.errorbar(s["n_train"], s[f"{c}_tstr_auroc_mean"], yerr=s[f"{c}_tstr_auroc_std"].fillna(0),
                            label=m, capsize=2, lw=1.2, ms=4, **STYLE.get(m, {}))
            ax.set_xscale("log")
            ax.set_title(f"{d}: {c.upper()}")
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("EXPLORATORY: TSTR AUROC per downstream classifier")
    fig.tight_layout()
    watermark(fig)
    out.mkdir(parents=True, exist_ok=True)
    stem = exploratory_name("per_classifier")
    paths = [out / f"{stem}.png", out / f"{stem}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for p in make():
        print(p)
    sys.exit(0)
