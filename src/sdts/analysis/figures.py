"""Figures, generated only from results/aggregate and results/analysis.

    python -m sdts.analysis.figures --all            # fig1..fig5 into results/figures/
    python -m sdts.analysis.figures --fig 1

Fig 1  primary metric against n, faceted by dataset, one line per method,
       seed-std error bars; crossover points (analysis 1) marked with a star.
Fig 2  fraction of ladder datasets where the best baseline beats every deep
       model, by rung: any margin, and margin beyond the pair std.
Fig 3  Kendall tau between adjacent rungs, per dataset and mean: all 7
       competing methods, and the 3 deep methods.
Fig 4  top-two gap against pair std, one point per (dataset, n); points
       below the diagonal are cells where the winner is inside the noise.
Fig 5  utility against privacy: primary metric vs DCR rate and vs
       membership-inference AUC, one point per (dataset, n, method).
Rows are selected with the preregistered budget rule (analysis.primary.select_budget).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sdts.analysis import primary as pr  # noqa: E402
from sdts.analysis.aggregate import OUT_DIR  # noqa: E402
from sdts.data import loaders  # noqa: E402

FIG_DIR = loaders.REPO_ROOT / "results" / "figures"
PRIMARY = pr.PRIMARY
BASELINES, DEEP, REFERENCE = pr.BASELINES, pr.DEEP, [pr.REFERENCE]
STYLE = {
    "real_subsample": dict(color="black", ls="--", marker="x"),
    "marginals": dict(color="#7f7f7f", marker="o"),
    "copula": dict(color="#1f77b4", marker="o"),
    "smote": dict(color="#2ca02c", marker="o"),
    "ucsmote": dict(color="#17becf", marker="o"),
    "ctgan": dict(color="#d62728", marker="s"),
    "tvae": dict(color="#ff7f0e", marker="s"),
    "tabddpm": dict(color="#9467bd", marker="s"),
}
ORDER = REFERENCE + BASELINES + DEEP


def _save(fig: plt.Figure, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return paths


def _grid(n_panels: int, width: float = 4.2, height: float = 3.4, ncol_max: int = 4):
    ncol = min(ncol_max, max(1, n_panels))
    nrow = (n_panels + ncol - 1) // ncol
    return plt.subplots(nrow, ncol, figsize=(width * ncol, height * nrow), squeeze=False)


def fig1(df: pd.DataFrame, out: Path = FIG_DIR, crossover: pd.DataFrame | None = None,
         name: str = "fig1_tstr_vs_n") -> list[Path]:
    datasets = sorted(df["dataset"].unique())
    fig, axes = _grid(len(datasets))
    for ax, d in zip(axes.flat, datasets):
        sub = df[df["dataset"] == d]
        for m in ORDER:
            s = sub[sub["method"] == m].sort_values("n_train")
            if s.empty:
                continue
            ax.errorbar(s["n_train"], s[f"{PRIMARY}_mean"], yerr=s[f"{PRIMARY}_std"].fillna(0),
                        label=m, capsize=3, lw=1.4, ms=5, **STYLE.get(m, {}))
            if crossover is not None and m in DEEP:
                cx = crossover[(crossover["dataset"] == d) & (crossover["method"] == m)]
                if len(cx) and pd.notna(cx["crossover_n"].iloc[0]):
                    n0 = int(cx["crossover_n"].iloc[0])
                    y0 = float(s[s["n_train"] == n0][f"{PRIMARY}_mean"].iloc[0])
                    ax.plot([n0], [y0], marker="*", ms=16, color=STYLE[m]["color"], mec="black", zorder=5)
        ax.set_xscale("log")
        ticks = sorted(int(x) for x in sub["n_train"].unique())
        ax.set_xticks(ticks, [str(t) for t in ticks], rotation=45)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(d)
        ax.set_xlabel("training rows n")
        ax.axhline(0.5, color="lightgray", lw=0.8, zorder=0)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("TSTR AUROC (mean of LR, RF, XGB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)), frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Utility against training size; error bars = std over seeds; star = crossover (analysis 1)", y=1.0)
    fig.tight_layout()
    return _save(fig, out, name)


def fig2(headline: pd.DataFrame, out: Path = FIG_DIR, name: str = "fig2_headline") -> list[Path]:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if len(headline):
        x = np.arange(len(headline))
        ax.bar(x - 0.2, headline["frac_baseline_wins"], 0.4, label="any margin", color="#2ca02c")
        ax.bar(x + 0.2, headline["frac_baseline_wins_beyond_noise"], 0.4, label="margin > pair std", color="#1b5e20")
        ax.set_xticks(x, [f"{int(n)}\n({k} ds)" for n, k in zip(headline["n_train"], headline["n_datasets"])])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of datasets")
    ax.set_xlabel("training rows n (number of ladder datasets with that rung)")
    ax.set_title("Best trivial baseline beats every deep model (analysis 2)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out, name)


def fig3(stability: pd.DataFrame, pairs: pd.DataFrame, out: Path = FIG_DIR, name: str = "fig3_ranking_stability") -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, col, title in zip(axes, ("tau_all7", "tau_deep3"), ("all 7 competing methods", "3 deep methods")):
        if len(stability):
            for d, g in stability.groupby("dataset"):
                ax.plot(g["n_hi"], g[col], marker="o", ms=4, lw=1, alpha=0.5, label=d)
            m = pairs.sort_values("n_hi")
            ax.plot(m["n_hi"], m[f"{col}_mean"], marker="s", ms=7, lw=2.5, color="black", label="mean")
            ax.set_xscale("log")
            pr_pairs = stability.drop_duplicates("n_hi").sort_values("n_hi")
            ax.set_xticks(pr_pairs["n_hi"].tolist(), [f"{int(lo)}->{int(hi)}" for lo, hi in
                                                      pr_pairs[["n_lo", "n_hi"]].to_numpy()], rotation=45)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.axhline(0.8, color="gray", ls=":", lw=1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Kendall tau, adjacent rungs: {title}")
        ax.set_xlabel("rung pair")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Kendall tau-b")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Ranking stability between adjacent rungs (analysis 3); dotted line = 0.8 falsification threshold")
    fig.tight_layout()
    return _save(fig, out, name)


def fig4(effect: pd.DataFrame, out: Path = FIG_DIR, name: str = "fig4_effect_vs_noise") -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    if len(effect):
        sc = ax.scatter(effect["pair_std"], effect["gap"], c=np.log10(effect["n_train"]), cmap="viridis", s=40,
                        edgecolor="black", lw=0.5)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("log10 n")
        lim = float(max(effect["pair_std"].max(), effect["gap"].max()) * 1.1) or 0.1
        ax.plot([0, lim], [0, lim], color="gray", ls="--", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        below = int(effect["inside_noise"].sum())
        ax.text(0.03, 0.95, f"{below} of {len(effect)} cells below the diagonal", transform=ax.transAxes, va="top")
    ax.set_xlabel("pair std of the top two methods")
    ax.set_ylabel("gap between top two methods (TSTR AUROC)")
    ax.set_title("Effect size against seed noise (analysis 4)")
    ax.grid(alpha=0.3)
    return _save(fig, out, name)


def fig5(df: pd.DataFrame, out: Path = FIG_DIR, name: str = "fig5_privacy_utility") -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, xcol, xlabel, ideal in zip(axes, ("metric_dcr_rate_mean", "metric_mia_auc_mean"),
                                       ("DCR rate (share of synthetic rows closer to train than test)",
                                        "membership-inference AUC"),
                                       (0.5, 0.5)):
        for m in ORDER:
            s = df[df["method"] == m]
            if s.empty:
                continue
            ax.scatter(s[xcol], s[f"{PRIMARY}_mean"], label=m, s=28 + 6 * np.log10(s["n_train"]), alpha=0.75,
                       color=STYLE[m]["color"], marker=STYLE[m]["marker"], edgecolor="black", lw=0.3)
        ax.axvline(ideal, color="gray", ls=":", lw=1)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("TSTR AUROC (mean of LR, RF, XGB)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Utility against privacy, one point per (dataset, n, method); dotted = no leakage; size grows with n")
    fig.tight_layout()
    return _save(fig, out, name)


def make_all(budget: int = pr.TUNED_BUDGET, summary_path: Path = OUT_DIR / "summary.parquet",
             analysis_dir: Path = pr.ANALYSIS_DIR, out: Path = FIG_DIR, which: list[int] | None = None) -> list[Path]:
    summary = pd.read_parquet(summary_path)
    df = pr.select_budget(summary, budget)
    which = which or [1, 2, 3, 4, 5]
    paths: list[Path] = []

    def read(name: str) -> pd.DataFrame:
        p = analysis_dir / f"{name}.csv"
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()

    if 1 in which:
        cx = read("crossover")
        paths += fig1(df, out, cx if len(cx) else None)
    if 2 in which:
        paths += fig2(read("headline"), out)
    if 3 in which:
        paths += fig3(read("stability"), read("stability_pairs"), out)
    if 4 in which:
        paths += fig4(read("effect_size"), out)
    if 5 in which:
        paths += fig5(df, out)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdts.analysis.figures", description=__doc__)
    ap.add_argument("--fig", type=int, action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--budget", type=int, default=pr.TUNED_BUDGET)
    ap.add_argument("--summary", type=Path, default=OUT_DIR / "summary.parquet")
    ap.add_argument("--analysis-dir", type=Path, default=pr.ANALYSIS_DIR)
    ap.add_argument("--out", type=Path, default=FIG_DIR)
    args = ap.parse_args(argv)
    which = None if args.all or not args.fig else args.fig
    for p in make_all(args.budget, args.summary, args.analysis_dir, args.out, which):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
