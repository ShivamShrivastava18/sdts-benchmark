"""Exploratory analyses: anything not in PREREGISTRATION.md Section 4.

Every output from this package is written under
``results/figures/exploratory`` or ``results/analysis/exploratory`` and
carries the word EXPLORATORY in its filename, its figure watermark and its
table caption. Nothing here may be cited as a preregistered result.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from sdts.data import loaders

EXPLORATORY_FIG_DIR = loaders.REPO_ROOT / "results" / "figures" / "exploratory"
EXPLORATORY_RESULT_DIR = loaders.REPO_ROOT / "results" / "analysis" / "exploratory"
LABEL = "EXPLORATORY"


def watermark(fig: plt.Figure) -> plt.Figure:
    fig.text(0.5, 0.5, LABEL, fontsize=48, color="red", alpha=0.18, ha="center", va="center", rotation=30, zorder=100)
    return fig


def exploratory_name(name: str) -> str:
    return name if name.startswith("EXPLORATORY_") else f"EXPLORATORY_{name}"
