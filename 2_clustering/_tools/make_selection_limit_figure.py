"""Two panels for the clustering selection result: the gates never bound, so the score
collapsed onto recall.

Run from the root of this package, the directory holding `book_identifiers.py`:

    python 2_clustering/_tools/make_selection_limit_figure.py

**Why this exists.** Section 4.3.1 of the report states three things in prose: that all 24
configurations cleared both eligibility gates, that 20 of them recorded no false merge at all,
and that a rule written to weight precision four times as heavily as recall therefore selected
on recall. The third is the conclusion and the hardest to see in a table, because it is a
statement about the *shape* of the grid rather than about any one row.

**What the panels show.** Left: every configuration against the two gates, which are nowhere
near the points they were meant to filter. Right: F0.5 against recall, with the curve
F0.5 = 1.25R/(0.25+R) that F0.5 reduces to when precision is exactly 1. The 20 zero-error
configurations lie on that curve to machine precision, so for five sixths of the grid the
selection score is a function of recall alone, and the selected configuration is the
rightmost point on it.

Reads the two frozen calibration grids and writes into the report's figure directory.
Nothing is recomputed: precision, recall, F0.5 and the largest-cluster share are taken as
scored.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(".").resolve()
assert (PROJECT / "book_identifiers.py").exists(), "run from the project root"

GRID = (PROJECT / "2_clustering_outputs"
        / "all_regions_v1_minside24_dinov2_vitb14_binarized_v1_clustering_benchmark_v1"
        / "method_comparison_rerun1")
OUT = PROJECT / "report" / "figures" / "fig_clu_selection_limit.png"

P_GATE, L_GATE = 0.90, 0.10          # the two declared eligibility constraints
NAVY, PALE, ORANGE, GREY = "#1f4e79", "#8fb4d9", "#e8703a", "#b0b7bd"

h = pd.read_csv(GRID / "hdbscan_calibration_grid.csv").assign(fam="HDBSCAN")
g = pd.read_csv(GRID / "graph_calibration_grid.csv").assign(fam="mutual-$k$NN")
a = pd.concat([h, g], ignore_index=True)
assert len(a) == 24, f"expected 24 configurations, found {len(a)}"

a["clean"] = a.false_positive == 0
sel = a.loc[a.f0_5.idxmax()]
assert sel.fam == "HDBSCAN" and sel.min_cluster_size == 10 and sel.min_samples == 5
assert a.recall.max() == sel.recall, "selected configuration is not the highest-recall one"

# F0.5 collapses to a function of recall wherever precision is exactly 1.
clean = a[a.clean]
resid = np.abs(clean.f0_5 - 1.25 * clean.recall / (0.25 + clean.recall)).max()
assert resid < 1e-12, f"identity does not hold: residual {resid:.2e}"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.9))

# ---- left: neither gate bound -------------------------------------------------
axL.axvline(P_GATE, color=ORANGE, lw=1.4, ls="--", zorder=2)
axL.axhline(L_GATE, color=ORANGE, lw=1.4, ls="--", zorder=2)
axL.text(P_GATE + 0.004, 0.015, "precision gate 0.90", color=ORANGE, fontsize=7.5,
         rotation=90, va="bottom", ha="left")
axL.text(1.008, L_GATE + 0.003, "largest-cluster limit 10%", color=ORANGE, fontsize=7.5,
         va="bottom", ha="right")
axL.annotate("", xy=(0.9865, 0.075), xytext=(0.9865, 0.098),
             arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9))
axL.text(0.982, 0.0865, "every configuration\nclears both", color="#555555", fontsize=7.5,
         ha="right", va="center")
for fam, mark in [("HDBSCAN", "o"), ("mutual-$k$NN", "^")]:
    s = a[a.fam == fam]
    axL.scatter(s.precision, s.largest_cluster_fraction, marker=mark, s=34,
                facecolor=NAVY if fam == "HDBSCAN" else PALE,
                edgecolor=NAVY, lw=0.8, zorder=3, label=fam)
axL.set_xlim(0.878, 1.012)
axL.set_ylim(0, 0.118)
axL.set_xlabel("Calibration precision", fontsize=9)
axL.set_ylabel("Largest cluster, share of corpus", fontsize=9)
axL.set_title("Neither eligibility gate excluded a configuration", fontsize=9.5)
axL.legend(fontsize=7.5, frameon=False, loc="upper left")
axL.grid(alpha=0.25, lw=0.5)
axL.tick_params(labelsize=8)

# ---- right: so the score is a function of recall -------------------------------
r = np.linspace(0.02, 0.64, 300)
axR.plot(r, 1.25 * r / (0.25 + r), color=GREY, lw=1.3, zorder=1,
         label=r"$F_{0.5}=1.25R/(0.25+R)$, the curve at $P=1$")
axR.scatter(clean.recall, clean.f0_5, marker="o", s=34, facecolor=NAVY, edgecolor=NAVY,
            lw=0.8, zorder=3, label="precision exactly 1.000 (20 of 24)")
dirty = a[~a.clean]
axR.scatter(dirty.recall, dirty.f0_5, marker="o", s=34, facecolor="white",
            edgecolor=NAVY, lw=1.0, zorder=3, label="one or more false merges (4)")
axR.scatter([sel.recall], [sel.f0_5], marker="*", s=230, facecolor=ORANGE,
            edgecolor=ORANGE, lw=0.8, zorder=4, label="selected by the rule")
axR.annotate("highest recall\nin the grid", xy=(sel.recall, sel.f0_5),
             xytext=(sel.recall - 0.20, sel.f0_5 - 0.16), fontsize=7.5, color=ORANGE,
             arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
axR.set_xlim(0, 0.66)
axR.set_ylim(0.15, 0.98)
axR.set_xlabel("Calibration recall", fontsize=9)
axR.set_ylabel(r"Calibration $F_{0.5}$", fontsize=9)
axR.set_title(r"$F_{0.5}$ therefore varies with recall alone", fontsize=9.5)
axR.legend(fontsize=7, frameon=False, loc="lower right")
axR.grid(alpha=0.25, lw=0.5)
axR.tick_params(labelsize=8)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")

print(f"24 configurations: {int(a.clean.sum())} with zero false merges")
print(f"worst precision {a.precision.min():.3f}, largest cluster share "
      f"{a.largest_cluster_fraction.max():.1%}")
print(f"identity residual over the {len(clean)} clean configurations: {resid:.2e}")
print(f"written to {OUT.relative_to(PROJECT)}")
