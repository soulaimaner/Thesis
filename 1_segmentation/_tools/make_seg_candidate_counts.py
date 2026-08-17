"""Distribution of candidate crops per scan for the frozen extraction run.

Run from the root of this package:

    python 1_segmentation/_tools/make_seg_candidate_counts.py

**Why this exists.** Section 4.1 reports a corpus-wide median of 32 candidates per scan
against a mean of 65.9 and calls the distribution strongly right-skewed. Two summary
statistics state that a skew exists without showing its shape, and the shape is what
governs downstream review effort: a long tail means a minority of scans carries most of
the crops a human would have to look at. This figure shows the distribution the two
statistics summarise, and the concentration that follows from it.

The source is the per-scan table of the frozen run, so the counts here are the same ones
that produce Table 4.1; the script prints the reconciliation it performs.

Both panels are descriptive. Neither is an accuracy measurement: a candidate is a region
the extractor proposed, not a verified ornament.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(".").resolve()
assert (PROJECT / "book_identifiers.py").exists(), "run from the project root"

RUN = PROJECT / "1_segmentation_outputs" / "otsu_g3_area40_pad3_v1"
OUT = PROJECT / "report" / "figures" / "fig_seg_candidate_counts.pdf"

# Collection names as the thesis uses them, not the folder names of the run.
COLLECTION = {"original": "Principal", "suppl": "Supplementary"}
COLOUR = {"Principal": "#4878A6", "Supplementary": "#D9843B"}

img = pd.read_csv(RUN / "image_summary.csv")
img = img[img.status == "processed"].copy()
img["collection"] = img["source"].map(COLLECTION)
assert img.collection.notna().all(), "unexpected source folder"

n_scans = len(img)
total = int(img.n_candidates.sum())
median = float(img.n_candidates.median())
mean = float(img.n_candidates.mean())

# Reconcile against the frozen summary that Table 4.1 is drawn from.
summary = pd.read_csv(RUN / "dataset_summary.csv").set_index("source")
assert int(summary.loc["total", "candidates"]) == total, "per-scan counts disagree with the run summary"
assert int(summary.loc["total", "images_processed"]) == n_scans
assert float(summary.loc["total", "median_candidates_per_image"]) == median

# Concentration: share of all candidates carried by the densest tenth of scans.
ranked = np.sort(img.n_candidates.values)[::-1]
top_decile = int(np.ceil(0.10 * n_scans))
share_top10 = ranked[:top_decile].sum() / total
cum_share = np.cumsum(ranked) / total
scan_share = np.arange(1, n_scans + 1) / n_scans

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

bins = np.arange(0, 620, 20)
ax1.hist(
    [img.loc[img.collection == c, "n_candidates"] for c in ["Principal", "Supplementary"]],
    bins=bins,
    stacked=True,
    color=[COLOUR["Principal"], COLOUR["Supplementary"]],
    label=["Principal", "Supplementary"],
    edgecolor="white",
    linewidth=0.4,
)
ax1.axvline(median, color="black", linestyle="-", linewidth=1.1)
ax1.axvline(mean, color="black", linestyle="--", linewidth=1.1)
ax1.annotate(
    f"median {median:.0f}",
    xy=(median, 150), xytext=(median + 105, 168),
    fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.7),
)
ax1.annotate(
    f"mean {mean:.1f}",
    xy=(mean, 100), xytext=(mean + 135, 118),
    fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.7),
)
ax1.set_xlabel("Candidate crops per scan")
ax1.set_ylabel("Scans")
ax1.set_xlim(0, 620)
ax1.legend(frameon=False, fontsize=8)
ax1.set_title("(a) Distribution of candidates per scan", fontsize=9, loc="left")

ax2.plot(scan_share, cum_share, color="#333333", linewidth=1.4)
ax2.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1.0)
ax2.axvline(0.10, color="black", linestyle="--", linewidth=0.9)
ax2.annotate(
    f"densest 10% of scans\ncarry {share_top10 * 100:.0f}% of candidates",
    xy=(0.10, share_top10), xytext=(0.22, share_top10 - 0.22),
    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7),
)
ax2.set_xlabel("Scans, ranked by candidate count")
ax2.set_ylabel("Cumulative share of candidates")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.02)
ax2.set_title("(b) Concentration of the review load", fontsize=9, loc="left")

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")

print(f"scans {n_scans}, candidates {total}, median {median}, mean {mean:.1f}")
print(f"max {int(ranked[0])}, 90th pct {np.percentile(img.n_candidates, 90):.0f}, "
      f"share carried by densest {top_decile} scans {share_top10:.3f}")
print(f"scans with 0 candidates: {(img.n_candidates == 0).sum()}")
print(f"wrote {OUT}")
