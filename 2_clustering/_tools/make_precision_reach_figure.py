"""Build the precision--review-burden figure used in Results Section 4.3.2.

The disagreement sample supports comparison of unilateral-merge precision, whereas its
same-design recovery ordering depends on weighting.  The second panel therefore shows
the observable corpus output that motivated retention: the number and median size of
clusters a reviewer would receive.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "2_clustering_outputs"
    / "all_regions_v1_minside24_dinov2_vitb14_binarized_v1_clustering_benchmark_v1"
    / "method_evaluation_v1_rerun1"
)
OUTPUTS = [
    ROOT / "2_clustering" / "figures" / "precision_reach_tradeoff_compact.png",
    ROOT / "report" / "figures" / "fig_clu_tradeoff.png",
]

unilateral = pd.read_csv(RESULTS / "unilateral_error_rates.csv").set_index("method")
fragmentation = pd.read_csv(RESULTS / "fragmentation.csv").set_index("method")

styles = {
    "HDBSCAN": {"marker": "o", "color": "#9c4a2f"},
    "mutual-kNN": {"marker": "s", "color": "#7a7a7a"},
    "CC @ 0.95": {"marker": "^", "color": "#3d3d3d"},
}
methods = list(styles)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
figure, (precision_axis, burden_axis) = plt.subplots(1, 2, figsize=(7.4, 2.75))

for x, (method, style) in enumerate(styles.items()):
    error = unilateral.loc[method]
    precision = 1.0 - error.error_rate
    lower = 1.0 - error.ci95_upper
    upper = 1.0 - error.ci95_lower
    precision_axis.errorbar(
        x,
        precision,
        yerr=[[precision - lower], [upper - precision]],
        marker=style["marker"],
        markersize=3.6,
        color=style["color"],
        elinewidth=0.8,
        capsize=2,
        linestyle="none",
        zorder=3,
    )

precision_axis.set(
    ylabel="Precision on unilateral merges",
    xlim=(-0.45, 2.45),
    ylim=(0.40, 1.00),
)
precision_axis.set_xticks(range(len(methods)), methods)

cluster_counts = fragmentation.loc[methods, "clusters"].astype(int)
median_sizes = fragmentation.loc[methods, "median_cluster_size"].astype(int)
bars = burden_axis.bar(range(len(methods)), cluster_counts, width=0.58,
                       color=[styles[method]["color"] for method in methods], linewidth=0)
for bar, count, median in zip(bars, cluster_counts, median_sizes):
    burden_axis.text(bar.get_x() + bar.get_width() / 2, count + 38,
                     f"{count:,}\nmedian {median}", ha="center", va="bottom", fontsize=7)
burden_axis.set(ylabel="Clusters returned for review", ylim=(0, 1320))
burden_axis.set_xticks(range(len(methods)), methods)

for axis in (precision_axis, burden_axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)

figure.tight_layout(pad=0.3)

for output in OUTPUTS:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
