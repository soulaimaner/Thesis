#!/usr/bin/env bash
# Refresh figures/ from the frozen run directory.
#
# The figures are the only copies in 2_clustering/. Everything else is referenced in
# place, so this script is what keeps the one exception honest: rerun it after any notebook that
# writes a figure, and the copies match their sources again. It fails loudly on a missing source
# rather than leaving a stale copy in place.
#
# Destination names match their source names, so a figure cited in the README can be traced back
# to the notebook that drew it. Override RUN_TAG to sync a different run:
#
#     RUN_TAG=_rerun2 ./sync_figures.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
RUN="$PROJECT/2_clustering_outputs/all_regions_v1_minside24_dinov2_vitb14_binarized_v1_clustering_benchmark_v1"
DEST="$HERE/figures"
RUN_TAG="${RUN_TAG-_rerun1}"

mkdir -p "$DEST"

# destination name <- source path, relative to the run directory
copy() {
  local name="$1" src="$RUN/$2"
  if [[ ! -f "$src" ]]; then
    echo "MISSING: $src" >&2
    return 1
  fi
  cp -f "$src" "$DEST/$name"
  echo "  $name"
}

echo "Syncing figures from $RUN (run tag '${RUN_TAG}')"

# 3_MethodSelection
#
# similarity_threshold_calibration.png is deliberately NOT synced from here.
# The notebook draws its own version of that plot, but the copy this chapter
# cites, and the copy the report uses, is the redrawn one written directly by
# _tools/make_threshold_calibration_figure.py. Syncing the notebook's version
# would silently replace the cited figure with a different image.
copy negatives_by_band.png                 "method_comparison${RUN_TAG}/figures/negatives_by_band.png"
copy method_calibration_comparison.png     "method_comparison${RUN_TAG}/figures/method_calibration_comparison.png"
copy selected_cluster_size_distribution.png "method_comparison${RUN_TAG}/figures/selected_cluster_size_distribution.png"
copy eligibility_gate_power.png            "method_comparison${RUN_TAG}/figures/eligibility_gate_power.png"

# 4_MethodComparison
copy precision_reach_tradeoff.png          "method_evaluation_v1${RUN_TAG}/figures/precision_reach_tradeoff.png"

# 5_ShortlistConstruction
copy stability_cutoff_sensitivity.png     "shortlist_v1${RUN_TAG}/stability_cutoff_sensitivity.png"

# 6_WithinClusterAudit
copy audit_error_structure.png             "audit_analysis_v1${RUN_TAG}/figures/audit_error_structure.png"

# 7_RepresentationCheck
copy representation_delta_auc.png          "representation_v1${RUN_TAG}/representation_delta_auc.png"

# _tools/audit_catalogue.py
copy catalogue_contributions.png           "catalogue_v1/catalogue_contributions.png"

echo "Done."
