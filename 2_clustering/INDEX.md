# Index: where everything this stage cites actually lives

Nothing here is a copy of code or data. This file points at the real artifacts, so there is only
ever one version of each.

**Run directory.** Every short path below is relative to

```
2_clustering_outputs/all_regions_v1_minside24_dinov2_vitb14_binarized_v1_clustering_benchmark_v1/
```

Three inputs sit outside it because later stages read the same stores: `feature_extraction_outputs/`
holds the embeddings, `all_regions_outputs/crops/` holds the crop images, and
`Fleurons/Fleurons_v2_plus_retrieval/` is the catalogue itself.

## Notebooks

Each asserts its working directory is `2_clustering`, so they run from here. Reading
order is execution order.

The README states what each notebook asks and answers. This table gives only where each one writes.

| Notebook | Reads | Writes to |
|---|---|---|
| `1_FeatureExtraction.ipynb` | `all_regions_outputs/` | `feature_extraction_outputs/` |
| `2_PairBenchmark.ipynb` | the embeddings | `pair_review/` |
| `3_MethodSelection.ipynb` | the embeddings, `pair_review/` | `method_comparison{RUN_TAG}/` |
| `4_MethodComparison.ipynb` | `method_comparison/`, `disagreement_v1/`, `pair_review/`, `robustness_v1/` | `method_evaluation_v1{RUN_TAG}/` |
| `5_ShortlistConstruction.ipynb` | `method_comparison/`, `robustness_v1/` | `shortlist_v1{RUN_TAG}/` |
| `6_WithinClusterAudit.ipynb` | `method_comparison/`, `within_cluster_audit_v1/` | `audit_analysis_v1{RUN_TAG}/` |
| `7_RepresentationCheck.ipynb` | the embeddings, `pair_review/`, `method_comparison/` | `representation_v1{RUN_TAG}/` |

Untagged directories are frozen inputs; `{RUN_TAG}` directories are where a re-run writes. README §10
explains why both exist.

## Scripts

In `_tools/`, run from this folder, except `make_selection_limit_figure.py`, which resolves its
paths from the working directory and is run from the package root as its docstring states. Each is a
one-off computation whose output was reviewed by hand.

| Script | Purpose | Writes to |
|---|---|---|
| `make_disagreement_pool.py` | Builds the 250-pair benchmark of notebook 4 and its blind sheets | `disagreement_v1/` |
| `make_disagreement_batches.py` | Renders the same pairs ten to a page for review | `disagreement_v1/disagreement_batches/` |
| `make_rereview_pool.py` | Draws the 60-pair blind re-review | `disagreement_v1/rereview_v1/` |
| `match_clusters_to_catalogue.py` | Ranks each shortlisted cluster against the curated fleurons | `catalogue_v1/` |
| `find_duplicate_classes.py` | Ranks every pair of catalogue classes, to surface duplicates | `catalogue_v1/` |
| `audit_catalogue.py` | Asserts the catalogue and reports which stage contributed each crop | `catalogue_v1/` |

Six additional scripts generate thesis-report tables and figures from frozen tables. None is a
`sync_figures.sh` target, because each writes its destination directly.

| Script | Writes | Runs from this package alone |
|---|---|---|
| `make_selection_limit_figure.py` | `report/figures/fig_clu_selection_limit.png` | yes |
| `make_precision_reach_figure.py` | `report/figures/fig_clu_tradeoff.png`, and a copy in `figures/` as `precision_reach_tradeoff_compact.png` | yes |
| `make_threshold_calibration_figure.py` | `figures/similarity_threshold_calibration.png`, copied to `report/figures/fig_clu_threshold.png` | yes |
| `make_confusable_examples_figure.py` | `report/figures/fig_clu_confusable.png` | **no**, it reads crop images from `all_regions_outputs/crops/`, which is not included |
| `make_catalogue_inventory.py` | `report/tab_catalogue_inventory.tex` | **no**, it reads the catalogue and the crop manifest |
| `make_catalogue_contact_sheet.py` | `report/figures/fig_catalogue_contact_sheet.png` | **no**, it reads the crop images |

The first three create `report/figures/` if it is absent; `make_threshold_calibration_figure.py`
copies into it only when it already exists. Because the weighting limitation stated in README §6.4
prevents a population recovery ranking, the report figure omits recovery and instead pairs
unilateral-merge precision with the observed number and median size of review clusters.

The pre-refactor curation notebook that these scripts replaced is not part of this package. It used
the old catalogue layout and is not executable; the current catalogue record is the three catalogue
scripts listed above plus README §8.

## Figures

`figures/` is the one exception to the rule above: it holds copies, refreshed from the run directory
by `sync_figures.sh`, which fails loudly on a missing source rather than leaving a stale copy. Names
match their sources. Five of them are cited in the README; the rest are kept because a notebook drew
them.

Two of the eleven files are written directly rather than synced, and `sync_figures.sh` deliberately
leaves both alone: `precision_reach_tradeoff_compact.png`, written by
`_tools/make_precision_reach_figure.py` for the report, and
`similarity_threshold_calibration.png`, written by `_tools/make_threshold_calibration_figure.py`.
Notebook 3 draws its own version of the latter into the run directory; the copy kept here and cited
in the report is the redrawn one, and syncing the notebook's version over it would substitute a
different image.

| Figure | Drawn by | README |
|---|---|---|
| `similarity_threshold_calibration.png` | `_tools/make_threshold_calibration_figure.py` | |
| `negatives_by_band.png` | `3_MethodSelection` | |
| `method_calibration_comparison.png` | `3_MethodSelection` | |
| `selected_cluster_size_distribution.png` | `3_MethodSelection` | |
| `eligibility_gate_power.png` | `3_MethodSelection` | Figure 2.1 |
| `precision_reach_tradeoff.png` | `4_MethodComparison` | Figure 2.2 |
| `stability_cutoff_sensitivity.png` | `5_ShortlistConstruction` | Figure 2.5 |
| `audit_error_structure.png` | `6_WithinClusterAudit` | Figure 2.3 |
| `representation_delta_auc.png` | `7_RepresentationCheck` | Figure 2.4 |
| `catalogue_contributions.png` | `_tools/audit_catalogue.py` | |

The figure set of the previous nine-notebook version of this stage is not part of this package. Its
sources remain in the run directory under `label_evaluation_v1/`, `robustness_v1/`,
`validation_results_v1/`, `catalogue_audit_v1/` and `within_cluster_audit_v1/`, so those figures can
be redrawn from the retained tables. Nothing in this chapter reads them.

## Human judgements

All produced by one reviewer, and read as frozen inputs. No notebook regenerates them.

| What | Path | Size |
|---|---|---|
| Pair benchmark | `pair_review/pair_review_labels.csv` | 180 pairs, 16 negatives, 11 of them in the calibration split |
| Disagreement benchmark | `disagreement_v1/disagreement_labels.csv` | 250 pairs, 35 negatives |
| Blind re-review of the above | `disagreement_v1/rereview_v1/rereview_labels.csv` | 60 pairs, 60/60 agreement |
| Within-cluster audit | `within_cluster_audit_v1/audit_pair_labels.csv` | 150 pairs |
| Blind re-review, 25 from the pair benchmark and 25 from the audit | `validation_v1/rereview_labels.csv` | 50 pairs; identity 36/36, all 9 disagreements on `non_fleuron` |
| Cluster review sheets | `method_comparison/selected_cluster_review.csv` | 20 clusters |

580 distinct pairs, 110 of them judged twice, 690 judgements in all. README §3.1 tabulates the same
sets with the section each is reported in.

## The catalogue

| What | Path | State |
|---|---|---|
| Hand curation, first clustering pass | `all_regions_outputs/Fleurons/` | untouched since 31 July |
| Frozen HDBSCAN clustering | `method_comparison/` | 140 clusters, 7,160 assigned crops |
| The shortlist reviewed | `robustness_v1/hdbscan_shortlist.csv` | 55 clusters, 5,017 crops |
| The catalogue | `Fleurons/Fleurons_v2_plus_retrieval/` | 93 fleurons, 8,552 crops |
| Per-crop provenance | `3_retrieval_outputs/occurrence_v1/catalogue_provenance.csv` | written by the retrieval stage |

**Counting hazard.** Folder names carry a stale crop count. Count links, never names.

**Reproducibility gap.** The merge into the catalogue was performed by moving symlinks by hand, so
`audit_catalogue.py` can verify the result but cannot reconstruct the steps.

## Superseded

The datable record behind the provenance claim of README §5 is `benchmark_protocol.json`, in the
run directory. It fixes the pair-sampling design, 180 pairs over six similarity bands with a 120/60
split at seed 42, and is dated ahead of both the completed labels and the frozen clustering. It does
not contain the numerical gates or tie-breakers, which README §5 reports as encoded rather than
pre-registered, and which §6.3 shows could not have changed the selected configuration in any case.

The written protocol that preceded it named no threshold, has been superseded by the README, and is
not part of this package. Neither is the rest of the development history: the previous nine-notebook
version, the pre-refactor curation notebook, the catalogue notebook the `_tools/` scripts replaced,
an earlier draft of this write-up, and the superseded figure sets. Nothing in this chapter reads any
of it, and no reported result depends on it.

## Where this stage lands in the thesis

In the thesis text: methodology in §3.4, results in §4.3, discussion in §5.3. The
results section carries subsections for each of notebooks 4 to 7 (`sec:res-clu-labels`,
`sec:res-clu-stability`, `sec:res-clu-audit`, `sec:res-clu-ablation`, `sec:res-clu-convergence`).
