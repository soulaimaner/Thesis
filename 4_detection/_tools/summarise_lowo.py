"""Pool the leave-one-work-out runs across targets into the tables the chapter reports.

Run from the root of this package (the directory holding `book_identifiers.py`):

    python 4_detection/_tools/summarise_lowo.py

Reads `4_detection_outputs/<target>_lowo/` for every target that has one and writes
`4_detection_outputs/lowo_summary/`. Scores are recomputed here from the stored predictions
and ground truth rather than read from each run's `pooled_summary.csv`, so the tables in the
chapter and the per-run artifacts cannot drift apart.

Matching is greedy and confidence-ordered at IoU 0.50, one prediction to at most one
ground-truth box. Uncertainty is estimated by resampling held-out bibliographic works,
because impressions from the same work are not independent.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(".").resolve()
assert (PROJECT_DIR / "book_identifiers.py").exists(), "run from the project root"
OUT = PROJECT_DIR / "4_detection_outputs" / "lowo_summary"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = ["Fleuron_2", "Fleuron_74", "Fleuron_73", "Fleuron_72"]
IOU_EVAL = 0.50
CONF_MIN = 0.25
BOOT_REPS = 10_000
BOOT_SEED = 20_260_806


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def score(pred, truth, iou_thr=IOU_EVAL, conf_min=CONF_MIN):
    tp = fp = 0
    used = set()
    by_scan = {s: g for s, g in truth.groupby("stem")} if len(truth) else {}
    for r in pred[pred.conf >= conf_min].sort_values("conf", ascending=False).itertuples():
        best, best_i = 0.0, None
        for t in by_scan.get(r.stem, pd.DataFrame()).itertuples():
            if t.Index in used:
                continue
            v = iou((r.x1, r.y1, r.x2, r.y2), (t.x1, t.y1, t.x2, t.y2))
            if v > best:
                best, best_i = v, t.Index
        if best >= iou_thr:
            tp += 1
            used.add(best_i)
        else:
            fp += 1
    fn = len(truth) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f2 = (5*p*r_ / (4*p + r_)) if (p + r_) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r_, f2=f2)


def work_cluster_intervals(rows, seed=BOOT_SEED, reps=BOOT_REPS):
    """Percentile intervals after resampling bibliographic works with replacement."""
    by_work = rows.groupby("work")[["tp", "fp", "fn"]].sum().sort_index()
    counts = by_work.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(counts), size=(reps, len(counts)))
    sampled = counts[indices].sum(axis=1)
    tp, fp, fn = sampled.T
    precision = np.divide(tp, tp + fp, out=np.full(reps, np.nan), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.full(reps, np.nan), where=(tp + fn) > 0)
    per_work_recall = np.divide(counts[:, 0], counts[:, 0] + counts[:, 2],
                                out=np.full(len(counts), np.nan),
                                where=(counts[:, 0] + counts[:, 2]) > 0)
    work_recall = np.nanmean(per_work_recall[indices], axis=1)
    return {
        "precision_lo": float(np.nanquantile(precision, 0.025)),
        "precision_hi": float(np.nanquantile(precision, 0.975)),
        "recall_lo": float(np.nanquantile(recall, 0.025)),
        "recall_hi": float(np.nanquantile(recall, 0.975)),
        "work_recall_lo": float(np.nanquantile(work_recall, 0.025)),
        "work_recall_hi": float(np.nanquantile(work_recall, 0.975)),
        "work_clusters": int(len(by_work)),
    }


pooled_rows, fold_rows = [], []
all_pred, all_truth = [], []

for target in TARGETS:
    base = PROJECT_DIR / "4_detection_outputs" / f"{target.lower()}_lowo"
    if not (base / "pooled_predictions.csv").exists():
        print(f"skipping {target}: no leave-one-work-out run found")
        continue
    pred = pd.read_csv(base / "pooled_predictions.csv")
    truth = pd.read_csv(base / "pooled_ground_truth.csv")
    folds = pd.read_csv(base / "folds.csv")
    proto = json.loads((base / "protocol.json").read_text())

    s = score(pred, truth)
    pooled_rows.append(dict(target=target, works=proto["folds"], scans=proto["scans"],
                            impressions=len(truth), predictions=len(pred), **s))
    all_pred.append(pred.assign(target=target))
    all_truth.append(truth.assign(target=target))

    for w in sorted(truth.fold.unique()):
        fs = score(pred[pred.fold == w], truth[truth.fold == w])
        meta = folds[folds.work == w].iloc[0]
        fold_rows.append(dict(target=target, work=w, impressions=fs["tp"] + fs["fn"],
                              tp=fs["tp"], fp=fs["fp"], fn=fs["fn"],
                              scans=int(meta.val_scans),
                              impressions_per_scan=round((fs["tp"]+fs["fn"])/max(1, meta.val_scans), 2),
                              train_boxes=int(meta.train_boxes),
                              train_share=round(meta.train_boxes/(meta.train_boxes+meta.val_boxes), 3),
                              precision=fs["precision"], recall=fs["recall"]))

pooled = pd.DataFrame(pooled_rows)
per_fold = pd.DataFrame(fold_rows)
for i, target in enumerate(pooled.target):
    ci = work_cluster_intervals(per_fold[per_fold.target == target], seed=BOOT_SEED + i + 1)
    for key, value in ci.items():
        pooled.loc[pooled.target == target, key] = value
pooled.to_csv(OUT / "pooled_by_target.csv", index=False)
per_fold.to_csv(OUT / "per_fold.csv", index=False)

print("=" * 78)
print("POOLED, leave-one-work-out, IoU 0.50, conf >= 0.25")
print(pooled[["target", "works", "impressions", "predictions", "tp", "fp", "fn",
              "precision", "recall", "f2"]].round(3).to_string(index=False))

if len(all_truth):
    P = pd.concat(all_pred, ignore_index=True)
    T = pd.concat(all_truth, ignore_index=True)
    T["key"] = T.target + "|" + T.stem
    P["key"] = P.target + "|" + P.stem
    overall = score(P.rename(columns={"key": "stem", "stem": "_scan"}),
                    T.rename(columns={"key": "stem", "stem": "_scan"}))
    overall_ci = work_cluster_intervals(per_fold)
    print(f"\nall targets pooled: {len(T)} impressions, "
          f"P {overall['precision']:.3f} [{overall_ci['precision_lo']:.3f}, "
          f"{overall_ci['precision_hi']:.3f}], "
          f"R {overall['recall']:.3f} [{overall_ci['recall_lo']:.3f}, "
          f"{overall_ci['recall_hi']:.3f}], "
          f"F2 {overall['f2']:.3f}")
    (OUT / "pooled_all_targets.json").write_text(json.dumps(
        {**{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in overall.items()}, **overall_ci, "impressions": int(len(T)),
          "interval_method": "percentile bootstrap over bibliographic works",
          "bootstrap_replicates": BOOT_REPS, "bootstrap_seed": BOOT_SEED}, indent=1))

# ---- the density relationship, measured across every fold of every target ----
# Micro-averaged within each group (pooled TP over pooled TP+FN), not a mean of per-fold
# recalls: a fold holding one impression would otherwise weigh as much as one holding 247,
# and no fold has to be excluded for being too small to average.
print("\n" + "=" * 78)
print("RECALL AGAINST HOW DENSELY THE FLEURON IS SET, one row per held-out work")
d = per_fold.copy()
d["group"] = np.where(d.impressions_per_scan >= 5, "densely set (>=5/scan)",
                      "sparsely set (<5/scan)")
grp = d.groupby("group").agg(folds=("work", "size"), impressions=("impressions", "sum"),
                             tp=("tp", "sum"), fn=("fn", "sum"), fp=("fp", "sum"))
grp["recall"] = grp.tp / (grp.tp + grp.fn)
grp["precision"] = grp.tp / (grp.tp + grp.fp)
print(grp.round(3).to_string())
grp.to_csv(OUT / "recall_by_density.csv")

if len(d) > 3:
    print(f"\nSpearman recall vs impressions per scan: "
          f"{d.recall.corr(d.impressions_per_scan, method='spearman'):+.3f}  (n={len(d)} folds)")
    print(f"Spearman recall vs training share:       "
          f"{d.recall.corr(d.train_share, method='spearman'):+.3f}")
    print("Density and training share are correlated with each other "
          f"({d.impressions_per_scan.corr(d.train_share, method='spearman'):+.3f}): the works "
          "holding many impressions are the works whose removal costs the most training data, "
          "so within one target the two cannot be separated.")

# ---- what the detector missed --------------------------------------------------
# Every ground-truth impression is matched once, then the found and the missed are compared on
# properties of the impression itself. If the two are indistinguishable, the failure is a
# property of the book rather than of the impression, which is what the fold recalls suggest.
print("\n" + "=" * 78)
print("WHAT THE DETECTOR MISSED, found against missed impressions")
miss_rows = []
corpus_dir = PROJECT_DIR.parent
image_dirs = {"original": corpus_dir / "Images", "suppl": corpus_dir / "images suppl"}
manifest = pd.read_csv(PROJECT_DIR / "all_regions_outputs" / "features" /
                       "features_manifest.csv")
scan_size = {}
for row in manifest.drop_duplicates("stem").itertuples():
    path = image_dirs[row.source] / row.rel_path
    if path.exists():
        with Image.open(path) as image:
            scan_size[row.stem] = image.size
for target in TARGETS:
    base = PROJECT_DIR / "4_detection_outputs" / f"{target.lower()}_lowo"
    if not (base / "pooled_predictions.csv").exists():
        continue
    pred = pd.read_csv(base / "pooled_predictions.csv")
    truth = pd.read_csv(base / "pooled_ground_truth.csv").reset_index(drop=True)
    used, found = set(), set()
    by_scan = {s: g for s, g in truth.groupby("stem")}
    for r in pred.sort_values("conf", ascending=False).itertuples():
        best, bi = 0.0, None
        for t in by_scan.get(r.stem, pd.DataFrame()).itertuples():
            if t.Index in used:
                continue
            v = iou((r.x1, r.y1, r.x2, r.y2), (t.x1, t.y1, t.x2, t.y2))
            if v > best:
                best, bi = v, t.Index
        if best >= IOU_EVAL:
            used.add(bi)
            found.add(bi)
    truth["target"] = target
    truth["found"] = truth.index.isin(found)
    truth["min_side"] = (truth[["x2", "y2"]].to_numpy() - truth[["x1", "y1"]].to_numpy()).min(axis=1)
    truth["area"] = (truth.x2 - truth.x1) * (truth.y2 - truth.y1)
    truth["per_scan"] = truth.groupby("stem").stem.transform("size")
    truth["scan_w"] = truth.stem.map(lambda stem: scan_size.get(stem, (np.nan, np.nan))[0])
    truth["scan_h"] = truth.stem.map(lambda stem: scan_size.get(stem, (np.nan, np.nan))[1])
    truth["clipped"] = ((truth.x1 <= 1) | (truth.y1 <= 1)
                        | (truth.x2 >= truth.scan_w - 1)
                        | (truth.y2 >= truth.scan_h - 1))
    miss_rows.append(truth)

if miss_rows:
    M = pd.concat(miss_rows, ignore_index=True)
    M.to_csv(OUT / "impression_level_outcomes.csv", index=False)
    comp = M.groupby(["target", "found"]).agg(
        n=("found", "size"), median_min_side=("min_side", "median"),
        median_area=("area", "median"), median_per_scan=("per_scan", "median"),
        clipped=("clipped", "sum")).round(0)
    print(comp.to_string())
    size_rows = []
    for target, group in M.groupby("target"):
        group = group.copy()
        group["size_q"] = pd.qcut(
            group.min_side, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"],
            duplicates="drop")
        q = group.groupby("size_q", observed=True).agg(
            n=("found", "size"), found=("found", "sum")).reset_index()
        q["target"] = target
        q["miss_rate"] = (1 - q.found / q.n).round(3)
        size_rows.append(q)
    size_effect = pd.concat(size_rows, ignore_index=True)
    size_effect.to_csv(OUT / "miss_rate_by_within_target_size_quartile.csv", index=False)
    print("\nmiss rate by size quartile of the impression (within each target):")
    print(size_effect[["target", "size_q", "n", "miss_rate"]].to_string(index=False))
    w = M.groupby(["target", "fold"]).agg(n=("found", "size"), found=("found", "sum"))
    w["miss_rate"] = (1 - w.found / w.n).round(3)
    w.to_csv(OUT / "miss_rate_by_work.csv")
    print(f"\nmiss rate varies from {w.miss_rate.min():.3f} to {w.miss_rate.max():.3f} "
          f"across the {len(w)} held-out works")

print("\nfolds retaining >= 85% of the training boxes, so training volume is not the constraint:")
rich = d[d.train_share >= 0.85].sort_values("impressions_per_scan", ascending=False)
print(rich[["target", "work", "impressions", "impressions_per_scan", "train_share",
            "recall"]].round(3).to_string(index=False))
if len(rich):
    rtp, rfn = rich[rich.impressions_per_scan >= 5][["tp", "fn"]].sum()
    stp, sfn = rich[rich.impressions_per_scan < 5][["tp", "fn"]].sum()
    if rtp + rfn and stp + sfn:
        print(f"  within these training-rich folds only: dense recall "
              f"{rtp/(rtp+rfn):.3f} ({rtp+rfn} impressions) against sparse "
              f"{stp/(stp+sfn):.3f} ({stp+sfn} impressions)")

per_fold.to_csv(OUT / "per_fold.csv", index=False)
print(f"\nwritten to {OUT.relative_to(PROJECT_DIR)}")
