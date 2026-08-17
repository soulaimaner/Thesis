"""Leave-one-work-out cross-validation for a single-target fleuron detector.

Run from the root of this package (the directory holding `book_identifiers.py`):

    python 4_detection/_tools/leave_one_work_out.py Fleuron_73

**Why this exists.** A single held-out split cannot validate a target whose impressions are
concentrated in one work. `Fleuron_73` carries 247 of its 366 impressions in `dicolosn73`, so
reserving a random 20% of works leaves a validation set of seven instances, while reserving
`dicolosn73` removes two thirds of the training data. Neither split measures anything. Under
leave-one-work-out, each work is held out in turn, every impression is predicted exactly once
by a model that never saw its work, and the predictions are pooled into one micro-averaged
figure over the whole class.

Pooled precision and recall are computed at IoU 0.50 by greedy confidence-ordered matching,
the same rule and the same threshold the candidate-extraction benchmark reports, so the two
stages can be read against each other directly.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

PROJECT_DIR = Path(".").resolve()
assert (PROJECT_DIR / "book_identifiers.py").exists(), (
    f"Run this script from the project root, not {PROJECT_DIR}.")
sys.path.insert(0, str(PROJECT_DIR))
from book_identifiers import build_work_map, volume_id  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("target")
parser.add_argument("--catalogue", default=None,
                    help="pre-detection catalogue root, relative to the project root. If omitted, "
                         "the preserved snapshot used by the reported target is selected.")
parser.add_argument("--tag", default="lowo")
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--iou-eval", type=float, default=0.50)
parser.add_argument("--model", default="yolo11n.pt",
                    help="detector weights to start from. The reported runs use yolo11n; a "
                         "larger model tests whether capacity, rather than the design's "
                         "concentration across works, is what limits transfer to an unseen book.")
parser.add_argument("--seed", type=int, default=42,
                    help="seeds both the negative-scan sample and training initialisation, so "
                         "repeating a run with a new seed measures the variance of the whole "
                         "procedure rather than of the optimiser alone.")
args = parser.parse_args()

TARGET = args.target
CORPUS_DIR = PROJECT_DIR.parent
IMAGE_DIRS = {"original": CORPUS_DIR / "Images", "suppl": CORPUS_DIR / "images suppl"}
MANIFEST_PATH = PROJECT_DIR / "all_regions_outputs" / "features" / "features_manifest.csv"
DEFAULT_CATALOGUES = {
    "Fleuron_2": "../curation_backups/2026-08-04_1644_pre_detection_foldin/Fleurons_v2_plus_retrieval",
    "Fleuron_74": "../curation_backups/2026-08-04_1644_pre_detection_foldin/Fleurons_v2_plus_retrieval",
    "Fleuron_73": "../curation_backups/2026-08-04_1735_pre_f72_f73_detection_foldin/Fleurons_v2_plus_retrieval",
    "Fleuron_72": "../curation_backups/2026-08-04_1735_pre_f72_f73_detection_foldin/Fleurons_v2_plus_retrieval",
}
catalogue_arg = args.catalogue or DEFAULT_CATALOGUES.get(TARGET)
if catalogue_arg is None:
    raise SystemExit("Pass --catalogue for a target without a recorded pre-detection snapshot.")
CATALOGUE_DIR = PROJECT_DIR / catalogue_arg
if not CATALOGUE_DIR.is_dir():
    raise SystemExit(f"Pre-detection catalogue is missing: {CATALOGUE_DIR}")
OUT = PROJECT_DIR / "4_detection_outputs" / f"{TARGET.lower()}_{args.tag}"
WORK_DIR = OUT / "_folds"
FOLD_OUT = OUT / "_fold_results"   # one CSV per completed fold, so a crash loses at most one
FOLD_OUT.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = args.seed
OUT.mkdir(parents=True, exist_ok=True)

# ---- catalogue and identity -------------------------------------------------
manifest = pd.read_csv(MANIFEST_PATH)
manifest["crop_name"] = manifest.crop_path.map(lambda p: Path(p).name)
catalogue = {}
for folder in CATALOGUE_DIR.iterdir():
    if folder.is_dir():
        for f in folder.iterdir():
            name = (Path(os.readlink(f)) if f.is_symlink() else f).name
            catalogue[name] = folder.name.split(" (")[0]
manifest["klass"] = manifest.crop_name.map(catalogue)
manifest["volume"] = manifest.rel_path.map(volume_id)
work_of, _ = build_work_map(sorted(manifest.volume.unique()))
manifest["work"] = manifest.volume.map(work_of)

target_rows = manifest[manifest.klass == TARGET]
target_scans = sorted(set(target_rows.stem))
work_of_scan = manifest.drop_duplicates("stem").set_index("stem").work.to_dict()
positive_works = sorted(set(target_rows.work))

# One seeded negative sample, shared by every fold, so folds differ only in which work is
# held out. `sorted` before shuffling: set iteration order over strings is not stable
# between processes under hash randomisation.
rng = np.random.RandomState(RANDOM_SEED)
negatives = sorted(set(manifest[manifest.klass.notna() & (manifest.klass != TARGET)].stem)
                   - set(target_scans))
rng.shuffle(negatives)
negatives = sorted(negatives[:len(target_scans)])

print(f"{TARGET}: {len(target_rows)} impressions, {len(target_scans)} scans, "
      f"{len(positive_works)} works, {len(negatives)} negative scans")
print("held out in turn:", ", ".join(positive_works))

geom = manifest.set_index(["stem"])  # for box lookup per scan


def write_scan(stem, split, root, positive):
    rows = manifest[manifest.stem == stem]
    src = IMAGE_DIRS[rows.source.iloc[0]] / rows.rel_path.iloc[0]
    if not src.exists():
        return 0, None
    img = Image.open(src)
    W, H = img.size
    img.convert("RGB").save(root / "images" / split / f"{stem}.png")
    lines, boxes = [], []
    if positive:
        for r in rows[rows.klass == TARGET].itertuples():
            lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
                (r.x_min + r.x_max) / 2 / W, (r.y_min + r.y_max) / 2 / H,
                (r.x_max - r.x_min) / W, (r.y_max - r.y_min) / H))
            boxes.append((r.x_min, r.y_min, r.x_max, r.y_max))
    (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
    return len(lines), boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


predictions, truths, fold_rows = [], [], []

for fold, held in enumerate(positive_works, 1):
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (WORK_DIR / d).mkdir(parents=True, exist_ok=True)

    pos_val = [s for s in target_scans if work_of_scan.get(s) == held]
    pos_train = [s for s in target_scans if work_of_scan.get(s) != held]
    neg_val = [s for s in negatives if work_of_scan.get(s) == held]
    neg_train = [s for s in negatives if work_of_scan.get(s) != held]

    n_train = sum(write_scan(s, "train", WORK_DIR, True)[0] for s in pos_train)
    for s in neg_train:
        write_scan(s, "train", WORK_DIR, False)
    n_val = 0
    for s in pos_val:
        k, boxes = write_scan(s, "val", WORK_DIR, True)
        n_val += k
        for b in (boxes or []):
            truths.append(dict(fold=held, stem=s, x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
    for s in neg_val:
        write_scan(s, "val", WORK_DIR, False)

    train_stems = {p.stem for p in (WORK_DIR / "images/train").glob("*.png")}
    val_stems = {p.stem for p in (WORK_DIR / "images/val").glob("*.png")}
    assert not (train_stems & val_stems), f"fold {held}: scan on both sides"

    (WORK_DIR / "data.yaml").write_text(
        f"path: {WORK_DIR}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['{TARGET}']\n")

    print(f"\n[{fold}/{len(positive_works)}] hold out {held}: "
          f"{n_train} train boxes / {n_val} val boxes over {len(pos_val)} scans", flush=True)

    run = f"{TARGET.lower()}_{args.tag}_{held}"
    fold_csv = FOLD_OUT / f"{held}.csv"
    weights = OUT / "runs" / run / "weights" / "best.pt"

    if fold_csv.exists():
        # Already scored by an earlier run. Its metadata is still recorded, otherwise a fully
        # resumed run would rebuild the pooled tables with an empty fold summary.
        try:
            n_done = len(pd.read_csv(fold_csv))
        except pd.errors.EmptyDataError:
            n_done = 0
        fold_rows.append(dict(work=held, train_boxes=n_train, val_boxes=n_val,
                              val_scans=len(pos_val), neg_val_scans=len(neg_val),
                              predictions=n_done))
        print(f"    already scored ({n_done} predictions), skipping", flush=True)
        continue
    if weights.exists():
        print(f"    reusing weights from an earlier run: {weights.name}", flush=True)
    else:
        model = YOLO(args.model)
        # No image caching. It was tried and measured: RAM caching added ~370 s of one-time
        # cost per fold and left the steady-state epoch time unchanged at ~5 s, because the
        # bottleneck is the per-epoch validation pass and dataloader overhead on a very small
        # dataset, not disk reads. With ~100 folds still to run it would have cost 10 hours.
        model.train(data=str(WORK_DIR / "data.yaml"), epochs=args.epochs, imgsz=640, batch=8,
                degrees=0, flipud=0.5, fliplr=0.5, scale=0.5, mosaic=1.0,
                project=str(OUT / "runs"), name=run, exist_ok=True,
                patience=25, seed=RANDOM_SEED, verbose=False, plots=False)

    det = YOLO(str(weights))
    n_pred = 0
    for s in pos_val + neg_val:
        rows = manifest[manifest.stem == s]
        path = IMAGE_DIRS[rows.source.iloc[0]] / rows.rel_path.iloc[0]
        if not path.exists():
            continue
        res = det.predict(str(path), conf=args.conf, verbose=False, device=0)[0]
        for box, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            predictions.append(dict(fold=held, stem=s, x1=float(box[0]), y1=float(box[1]),
                                    x2=float(box[2]), y2=float(box[3]), conf=float(cf)))
            n_pred += 1
    fold_rows.append(dict(work=held, train_boxes=n_train, val_boxes=n_val,
                          val_scans=len(pos_val), neg_val_scans=len(neg_val),
                          predictions=n_pred))
    # Explicit columns, so a fold that produced no prediction still writes a parseable file
    # rather than a zero-byte one the aggregation cannot read.
    PRED_COLS = ["fold", "stem", "x1", "y1", "x2", "y2", "conf"]
    TRUTH_COLS = ["fold", "stem", "x1", "y1", "x2", "y2"]
    pd.DataFrame([p for p in predictions if p["fold"] == held],
                 columns=PRED_COLS).to_csv(fold_csv, index=False)
    pd.DataFrame([t for t in truths if t["fold"] == held],
                 columns=TRUTH_COLS).to_csv(FOLD_OUT / f"{held}__truth.csv", index=False)
    print(f"    {n_pred} predictions, checkpointed", flush=True)

# Rebuild from the per-fold checkpoints rather than from memory, so a run resumed after a
# crash produces exactly the same pooled result as an uninterrupted one.
def _read_checkpoints(pattern, exclude_truth):
    frames = []
    for f in sorted(FOLD_OUT.glob(pattern)):
        if exclude_truth and f.name.endswith("__truth.csv"):
            continue
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue               # a fold that produced nothing, written before the
                                   # explicit-columns fix above; it contributes no rows
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

pred = _read_checkpoints("*.csv", exclude_truth=True)
truth = _read_checkpoints("*__truth.csv", exclude_truth=False)
pred.to_csv(OUT / "pooled_predictions.csv", index=False)
truth.to_csv(OUT / "pooled_ground_truth.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "folds.csv", index=False)


def pooled_scores(pred, truth, iou_thr, conf_min):
    """Greedy confidence-ordered matching, one prediction to at most one ground-truth box."""
    tp = fp = 0
    used = set()
    for r in pred[pred.conf >= conf_min].sort_values("conf", ascending=False).itertuples():
        cands = truth[truth.stem == r.stem]
        best, best_i = 0.0, None
        for t in cands.itertuples():
            if (t.stem, t.Index) in used:
                continue
            v = iou((r.x1, r.y1, r.x2, r.y2), (t.x1, t.y1, t.x2, t.y2))
            if v > best:
                best, best_i = v, t.Index
        if best >= iou_thr:
            tp += 1
            used.add((r.stem, best_i))
        else:
            fp += 1
    fn = len(truth) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f2 = (5 * p * rc / (4 * p + rc)) if (p + rc) else 0.0
    return dict(conf_min=conf_min, iou=iou_thr, tp=tp, fp=fp, fn=fn,
                precision=p, recall=rc, f2=f2)


sweep = [pooled_scores(pred, truth, args.iou_eval, c)
         for c in [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]]
sweep += [pooled_scores(pred, truth, 0.25, 0.25)]
summary = pd.DataFrame(sweep)
summary.to_csv(OUT / "pooled_summary.csv", index=False)

(OUT / "protocol.json").write_text(json.dumps(dict(
    target=TARGET, catalogue=str(catalogue_arg), folds=len(positive_works),
    model=args.model,
    works=positive_works, impressions=int(len(truth)), scans=len(target_scans),
    negative_scans=len(negatives), epochs=args.epochs, seed=RANDOM_SEED,
    conf_floor=args.conf, iou_eval=args.iou_eval), indent=1))

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)  # per-fold scratch; the folds are reproducible from protocol.json

print(f"\n{'='*60}\npooled over {len(positive_works)} folds, "
      f"{len(truth)} ground-truth impressions, {len(pred)} predictions")
print(summary.round(4).to_string(index=False))
print(f"written to {OUT}")
