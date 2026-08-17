"""Train one single-target fleuron detector, end to end, for one catalogue class.

Run from the root of this package (the directory holding `book_identifiers.py`), which is where every path below resolves from:

    python 4_detection/_tools/train_single_fleuron_detector.py Fleuron_73

The train/validation split is made over **works**, not scans and not digitised volumes,
using the corpus-wide rule in `book_identifiers.py`. Volumes of one multi-volume set were
printed in the same shop from the same ornament stock, so holding out volume 3 while
training on volume 4 does not test generalisation to an unseen book.
"""

import argparse
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
parser.add_argument("target", help='catalogue class, e.g. "Fleuron_73"')
parser.add_argument("--catalogue", default=None,
                    help="pre-detection catalogue root, relative to the project root. If omitted, "
                         "the preserved snapshot used by the reported target is selected.")
parser.add_argument("--dataset-tag", default="v1",
                    help="suffix for the dataset and run directories. Use a fresh tag rather "
                         "than overwriting a dataset whose results are already reported.")
parser.add_argument("--infer", action="store_true",
                    help="after training, run the detector over the corpus and export review "
                         "crops. OFF by default: re-running it against an existing, "
                         "hand-reviewed directory would desynchronise the frozen review labels.")
args = parser.parse_args()

TARGET = args.target
CORPUS_DIR = PROJECT_DIR.parent
IMAGE_DIRS = {"original": CORPUS_DIR / "Images", "suppl": CORPUS_DIR / "images suppl"}

# The CURRENT manifest (all_regions_outputs/features/), not the notebook's configured
# feature_extraction_outputs/.../features_manifest.csv, which is a 2026-07-31 snapshot
# predating the retrieval gap-class fold-in, the Fleuron_38/17 split, and this chapter's own.
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

RUN_NAME = f"{TARGET.lower()}_{args.dataset_tag}"
OUTPUT_DIR = PROJECT_DIR / "4_detection_outputs" / RUN_NAME
REVIEW_DIR = OUTPUT_DIR / "review_new_detections"
RUNS_DIR = PROJECT_DIR / "4_detection_outputs" / "runs"

IOU_MATCH = 0.40
FRAGMENT_RATIO = 0.67
MERGE_RATIO = 1.50
CONF_MIN = 0.25
RANDOM_SEED = 42
rng = np.random.RandomState(RANDOM_SEED)

print(f"\n{'='*60}\n{TARGET}  [{RUN_NAME}]\n{'='*60}")
print(f"catalogue: {catalogue_arg}")

# The dataset directory is rebuilt from scratch on every run. Writing into a directory that
# already holds a previous split leaves that split's scans in place, because the writer skips
# files that exist; a scan the earlier split put in train/ then survives alongside the same
# scan in val/, and the validation figure is measured on training data. That is exactly what
# happened to Fleuron_73 on 2026-08-04, and is why this guard exists. The review directory is
# never touched: it holds human decisions that cannot be regenerated.
for split_dir in ["images/train", "images/val", "labels/train", "labels/val"]:
    target_dir = OUTPUT_DIR / split_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

# ---- 1. load manifest + catalogue class per crop ---------------------------
manifest = pd.read_csv(MANIFEST_PATH)
manifest["crop_name"] = manifest.crop_path.map(lambda p: Path(p).name)

catalogue = {}
for folder in CATALOGUE_DIR.iterdir():
    if folder.is_dir():
        for f in folder.iterdir():
            name = (Path(os.readlink(f)) if f.is_symlink() else f).name
            catalogue[name] = folder.name.split(" (")[0]

manifest["klass"] = manifest.crop_name.map(catalogue)

# Identity at two levels. `volume_id` is one physical volume; `work_id` merges the volumes of
# a multi-volume set. The split below is made at work level. The earlier rule extracted a book
# from the filename with `^([A-Za-z]+[0-9]*)`, which both split sibling volumes apart
# (`vaoe1`, `vaoe2`) and merged unrelated works together wherever the shelfmark carries a
# non-ASCII letter (`moméslsn50` and `moœurovi65` both truncating towards `mo`).
manifest["volume"] = manifest.rel_path.map(volume_id)
work_of, multivolume_sets = build_work_map(sorted(manifest.volume.unique()))
manifest["work"] = manifest.volume.map(work_of)

target_rows = manifest[manifest.klass == TARGET]
target_scans = set(target_rows.stem)
print(f"{TARGET}: {len(target_rows)} impressions on {len(target_scans)} scans, "
      f"{target_rows.volume.nunique()} volumes, {target_rows.work.nunique()} works")
print(f"box size median {target_rows.width.median():.0f}x{target_rows.height.median():.0f} px")
touched = {w: m for w, m in multivolume_sets.items() if w in set(target_rows.work)}
if touched:
    print("multi-volume sets carrying this target (held out or trained on as a unit):")
    for work, members in touched.items():
        print(f"  {work}: {', '.join(members)}")

# ---- 2. positive/negative scans, work-disjoint split ------------------------
other_scans_df = manifest[manifest.klass.notna() & (manifest.klass != TARGET)]
# `sorted`, not `list`: iteration order over a set of strings varies between processes under
# Python's default hash randomisation, so the seeded shuffle below did not in fact produce the
# same negative sample twice.
neg_scans = sorted(set(other_scans_df.stem) - target_scans)
rng.shuffle(neg_scans)
neg_scans = set(neg_scans[:len(target_scans)])

pos_works = sorted(manifest[manifest.stem.isin(target_scans)].work.dropna().unique())
neg_works = sorted(manifest[manifest.stem.isin(neg_scans)].work.dropna().unique())

# Reserve validation works separately from the positive pool and the negative pool, so
# validation gets a meaningful number of target scans regardless of how many negative-only
# works exist. Taking 20% of the combined pool left one 14-volume target with two validation
# scans, because negative scans are drawn from many single-scan sources that dominate the pool.
rng.shuffle(pos_works)
n_val_pos_works = max(1, round(len(pos_works) * 0.2))
val_works = set(pos_works[:n_val_pos_works])
train_works = set(pos_works[n_val_pos_works:])

neg_works = [w for w in neg_works if w not in val_works and w not in train_works]
rng.shuffle(neg_works)
n_val_neg_works = max(1, round(len(neg_works) * 0.2))
val_works |= set(neg_works[:n_val_neg_works])
train_works |= set(neg_works[n_val_neg_works:])

assert not (val_works & train_works), "work assigned to both sides of the split"

work_of_scan = manifest.drop_duplicates("stem").set_index("stem").work.to_dict()

def split_scans(scans):
    tr = [p for p in sorted(scans) if work_of_scan.get(p) in train_works]
    va = [p for p in sorted(scans) if work_of_scan.get(p) in val_works]
    return tr, va

pos_train, pos_val = split_scans(target_scans)
neg_train, neg_val = split_scans(neg_scans)
print(f"train: {len(pos_train)} positive + {len(neg_train)} negative scans")
print(f"val:   {len(pos_val)} positive + {len(neg_val)} negative scans "
      f"({len(val_works)} validation works)")

# The split is only as good as what actually reaches disk, so it is checked there, after the
# write, rather than trusted here.

# ---- 3. write YOLO dataset ---------------------------------------------------
def write_split(scans, split, is_positive):
    n_boxes = 0
    for stem in scans:
        rows = manifest[manifest.stem == stem]
        source = rows.source.iloc[0]
        rel_path = rows.rel_path.iloc[0]
        src_img = IMAGE_DIRS[source] / rel_path
        if not src_img.exists():
            continue
        img = Image.open(src_img)
        W, H = img.size
        dst_img = OUTPUT_DIR / "images" / split / f"{stem}.png"
        if not dst_img.exists():
            img.convert("RGB").save(dst_img)

        label_path = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
        lines = []
        if is_positive:
            hits = rows[rows.klass == TARGET]
            for r in hits.itertuples():
                xc = (r.x_min + r.x_max) / 2 / W
                yc = (r.y_min + r.y_max) / 2 / H
                w = (r.x_max - r.x_min) / W
                h = (r.y_max - r.y_min) / H
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                n_boxes += 1
        label_path.write_text("\n".join(lines))
    return n_boxes

n_train_boxes = write_split(pos_train, "train", True) + write_split(neg_train, "train", False)
n_val_boxes = write_split(pos_val, "val", True) + write_split(neg_val, "val", False)
print(f"boxes: {n_train_boxes} train, {n_val_boxes} val")

# ---- 3a. verify the split on disk, which is what the trainer actually reads ----
# The counts printed above are what this process wrote. The trainer reads the directory, so
# the directory is what has to be checked.
train_stems = {p.stem for p in (OUTPUT_DIR / "images/train").glob("*.png")}
val_stems = {p.stem for p in (OUTPUT_DIR / "images/val").glob("*.png")}
shared_scans = train_stems & val_stems
shared_works = {work_of_scan.get(s) for s in train_stems} & {work_of_scan.get(s) for s in val_stems}
disk_train_boxes = sum(len([ln for ln in p.read_text().splitlines() if ln.strip()])
                       for p in (OUTPUT_DIR / "labels/train").glob("*.txt"))
disk_val_boxes = sum(len([ln for ln in p.read_text().splitlines() if ln.strip()])
                     for p in (OUTPUT_DIR / "labels/val").glob("*.txt"))
if shared_scans:
    raise SystemExit(f"ABORT: {len(shared_scans)} scans in both splits: {sorted(shared_scans)}")
if shared_works - {None}:
    raise SystemExit(f"ABORT: works on both sides of the split: {sorted(shared_works - {None})}")
if (disk_train_boxes, disk_val_boxes) != (n_train_boxes, n_val_boxes):
    raise SystemExit(f"ABORT: wrote {n_train_boxes}/{n_val_boxes} boxes but the directory holds "
                     f"{disk_train_boxes}/{disk_val_boxes}; a previous split has survived.")
print(f"split verified on disk: {len(train_stems)} train / {len(val_stems)} val scans, "
      f"no shared scan, no shared work")

(OUTPUT_DIR / "data.yaml").write_text(
    f"path: {OUTPUT_DIR}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['{TARGET}']\n")

pd.DataFrame(
    [dict(stem=s, split="train", work=work_of_scan.get(s)) for s in sorted(train_stems)]
    + [dict(stem=s, split="val", work=work_of_scan.get(s)) for s in sorted(val_stems)]
).to_csv(OUTPUT_DIR / "split.csv", index=False)

# ---- 4. train (flips only, established best practice) ----------------------
model = YOLO("yolo11n.pt")
model.train(data=str(OUTPUT_DIR / "data.yaml"), epochs=80, imgsz=640, batch=8,
            degrees=0, flipud=0.5, fliplr=0.5, scale=0.5, mosaic=1.0,
            project=str(RUNS_DIR), name=RUN_NAME, exist_ok=True,
            patience=25, seed=RANDOM_SEED, verbose=False, plots=True)

WEIGHTS = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
print(f"weights: {WEIGHTS}  present: {WEIGHTS.exists()}")

metrics = YOLO(str(WEIGHTS)).val(data=str(OUTPUT_DIR / "data.yaml"), split="val", verbose=False,
                                 project=str(RUNS_DIR), name=f"{RUN_NAME}_val", exist_ok=True)
print(f"val: P={metrics.box.mp:.3f} R={metrics.box.mr:.3f} "
      f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")

if not args.infer:
    print("\ncorpus inference and review export skipped (pass --infer to run them).")
    print(f"{TARGET} DONE")
    sys.exit(0)

if REVIEW_DIR.exists():
    raise SystemExit(
        f"ABORT: {REVIEW_DIR} already exists. It holds human review decisions that cannot be "
        f"regenerated; re-exporting over them would desynchronise the frozen labels. Use a "
        f"fresh --dataset-tag.")

# ---- 5. detect over the whole corpus -----------------------------------------
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

det_model = YOLO(str(WEIGHTS))
scans = manifest[["rel_path", "source", "stem"]].drop_duplicates()
det_rows = []
for p in scans.itertuples():
    path = IMAGE_DIRS[p.source] / p.rel_path
    if not path.exists():
        continue
    res = det_model.predict(str(path), conf=CONF_MIN, verbose=False, device=0)[0]
    for box, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        det_rows.append(dict(stem=p.stem, rel_path=p.rel_path, source=p.source,
                              x1=box[0], y1=box[1], x2=box[2], y2=box[3], conf=float(conf)))
detections = pd.DataFrame(det_rows)
print(f"detections: {len(detections)} over {detections.stem.nunique()} scans")

def attribute(det):
    by_scan = {s: g for s, g in manifest.groupby("stem")}
    bucket = pd.Series(index=det.index, dtype=object)
    matched = pd.Series(index=det.index, dtype=object)
    ratio = pd.Series(index=det.index, dtype=float)
    for i, r in det.iterrows():
        cand = by_scan.get(r.stem)
        if cand is None or not len(cand):
            bucket[i] = "no_otsu_candidate"; continue
        boxes = cand[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
        klass = cand.klass.to_numpy()
        overlaps = np.array([iou([r.x1, r.y1, r.x2, r.y2], b) for b in boxes])
        if overlaps.max() < IOU_MATCH:
            bucket[i] = "no_otsu_candidate"; continue
        j = overlaps.argmax(); k = klass[j]; matched[i] = k
        box = boxes[j]
        det_area = (r.x2 - r.x1) * (r.y2 - r.y1)
        ratio[i] = ((box[2] - box[0]) * (box[3] - box[1]) / det_area) if det_area > 0 else np.nan
        if k == TARGET: bucket[i] = "known_target"
        elif pd.notna(k): bucket[i] = "other_catalogued"
        elif ratio[i] < FRAGMENT_RATIO: bucket[i] = "otsu_box_fragment"
        elif ratio[i] > MERGE_RATIO: bucket[i] = "otsu_box_merged"
        else: bucket[i] = "otsu_box_ok_uncurated"
    return det.assign(bucket=bucket, matched_class=matched, otsu_area_ratio=ratio)

detections = attribute(detections)
detections.to_csv(OUTPUT_DIR / "detections_flipsonly.csv", index=False)
print(detections.bucket.value_counts())

# ---- 6. export candidates for review ----------------------------------------
REVIEW_DIR.mkdir(exist_ok=True)
new_buckets = ["no_otsu_candidate", "otsu_box_fragment", "otsu_box_merged", "otsu_box_ok_uncurated"]
confusion_bucket = ["other_catalogued"]
to_review = detections[detections.bucket.isin(new_buckets + confusion_bucket)].copy()
to_review["band"] = pd.cut(to_review.conf, [0, 0.5, 0.75, 0.9, 1.01],
                            labels=["0.25-0.50", "0.50-0.75", "0.75-0.90", "0.90-1.00"],
                            right=False)
to_review = to_review.sort_values("conf", ascending=False).reset_index(drop=True)
to_review.to_csv(REVIEW_DIR / "new_detections.csv", index=False)

BUCKET_DIRS = {
    "no_otsu_candidate": "A_no_otsu_candidate",
    "otsu_box_fragment": "B_otsu_box_fragment",
    "otsu_box_merged": "C_otsu_box_merged",
    "otsu_box_ok_uncurated": "D_otsu_box_ok_uncurated",
    "other_catalogued": "E_matches_a_different_ornament",
}
for b, dirname in BUCKET_DIRS.items():
    (REVIEW_DIR / dirname).mkdir(exist_ok=True)

for idx, r in to_review.iterrows():
    path = IMAGE_DIRS[r.source] / r.rel_path
    img = Image.open(path).convert("RGB")
    crop = img.crop((int(r.x1), int(r.y1), int(r.x2), int(r.y2)))
    conf_tag = int(round(r.conf * 1000))
    fname = f"{conf_tag:04d}_{idx:04d}_{r.stem}.png"
    crop.save(REVIEW_DIR / BUCKET_DIRS[r.bucket] / fname)

print(f"exported {len(to_review)} candidates for review to {REVIEW_DIR}")
print(f"{TARGET} DONE")
