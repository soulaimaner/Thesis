"""A training-free baseline for the detection stage: multi-scale template matching.

Run from the root of this package (the directory holding `book_identifiers.py`):

    python 4_detection/_tools/template_matching_baseline.py Fleuron_2 --catalogue <snapshot>

**Why this exists.** The chapter reports that a trained detector recovers impressions
connected-component extraction never produced a region for. It does not establish that
*training* was necessary. Normalised cross-correlation against a catalogue crop is the obvious
alternative: it needs no model, no labels beyond the exemplar, and no GPU, and it is the first
thing anyone would try. If it recovers the same impressions, the chapter's method choice is
unjustified; if it does not, the choice is measured rather than assumed.

**Design, and the two ways it is deliberately not handicapped.** The matcher is given the same
advantages the detector had. Impressions vary in size, so each template is swept over a range of
scales; the detector was trained with horizontal and vertical flips, so each template is matched
in its four flip variants too. The score threshold is swept rather than fixed, so the baseline is
read off its whole operating curve instead of one arbitrary point.

**Two ways it is deliberately handicapped, both matching the detector's conditions.** Templates
are drawn from the catalogue as it stood *before* the detector's recoveries were folded in, so
the impressions being searched for are not themselves in the template set. Templates from the
bibliographic work under search are excluded, preventing same-work near-duplicates.

Preprocessing follows the pipeline: images are binarised before matching, which is what the
descriptor stage does and what makes matching robust to inking and paper tone.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(".").resolve()
assert (PROJECT_DIR / "book_identifiers.py").exists(), "run from the project root"
import sys
sys.path.insert(0, str(PROJECT_DIR))
from book_identifiers import build_work_map, volume_id  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("target")
parser.add_argument("--catalogue", required=True,
                    help="pre-fold-in catalogue snapshot, so recoveries are not in the templates")
parser.add_argument("--templates", type=int, default=8)
parser.add_argument("--tag", default="tm")
args = parser.parse_args()

TARGET = args.target
CORPUS_DIR = PROJECT_DIR.parent
IMAGE_DIRS = {"original": CORPUS_DIR / "Images", "suppl": CORPUS_DIR / "images suppl"}
CATALOGUE_DIR = PROJECT_DIR / args.catalogue
OUT = PROJECT_DIR / "4_detection_outputs" / f"{TARGET.lower()}_{args.tag}"
OUT.mkdir(parents=True, exist_ok=True)

SCALES = [0.6, 0.75, 0.9, 1.0, 1.1, 1.3, 1.6]
FLIPS = [(False, False), (True, False), (False, True), (True, True)]
IOU_EVAL = 0.50
NMS_IOU = 0.30
SCORE_FLOOR = 0.30
PEAK_WIN = 9          # a correlation peak spans several pixels; keep only its summit
TOP_K_PER_CONFIG = 25  # per template/flip/scale, before non-maximum suppression

# ---- catalogue as it stood before the fold-in --------------------------------
manifest = pd.read_csv(PROJECT_DIR / "all_regions_outputs" / "features" / "features_manifest.csv")
manifest["crop_name"] = manifest.crop_path.map(lambda p: Path(p).name)
catalogue = {}
for folder in CATALOGUE_DIR.iterdir():
    if folder.is_dir():
        for f in folder.iterdir():
            catalogue[(Path(os.readlink(f)) if f.is_symlink() else f).name] = folder.name.split(" (")[0]
manifest["klass"] = manifest.crop_name.map(catalogue)
manifest["volume"] = manifest.rel_path.map(volume_id)
_work_of, _ = build_work_map(sorted(manifest.volume.unique()))
manifest["work"] = manifest.volume.map(_work_of)
work_of_scan = manifest.drop_duplicates("stem").set_index("stem").work.to_dict()
target_rows = manifest[manifest.klass == TARGET]

# ---- ground truth: catalogued impressions + the confirmed recoveries ---------
reviewed = pd.read_csv(PROJECT_DIR / "4_detection_outputs" / f"{TARGET.lower()}_v1" /
                       "review_new_detections" / "new_detections_reviewed.csv")
recov = reviewed[(reviewed.confirmed) & (reviewed.bucket != "other_catalogued")]

truth = pd.concat([
    target_rows.assign(kind="catalogued").rename(
        columns={"x_min": "x1", "y_min": "y1", "x_max": "x2", "y_max": "y2"}
    )[["stem", "rel_path", "source", "x1", "y1", "x2", "y2", "kind"]],
    recov.assign(kind="recovered")[["stem", "rel_path", "source", "x1", "y1", "x2", "y2", "kind"]],
], ignore_index=True)

scans = truth[["stem", "rel_path", "source"]].drop_duplicates("stem").reset_index(drop=True)
print(f"{TARGET}: {len(truth)} ground-truth impressions "
      f"({(truth.kind=='catalogued').sum()} catalogued, {(truth.kind=='recovered').sum()} "
      f"recovered by the detector) on {len(scans)} scans", flush=True)


def binarise(gray):
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


# ---- templates ---------------------------------------------------------------
members = sorted([p for p in (CATALOGUE_DIR).iterdir() if p.is_dir()
                  and p.name.split(" (")[0] == TARGET][0].iterdir())
idx = np.linspace(0, len(members) - 1, min(args.templates, len(members))).astype(int)
templates = []
for p in [members[i] for i in idx]:
    real = Path(os.readlink(p)) if p.is_symlink() else p
    im = cv2.imread(str(real), cv2.IMREAD_GRAYSCALE)
    if im is None or min(im.shape) < 8:
        continue
    src_stem = manifest.loc[manifest.crop_name == real.name, "stem"]
    st = src_stem.iloc[0] if len(src_stem) else None
    templates.append(dict(img=binarise(im), stem=st, work=work_of_scan.get(st)))
print(f"templates: {len(templates)} drawn from {CATALOGUE_DIR.name}", flush=True)


def nms(dets, thr=NMS_IOU):
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            ix1, iy1 = max(d["x1"], k["x1"]), max(d["y1"], k["y1"])
            ix2, iy2 = min(d["x2"], k["x2"]), min(d["y2"], k["y2"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            u = ((d["x2"]-d["x1"])*(d["y2"]-d["y1"]) +
                 (k["x2"]-k["x1"])*(k["y2"]-k["y1"]) - inter)
            if u > 0 and inter / u > thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


rows = []
for i, s in enumerate(scans.itertuples(), 1):
    path = IMAGE_DIRS[s.source] / s.rel_path
    if not path.exists():
        continue
    scan = binarise(np.array(Image.open(path).convert("L")))
    H, W = scan.shape
    found = []
    for t in templates:
        if t["work"] is not None and t["work"] == work_of_scan.get(s.stem):
            continue      # same work as the scan under search: the detector never got this
        for fx, fy in FLIPS:
            base = t["img"]
            if fx:
                base = base[:, ::-1]
            if fy:
                base = base[::-1, :]
            for sc in SCALES:
                th, tw = int(base.shape[0]*sc), int(base.shape[1]*sc)
                if th < 8 or tw < 8 or th >= H or tw >= W:
                    continue
                tmpl = cv2.resize(base, (tw, th), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(scan, tmpl, cv2.TM_CCOEFF_NORMED)
                # Local maxima only: a correlation peak is one detection, not the hundreds of
                # neighbouring pixels that also clear the threshold.
                peak = cv2.dilate(res, np.ones((PEAK_WIN, PEAK_WIN), np.uint8))
                ys, xs = np.where((res >= SCORE_FLOOR) & (res >= peak))
                if len(ys) > TOP_K_PER_CONFIG:
                    order = np.argsort(res[ys, xs])[::-1][:TOP_K_PER_CONFIG]
                    ys, xs = ys[order], xs[order]
                for y, x in zip(ys, xs):
                    found.append(dict(stem=s.stem, x1=float(x), y1=float(y),
                                      x2=float(x+tw), y2=float(y+th),
                                      score=float(res[y, x])))
    for d in nms(found):
        rows.append(d)
    if i % 10 == 0 or i == len(scans):
        print(f"  {i}/{len(scans)} scans, {len(rows)} detections so far", flush=True)

pred = pd.DataFrame(rows)
pred.to_csv(OUT / "template_detections.csv", index=False)
truth.to_csv(OUT / "ground_truth.csv", index=False)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / u if u > 0 else 0.0


def sweep(kind=None):
    t = truth if kind is None else truth[truth.kind == kind]
    out = []
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        p = pred[pred.score >= thr] if len(pred) else pred
        tp = fp = 0
        used = set()
        by_scan = {k: g for k, g in t.groupby("stem")} if len(t) else {}
        for r in (p.sort_values("score", ascending=False).itertuples() if len(p) else []):
            best, bi = 0.0, None
            for g in by_scan.get(r.stem, pd.DataFrame()).itertuples():
                if g.Index in used:
                    continue
                v = iou((r.x1, r.y1, r.x2, r.y2), (g.x1, g.y1, g.x2, g.y2))
                if v > best:
                    best, bi = v, g.Index
            if best >= IOU_EVAL:
                tp += 1
                used.add(bi)
            else:
                fp += 1
        fn = len(t) - tp
        out.append(dict(threshold=thr, tp=tp, fp=fp, fn=fn,
                        precision=tp/(tp+fp) if tp+fp else 0.0,
                        recall=tp/(tp+fn) if tp+fn else 0.0))
    return pd.DataFrame(out)


overall, recovered_only = sweep(), sweep("recovered")
overall.to_csv(OUT / "sweep_all.csv", index=False)
recovered_only.to_csv(OUT / "sweep_recovered_only.csv", index=False)
(OUT / "protocol.json").write_text(json.dumps(dict(
    target=TARGET, catalogue=str(args.catalogue), templates=len(templates),
    scales=SCALES, flips=len(FLIPS), score_floor=SCORE_FLOOR, nms_iou=NMS_IOU,
    iou_eval=IOU_EVAL, scans=len(scans), truth=len(truth)), indent=1))

print(f"\nALL {len(truth)} ground-truth impressions:")
print(overall.round(3).to_string(index=False))
print(f"\nRECALL ONLY, over the {int((truth.kind=='recovered').sum())} impressions the detector "
      f"recovered (the ones the chapter claims need a detector).")
print("Precision is omitted here: against this subset alone, every correct detection of a "
      "catalogued\nimpression would count as a false positive. Read precision from the table "
      "above.")
print(recovered_only[["threshold", "tp", "fn", "recall"]].round(3).to_string(index=False))
print(f"\nwritten to {OUT.relative_to(PROJECT_DIR)}")
