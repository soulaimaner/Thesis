"""Export a blind re-review sample, so the chapter's confirmations carry a reliability estimate.

Run from the root of this package (the directory holding `book_identifiers.py`):

    python 4_detection/_tools/build_reliability_sample.py

**Why this exists.** 702 confirmed recoveries carry this chapter, and they rest on one reviewer
judging crops whose filenames encoded the detector's confidence, in folders already sorted by
attribution bucket. Neither the reliability of those judgements nor their independence from the
detector's own score has been measured. Chapter 2 quantifies reviewer disagreement on a
250-pair benchmark; this stage has no equivalent.

**What it does.** Draws a stratified sample of previously reviewed detections, re-cuts each crop
from its source scan, and writes it under a neutral serial name in randomised order. The score,
the rank, the attribution bucket and the original verdict are all absent from what the reviewer
sees. The target is not hidden, since a reviewer cannot judge "is this Fleuron_2" without being
told which design is meant; each target folder therefore carries a reference sheet of known
impressions drawn from the catalogue.

**How to review.** Same workflow as the original pass: open a target folder in a file manager,
compare each crop against `_reference_<target>.png`, and delete the crops that are *not* that
design. Keep the ones that are. Do not open `key.csv` until you are finished, and do not consult
the original review folders.

Nothing here is written back to any existing review directory.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(".").resolve()
assert (PROJECT_DIR / "book_identifiers.py").exists(), "run from the project root"
CORPUS_DIR = PROJECT_DIR.parent
IMAGE_DIRS = {"original": CORPUS_DIR / "Images", "suppl": CORPUS_DIR / "images suppl"}
CATALOGUE_DIR = PROJECT_DIR / "Fleurons" / "Fleurons_v2_plus_retrieval"
OUT = PROJECT_DIR / "4_detection_outputs" / "reliability_check"

TARGETS = [2, 74, 73, 72]
PER_TARGET_CONFIRMED = 13     # stratified so both verdicts are represented well enough
PER_TARGET_REJECTED = 12      # to estimate agreement within each stratum separately
SEED = 42
PAD = 6                       # px of context, matching the original export

if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(
        f"ABORT: {OUT} already exists and is not empty. It may hold a review in progress, "
        f"which cannot be regenerated. Move it aside before rebuilding.")

rng = np.random.RandomState(SEED)
rows = []
for t in TARGETS:
    src = (PROJECT_DIR / "4_detection_outputs" / f"fleuron_{t}_v1" /
           "review_new_detections" / "new_detections_reviewed.csv")
    d = pd.read_csv(src)
    d["target"] = f"Fleuron_{t}"
    for verdict, n in [(True, PER_TARGET_CONFIRMED), (False, PER_TARGET_REJECTED)]:
        pool = d[d.confirmed == verdict]
        take = min(n, len(pool))
        if take:
            rows.append(pool.sample(take, random_state=rng).assign(
                stratum=("confirmed" if verdict else "rejected"),
                stratum_size=len(pool), stratum_drawn=take))
sample = pd.concat(rows, ignore_index=True)

# One shuffle over the whole sample, so serial numbers carry no information about target,
# verdict, bucket or confidence.
sample = sample.sample(frac=1.0, random_state=rng).reset_index(drop=True)
sample["serial"] = [f"{i+1:03d}" for i in range(len(sample))]

# ---- reference sheets, six catalogue impressions per target ------------------
catalogue = {}
for folder in CATALOGUE_DIR.iterdir():
    if folder.is_dir():
        for f in folder.iterdir():
            catalogue.setdefault(folder.name.split(" (")[0], []).append(f)

for t in TARGETS:
    name = f"Fleuron_{t}"
    (OUT / name).mkdir(parents=True, exist_ok=True)
    members = sorted(catalogue.get(name, []))[:60]
    picks = [members[i] for i in np.linspace(0, len(members)-1, min(6, len(members))).astype(int)]
    ims = [Image.open(p).convert("L") for p in picks]
    if not ims:
        continue
    h = 120
    ims = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in ims]
    sheet = Image.new("L", (sum(i.width + 8 for i in ims) + 8, h + 16), 255)
    x = 8
    for im in ims:
        sheet.paste(im, (x, 8))
        x += im.width + 8
    sheet.save(OUT / name / f"_reference_{name}.png")

# ---- the crops ---------------------------------------------------------------
written = 0
for r in sample.itertuples():
    path = IMAGE_DIRS[r.source] / r.rel_path
    if not path.exists():
        continue
    img = Image.open(path).convert("L")
    crop = img.crop((max(0, int(r.x1) - PAD), max(0, int(r.y1) - PAD),
                     min(img.size[0], int(r.x2) + PAD), min(img.size[1], int(r.y2) + PAD)))
    crop.save(OUT / r.target / f"{r.serial}.png")
    written += 1

key = sample[["serial", "target", "stem", "conf", "bucket", "confirmed",
              "stratum", "stratum_size", "stratum_drawn"]].sort_values("serial")
key.to_csv(OUT / "key.csv", index=False)
(OUT / "protocol.json").write_text(json.dumps(dict(
    seed=SEED, per_target_confirmed=PER_TARGET_CONFIRMED,
    per_target_rejected=PER_TARGET_REJECTED, drawn=int(len(sample)), written=int(written),
    padding_px=PAD, blind_to=["confidence", "rank", "attribution bucket", "original verdict"],
    not_blind_to=["which target the crop is claimed to be"]), indent=1))

(OUT / "HOW_TO_REVIEW.txt").write_text(
    "Blind re-review, for a reliability estimate on the chapter's confirmations.\n\n"
    "In each target folder, compare every numbered crop against _reference_<target>.png and\n"
    "DELETE the crops that are NOT that design. Keep the ones that are.\n\n"
    "The crops carry no score, no rank, no bucket and no record of the earlier verdict, and\n"
    "the numbering is shuffled across all four targets. Please do not open key.csv, and do not\n"
    "look at the original review folders, until you have finished.\n\n"
    "When done, tell me and I will score the agreement.\n")

print(f"wrote {written} crops across {sample.target.nunique()} targets to "
      f"{OUT.relative_to(PROJECT_DIR)}")
print(sample.groupby(["target", "stratum"]).size().unstack(fill_value=0).to_string())
print(f"\nreview instructions: {(OUT / 'HOW_TO_REVIEW.txt').relative_to(PROJECT_DIR)}")
