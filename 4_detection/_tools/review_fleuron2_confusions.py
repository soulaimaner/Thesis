"""Export, and then score, the 15 `Fleuron_2` detections that fell on a different catalogued
fleuron and were never reviewed.

Run from the root of this package (the directory holding `book_identifiers.py`):

    python 4_detection/_tools/review_fleuron2_confusions.py          # export for review
    python 4_detection/_tools/review_fleuron2_confusions.py --score  # after reviewing

**Why this exists.** Every detection this chapter reports is either reproduced from the
catalogue, reviewed by hand, or a confusion whose verdict was recorded. `Fleuron_2` is the one
exception. Its corpus run produced 360 detections: 254 reproduced catalogued impressions of the
target and 91 were candidate recoveries that were reviewed, leaving 15 that landed on a region
the catalogue had already assigned to a different design. Those 15 were never exported. The
target was the chapter's first, and its export wrote only the two recovery folders; by the time
the other three targets were run the export also wrote `E_matches_a_different_ornament`, which
is where their 103, 4 and 1 confusions were judged.

**What the gap costs.** The chapter states that not one confusion was judged to be the target,
so the prior curation was correct in every disputed case. That is a claim about the curation,
and it currently rests on 108 of the 123 confusions the four targets actually produced, with the
remaining 15 absent from the denominator rather than resolved. Either verdict is worth having:
if the detector is wrong the claim simply becomes complete, and if the catalogue is wrong then a
detector trained on the catalogue has found a curation error, which is a stronger result than
the chapter claims anywhere.

**Blinding.** The reviewer is not shown the detector's confidence, the catalogue's competing
class, or the original order. Section 4.5 of the chapter README records that the first review
pass saw the confidence in every filename and reports its confirmation-by-confidence gradient as
review burden rather than as calibration evidence; there is no reason to repeat that here. What
the reviewer is told is which design is meant, since "is this `Fleuron_2`" cannot be answered
otherwise, and a reference sheet of catalogue impressions is written beside the crops.

**Nothing is written into `review_new_detections/`.** That directory holds the frozen 91-row
review the chapter's `Fleuron_2` figures rest on, and it cannot be regenerated.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(".").resolve()
assert (PROJECT_DIR / "book_identifiers.py").exists(), "run from the project root"
CORPUS_DIR = PROJECT_DIR / ".."
IMAGE_DIRS = {"original": CORPUS_DIR / "Images", "suppl": CORPUS_DIR / "images suppl"}
CATALOGUE_DIR = PROJECT_DIR / "Fleurons" / "Fleurons_v2_plus_retrieval"

TARGET = "Fleuron_2"
DETECTIONS = (PROJECT_DIR / "4_detection_outputs" / "fleuron_2_v1" /
              "detections_flipsonly.csv")
OUT = PROJECT_DIR / "4_detection_outputs" / "fleuron_2_confusion_review"
SEED = 42
PAD = 6          # px of context, matching the original export and the reliability sample

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--score", action="store_true",
                    help="score a completed review instead of exporting a new one")
args = parser.parse_args()


# ---- scoring -----------------------------------------------------------------
if args.score:
    key = pd.read_csv(OUT / "key.csv")
    key["serial"] = key.serial.astype(str).str.zfill(3)
    # A crop that survived the review was judged to be the target; a deleted one was not.
    key["confirmed"] = [(OUT / "crops" / f"{s}.png").exists() for s in key.serial]
    key.to_csv(OUT / "confusions_reviewed.csv", index=False)

    n, conf = len(key), int(key.confirmed.sum())
    print("=" * 74)
    print(f"{TARGET} confusions: {conf} of {n} judged to be the target")
    if conf == 0:
        print("\nAll 15 are genuine detector errors. The chapter's claim that no confusion was")
        print("judged to be the target now covers all 123 confusions rather than 108, and the")
        print("per-target arithmetic closes: 254 known + 91 reviewed + 15 confusions = 360.")
    else:
        print(f"\n{conf} detection(s) the catalogue had filed elsewhere are impressions of")
        print(f"{TARGET}. These are curation errors found by the detector, not detector errors.")
        print("The chapter's claim that prior curation was correct in every disputed case does")
        print("not hold as written and must be restated with this count.")
    print("\nby the class the catalogue had assigned:")
    print(key.groupby("catalogue_says").confirmed
             .agg(cases="size", judged_to_be_target="sum").to_string())
    print("\nagainst detection confidence (hidden from the reviewer):")
    print(key.sort_values("conf", ascending=False)
             [["serial", "conf", "catalogue_says", "confirmed"]]
             .to_string(index=False))

    (OUT / "summary.json").write_text(json.dumps(dict(
        target=TARGET, reviewed=n, judged_to_be_target=conf,
        by_catalogue_class={k: int(v) for k, v in
                            key.groupby("catalogue_says").confirmed.sum().items()}), indent=1))
    print(f"\nwritten to {(OUT / 'confusions_reviewed.csv').relative_to(PROJECT_DIR)}")
    raise SystemExit

# ---- export ------------------------------------------------------------------
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(
        f"ABORT: {OUT} already exists and is not empty. It may hold a review in progress, "
        f"which cannot be regenerated. Move it aside before rebuilding, then re-run.")

det = pd.read_csv(DETECTIONS)
conf = det[det.bucket == "other_catalogued"].copy()
assert len(conf), "no other_catalogued detections found; has the detection CSV changed?"

# One shuffle, so the serial numbers carry no information about confidence or catalogue class.
rng = np.random.RandomState(SEED)
conf = conf.sample(frac=1.0, random_state=rng).reset_index(drop=True)
conf["serial"] = [f"{i + 1:03d}" for i in range(len(conf))]

(OUT / "crops").mkdir(parents=True)

# ---- reference sheet, six catalogue impressions of the target ----------------
members = sorted([p for p in CATALOGUE_DIR.iterdir()
                  if p.is_dir() and p.name.split(" (")[0] == TARGET][0].iterdir())
picks = [members[i] for i in np.linspace(0, len(members) - 1, min(6, len(members))).astype(int)]
ims = []
for p in picks:
    real = Path(os.readlink(p)) if p.is_symlink() else p
    im = Image.open(real).convert("L")
    h = 120
    ims.append(im.resize((max(1, int(im.width * h / im.height)), h)))
sheet = Image.new("L", (sum(i.width + 8 for i in ims) + 8, 136), 255)
x = 8
for im in ims:
    sheet.paste(im, (x, 8))
    x += im.width + 8
sheet.save(OUT / f"_reference_{TARGET}.png")

# ---- the crops ---------------------------------------------------------------
written = 0
for r in conf.itertuples():
    path = IMAGE_DIRS[r.source] / r.rel_path
    if not path.exists():
        print(f"  missing scan for {r.stem}, skipped")
        continue
    img = Image.open(path).convert("L")
    img.crop((max(0, int(r.x1) - PAD), max(0, int(r.y1) - PAD),
              min(img.size[0], int(r.x2) + PAD),
              min(img.size[1], int(r.y2) + PAD))).save(OUT / "crops" / f"{r.serial}.png")
    written += 1

conf.rename(columns={"matched_class": "catalogue_says"})[
    ["serial", "stem", "conf", "catalogue_says", "x1", "y1", "x2", "y2"]
].sort_values("serial").to_csv(OUT / "key.csv", index=False)

(OUT / "protocol.json").write_text(json.dumps(dict(
    target=TARGET, source=str(DETECTIONS.relative_to(PROJECT_DIR)),
    bucket="other_catalogued", proposed=int(len(conf)), written=int(written),
    seed=SEED, padding_px=PAD,
    blind_to=["detection confidence", "the class the catalogue assigned", "detection order"],
    not_blind_to=["which design the detection is claimed to be"]), indent=1))

(OUT / "HOW_TO_REVIEW.txt").write_text(
    f"{written} detections the {TARGET} detector made on regions the catalogue had already\n"
    f"assigned to a different design. They were never reviewed. Every other target's\n"
    f"confusions were.\n\n"
    f"ONE QUESTION PER CROP: is this an impression of {TARGET}?\n\n"
    f"Open crops/ in a file manager, compare each crop against _reference_{TARGET}.png, and\n"
    f"DELETE the crops that are NOT {TARGET}. Keep only the ones that are.\n\n"
    f"Expect to delete most or all of them. Each of these is a disagreement between the\n"
    f"detector and the curation, and the detector is usually the one that is wrong. A crop you\n"
    f"keep is the other case: an impression that was filed under the wrong design.\n\n"
    f"The crops carry no score, no catalogue label and no ordering. Please do not open key.csv\n"
    f"until you have finished.\n\n"
    f"When done:\n"
    f"    python 4_detection/_tools/review_fleuron2_confusions.py --score\n")

print(f"wrote {written} crops to {(OUT / 'crops').relative_to(PROJECT_DIR)}")
print(f"reference sheet: _reference_{TARGET}.png")
print(f"\ninstructions: {(OUT / 'HOW_TO_REVIEW.txt').relative_to(PROJECT_DIR)}")
print(f"when finished: python 4_detection/_tools/review_fleuron2_confusions.py --score")
