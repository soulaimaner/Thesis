"""Separate two failure modes the attribution buckets conflate, and price the size filter.

Run from the root of this package (the directory holding `book_identifiers.py`):

    python 4_detection/_tools/refine_attribution.py

**The problem this corrects.** Detections were attributed against `features_manifest.csv`, the
candidate set that was embedded. That set is what survived the `MIN_SIDE >= 24` filter: Otsu
proposed 30,804 regions and 9,054 were discarded before embedding. A detection landing on a
discarded region therefore scored as `no_otsu_candidate`, and was reported as an impression the
extraction stage never found. Those are two different failures with two different remedies, and
only one of them is the connected-component ceiling this chapter is about.

This pass re-tests every confirmed `no_otsu_candidate` recovery against the **unfiltered** Otsu
output and splits the bucket in two. It reads the frozen review tables and writes new files; it
never modifies them, and it retrains nothing.

It also prices the filter: for each candidate threshold, how many extra regions the corpus would
admit, and how many of those are known to be genuine impressions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(".").resolve()
assert (PROJECT_DIR / "book_identifiers.py").exists(), "run from the project root"
OUT = PROJECT_DIR / "4_detection_outputs" / "attribution_refined"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [2, 74, 73, 72]
IOU_MATCH = 0.40
MIN_SIDE_APPLIED = 24


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


full = pd.read_csv(PROJECT_DIR / "all_regions_outputs" / "candidate_boxes.csv")
full["min_side"] = full[["width", "height"]].min(axis=1)
dropped = full[full.min_side < MIN_SIDE_APPLIED]
print(f"Otsu proposed {len(full):,} regions; {len(dropped):,} fall below MIN_SIDE >= "
      f"{MIN_SIDE_APPLIED} and were never embedded.")

by_scan = {s: g[["x_min", "y_min", "x_max", "y_max", "min_side"]].to_numpy()
           for s, g in full.groupby("stem")}

rows = []
for t in TARGETS:
    src = (PROJECT_DIR / "4_detection_outputs" / f"fleuron_{t}_v1" /
           "review_new_detections" / "new_detections_reviewed.csv")
    d = pd.read_csv(src)
    d["target"] = f"Fleuron_{t}"
    for r in d.itertuples():
        best, side = 0.0, np.nan
        if r.bucket == "no_otsu_candidate":
            for c in by_scan.get(r.stem, []):
                v = iou((r.x1, r.y1, r.x2, r.y2), (c[0], c[1], c[2], c[3]))
                if v > best:
                    best, side = v, c[4]
        rows.append(dict(target=r.target, stem=r.stem, conf=r.conf, bucket=r.bucket,
                         confirmed=r.confirmed, unfiltered_iou=best,
                         unfiltered_min_side=side if best >= IOU_MATCH else np.nan))

R = pd.DataFrame(rows)
R["bucket_refined"] = R.bucket
mask = (R.bucket == "no_otsu_candidate") & (R.unfiltered_iou >= IOU_MATCH)
R.loc[mask, "bucket_refined"] = "otsu_below_min_side"
R.loc[(R.bucket == "no_otsu_candidate") & ~mask, "bucket_refined"] = "no_candidate_any_size"
R.to_csv(OUT / "recoveries_refined.csv", index=False)

conf = R[R.confirmed]
print("\n" + "=" * 74)
print("CONFIRMED RECOVERIES, original bucket against refined bucket")
tab = (conf.groupby(["bucket", "bucket_refined"]).size()
       .rename("confirmed").reset_index().sort_values("confirmed", ascending=False))
print(tab.to_string(index=False))

split = conf[conf.bucket == "no_otsu_candidate"].bucket_refined.value_counts()
n_none = int(split.get("no_candidate_any_size", 0))
n_small = int(split.get("otsu_below_min_side", 0))
print(f"\nThe headline bucket splits: {n_none} never extracted at any size, "
      f"{n_small} extracted but discarded by the size filter "
      f"({n_small/(n_none+n_small):.1%} of the {n_none+n_small} previously reported as one number).")
print("\nby target:")
print(conf[conf.bucket == "no_otsu_candidate"]
      .groupby(["target", "bucket_refined"]).size().unstack(fill_value=0).to_string())

small = conf[conf.bucket_refined == "otsu_below_min_side"]
if len(small):
    print(f"\nthe discarded regions behind those {len(small)} recoveries: min side "
          f"median {small.unfiltered_min_side.median():.0f} px, range "
          f"{small.unfiltered_min_side.min():.0f}-{small.unfiltered_min_side.max():.0f} px")

# ---- price the filter --------------------------------------------------------
# The cost of a lower threshold is exact and corpus-wide. The gain is known only for the four
# designs a detector was trained for, so it is a floor for those four and unknown for the rest.
print("\n" + "=" * 74)
print("WHAT A LOWER THRESHOLD WOULD COST AND RETURN")
price = []
for thr in [24, 22, 20, 18, 16, 14, 12]:
    admitted = int(((full.min_side >= thr) & (full.min_side < MIN_SIDE_APPLIED)).sum())
    recovered = int((small.unfiltered_min_side >= thr).sum()) if len(small) else 0
    price.append(dict(min_side=thr, extra_regions_admitted=admitted,
                      known_impressions_recovered=recovered,
                      regions_per_impression=(admitted/recovered) if recovered else np.nan))
price = pd.DataFrame(price)
print(price.round(1).to_string(index=False))
price.to_csv(OUT / "min_side_price.csv", index=False)

(OUT / "summary.json").write_text(json.dumps(dict(
    otsu_regions_total=int(len(full)), regions_below_filter=int(len(dropped)),
    min_side_applied=MIN_SIDE_APPLIED, iou_match=IOU_MATCH,
    confirmed_no_candidate_original=int(n_none + n_small),
    confirmed_never_extracted=n_none, confirmed_below_size_filter=n_small), indent=1))
print(f"\nwritten to {OUT.relative_to(PROJECT_DIR)}")
