"""Score the blind re-review against the original review, giving the chapter a reliability figure.

Run from the root of this package (the directory holding `book_identifiers.py`), after the sample in
`4_detection_outputs/reliability_check/` has been reviewed:

    python 4_detection/_tools/score_reliability.py

**What it measures.** The 702 confirmed recoveries this chapter reports rest on a single pass in
which the reviewer could see the detector's confidence in every filename and worked through
folders already sorted by attribution bucket. This compares a second, blind pass over a
stratified sample of the same detections against the first, and reports raw agreement, Cohen's
kappa, and agreement within each original verdict separately.

**Why the strata are reported separately.** The sample deliberately over-represents rejections
(12 drawn per target against 13 confirmations, from populations that are nothing like balanced),
so a single pooled agreement figure would describe the sample rather than the review. Each
stratum carries its sampling weight, and the population confirmation rate implied by the second
pass is reconstructed from those weights.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(".").resolve()
assert (PROJECT_DIR / "book_identifiers.py").exists(), "run from the project root"
BASE = PROJECT_DIR / "4_detection_outputs" / "reliability_check"
key = pd.read_csv(BASE / "key.csv")
key["serial"] = key.serial.astype(str).str.zfill(3)

# A crop that survived the second pass was judged to be the target; a deleted one was not.
key["rereview"] = [(BASE / r.target / f"{r.serial}.png").exists() for r in key.itertuples()]
key["agree"] = key.rereview == key.confirmed
key.to_csv(BASE / "rereview_scored.csv", index=False)

n = len(key)
agree = int(key.agree.sum())
print("=" * 74)
print(f"BLIND RE-REVIEW: {agree} of {n} judgements agree with the original pass "
      f"({agree/n:.1%})")

# Cohen's kappa on the sample as drawn.
a = key.confirmed.to_numpy().astype(bool)
b = key.rereview.to_numpy().astype(bool)
po = (a == b).mean()
pe = (a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean()))
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
print(f"Cohen's kappa on the sample as drawn: {kappa:.3f} "
      f"(observed {po:.3f}, chance {pe:.3f})")
print("The sample is stratified, not representative, so read the strata below rather than "
      "this single number.")

print("\n" + "=" * 74)
print("BY ORIGINAL VERDICT")
rows = []
for verdict, label in [(True, "originally confirmed"), (False, "originally rejected")]:
    s = key[key.confirmed == verdict]
    kept = int(s.rereview.sum())
    same = int(s.agree.sum())
    rows.append(dict(stratum=label, drawn=len(s), kept_now=kept, agreed=same,
                     agreement=same/len(s) if len(s) else np.nan))
strata = pd.DataFrame(rows)
print(strata.round(3).to_string(index=False))

flipped_out = key[(key.confirmed) & (~key.rereview)]
flipped_in = key[(~key.confirmed) & (key.rereview)]
print(f"\nreversals: {len(flipped_out)} previously confirmed now rejected, "
      f"{len(flipped_in)} previously rejected now confirmed")

print("\n" + "=" * 74)
print("BY TARGET")
t = key.groupby("target").agg(n=("agree", "size"), agreed=("agree", "sum"))
t["agreement"] = (t.agreed / t.n).round(3)
print(t.to_string())

print("\n" + "=" * 74)
print("BY ATTRIBUTION BUCKET")
bk = key.groupby("bucket").agg(n=("agree", "size"), agreed=("agree", "sum"))
bk["agreement"] = (bk.agreed / bk.n).round(3)
print(bk.sort_values("n", ascending=False).to_string())

# ---- reweight to the population ---------------------------------------------
# Each drawn detection stands for stratum_size / stratum_drawn of its stratum, so the second
# pass implies a confirmation rate over all 943 reviewed detections without re-reviewing them.
key["weight"] = key.stratum_size / key.stratum_drawn
pop = key.weight.sum()
implied = (key.weight * key.rereview).sum() / pop
original = (key.weight * key.confirmed).sum() / pop
print("\n" + "=" * 74)
print("IMPLIED POPULATION CONFIRMATION RATE")
print(f"weighted population represented: {pop:.0f} detections")
print(f"original pass:      {original:.3f}")
print(f"blind second pass:  {implied:.3f}")
print(f"difference:         {implied - original:+.3f}")

(BASE / "reliability_summary.json").write_text(json.dumps(dict(
    n=n, agreement=agree/n, kappa=float(kappa),
    agreement_confirmed=float(strata.loc[0, "agreement"]),
    agreement_rejected=float(strata.loc[1, "agreement"]),
    reversals_confirmed_to_rejected=int(len(flipped_out)),
    reversals_rejected_to_confirmed=int(len(flipped_in)),
    original_rate=float(original), rereview_rate=float(implied)), indent=1))
print(f"\nwritten to {(BASE / 'rereview_scored.csv').relative_to(PROJECT_DIR)}")
