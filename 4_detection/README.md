# Chapter 4: Detection of Unextracted Impressions

This is the fourth of four stages in the ornament-identification pipeline:

```
   Candidate extraction  →  find candidate regions on each ornament scan
   Clustering            →  discover which candidates are the same fleuron
   Retrieval             →  given a verified fleuron, find more of its impressions
→  Detection             →  find impressions that were never extracted at all
```

**Where we are and where the pipeline ends:**

| Step | Plain-language answer |
|---|---|
| **Input** | The catalogue frozen before each target's detections were folded in, plus all 613 scans |
| **This chapter** | For four case-study identities, locate impressions without depending on an extracted crop |
| **Output** | 702 confirmed recoveries: 644 attributable to extraction/size failures and 58 to adequate but uncurated regions |
| **Central finding** | 365 confirmed impressions have no overlapping unfiltered extraction box at IoU 0.40 |
| **End point** | Confirmed detections enter the catalogue; occurrence tables then recount the completed evidence |

Terminology (*composite ornament*, *fleuron*, *scan*, *candidate*, *crop*, *impression*) is
defined once in the [root README](../README.md) §2 and used identically across all four stages.
Chapter 1 measured the limit that makes this chapter necessary: candidate extraction recalls
0.746 of hand-drawn glyph boxes, and the reason is structural rather than a matter of tuning.
Chapter 3 could not address it, because retrieval ranks crops that already exist. This chapter
asks what a method that does not depend on ink connectivity can recover.

**How the chapter is organised.** Section 1 states the question and what the stage inherits.
Sections 2 and 3 give the design and the protocol, including one setting that was wrong and the
correction it forced. Section 4 reports the corpus run and the human review that is the
chapter's result, together with a correction to how that result was attributed. Section 5
withdraws the chapter's original validation figures and reports what replaced them. Section 6
asks whether a trained detector was necessary at all. Sections 7 to 11 qualify, discuss and
reproduce.

| Notebook | Question | Section |
|---|---|---|
| `1_SingleFleuronDetector` | How is a single-target detector built, and what does it find? | §2–§4 |
| `2_FourTargets` | What did four detectors recover, and what did review confirm? | §4 |
| `3_Validation` | How well does it work on a work it has never seen? | §5 |
| `4_Baselines` | Was training necessary, against a method that needs none? | §6 |

Bibliographic identity follows the corpus-wide rule in `book_identifiers.py` at the project
root. The distinction between a digitised volume and a bibliographic work is not incidental
here: §5.1 reports a defect that arises directly from conflating them.

## 1. Task and the Argument for the Stage

The question this chapter answers is:

> Can a detector trained on verified impressions of a fleuron recover further impressions that
> the extraction stage never produced a region for?

Chapter 1 measured candidate-extraction recall at 0.746 at IoU 0.50 and named the mechanism:
where a fleuron's ink touches neighbouring ink, connected-component labelling merges the two
into one region and the fleuron never receives its own box. That is a property of the method
family. No threshold recovers a region that was never separated.

Chapter 3 cannot reach these impressions either, and the reason is worth stating precisely.
Retrieval scores crops against a catalogue. If no crop was ever cut from the scan, there is
nothing to score. The two chapters address different kinds of miss, and conflating them would
misstate what either recall figure means.

| Kind of miss | Where it is | Remedy |
|---|---|---|
| The crop exists but nothing grouped it | among the unassigned crops | retrieval, Chapter 3 |
| The fleuron was never cut from the scan | not in the candidate set at all | detection, this chapter |

A detector does not threshold pixels. It learns appearance, so connected ink is not an obstacle.

## 2. Design: One Fleuron at a Time

**A general all-fleuron detector is not trainable from this catalogue.** Over the 514 scans carrying
at least one catalogued crop, the catalogue covers a median of 39% of the candidate regions on a
scan, and rendering its labels back onto scans confirms many genuine fleurons carry none. A detector
trained on such scans is taught that most fleurons are background. Only 28 scans reach 90% label
coverage, too few to train on. Both figures are computed against the pre-detection snapshot of §3;
the final catalogue, larger by this chapter's own recoveries, raises them to 42% and 33 scans, which
does not change the conclusion.

**Restricting the target to one fleuron removes the problem rather than mitigating it.** Every
*other* fleuron on a training scan is then a correct negative rather than a missing label: for the
first target, the scans carrying it hold 845 other catalogued fleurons the model should reject. The
only remaining gap is uncatalogued impressions of the target itself, which is precisely what the
experiment sets out to measure. This is the positive-unlabelled setting in its mildest form (Bekker
& Davis, 2020), and confining it to one class is what makes it tractable.

Negative scans are scans holding catalogued fleurons but no instance of the target, so the model
must discriminate between designs rather than merely locate fleuron-like shapes. §4 shows why
that distinction matters.

## 3. Protocol

| Setting | Value |
|---|---|
| Model | YOLO11n (Redmon et al., 2016, and successors), 640 px, 80 epochs, early stopping patience 25 |
| Augmentation | `flipud=0.5`, `fliplr=0.5`, `scale=0.5`, **`degrees=0`** |
| Matching | a detection is credited to a candidate region above IoU 0.40 |
| Attribution | candidate area / detection area, thresholds 0.67 and 1.5 |
| Operating point | detections retained above confidence 0.25 |
| Review | every proposal judged by hand; the detector proposes, the reviewer disposes |

**The catalogue input is frozen before each detection fold-in.** Detection results were added to
the live catalogue after review, so rebuilding training data from that live end-state would leak
the answers back into the experiment. The four reported targets use two preserved snapshots:

| Targets | Pre-fold-in catalogue snapshot | Verified target impressions |
|---|---|---:|
| `Fleuron_2`, `Fleuron_74` | `../curation_backups/2026-08-04_1644_pre_detection_foldin/Fleurons_v2_plus_retrieval/` | 265, 98 |
| `Fleuron_73`, `Fleuron_72` | `../curation_backups/2026-08-04_1735_pre_f72_f73_detection_foldin/Fleurons_v2_plus_retrieval/` | 366, 678 |

The targets were run sequentially, which is why two snapshots are required. Stored protocol files
record the same paths and counts; the current live catalogue is an output, not a valid training
input for reproducing these experiments.

Those two paths are reproduced exactly as the `catalogue` field of each `protocol.json` records
them, and are relative to the working project root the runs were launched from, one level above the
corpus scans. **Neither snapshot is included in this package**, for the same reason the live
catalogue is not (root README §6): both are directories of symlinks into the crop store. They are
listed here because they identify which catalogue state each detector was trained against, which is
what makes the run auditable; reproducing a training run requires obtaining them separately.

**Four targets, chosen sequentially, and reported as case studies rather than as a sample.**
`Fleuron_2` came first for its distinctive outline. `Fleuron_74` followed as a control, to test
whether the result depended on that distinctiveness. `Fleuron_73` was chosen because it was the
design `Fleuron_74` kept confusing with its target, to test whether the confusion ran both ways.
`Fleuron_72` was added as a second large, well-attested class. Each was selected in response to
what the previous one measured, which is a defensible way to proceed and is not a sampling rule.
Nothing in this chapter supports a claim about the other 88 catalogue classes, and none is made.

| Target | Impressions | Scans | Works | Training boxes | Role |
|---|---:|---:|---:|---:|---|
| `Fleuron_2` | 265 | 78 | 17 | 186 | first target |
| `Fleuron_74` | 98 | 33 | 12 | 82 | control |
| `Fleuron_73` | 366 | 40 | 14 | 364† | reciprocal test |
| `Fleuron_72` | 678 | 36 | 9 | 539 | second large class |

† Actual boxes read from the on-disk corpus-run training directory. A stale-directory defect
duplicated 76 `Fleuron_73` labels across the original split; §5.1 withdraws that split's metrics.

### 3.1 Rotation augmentation was harmful, and it inflated the first result

The first run used `degrees=180`, on the reasoning that the same block appears at several
orientations and at scales from 33 to 129 px. That reasoning is wrong for an axis-aligned
detector, and the error is recorded because its effect on the result was large.

When the framework rotates a training image by an arbitrary angle it recomputes the axis-aligned
box enclosing the rotated object. That box is always larger, and closer to square, than the
original, so the model is trained on inflated targets and learns to reproduce them. Flips do not
have this property: they map an axis-aligned box onto one of identical dimensions, so mirrored
impressions are learned without any loss of box tightness.

Changing **only** `degrees` from 180 to 0:

| | Run 1 (`degrees=180`) | Run 2 (flips only) |
|---|---:|---:|
| Precision | 0.686 | **0.879** |
| Recall | 0.544 | **0.917** |
| mAP50 | 0.633 | **0.935** |
| mAP50-95 | 0.405 | **0.806** |
| Near-square detections (AR 0.9–1.1) | 80% | 45% |
| Confusions with other fleurons | 70 | 15 |

mAP50-95 is the metric most sensitive to box tightness. Its larger change, together with the
reduction in near-square detections, is consistent with the predicted localisation damage. The
defective split means this contrast diagnoses the setting rather than estimating generalisation.

**The correction changed the finding, not only the metrics.** An inflated box fails the IoU 0.40
test against the correct candidate region, so a correctly detected, already-extracted impression
was filed as one extraction had never found. Run 1's headline of 162 such impressions was
substantially an artefact of its own boxes. Run 1 is reported here rather than discarded because
a reader encountering only the second run would have no way to see that the first result had
been inflated by the analyst's own setting.

*The two columns above are internally comparable, since both runs were scored the same way. The
absolute values are superseded by §5, which shows the split that produced them did not hold out
what it claimed to.*

## 4. The Corpus Run

Each detector was applied to all 613 scans and every detection compared with the candidate
regions extraction had produced for that scan.

| Target | Detections | Reproduced known | Reviewed | Recoveries confirmed | Confusions |
|---|---:|---:|---:|---|---:|
| `Fleuron_2` | 360 | 254 | 106 | 54 / 91 = 0.593 | 15 |
| `Fleuron_74` | 291 | 97 | 194 | 33 / 91 = 0.363 | 103 |
| `Fleuron_73` | 501 | 364 | 137 | 121 / 133 = 0.910 | 4 |
| `Fleuron_72` | 1,194 | 673 | 521 | 494 / 520 = 0.950 | 1 |

**958 detections reviewed by hand, 702 recoveries confirmed.** Every one of the 2,346 detections
the four detectors produced is either reproduced from the catalogue, reviewed as a recovery, or
reviewed as a confusion; the notebook asserts this rather than leaving it to be checked.

Recoveries and confusions are counted on separate denominators. A detection landing on a
different catalogued fleuron is an error of identity, not a candidate recovery, and folding the
two together would misstate both. **Not one of the 123 confusions was judged to be the target**,
so the prior curation was correct in every disputed case.

`Fleuron_2`'s 15 confusions were judged last, and until then were absent from that claim rather
than resolved by it. It was the chapter's first target and its export wrote only the two recovery
folders; the folder for confusions was added before the other three targets ran, so theirs were
reviewed and its were not. The chapter reported 108 confusions, which was 108 of 123. They have
since been reviewed under the same single question, with the detector's score and the
catalogue's competing class hidden, and none is an impression of the target.

### 4.1 Attribution requires box geometry, not merely presence

A candidate region may exist at the location and still be unusable: the connectivity failure that
merges a fleuron with adjacent ink can equally truncate it into a fragment. Both produce a region a
curator would reject, and treating either as "extracted" would attribute an extraction failure to
clustering. Detections are therefore sub-classified by the ratio of candidate area to detection
area, under two thresholds declared by inspection of the geometry and not tuned against the review
outcome. These are diagnostic rules rather than reviewer-adjudicated causes: "fragment" and "merged"
follow from the area ratio, and "no region" means no unfiltered Stage 1 box overlaps at IoU 0.40.

### 4.2 One bucket was measuring two different failures

The bucket recording "no candidate region" was originally read as "extraction never found this".
That is not what it measured. Detections were matched against the candidate set that reached the
embedding stage, and a size filter sits between the two: candidates whose shorter side falls
below 24 px are discarded before any crop is embedded, removing **9,054 of the 30,804** regions
extraction proposed corpus-wide. A detection landing on a discarded region recorded no candidate
at all, indistinguishably from one landing where extraction genuinely produced nothing.

Re-testing every confirmed recovery in that bucket against the **unfiltered** extraction output
separates them:

| What extraction had produced | Proposed | Confirmed | Precision |
|---|---:|---:|---:|
| No region at any size | 401 | **365** | 0.910 |
| A region, discarded by the size filter | 75 | **71** | 0.947 |
| A region that was a fragment | 52 | 44 | 0.846 |
| A region merged with a neighbour | 185 | 164 | 0.886 |
| An adequate region, never curated | 122 | 58 | 0.475 |
| **Total** | **835** | **702** | **0.841** |

**The chapter's central claim is 365, not 436.** The 71 are not a connected-component failure
and are not reported as one: extraction found them and a later decision discarded them. They
concentrate in two targets, `Fleuron_72` (62) and `Fleuron_2` (9), and the regions behind them
have a shorter side of 17 to 23 px.

**644 of the 702 confirmed recoveries were lost before any crop reached the catalogue**: 573 to
extraction, which produced no region, a fragment, or a merge, and 71 to the size filter that
followed it. The remaining 58 had an adequate region that was never carried into the catalogue.
Counting by presence of a region alone would have credited 122 of them to clustering.

### 4.3 What the size filter costs, priced rather than asserted

The second group has a remedy the first does not: lower the threshold. That remedy can be priced
exactly, because the cost is a corpus-wide count.

| `MIN_SIDE` | Extra regions admitted | Known impressions recovered | Regions per impression |
|---:|---:|---:|---:|
| 24 (applied) | 0 | 0 | |
| 20 | 3,527 | 60 | 59 |
| 18 | 5,656 | 69 | 82 |
| 16 | 7,554 | 71 | 106 |

Recovering all 71 means admitting 7,554 further regions, roughly 106 for each impression gained,
against the one confirmed recovery per 3.4 crops inspected that Chapter 3 achieves on ranked
retrieval. **The filter is well placed and these 71 impressions are its measured cost.** The
gain is a floor, since it counts only impressions of the four designs a detector was trained
for; the cost is exact and corpus-wide, and it would take an implausible yield among the
remaining 88 designs to reverse the conclusion.

The threshold was not changed. It is applied upstream of the embedding, the clustering, the
catalogue and Chapter 3's occurrence tables, so altering it invalidates all of them, and the 71
impressions are not lost in any case: the detector recovered them and they sit in the catalogue.
What was wrong was the attribution, and correcting that cost one measurement rather than four
re-run chapters.

### 4.4 The review is repeatable

The 702 confirmations carry this chapter and rested on one reviewer working through folders
sorted by attribution bucket, with the detector's confidence visible in every filename. Neither
the reliability of those judgements nor their independence from the score had been measured.

A target-and-verdict-stratified sample of 100 detections was re-reviewed blind: scores stripped,
order randomised across all four targets, no record of the earlier verdict, and the target named
only because a reviewer cannot judge "is this `Fleuron_2`" without being told which design is meant.

| | Value |
|---|---|
| Agreement | **97 of 100** |
| Cohen's kappa (1960) | **0.940** |
| Agreement on originally confirmed | 0.981 |
| Agreement on originally rejected | 0.958 |
| Reversals | 1 confirmed → rejected, 2 rejected → confirmed |
| Implied confirmation rate over the 943 sampled from | 0.738, against 0.744 for the first pass |

The bucket carrying the chapter's central claim agreed **37 of 37**. Attribution was observed,
not controlled, in a sample allocated as 13 confirmed and 12 rejected proposals per target. The
bucket is `no_otsu_candidate` as it stood before §4.2 split it: it holds the 365 recoveries with
no overlapping unfiltered box together with the 71 the size filter discarded, and the re-review does not
resolve the two separately. Kappa is prevalence-dependent and applies to this deliberately
verdict-balanced sampling frame.

Disagreement concentrates where the chapter already reports the weakest evidence: the
adequate-but-uncurated bucket (0.929 agreement, and the worst precision at 0.475) and the
fragment bucket (0.889).

The sample was drawn from the 943 detections reviewed in the first pass, so the reweighted rate
describes those rather than all 958. `Fleuron_2`'s 15 confusions were judged after it and carry
no reliability estimate of their own.

**This is intra-rater reliability.** It measures whether the reviewer repeats their own
judgements, not whether those judgements are correct. Both passes carry the same view of what
counts as the design, and a second reviewer could disagree systematically without this detecting
it. Chapter 2's disagreement benchmark has the same property, and neither is a substitute for
independent adjudication.

### 4.5 Confirmation against detection confidence

Confirmation rises with the detector's confidence, from 0.690 in the 0.25–0.50 band to 1.000
above 0.90. **This cannot be read as evidence that the score is calibrated.** The exported crops
were named with the confidence in the filename, so the reviewer saw the score while judging, and
any tendency to resolve doubt in its favour would produce exactly this gradient. It is reported
as a guide to review burden at a given operating point and nothing more. §5 establishes the
score's behaviour with no human in the loop. A future export should strip the score from the
filename and randomise the order within each folder, which costs nothing and removes the
confound; the re-review of §4.4 did exactly that.

The same defect is recorded against the score-band table of Chapter 3, and is disclosed at both
points of use rather than left for a reader to assemble.

## 5. Validation

### 5.1 The original split did not hold out what it claimed to

The protocol specified a book-disjoint validation set. Audited against `book_identifiers.py`,
three of the four splits do not meet it, for two separate reasons.

**Sibling volumes were treated as different books.** The split grouped scans by a prefix taken
from the filename, so `vaoe1`, `vaoe2`, `vaoe3` and `vaoe4`, four volumes of one work printed in
the same shop from the same ornament stock, landed on opposite sides. Chapter 1 measures
sibling-volume sensitivity explicitly for the extraction benchmark; this stage was held to a
weaker standard than the chapter it cites.

**One target trained on its own validation set.** The build script wrote into the dataset
directory without clearing it and skipped files already present. `Fleuron_73`'s directory was
written twice, one minute apart, under two split rules; five scans the first pass placed in
`train/` survived into the second, which placed the same scans in `val/`, carrying 76 boxes with
them. The trainer reads the directory, so it trained on 364 boxes while the script reported the
288 that pass had written.

| Target | Validation boxes | In a work that also supplies training scans | Genuinely unseen |
|---|---:|---:|---:|
| `Fleuron_2` | 79 | 75 | **4** |
| `Fleuron_74` | 16 | 0 | 16 |
| `Fleuron_73` | 78 | 76 | **2** |
| `Fleuron_72` | 139 | 72 | 67 |

**All validation figures reported before this audit are withdrawn**, including those in §3.1.
The script now clears both directories before writing and verifies on disk, after the write,
that no scan and no work appears on both sides, aborting if either does.

**None of this touches the chapter's result.** The 702 confirmations come from a human reviewing
crops, not from a metric.

### 5.2 A single split cannot validate these targets

Rebuilding the split correctly was tried and failed, for a reason worth reporting. `Fleuron_73`
carries 247 of its 366 impressions in one work. A random 20% of works held out leaves seven
validation instances; holding out the dominant work removes two thirds of the training data.
Neither measures anything, and that concentration, not the file-handling defect alone, is why
the original run drifted into scoring itself on training data.

### 5.3 Leave-one-work-out

Each work is held out in turn and a detector trained from scratch on the rest, so every
impression is predicted exactly once by a model that never saw its work. Predictions are pooled
and scored by greedy, confidence-ordered matching at IoU 0.50 (Everingham et al., 2010). Chapter 1
instead uses maximum-cardinality assignment because its candidates have no confidence score. The four targets carry
17, 12, 14 and 9 works, so one pass is 52 folds and **208 models were trained** in total: three
seeds of YOLO11n, and the YOLO11s pass of §5.5.

| Target | Training-box range | Impressions | Precision | Recall | Recall, each work weighted equally |
|---|---:|---:|---:|---:|---:|
| `Fleuron_2` | 169–264 | 265 | 0.787 | 0.864 | 0.857 |
| `Fleuron_74` | 78–96 | 98 | 0.544 | 0.949 | 0.929 |
| `Fleuron_73` | 119–365 | 366 | 0.791 | 0.486 | 0.778 |
| `Fleuron_72` | 427–675 | 678 | 0.734 | 0.827 | 0.818 |

Pooled across targets, precision is 0.731 (work-clustered bootstrap 95% CI 0.646–0.807),
recall is 0.754 (0.680–0.930), and F₂ is 0.749. The intervals resample the 27 distinct
bibliographic works 10,000 times; treating 1,407 impressions as independent is anti-conservative.

**Two averages are reported because they answer different questions and diverge sharply where a
design is concentrated.** Impression-weighted recall treats every impression equally, so
`Fleuron_73`'s dominant work supplies two thirds of the test cases *and* is the fold where the model
trains on a third of the data; its 0.486 is close to the score of that one handicapped model.
Work-weighted recall treats every work equally, answering "will this transfer to a book the model
has not seen", and gives 0.778.

**These figures describe 52 models, none of which produced the corpus run.** Each fold trains
without one work, while the deployed detector of §4 trained on all of them and was then applied to
the same corpus, so cross-validated recall measures known-catalogue transfer to an unseen work and
is not a bound on the corpus-run models' unknown recall. Precision is conservative for a different
reason: the labels are the catalogue, incomplete by construction, so a correct uncatalogued
impression scores as a false positive.

### 5.4 What the detector misses

Impressions missed by the model that never saw their work are compared with those found, on
properties of the impression itself. Quartiles are taken **within** each target, since the four
designs differ in typical size and pooled quartiles would confound "small impressions are
missed" with "the target whose impressions are small is harder".

| Target | Q1 smallest | Q2 | Q3 | Q4 largest |
|---|---:|---:|---:|---:|
| `Fleuron_2` | 0.313 | 0.164 | 0.031 | 0.030 |
| `Fleuron_72` | 0.510 | 0.045 | 0.026 | 0.000 |
| `Fleuron_74` | 0.000 | 0.182 | 0.036 | 0.000 |
| `Fleuron_73` | 0.465 | 0.532 | 0.590 | 0.441 |

**Small impressions are missed far more often, on three of the four targets.** `Fleuron_73` is
flat, so the effect is not universal, and analysing that target alone would have found nothing.
Boundary truncation is not a major observed failure mode in this small subset: 11 of the 1,407
impressions touch a scan edge, and 10 of those 11 were found. The first version of this check tested the left and top edges only, and so
could not have observed the 11, all of which sit on a right or bottom edge; the test now covers
all four sides against the scan's own dimensions.

### 5.5 Model capacity, and the variation that bounds every comparison

Every figure above uses YOLO11n, the smallest model in the family, chosen without comparison.
The whole cross-validation was repeated with YOLO11s, and separately with two further random
seeds, which reseed both the negative-scan sample and the training initialisation.

| Target | Training-box range | Recall change from a larger model | Recall spread over three seeds | Reading |
|---|---:|---:|---:|---|
| `Fleuron_74` | 78–96 | **−0.204** | 0.102 | exceeds the observed n-seed spread; a **loss** |
| `Fleuron_2` | 169–264 | +0.083 | 0.083 | comparable to the spread |
| `Fleuron_73` | 119–365 | **+0.270** | 0.090 | exceeds the observed n-seed spread |
| `Fleuron_72` | 427–675 | +0.150 | 0.198 | **within** the spread |

**A larger detector is not uniformly better in this descriptive comparison.** It raises recall on
some targets and lowers it on another, while precision falls on all four by 0.076 to 0.224. Only
YOLO11s seed 42 was run against three YOLO11n seeds, so these are capacity indications rather than a
factorial estimate of architecture effects, and the comparison bounds seed noise on the smaller
model only.

**The finding with the widest consequence is the seed variation itself.** Across three runs
differing only in seed, precision ranges 0.544 to 0.825 on `Fleuron_74` and recall 0.765 to 0.963 on
`Fleuron_72`. **No single run supports a comparison finer than roughly 0.1 in recall or 0.28 in
precision**, and nothing in this chapter had been run twice before this experiment. This bounds
every comparison the chapter makes between detectors, including the control experiment of §4:
`Fleuron_74`'s precision of 0.544 is the lowest of its three seeds, the others giving 0.825 and
0.822, so its apparent precision collapse is substantially an unlucky draw.

## 6. Was a Trained Detector Necessary?

The chapter establishes that a detector recovers impressions extraction never produced a region
for. It does not follow that *training* was required, and the obvious alternative needs no model
at all: normalised cross-correlation of a catalogue crop against the scan (Lewis, 1995), the
first thing anyone would try and standard for decades.

The baseline is given the same advantages the detector had (seven scales, all four flip
orientations, a swept threshold) and the same handicaps: templates are drawn from the catalogue
snapshot taken *before* the recoveries were folded in, and never from the work under search, so
the comparison is work-disjoint on both sides. Both are scored against the same ground truth,
the catalogued impressions that existed at training time, with the baseline's operating point
chosen by F₂ rather than by whichever threshold maximises recall.

| Target | Detector (YOLO11n) | Detector (YOLO11s) | Template matching | Recoveries the baseline also finds |
|---|---|---|---|---|
| `Fleuron_2` | 0.787 / 0.864 | 0.711 / 0.947 | **0.830 / 0.774** | 31 of 54 |
| `Fleuron_74` | 0.544 / 0.949 | 0.417 / 0.745 | **0.776 / 0.459** | 11 of 33 |
| `Fleuron_73` | 0.791 / 0.486 | 0.681 / 0.757 | **0.658 / 0.836** | 79 of 121 |
| `Fleuron_72` | 0.734 / 0.827 | 0.511 / 0.978 | **0.658 / 0.850** | 222 of 494 |

**The necessity of training is not established by this evidence.** A training-free matcher exceeds
the detector's recall on two targets and its precision on two, and independently reaches 343 of the
702 impressions this chapter reports as recoveries. The precision-recall trade varies by target, and
the different evaluation scan sets prevent a definitive ranking. The supportable statement is that
**the two methods reach overlapping sets of impressions** and offer target-dependent operating
trade-offs. Correlation is a credible first baseline for a new corpus; these four adaptive cases
establish no universal ordering.

Two qualifications bound the table in the baseline's disfavour. It uses one seed per detector
configuration, and §5.5 shows seed variation up to 0.2 in recall, so several differences are not
interpretable. And the two methods are charged for false positives on different scan sets:
correlation ran only on scans already known to carry the target, so it is never charged for a false
positive on a scan carrying none, while the detector's cross-validated figures cover the held-out
work's positive scans *together with* a sample of negative scans and charge every false positive
there, 132 scans against the baseline's 80 for `Fleuron_2`, 52 against 37, 62 against 44, and 48
against 46. Neither is a corpus-wide false-positive rate. Only the corpus run of §4 covers all 613
scans, and that is not what this table reports.

## 7. Detection and Retrieval Are Complementary

The `Fleuron_74` failure is one of identification, not localisation: the detector found the
fleurons and misnamed them. Identification is what Chapter 3 does, and it operates on the
extracted crop where interior detail survives.

The stored output table says that each of the 103 `Fleuron_74` confused detections was cropped,
binarised, embedded, and matched with the overlapping catalogue crop excluded. It records all 103
as assigned to their reviewed class, including all 62 `Fleuron_73` cases. The code that generated
the table and the neighbour identifiers needed to audit the exclusion were not preserved, so this
is an exploratory result rather than an independently reproducible validation.

| | Locates an unextracted impression | Distinguishes similar designs |
|---|---|---|
| Retrieval alone | no, it ranks crops that already exist | yes |
| Detection alone | yes | no, at scan scale it sees outline only |
| **Detection then retrieval** | **yes** | promising, not yet reproducibly validated |

One limitation is worth naming rather than leaving implicit: this was tested on confusions,
which are by construction locations where a catalogued crop already exists. It has **not** been
tested on the 365 recoveries where extraction produced nothing, which is the case the chapter
exists for. §9 records it as outstanding.

## 8. Discussion and Limitations

**The primary gain is completeness, with limited added reach.** The reviewed additions create
seven new work-class occurrences: one for `Fleuron_74`, two for `Fleuron_73`, and four for
`Fleuron_72`; `Fleuron_2`'s reach is unchanged. These are genuine new occurrences for the selected
targets, but the adaptive four-target design cannot support a catalogue-wide distribution claim.
The fair primary recurrence summary therefore excludes detection; including it as a sensitivity
check changes 508 work-class occurrences to 515 while leaving the recurrent-class count, median
reach, and maximum reach unchanged.

**Validation precision is a floor, not an estimate.** The validation labels are the catalogue,
and the catalogue is incomplete by construction, so a correct detection of an uncatalogued
impression scores as a false positive. The quantity being penalised is exactly the quantity this
stage exists to recover.

**Run-to-run variation is large and was unmeasured until §5.5.** Every detection figure published
before that experiment is a single draw from a distribution whose spread reaches 0.2 in recall.

**The review has no independent adjudication.** §4.4 establishes that the reviewer repeats
themselves, not that they are right.

**Claims this chapter supports:** that impressions exist which connected-component extraction
could not produce a region for, and that a detector trained on the pipeline's own verified
output recovers them without additional manual annotation; that 365 such impressions were
confirmed by hand across four designs; that 644 of 702 confirmed recoveries were lost before the
catalogue, 573 by extraction and 71 by the size filter, rather than by clustering; and that
detection and retrieval show potentially complementary localisation and identity behaviour in
the stored exploratory result of §7.

**Claims it does not support:** that this generalises to the other 88 classes; that the detector
would discover designs absent from the catalogue, since it can only learn what extraction
surfaced at least sometimes; that a trained detector was necessary, which §6 examines and does
not settle in its favour; or that any two detection numbers differing by less than the seed
spread of §5.5 are different.

## 9. Outstanding Work

**The two-stage architecture needs a reproducible test where it matters.** First regenerate §7
with query and neighbour identifiers that prove the exclusion, then embed the 702 confirmed
recovery crops and check, leave-one-out, whether retrieval assigns them to the right class.

**The chapter has never been measured against ground truth the pipeline did not generate.**
Chapter 1's 30 held-out benchmark scans carry 661 hand-drawn glyph boxes, of which 100 are
impressions of these four targets. Measuring how many of the boxes extraction missed the
detector finds would give the complementarity claim external evidence rather than internal.

**The corpus run used the smaller model.** §5.5 shows a larger one raises recall on two targets,
so more impressions are likely reachable. Re-running corpus inference would desynchronise the
frozen review labels that 702 confirmations rest on, and the chapter's finding does not depend
on maximising the count, so this is recorded rather than done.

**The baseline was not run corpus-wide.** §6 measures correlation only on scans known to carry
the target, so its false-positive rate over the whole corpus is unknown while the detector's is
measured.

## 10. Reproducibility

The per-target directories named in this section belong to the working project rather than to this
repository, which carries the code and the written account alone. All four notebooks are stored
executed, so every result reported above is readable here exactly as it was produced; the
prediction, fold and review tables they read travel with the thesis deposit instead.

Four notebooks form the executable record, and each runs from this folder, resolving every path
relative to it. Training is driven by scripts in `_tools/`, called rather than duplicated, since
a notebook is a poor place for a job that trains 208 models.

| # | Notebook | Input → output |
|---|---|---|
| 1 | `1_SingleFleuronDetector.ipynb` | catalogue + scans → `fleuron_2_v1/` |
| 2 | `2_FourTargets.ipynb` | frozen review labels + unfiltered candidates → `four_targets/` |
| 3 | `3_Validation.ipynb` | cross-validation runs → `validation/` |
| 4 | `4_Baselines.ipynb` | template-matching runs + cross-validation → `baseline_comparison/` |

| Tool | Purpose |
|---|---|
| `train_single_fleuron_detector.py` | trains one detector, runs the corpus, exports review sheets |
| `leave_one_work_out.py` | cross-validation; one model per work, checkpointed per fold |
| `template_matching_baseline.py` | the training-free baseline of §6 |
| `summarise_lowo.py` | rescoring, work-clustered intervals, and corrected edge/within-target size diagnostics for §5.3 |
| `build_reliability_sample.py` | exports the blind re-review sample of §4.4 |
| `score_reliability.py` | scores it against the original review |
| `review_fleuron2_confusions.py` | exports and scores the 15 late-reviewed `Fleuron_2` confusions of §4 |
| `refine_attribution.py` | the size-filter separation of §4.2 |
| `make_training_curves_figure.py` | the augmentation-comparison curves reproduced in the report appendix |
| `progress.py` | reports which runs have completed, read from disk |

**Human decisions are never overwritten.** The review directories carry their labels in their
contents. `train_single_fleuron_detector.py` refuses to export over an existing review directory
and keeps corpus inference behind an explicit flag; `build_reliability_sample.py` refuses to
rebuild a sample that already exists. Every notebook writes through a guarded directory that
refuses to overwrite a populated one, so re-running means moving the previous output aside
rather than disabling the guard.

**Three defects in the training script were found and fixed**, each recorded in the code at the
point it was wrong: the dataset directory is now cleared before writing and verified on disk
afterwards; the split is made over works rather than filename prefixes; and the negative sample
is drawn from a sorted list, since iteration order over a set of strings varies between
processes and the seeded shuffle was not in fact reproducible.

Both training scripts now select the appropriate pre-detection snapshot from §3 by target when
`--catalogue` is omitted. Passing a catalogue explicitly remains possible for a new experiment;
the live final catalogue must not be used to reproduce the reported runs.

**Figures.**

| Figure | Section | File |
|---|---|---|
| Recoveries by what extraction had produced, and confirmation by target | §4 | `four_targets/four_targets.png` |
| Cross-validated recall with its seed range, and capacity against noise | §5 | `validation/validation.png` |
| The detector against a training-free baseline | §6 | `baseline_comparison/baseline_vs_detector.png` |

## 11. Superseded

An earlier multi-class attempt preceded this chapter and is not part of this package. Two of its
findings shaped the protocol above and are recorded here because the design decisions they forced
would otherwise look arbitrary: sparse catalogue labels teach a detector that unlabelled fleurons
are background, which is why §2 confines each detector to a single target; and raw extraction
output mixes whole ornament bands with single units in one box set, which is why §4.1 attributes
detections by box geometry rather than by the presence of a candidate region. Its notebooks and
training outputs are not retained, and no reported result depends on them.

Four directories prefixed `_superseded_` in `4_detection_outputs/` hold analyses replaced by a
corrected version, kept because each is the evidence for why the correction was needed:

| Directory | Replaced because |
|---|---|
| `_superseded_fleuron_2_tm_samescan/` | template matching excluded the same scan rather than the same work |
| `_superseded_baseline_yolo11n_only/` | the baseline comparison of §6 ran against the smaller model only |
| `_superseded_validation_pooled_quartiles/` | the false-negative analysis of §5.4 pooled quartiles across targets |
| `_superseded_four_targets_pre_confusion_review/` | the §4 tables predate the late review of `Fleuron_2`'s 15 confusions |

## References

Bekker, J., & Davis, J. (2020). Learning from positive and unlabeled data: a survey.
*Machine Learning*, 109(4), 719–760.

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological
Measurement*, 20(1), 37–46.

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal
Visual Object Classes (VOC) challenge. *International Journal of Computer Vision*, 88(2),
303–338.

Lewis, J. P. (1995). Fast normalized cross-correlation. *Vision Interface*, 120–123.

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: unified,
real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition*, 779–788.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
*Journal of the American Statistical Association*, 22(158), 209–212.
