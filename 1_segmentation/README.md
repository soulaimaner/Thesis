# Chapter 1: Candidate-Region Extraction

This is the first of four stages in the ornament-identification pipeline:

```
→  Candidate extraction  →  find candidate regions on each ornament scan
   Clustering            →  discover which candidates are the same fleuron
   Retrieval             →  given a verified fleuron, find more of its impressions
   Detection             →  find impressions that were never extracted at all
```

Terminology (*composite ornament*, *fleuron*, *scan*, *candidate*, *crop*, *impression*) is defined
once in the [root README](../README.md) §2 and used identically across all four stages. One term
belongs to this chapter alone: a *glyph* is the unit annotated in its benchmark, one intended
fleuron on a scan, whether or not its ink is physically separate from its neighbours. Each scan in
the corpus shows one composite ornament, and this chapter proposes the rectangular regions on it
that may contain the constituent fleurons. Every later stage operates on the resulting crops.

**Organisation.** Sections 1 and 2 state the task and build the annotated benchmark with its frozen
volume-disjoint split. Sections 3 to 5 compare three method families under a criterion fixed in
advance and open the held-out split once. Sections 6 to 8 state the limits, apply the frozen
configuration to the corpus, and record reproduction.

| Notebook | Question | Section |
|---|---|---|
| `1_SegmentationBenchmark` | Which extraction method, judged on annotated scans under a frozen split? | §2–§5 |
| `2_CandidateExtraction` | What does the selected configuration produce over the full corpus? | §7 |

## 1. Task, Terminology, and Selection Criterion

This chapter selects the method used to extract candidate fleuron regions before feature extraction
and clustering. Its output is a set of rectangular candidate crops per scan.

The evaluation annotations are bounding boxes rather than pixel masks, so this benchmark measures
glyph localisation and candidate-crop proposal quality, not pixel-level segmentation accuracy. It
says nothing about exact fleuron contours. The practical research question is:

> Which candidate-generation method provides the best balance between recovering potential fleurons
> and limiting irrelevant or badly localised crops across heterogeneous historical printed material?

Writing TP for candidates matched to a reference fleuron, FP for candidates matched to nothing, and
FN for reference fleurons left unproposed,

    precision = TP / (TP + FP)        recall = TP / (TP + FN)

and their costs are not symmetric at this stage. A false candidate can still be rejected later, by
the minimum-side filter the feature stage applies, by clustering, or by human review of the
resulting groups. A fleuron that is never proposed enters none of those stages and cannot be
recovered by any of them. Selection therefore uses

    F₂ = 5 · precision · recall / (4 · precision + recall)

the recall-weighted member of the F-measure family (van Rijsbergen, 1979), weighting recall four
times as strongly as precision.

The method was selected on the calibration split alone, and the held-out split was opened only after
the method family and every parameter had been frozen. Because the split is fixed before any method
runs, it cannot be influenced by knowledge of how the methods perform. This ordering is the
methodological core of the chapter.

## 2. The Annotation Benchmark

### 2.1 Composition

| Quantity | Value |
|---|---:|
| Annotated images | 40 |
| Digitised volumes | 24 |
| Glyph boxes | 881 |
| Ignore regions | 5 |
| BMP images | 20 |
| JPEG images | 20 |
| Minimum glyphs per image | 3 |
| Median glyphs per image | 19.5 |
| Maximum glyphs per image | 61 |

Every potential fleuron on each selected scan was annotated with a `Glyph` rectangle; genuinely
uninterpretable regions were marked `Ignore_group`. Boxes were centred by hand on the intended
fleuron, and small variation in tightness with occasional unavoidable neighbouring extremities was
accepted as part of the process.

The grouping unit is the digitised volume, not the individual glyph: placing many boxes on one scan
makes the box-level measurement more precise but adds no independent volumes or scanning conditions.
For backward compatibility the executable tables call this field `book`; it is derived from the
leading catalogue prefix of each filename and denotes a volume, not necessarily a distinct
bibliographic work. The corpus-wide rule decoding both levels of identity is `book_identifiers.py`
at the package root, and the consequences of grouping at volume level are quantified in §5.3.

### 2.2 Annotation Sources and Audit

The authoritative record is the raw Label Studio export
(`annotations/project-5-at-2026-07-31-13-30-81dfaca9.json`); a derived image manifest and box table
are stored under `annotations/benchmark/`. All 40 images were reviewed visually in five batches, to
confirm that every potential glyph carried a label, that each box identified one intended glyph,
that any neighbouring material caught inside a box was unavoidable rather than careless, and that
ignore regions were genuinely unresolvable rather than merely difficult.

The raw and COCO exports have identical label counts, but two raw boxes and two COCO boxes differ in
their coordinates. The benchmark uses the raw records in those cases, as declared in advance. The
cause of the divergence was not traced to a specific export step, so the precedence of the raw
export rests on a rule fixed before the comparison rather than on established provenance; the four
records are too few to affect any reported figure.

### 2.3 Frozen Volume-Disjoint Split

| Split | Images | Volumes | Glyph boxes | Ignore regions | BMP | JPEG |
|---|---:|---:|---:|---:|---:|---:|
| Calibration | 10 | 6 | 220 | 0 | 5 | 5 |
| Held-out evaluation | 30 | 18 | 661 | 5 | 15 | 15 |

The split is grouped by digitised volume, so no volume contributes scans to both sides. This is more
defensible than a scan-level random split, since fleurons, printing style, degradation and
digitisation properties can repeat within a volume. It was constructed by searching every
combination of the required number of calibration volumes and scoring each deterministically against
25% targets for image count, glyph count, BMP count and JPEG count; no property related to image
quality or segmentation difficulty entered the choice. It was fixed before any evaluation was run
and never rewritten.

## 3. Compared Methods

Three families were compared, spanning the range of approaches available: a global threshold, a
local threshold, and a learned foundation model. Global and local thresholding followed by
connected-component analysis (Rosenfeld & Pfaltz, 1966) remain the standard classical baselines in
document-image binarisation and recur as reference methods in the DIBCO series (Pratikakis et al.,
2019).

### 3.1 Otsu Connected Components

Otsu (1979) selects a per-image global threshold by minimising intra-class intensity variance in the
grey-level histogram. It is the existing baseline for this corpus and was tested first because
fleurons are distinguished from paper mainly by ink density rather than colour; because the corpus
mixes collections with visibly different exposure, contrast and paper tone, which a scan-specific
threshold accommodates and a fixed corpus-wide cutoff would not; and because it needs no training
data, no GPU and only three parameters.

Per image: load and convert to RGB; convert to grayscale; apply per-image min-max intensity
normalisation to `[0, 255]`; apply a `3 × 3` Gaussian blur; apply inverted global Otsu thresholding;
extract 8-connected foreground components; remove components below `40` foreground pixels; convert
each retained component to its bounding rectangle; expand each rectangle by `3` pixels on every
side, clipped to the image boundary.

Each step, and what it can and cannot be claimed to do:

| Step | Purpose | Limit on the claim |
|---|---|---|
| Grayscale | routine preprocessing | not a contribution |
| Min-max normalisation | common 8-bit scale across scans; reproducible | never ablated against its absence |
| Gaussian blur | suppresses high-frequency scan noise | tuned over `{3, 5, 7}`; no-blur never evaluated |
| Minimum-area filter | removes speckles and fragments | lower thresholds raise recall and fragmentation together |
| Box padding | preserves extremities and context for encoding | affects box IoU directly, so fixed after calibration |

No morphology, adaptive thresholding, maximum-area filter or learned component was introduced, so
the grid tests the method itself rather than additional machinery.

The calibration grid comprised 27 settings: Gaussian size `{3, 5, 7}`, minimum component area
`{40, 80, 160}` and padding `{0, 3, 6}`. It was anchored on the pre-existing baseline of `(5, 80,
3)`, values chosen informally during earlier exploratory work rather than by controlled comparison,
with one neighbouring value either side of each parameter. The grid is deliberately small: it
establishes whether the existing setting is locally reasonable, not an open-ended search.

### 3.2 Sauvola Connected Components

Sauvola and Pietikäinen (2000) adapt the threshold to the local mean and standard deviation within a
sliding window, and the method remains a standard baseline for degraded document images where
contrast varies spatially. It was included to test whether local adaptation could remove the
foreground bridges caused by uneven paper tone or weak local contrast, and was applied to the same
min-max-normalised grayscale image as Otsu, with connected components and area filtering applied
identically afterwards.

Its 27-setting grid was window size `{15, 31, 45}`, sensitivity `{0.15, 0.25, 0.35}` and minimum
component area `{40, 80, 160}`, with padding fixed at `3`. Every window fits inside the smallest
image dimension in the benchmark. Padding was frozen at the Otsu-selected value because it governs
the output crop convention rather than the thresholding method; this gives Otsu no advantage from
having searched one dimension more, since padding was free in the Otsu grid and that search returned
the same value of `3`. The selected configuration was window `45`, sensitivity `0.25`, minimum area
`40`, padding `3`.

### 3.3 Automatic SAM-B Proposals

The Segment Anything Model (Kirillov et al., 2023) was included as a training-free learned baseline,
to test whether a foundation segmentation model could separate glyphs that connected components
merge. It was applied automatically over a fixed point grid and never prompted with the manual
annotation boxes, since prompting with the reference would leak the target at inference time.

Raw configuration: checkpoint `sam_b.pt` (local), input size `512`, automatic point stride `16`,
point batch size `32`, raw predicted-IoU floor `0.50`, stability floor `0.80`, internal NMS IoU
`0.70`. The generation floors were kept permissive so proposals were retained for filtering rather
than discarded early. Because re-filtering cached proposals is inexpensive, a finer post-filter grid
of 12 settings was then applied over predicted-IoU `{0.70, 0.85, 0.95}`, minimum mask area
`{40, 80}` and maximum box-to-scan area ratio `{0.20, 0.35}`, with output padding fixed at `3` to
match the crop convention of both classical methods. The maximum-area filter controls whole-image
and large nested masks, while predicted-IoU and mask area control small fragments. The selected
filter was predicted-IoU `0.70`, minimum mask area `40`, maximum box ratio `0.20`.

Proposals were generated once on GPU (NVIDIA GeForce GTX 1080 Ti) using the frozen `16 × 16` point
grid, at a mean measured runtime of about `0.81` seconds per calibration image, timed on the actual
generation pass including image encoding rather than on a cached rerun. The completed SAM benchmark
was not rerun or retuned after the held-out protocol was frozen, and only one SAM configuration grid
was tested, so this experiment does not establish the best performance obtainable from every
possible SAM configuration.

## 4. Evaluation Protocol

The evaluator, the IoU thresholds and the selection rule were all fixed before any method was
scored.

**Matching.** Predicted and annotated boxes are matched one-to-one by maximum-cardinality Hungarian
assignment (Kuhn, 1955). Eligible matches must meet the stated IoU threshold; the `1000 + IoU`
assignment score prioritises the number of valid matches before preferring higher-IoU assignments.
Because candidate boxes carry no confidence score, ranking-based measures such as average precision
do not apply, and a fixed-threshold protocol in the style of Everingham et al. (2010) is used
instead. Two thresholds are reported: IoU `0.50` as the primary, stricter localisation result, and
IoU `0.25` as a secondary threshold, useful because the practical output is a crop proposal and
small box-tightness differences may still yield a usable crop.

**Metrics.** Precision, recall and F₂ are computed at each IoU threshold from the definitions in §1.
The primary metric is micro F₂ at IoU 0.50, computed after summing TP, FP and FN over all scans.
Ties break first by higher recall, then by fewer proposals per scan. Macro scan recall, recall at
IoU 0.25, proposals per scan, fragmentation and merging are diagnostics and may not substitute for
the primary criterion.

**Ignore regions.** A prediction is excluded from scoring when its centre lies inside an
`Ignore_group` rectangle, or when at least 50% of its own area lies inside one. Twenty-six
evaluation candidates were excluded by this rule.

**Structural diagnostics.** Two non-primary diagnostics describe failure structure and are reported
as counts rather than accuracy metrics. A fragmented glyph is one where at least two candidates each
have at least 50% of their own area inside a single reference glyph; a merged candidate is one
candidate covering at least 50% of each of at least two reference glyphs.

## 5. Results

### 5.1 Calibration

Against the original informal baseline, the selected configuration gains `0.095` of recall@0.50 and
`0.053` of F₂@0.50 and cuts merged candidates from 23 to 19, paying in precision, in a quarter more
candidates per scan, and in more fragmented glyphs. This is the intended recall-oriented trade for
candidate generation, and the table gives both sides of it.

| Setting | Gaussian | Min. area | Padding | Precision@0.50 | Recall@0.50 | F₂@0.50 | Recall@0.25 | Candidates/scan | Fragmented | Merged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original baseline | 5 | 80 | 3 | 0.677 | 0.714 | 0.706 | 0.777 | 23.2 | 34 | 23 |
| Calibration-selected | 3 | 40 | 3 | 0.610 | 0.809 | 0.759 | 0.864 | 29.2 | 46 | 19 |

**Figure 1.1: The Otsu calibration trade-off**
(`annotations/benchmark/otsu_calibration_tradeoff.png`). Each point is one of the 27 grid
configurations, placed by candidates proposed per calibration scan against glyph recall at IoU 0.50;
colour distinguishes the minimum-area setting. The pre-declared F₂ rule selects the starred
configuration at the recall-maximising end of the frontier, over the exploratory baseline marked
with a cross. The figure shows the price as directly as the gain: recall rises only by moving right,
that is by proposing more candidates for later stages to filter.

The three selected configurations on the same 220 glyph boxes across the same ten calibration
images:

| Method | Precision@0.50 | Recall@0.50 | F₂@0.50 | Recall@0.25 | Macro scan recall@0.50 | Candidates/scan | Fragmented | Merged | Seconds/scan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Otsu | **0.610** | **0.809** | **0.759** | 0.864 | **0.843** | **29.2** | **46** | 19 | **~0.002** (CPU) |
| Sauvola | 0.493 | 0.791 | 0.706 | **0.877** | 0.798 | 35.3 | 50 | 18 | ~0.011 (CPU) |
| SAM-B | 0.317 | 0.655 | 0.540 | 0.768 | 0.648 | 45.4 | 94 | **16** | 0.81 (GPU) |

Otsu achieved the best primary score and the best recall at IoU 0.50 while producing fewer
candidates than either alternative. Sauvola slightly improved recall at IoU 0.25 and reduced merges
by one, but its lower precision and higher fragmentation reduced F₂; local thresholding therefore
did not deliver the reduction in merging that motivated testing it, which is what prompted moving to
a different method family rather than to further threshold variants. SAM sometimes separated
touching glyphs visually and produced the fewest merged-candidate diagnostics, but missed
well-separated glyphs on some scans and generated many nested or irrelevant boxes. Otsu was retained
and the other two treated as rejected comparison baselines.

Runtime is measured over each method's full processing pass on the same calibration scans, including
image loading and, for SAM, encoding and automatic mask generation. The classical methods are timed
three times per scan and use the median repeat; SAM records the single uncached pass from which its
proposals were cached. Scan times are then averaged, so the column reports the cost of processing a
corpus rather than the median cost of a scan. The classical methods ran on CPU and SAM on GPU, so
each is compared at the cost it would actually be deployed at rather than on equal hardware. This is
the only quantity here that does not reproduce exactly, since it measures the machine rather than
the data, and the unequal repeat protocol makes it an operational order-of-magnitude comparison
rather than an inferential claim: Otsu is more than two orders of magnitude cheaper per scan than
SAM-B and several times cheaper than Sauvola.

These are point estimates on six calibration volumes, and six volumes are few. The same volume-level
bootstrap later used for the held-out result (10,000 resamples, shared across the three methods so
each replicate compares them on the same volumes and the differences are paired) shows what this
split can and cannot establish.

| Quantity | Estimate | 95% lower | 95% upper | Share of resamples favouring Otsu |
|---|---:|---:|---:|---:|
| Otsu F₂@0.50 | 0.759 | 0.672 | 0.825 | n/a |
| Sauvola F₂@0.50 | 0.706 | 0.483 | 0.834 | n/a |
| SAM-B F₂@0.50 | 0.540 | 0.390 | 0.701 | n/a |
| Otsu − Sauvola | 0.054 | −0.066 | 0.266 | 0.650 |
| Otsu − SAM-B | 0.220 | −0.020 | 0.401 | 0.970 |

Six volumes do not resolve the paired difference between the two classical methods: Otsu leads
Sauvola in only 65% of resamples and the interval comfortably contains zero. The margin over SAM-B
is far clearer at 97%, though its interval also just reaches zero. No claim is therefore made that
Otsu is significantly more accurate than Sauvola. The rule was pre-declared on the micro F₂ point
estimate and applied exactly as declared, but the case for retaining Otsu does not rest on the F₂
gap alone: Sauvola produces 21% more candidates per scan at markedly lower precision and several
times the runtime, and it failed at the very mechanism that motivated testing it. What separates the
two classical methods here is cost and behaviour on merges, not a demonstrated accuracy advantage.

### 5.2 Touching-Glyph Comparison

The three methods were also compared directly on the four calibration scans with the most Otsu
merged candidates, to judge whether SAM's proposals could resolve merges connected components
cannot.

| Scan | Otsu R@0.50 / merges | Sauvola R@0.50 / merges | SAM-B R@0.50 / merges / candidates |
|---|---:|---:|---:|
| `foropapr68.ph.t1.vii.bmp` (bandeau border) | 0.85 / 5 | 0.85 / 5 | 0.43 / 5 / 46 |
| `lmfa1ut1760_vb.jpg` (I-monogram frame) | 0.52 / 4 | 0.67 / 3 | 0.97 / 1 / 54 |
| `crcolonsn72.t265le.bmp` (S-monogram frame) | 0.64 / 3 | 0.72 / 3 | 0.64 / 4 / 65 |
| `auj1am1760_024.jpg` (fleuron scrollwork) | 0.76 / 3 | 0.84 / 3 | 0.96 / 1 / 62 |

SAM separates touching glyphs sometimes, and inconsistently. On the I-monogram and fleuron scans it
clearly separates motifs both classical methods leave merged, recall rising to 0.96 and 0.97 with
merges dropping to 1 against 3 and 4. On the bandeau border it does the opposite: recall collapses
to 0.43 against 0.85 with no reduction in merges, because it misses glyphs that are already clearly
separated rather than resolving any that are touching. On the S-monogram scan it only ties Otsu's
recall while Sauvola outperforms both, and its merge count is the worst of the three.

Its extra boxes are mostly noise even where it performs well: on the two scans where recall improves
sharply, candidate counts of 54 and 62 far outpace what that recall implies about the true glyph
count, so a large share are nested fragments or spurious detections, and where SAM does not help
essentially all of its extra candidates fall into this category, matching nothing at all. Its
capacity to separate touching ink is therefore real but scan-dependent and unreliable, consistent
with its weaker aggregate scores.

**Figure 1.2: Method-family comparison on touching-glyph scans**
(`annotations/benchmark/touching_glyph_three_method_comparison.png`). Solid green and red rectangles
are matched and missed manual glyph annotations at IoU 0.50; dashed cyan and gold rectangles are
matched and unmatched candidate proposals. These scans were chosen from the calibration split
specifically to illustrate merge-heavy cases and are not representative performance estimates.

### 5.3 Held-Out Evaluation

The frozen Otsu configuration was applied exactly once to the 30 evaluation scans from 18 held-out
volumes, with no parameter selection remaining.

| Quantity | Result |
|---|---:|
| Scans | 30 |
| Volumes | 18 |
| Reference glyphs | 661 |
| Scored candidates | 837 |
| Ignored candidates | 26 |
| Candidates/scan | 27.9 |
| TP@0.50 | 493 |
| FP@0.50 | 344 |
| FN@0.50 | 168 |
| Precision@0.50 | 0.589 |
| Recall@0.50 | 0.746 |
| F₂@0.50 | 0.708 |
| Macro scan recall@0.50 | 0.659 |
| Recall@0.25 | 0.814 |
| F₂@0.25 | 0.773 |
| Fragmented-glyph count | 144 |
| Merged-candidate count | 62 |

At IoU 0.25, 538 glyphs matched against 493 at IoU 0.50. Those 45 additional lower-IoU matches show
that some errors are localisation or box-tightness mismatches rather than complete failures to
propose the region. Micro recall@0.50 (`0.746`) exceeds macro scan recall (`0.659`), so performance
is uneven across scans and glyph-dense scans contribute more strongly to the micro result.

The 95% percentile intervals below use 10,000 bootstrap resamples of the 18 evaluation volumes,
resampled at the volume rather than the glyph level so the uncertainty respects the split's grouping
unit (Davison & Hinkley, 1997; Field & Welsh, 2007). Because those volumes represent 17 bibliographic
works, these are volume-clustered rather than fully work-clustered intervals; the distinction is
material to the independence claim and is revisited below and in §6.

| Metric | Estimate | 95% lower | 95% upper |
|---|---:|---:|---:|
| Precision@0.50 | 0.589 | 0.486 | 0.714 |
| Recall@0.50 | 0.746 | 0.654 | 0.830 |
| F₂@0.50 | 0.708 | 0.635 | 0.782 |

| Split | Precision@0.50 | Recall@0.50 | F₂@0.50 | Recall@0.25 | Candidates/scan |
|---|---:|---:|---:|---:|---:|
| Calibration | 0.610 | 0.809 | 0.759 | 0.864 | 29.2 |
| Held-out evaluation | 0.589 | 0.746 | 0.708 | 0.814 | 27.9 |
| Evaluation minus calibration | -0.021 | -0.063 | -0.051 | -0.050 | -1.3 |

The moderate F₂ decrease of `0.051` reflects reasonable but imperfect generalisation to held-out
volumes. It is not equivalence between the two splits, and is reported explicitly rather than
absorbed into a single combined figure.

Because the legacy `book` field is the filename's catalogue prefix, four evaluation volumes
(`auj2am1760`, `lmfa2ut1760`, `vaoe1lhgo1760`, `vaoe2lhgo1761`) resolve to works that also appear in
calibration, so they are sibling volumes of the same multi-volume sets, printed and digitised under
closely related conditions. The split is therefore disjoint at volume level but not necessarily at
the level of the underlying work. Recomputing the frozen held-out metrics without those four
volumes, from the locked per-scan results and with no re-segmentation, shows the residual
relatedness does not flatter the headline figure: the fourteen fully unrelated volumes score
slightly higher than the full evaluation set, and the siblings are the harder subset.

| Subset | Scans | Volumes | Glyphs | Precision@0.50 | Recall@0.50 | F₂@0.50 |
|---|---:|---:|---:|---:|---:|---:|
| All evaluation volumes | 30 | 18 | 661 | 0.589 | 0.746 | 0.708 |
| Without calibration siblings | 19 | 14 | 436 | 0.575 | 0.807 | 0.747 |
| Calibration siblings only | 11 | 4 | 225 | 0.627 | 0.627 | 0.627 |

The three identical figures in the sibling row follow from that subset's equal counts of false
positives and false negatives, 84 of each.

| Format | Scans | Glyphs | Precision@0.50 | Recall@0.50 | F₂@0.50 | Macro scan recall@0.50 |
|---|---:|---:|---:|---:|---:|---:|
| BMP | 15 | 350 | 0.629 | 0.829 | 0.779 | 0.755 |
| JPEG | 15 | 311 | 0.540 | 0.653 | 0.627 | 0.564 |

BMP scans performed better here. This is a descriptive association rather than a causal effect of
file format, since format is confounded with volume, source collection, scan quality, printing style
and degradation.

**Figure 1.3: Calibration, held-out generalisation, and format breakdown**
(`annotations/benchmark/otsu_final_evaluation_summary.png`). Precision, recall and F₂ at IoU 0.50 on
calibration scans and on 30 held-out scans from 18 held-out volumes (left), and descriptive held-out
results for BMP and JPEG scans (right). Bootstrap intervals are in the table above rather than in
the figure.

### 5.4 Qualitative Error Analysis

The five scans with the lowest F₂@0.50 were selected after final scoring for post-hoc limitation
analysis; they were not used to change the method.

| Scan | Format | Glyphs | Candidates | Recall@0.50 | F₂@0.50 | Merged candidates | Dominant explanation |
|---|---|---:|---:|---:|---:|---:|---|
| `imgocr1761_166a.jpg` | JPEG | 4 | 2 | 0.000 | 0.000 | 2 | Touching/merged ink: four scroll motifs fused into two connected components by connecting strokes. |
| `lmfa2ut1760_016.jpg` | JPEG | 9 | 1 | 0.000 | 0.000 | 1 | Touching/merged ink, the most extreme case: the whole ~9-part cross fleuron is one unbroken component. |
| `ropoamre51.t1.65.bmp` | BMP | 6 | 4 | 0.167 | 0.179 | 2 | Touching/merged ink: edge-to-edge repeat units bridged by ink into each surviving candidate. |
| `auj2am1760_014.jpg` | JPEG | 13 | 6 | 0.308 | 0.345 | 2 | Touching/merged ink: a connecting rule along the strip fuses most repeated units into two blocks; only units with a clean gap match individually. |
| `prmdfrkn64.nu.t1._i.bmp` | BMP | 11 | 7 | 0.364 | 0.392 | 1 | Mostly touching/merged ink, but two small corner fleurons are missed outright, more consistent with a size-filter or box-alignment effect than a merge. |

Visual review confirmed the dominant failure mode: multiple intended glyphs are physically connected
by touching ink or decorative strokes and become one large connected component. This is a structural
limitation of connected-component extraction. Adjusting the threshold may remove artificial bridges,
but it cannot reliably split fleurons whose foreground ink is genuinely connected. Lower-IoU matches
can still correspond to useful candidate crops, and repeated fleurons joined by horizontal strokes
are particularly likely to merge. SAM can occasionally split touching instances, but its overall
misses and extra nested boxes make it unsuitable as the primary method under the tested conditions.
This analysis supports retaining Otsu while documenting touching-glyph merging as the principal
limitation; it justifies no post-hoc parameter change.

**Figure 1.4: Five lowest-scoring held-out scans**
(`annotations/benchmark/otsu_final_evaluation_lowest_scans_overlays.png`, with individual
high-resolution overlays under `annotations/benchmark/otsu_final_evaluation_lowest_scan_overlays/`).
Solid green/red boxes are matched/missed annotations; dashed cyan/gold boxes are matched/unmatched
Otsu candidates. These were inspected only after the method and parameters were frozen, and are
deliberately the worst-scoring scans rather than typical ones.

## 6. Discussion and Limitations

The fixed configuration used for full-corpus candidate generation:

```text
method: global Otsu + 8-connected components
grayscale: yes
min-max normalisation: [0, 255]
Gaussian kernel: 3 × 3
threshold: inverted Otsu
minimum component area: 40 pixels
box padding: 3 pixels
```

It was selected because it maximised the predeclared primary metric on calibration data; it
prioritised recall, appropriate before downstream filtering; it achieved F₂ `0.708` on 18 held-out
volumes; it was simpler than SAM-B and more than two orders of magnitude cheaper per scan even
comparing CPU against SAM on GPU; and neither alternative solved touching-glyph merges sufficiently
to justify their additional candidates or complexity.

**What the annotations can support.** They are boxes rather than masks, so §1's restriction to
localisation and crop-proposal quality holds throughout. Tightness varies because annotation was
manual, and some boxes necessarily include extremities of adjacent fleurons. Touching fleurons were
annotated as separate intended glyphs even where their ink is connected, so the merge failures above
are counted against the method by construction. The 40 scans were chosen for visual diversity rather
than drawn as a probability sample, so they license no population-level generalisation. All boxes
come from a single annotator, so no inter-annotator agreement exists and no claim of human-level
ground truth can be made. Independence is qualified too: `book` identity derives from filename
prefixes, so the split separates volumes rather than works. Decoding the shelfmark convention with
`book_identifiers.py` confirms the consequence rather than leaving it a possibility, since the 18
evaluation volumes belong to 17 works, four of them siblings of calibration volumes, putting 11 of
the 30 evaluation scans in works calibration had already seen. Section 5.3 shows the effect runs
against the method rather than for it, but complete independence cannot be claimed.

**What the comparison establishes about the methods.** Every ablation was partial: min-max
normalisation was never tested against its absence, and blur was tuned over `{3, 5, 7}` but never
compared against no blur, so neither can be claimed to have improved performance. The selected
configuration sits on the grid boundary in two of three dimensions, at the smallest kernel and
smallest minimum area tested, so calibration establishes it is the best of the 27 settings examined,
not that smaller values would not score higher; the grid was anchored to the existing baseline
rather than widened until the optimum became interior. Only one constrained automatic SAM-B setup
was tested, on one checkpoint and one point grid, so the result bears on this configuration and this
corpus, not on SAM in general; prompted, fine-tuned and larger variants were not evaluated. And the
calibration split is small in the dimension carrying the uncertainty: per §5.1, six volumes cannot
separate Otsu from Sauvola on F₂, so the comparison establishes which method scored highest under a
rule fixed in advance rather than that the two differ in accuracy.

**Interpreting the observed differences.** Performance varied substantially across scans, macro scan
recall (`0.659`) falling well below micro recall (`0.746`), so the headline figure is carried
disproportionately by glyph-dense scans. The BMP/JPEG gap is descriptive only, for the confounding
reasons in §5.3. The false-candidate cost of the recall-weighted criterion is deferred rather than
eliminated, since every unmatched candidate becomes downstream filtering work; Chapter 2's
within-cluster audit, run on the earlier extraction of §7, confirms non-fleuron material reaches
later stages, though it sampled by stratum rather than in proportion to the corpus and so estimates
no corpus-wide contamination rate.

The principal limitation is that physically connected ink fuses separately intended glyphs into one
component, and threshold and area parameters alone cannot split genuinely connected foreground
without an additional separation mechanism. Because the five scans were selected purposively after
scoring, this establishes a recurrent failure mode, not the cause distribution of all 168 false
negatives or a ceiling on further tuning. It motivates an instance-aware separator or learned
detector, on a new calibration/evaluation protocol.

## 7. Full-Corpus Application

The frozen method was applied to the full corpus in `2_CandidateExtraction.ipynb`. All 613 scans
(455 original, 158 supplementary) were processed successfully, producing 40,397 validated raw RGB
crops, a mean of 65.9 candidates per scan and a median of 32.0. The previous, informally configured
baseline run produced 30,804 candidates, so the frozen recall-oriented configuration yields 9,593
more (+31.1%), which is additional downstream workload rather than a second accuracy estimate. Crops
are stored as raw RGB exactly as they appear on the scan; the binarisation used for feature
extraction is applied at the next stage.

That this run applies the benchmarked method, rather than a second implementation sharing its
parameter values, is verified rather than asserted: the 30 held-out scans belong to the corpus too,
so this run produced candidates for them independently, and comparing those against the frozen
candidates the held-out figures were measured on yields an exact match on all 863 boxes and their
component areas. The §5.3 figures therefore characterise this extraction and not a near relative of
it. The run protocol records the declared method, its frozen parameters, and a SHA-256 content
digest of all 613 source images, keeping bookkeeping such as notebook filename and run identifier
out of that identity, so it detects source-content and declared-parameter changes without
invalidating a run after a repository reorganisation. It does not detect an implementation change
made under the same method identifier, as §8 makes explicit.

One methodological discrepancy connects this chapter to the next. Clustering was built on the
30,804-candidate pre-benchmark extraction rather than this frozen 40,397-candidate run, because it
began before the benchmark was finalised and redirecting it would have invalidated roughly 330
existing human review labels. The measured comparison in Chapter 2 §2.2 shows 98.8% of the crops
clustering actually uses (21,489 of 21,750) have a counterpart in this chapter's frozen output, and
that of the 9,996 candidates this run holds and the earlier one does not, 90.5% fall below the
minimum-side filter clustering already applies. The mismatch is real but small in effect.

Chapter 2 then binarises, normalises and encodes each crop with DINOv2 (Oquab et al., 2024),
excluding crops whose shorter side falls below 24 pixels as a feature-stage noise-control rule.
Applied to this frozen run that filter would retain 22,421 of the 40,397 candidates, against the
21,750 the reported clustering actually uses, since it inherits the earlier extraction; the two
figures are not interchangeable. Because the filter can also exclude genuine small fleurons, it is a
property of the feature stage rather than of the segmentation method and is not part of this
benchmark's accuracy.

## 8. Reproducibility

The paths named in this section belong to the working project rather than to this repository, which
carries the code and the written account alone. Both notebooks are stored executed, so every table
and figure reported above is readable here exactly as it was produced; the frozen tables, protocol
records and helper-script inputs they read travel with the thesis deposit instead.

The primary executable record is `1_SegmentationBenchmark.ipynb`; full-corpus extraction is
performed in `2_CandidateExtraction.ipynb`, with isolated outputs under
`1_segmentation_outputs/otsu_g3_area40_pad3_v1/`. Frozen configuration files for all three compared
methods, and the final evaluation protocol, are stored as JSON under `annotations/benchmark/`.

The held-out run is cached behind a protocol record holding the selected configuration, evaluation
task and volume membership, counts, and the byte sizes of the split and annotation tables. This
guards against many accidental changes but is not a cryptographic content lock: a same-size
annotation edit or an implementation change can escape it. The split search re-runs on every
execution and asserts that it still selects the six frozen calibration volumes. The full-corpus run
adds a SHA-256 digest over all 613 source-image contents, but its identity likewise excludes a hash
of the implementation. Archival reproduction should therefore add content hashes for the annotation,
configuration, SAM checkpoint, frozen result tables and relevant code, plus an exact environment
lock file.

Four helper scripts in `_tools/` regenerate this chapter's figures from the frozen tables without
re-running segmentation: `make_calibration_tradeoff_figure.py` redraws Figure 1.1 from the 27-point
grid, `make_method_grids_figure.py` places all three calibration grids in one comparison space, and
`make_heldout_by_book_figure.py` draws the per-book held-out spread behind its book-clustered
interval. Each reads only `annotations/benchmark/`, writes its figure beside the tables it was drawn
from, and reproduces byte for byte against the archived copy. The fourth,
`make_seg_candidate_counts.py`, draws the per-scan candidate-count figure used in the thesis from
the frozen full-corpus summaries under `1_segmentation_outputs/`.

Re-executing the notebook end to end and comparing regenerated tables against the frozen ones
confirms the recorded deterministic quantities: grid rows, per-scan metrics, held-out figures and
bootstrap intervals. Runtime measures the machine rather than the data and varies between runs; as
noted in §5.1, its unequal repeat protocol supports only an order-of-magnitude comparison.

Key package versions at execution time were Python 3.10.19, OpenCV 4.13.0, NumPy 2.2.6, SciPy
1.15.3, scikit-image 0.25.2, Ultralytics 8.4.9 and PyTorch 2.5.1+cu121, with the SAM checkpoint
(`sam_b.pt`) run locally on an NVIDIA GeForce GTX 1080 Ti and the two classical methods on CPU. The
checkpoint is identified by filename and byte size rather than by SHA-256 digest.

## References

Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and their Application.* Cambridge
University Press.

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL
Visual Object Classes (VOC) Challenge. *International Journal of Computer Vision*, 88(2), 303–338.

Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical
Society: Series B*, 69(3), 369–390.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S.,
Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). Segment Anything. *Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV)*, 4015–4026.

Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics
Quarterly*, 2(1–2), 83–97.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., et al. (2024). DINOv2:
Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research*.

Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on
Systems, Man, and Cybernetics*, 9(1), 62–66.

Pratikakis, I., Zagoris, K., Karagiannis, X., Tsochatzidis, L., Mondal, T., & Marthot-Santaniello,
I. (2019). ICDAR 2019 Competition on Document Image Binarization (DIBCO 2019). *Proceedings of the
International Conference on Document Analysis and Recognition (ICDAR)*, 1547–1556.

Rosenfeld, A., & Pfaltz, J. L. (1966). Sequential Operations in Digital Picture Processing. *Journal
of the ACM*, 13(4), 471–494.

Sauvola, J., & Pietikäinen, M. (2000). Adaptive document image binarization. *Pattern Recognition*,
33(2), 225–236.

van Rijsbergen, C. J. (1979). *Information Retrieval* (2nd ed.). Butterworths.
