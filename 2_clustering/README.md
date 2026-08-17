# Chapter 2: Unsupervised Clustering

This is the second of four stages in the ornament-identification pipeline:

```
   Candidate extraction  →  find candidate regions on each ornament scan
→  Clustering            →  discover which candidates are the same fleuron
   Retrieval             →  given a verified fleuron, find more of its impressions
   Detection             →  find impressions that were never extracted at all
```

**Where we are and where we are going:**

| Step | Plain-language answer |
|---|---|
| **Input** | 21,750 candidate crops from the earlier extraction after the 24-pixel filter, with no fleuron-identity labels |
| **This chapter** | Group visually similar crops so a person can review groups instead of every crop |
| **Output** | Proposed clusters, a 55-cluster review shortlist, verified catalogue identities, and a large rejected remainder |
| **Next** | Retrieval searches existing crops outside the catalogue; detection later returns to the scans for impressions with no usable crop |

Terminology (*composite ornament*, *fleuron*, *scan*, *candidate*, *crop*, *impression*) is
defined once, earlier in this thesis, and used identically across all four stages. Chapter 1
produces candidate crops but says nothing about which of them show the same design. This chapter
addresses that question, and its deliverable is a catalogue: folders of human-verified crops, one
per fleuron identity.

**Organisation.** Sections 1 to 3 state the task, the representation, and the human annotation the
stage is measured against. Sections 4 and 5 name the methods compared and the encoded selection
protocol. Section 6 reports the results, including the point at which the recorded rule was found to
lack the resolution to apply itself, and the second instrument built in its place. Sections 7 to 10
state the limits, assemble the catalogue, measure what the stage leaves behind, and record
reproduction. Notebooks 2 to 4 each exist because the step before them has a measured limit; 5 and 6
ask what the selected method's output is worth; 7 tests, after the fact, a choice made before the
stage began.

| Notebook | Question | Section |
|---|---|---|
| `1_FeatureExtraction` | How is a crop represented, and at what cost? | §2 |
| `2_PairBenchmark` | What counts as the same fleuron, and who says so? | §3 |
| `3_MethodSelection` | Which method under the encoded rule, and what could that rule decide? | §6.1–§6.3 |
| `4_MethodComparison` | Is that the right method, judged on evidence that can discriminate? | §6.4 |
| `5_ShortlistConstruction` | Which clusters are worth a reviewer's time? | §6.5 |
| `6_WithinClusterAudit` | Of the pairs the method merged, how many are wrong? | §6.6 |
| `7_RepresentationCheck` | Was the descriptor worth its cost? | §6.7 |

The catalogue of §8 is assembled by the scripts in `_tools/` rather than by a notebook.

## 1. Task, Terminology, and Selection Criterion

The question of this stage is whether visual-similarity grouping of automatically segmented
candidates can produce a clean, human-verified catalogue of recurrent printers' fleurons. The
argument for grouping rests on one observation: manual identification does not scale, and grouping
is what makes it scale.

| Task | Human decisions | Feasible |
|---|---:|---|
| Inspect every candidate crop | 21,750 | no |
| Inspect every cluster | 140 | yes |
| Inspect the filtered shortlist | 55 | yes, in an afternoon |

That is a reduction in labour of roughly two orders of magnitude, bought at a price in false merges
and non-fleuron contamination. Measuring the price is the substance of this stage. The claim
defended here is not that the method is an accurate classifier, but that it is a human-in-the-loop
workflow whose error structure is sampled and made visible.

The stage cannot be supervised. The scan-level annotations of Chapter 1 locate glyphs but do not
record which fleuron a glyph is, and no external resource records that for this material, so no
classifier can be trained and no accuracy against an external reference can be computed. Every
reference used here was manufactured by hand, in quantities small enough that the quantity itself
has to be reported. Section 3 gives those quantities and §6.3 shows what follows from them.

Writing TP for pairs of crops correctly placed in one cluster, FP for pairs placed together that
show different designs, and FN for pairs of one design left apart, the two error rates are

    precision = TP / (TP + FP)        recall = TP / (TP + FN)

and their costs are not symmetric, in the opposite direction to Chapter 1. A candidate left
unassigned is merely unused, and the retrieval stage of Chapter 3 exists to recover it. A false
merge corrupts an identity and propagates into every historical claim built on it, because
retrieval against a corrupted anchor will faithfully confirm impressions of both designs. Method
selection therefore uses

    F₀.₅ = 1.25 · precision · recall / (0.25 · precision + recall)

the precision-weighted member of the F-measure family (van Rijsbergen, 1979), which weights
precision four times as strongly as recall. Chapter 1 selects on F₂, the mirror member of the same
family: extraction is scored on recall, grouping on precision.

The notebook applies the selection rule of §5 before opening the held-out pairs, and the archived
outputs record that the choice was frozen before evaluation. What §6.3 then reports is a separate
matter: the rule was applied exactly as encoded, and the annotation available to it could not
resolve the differences it was asked to rank. Section 5 justifies each element of the rule from the
cost structure above, and reports the sweep showing that neither gate value determined the outcome.

## 2. Representation and Inputs

`1_FeatureExtraction.ipynb`

### 2.1 From Crops to Vectors

Every candidate crop is converted to grayscale, binarised independently with Otsu (1979), padded to
a square canvas rather than stretched, resized to 224 × 224 and encoded with DINOv2 ViT-B/14
(Oquab et al., 2024). Vectors are L2 normalised, so cosine similarity is the metric throughout.
Crops whose shorter side is at least 24 pixels are embedded, which leaves **21,750 vectors of 768
dimensions**, each row-aligned to a manifest so that every later step can address a crop by index.

Binarisation before encoding is a deliberate attempt to reduce a known scan-format confound. The
corpus mixes bitonal `.bmp` and continuous-tone `.jpg` scans. The diagnostic run here reads the
format from the file itself and needs no identity
labels: a probe balanced across the two formats, up to 6,000 crops each, still predicts the source
format of a binarised crop with **79.4% accuracy against a 50% baseline**. The residual is
therefore substantial. The before-and-after comparison that quantifies what binarisation bought is
reported in the methodology chapter.

A 50-dimensional PCA projection (Hotelling, 1933) is derived here and used by the density method of
§4.1, because neighbour distances concentrate in high dimension and that contrast is what a density
method depends on. The projection is re-normalised afterwards, since PCA discards components and
does not preserve vector length. Its cost is measured in §6.7. UMAP (McInnes et al., 2018) is
computed for viewing only and enters no decision.

### 2.2 The Inherited Extraction

The input is the earlier Otsu extraction rather than Chapter 1's frozen run, because clustering
began before the segmentation benchmark was finalised and redirecting it would have invalidated
roughly 330 existing human labels. The notebook measures the cost of that decision directly rather
than asserting it is small.

| Quantity | Crops |
|---|---:|
| Earlier extraction, this stage's input | 30,804 |
| Chapter 1's frozen extraction | 40,397 |
| Embedded here after the 24-pixel filter | 21,750 |
| Of those, also present in the frozen extraction at IoU ≥ 0.50 | 21,489 (**98.8%**) |
| Boxes the frozen extraction holds and this one does not | 9,996 |
| Of those, below the 24-pixel filter and so dropped here in any case | 9,051 (**90.5%**) |

The mismatch is real and its effect on this stage is small. It is disclosed again in §7 and in
Chapter 1 §7.

## 3. The Identity Annotation

`2_PairBenchmark.ipynb`

### 3.1 Composition

The corpus carries no fleuron labels, so human judgement supplies the semantic reference. Two crops
were presented side by side and labelled `same_fleuron`, `different`, `non_fleuron` or `unclear`,
where *same* denotes a recurring visual design and never proof of the same physical block. The last
label exists because a recurring border or rule can be correctly grouped by visual similarity while
having no place in a catalogue.

Three sets were drawn. They are not three attempts at one measurement: each draws from a different
population and answers a question the others cannot, which is why this chapter reports precision
more than once. Two further sets re-present pairs already drawn from them, to measure whether the
reviewer repeats themselves.

| Set | Pairs | Drawn from | Used in |
|---|---:|---|---|
| Pair benchmark | 180 | six cosine similarity bands over the whole corpus | §6.1–§6.3 |
| Disagreement benchmark | 250 | the regions where the three candidate methods treat a pair differently | §6.4 |
| Within-cluster audit | 150 | inside the shortlisted clusters, stratified by similarity to the cluster centroid | §6.6 |
| *Re-presented, 25 from the pair benchmark and 25 from the audit* | 50 | renumbered and reshuffled | §7 |
| *Re-presented from the disagreement benchmark* | 60 | renumbered, reshuffled, and the two crops exchanged | §6.4 |

**The results below rest on 580 distinct pairs, 110 of them judged twice, for 690 judgements.** They
come from one reviewer, so no inter-annotator agreement exists; the two re-presented sets measure
consistency with that reviewer instead, and are reported at §6.4 and §7.

### 3.2 The Frozen Split, and Where the Negative Evidence Sits

The 180 pair-benchmark pairs were split **120 for calibration and 60 held out** before any grouping
method was run. Sampling from high-similarity bands is what makes a pair worth a reviewer's time,
and it is also the reason the annotation contains only **16 negative pairs**, eleven of them in the
calibration split. Precision can only be observed on pairs whose true label is `different`, so that
number, not the 180, governs everything §6.2 can conclude. Section 6.3 measures the consequence.

## 4. Compared Methods

Three families were compared, chosen to span the ways a similarity geometry can be turned into
groups: a density method on a reduced projection, a two-stage graph method built to resist chaining,
and the plain thresholded graph that both were designed to improve on. All three read the same
21,750 vectors and the same cosine metric.

### 4.1 HDBSCAN on a 50-Dimensional Projection

Six settings of (`min_cluster_size`, `min_samples`), from (3, 1) to (10, 10), fitted with
`cluster_selection_method='eom'` on the 50-dimensional projection of §2.1 (Campello et al., 2013;
McInnes et al., 2017). The two parameters do different jobs: `min_cluster_size` is the smallest
group the method will agree to call a cluster, so it sets a floor on how rare a fleuron can be and
still be found at all, while `min_samples` controls how conservative the density estimate is,
buying precision with coverage. That floor reaches beyond this chapter: a fleuron appearing fewer
times than `min_cluster_size` cannot be discovered here, which is why recurrence tables are deferred
to Chapter 3 rather than built from the clustering.

### 4.2 Mutual-kNN Cores with Centroid Expansion

Eighteen combinations of core similarity {0.97, 0.98, 0.99} and expansion similarity
{0.90, 0.92, 0.94, 0.95, 0.96, 0.97}, in two stages, each stage a defence against a specific
failure. Exact cosine neighbours are computed once at k = 20 and reused by all eighteen. An edge
survives only where both crops list each other above the core similarity, since mutuality stops one
popular crop acting as a hub that pulls unrelated designs together; the connected components of that
strict graph become the cores. Every candidate is then compared against the core centroids and joins
the best one it clears the expansion threshold against. Those centroids are computed once and never
updated, so a newly assigned candidate cannot drag a cluster towards itself and no chain of near
neighbours can walk a cluster across a design boundary.

### 4.3 Connected Components on a Thresholded Graph

Connected components on a graph thresholded at a fixed similarity is the simplest member of the
family, and it was excluded at pre-registration on the assumption that transitive closure would
chain unrelated designs together. That assumption was never measured before the choice was made.
Section 6.4 tests it, at seven thresholds from 0.85 to 0.97, and the outcome is not the one the
protocol assumed.

## 5. Evaluation Protocol

The notebook encodes the selection rule and metrics before the scoring cells, and the identity
threshold is calibrated before any clustering is fitted. The second annotation instrument was
designed after the first benchmark's limitation was known, and §6.4 says so where it is introduced.

**The rule.** A configuration is eligible if its pair precision reaches **0.90** and its largest
cluster holds under **10%** of the corpus. Among eligible configurations, maximise F₀.₅, then
coverage, then recall. All four are declared as constants at the top of `3_MethodSelection.ipynb`,
above the scoring cells, so a change to any of them is visible in the notebook.

**Why each element has that shape.** The rule is derived from the cost structure stated in §1
rather than adopted by convention.

*F₀.₅* follows from the cost asymmetry of §1, and that a missed merge is recoverable is a count
rather than an argument: §9 shows 2,455 crops this method rejected are verified catalogue members.
Precision therefore outweighs recall, and β = 0.5 weights it four times over. Chapter 1 chooses F₂
for the mirror reason.

*The precision gate* is a floor on purity rather than an objective, refusing a method wrong in more
than one merge in ten however much it recovers. Its particular value carries no argument of its own,
and the sweep below shows it carries no consequence either.

*The largest-cluster gate* guards against the degenerate solution, where a method collecting the
corpus into one group drives recall towards 1 and scores well on any F-measure while producing
nothing reviewable. This is not hypothetical: §6.4 finds connected components at the identity
threshold placing 83% of the corpus in a single component, and this constraint alone rejects it. The
gate is load bearing against a candidate outside the grid though it excluded none inside it.

*The tie-breakers* prefer, among equally precise configurations, the one leaving less material
unassigned, since that is work handed to the next stage. This is the weakest of the four: coverage
and recall are near-collinear here, and §6.3 shows F₀.₅ collapsed onto recall in any case, so the
ordering never arbitrated anything.

**Neither gate value determined the outcome.** `3_MethodSelection.ipynb` §7.1 sweeps each across its
range with the other held as encoded. The rule selects the same configuration at **every precision
gate from 0.00 to 1.00**, and at every largest-cluster gate **from 0.06 upward**; only below the
selected configuration's own largest cluster of 5.1% does the ranking change. That is a stronger
guarantee than declaring the values in advance, which shows only that they were not adjusted after
the grid was seen. The sweep shows that adjusting them could not have changed the answer.

**Provenance, stated exactly.** `benchmark_protocol.json` records the pair-sampling design, 180
pairs over six similarity bands with a 120/60 split at seed 42, and is dated ahead of both the
completed labels and the frozen clustering. The numerical gates and tie-breakers are not in that
file. Nor were they in the earlier written protocol that preceded it, which committed only to the
intent of "greatest useful coverage subject to high pair precision and acceptable cluster-size
behaviour" and named no threshold; that document has been superseded by this README and is not part
of this package. The sampling design is therefore datable; the exact rule is reported as encoded and
applied before the held-out split was opened, and is not described as pre-registered.

**Scoring.** A configuration predicts `same` for a pair when both crops carry the same non-noise
cluster label. Precision, recall and F₀.₅ are computed over the pairs labelled `same_fleuron` or
`different`; `non_fleuron` pairs are scored separately as a contamination diagnostic and never enter
the primary metric, and the single pair the reviewer marked `unclear` enters neither. Coverage is
the share of the 21,750 crops assigned to any cluster.

**Three samples, three questions.** No single annotation can answer this stage's questions, because
each rate conditions on a different event. Every figure in §6 is quoted with the sample it was drawn
from.

| Sample | Estimates | Reported in |
|---|---|---|
| Pairs drawn by similarity from the whole corpus | whether a method can separate designs at all | §6.1–§6.3 |
| Pairs where the candidate methods disagree | which method is better | §6.4 |
| Pairs the selected method placed in one shortlisted cluster | how much of its reviewable output is wrong | §6.6 |

None of the three is a probability sample of the corpus, so no rate below describes the corpus.

**Uncertainty.** Binomial rates carry Wilson intervals (Wilson, 1927). The within-cluster audit is
resampled by cluster rather than by pair (Field & Welsh, 2007), since pairs from one cluster are not
independent. Rates over the stratified disagreement sample are weighted back to the composition of
the region sampled, because a pooled rate over fixed quotas measures the sampling design rather than
the corpus. Method comparisons on the same pairs use McNemar's test (McNemar, 1947) rather than
independent-sample tests; comparisons across disjoint strata use Fisher's exact test (Fisher, 1935).
Ordered trends use the Cochran-Armitage test (Cochran, 1954; Armitage, 1955) with a permutation
check.

## 6. Results

### 6.1 Threshold Calibration

A single global threshold on cosine similarity is calibrated first, before any clustering, to
establish that the representation carries identity signal at all. Thresholds from 0.80 to 0.99 in
steps of 0.005 were scored on the calibration pairs, those below 0.90 precision rejected, and the
survivor with the highest F₀.₅ selected.

The selected threshold is **0.89**, reaching precision 0.974 and recall 0.826 on calibration and
**precision 1.000, recall 0.881 on the held-out pairs**. The representation therefore carries enough
identity signal for a bare threshold to recover most matches without an evident false merge, which
is what the stage needs before a grouping method is worth building; how precisely it does so is
bounded rather than established by the five held-out negatives of §6.2, and §6.7 scores the
descriptor threshold free. The 12% of same-design pairs falling below the threshold are a floor any
method built on this similarity inherits rather than a fault it introduces. The threshold is not a
parameter of either clustering method: it is a diagnostic that a grouping method must beat to
justify its complexity, and the identity threshold reused whenever two crops must be judged the same
design.

### 6.2 Selection Under the Rule

**Twenty-four configurations** were scored on the calibration pairs alone: the six HDBSCAN settings
of §4.1 and the eighteen mutual-kNN configurations of §4.2. The rule selected **HDBSCAN with
`min_cluster_size=10` and `min_samples=5`**, calibration F₀.₅ 0.877 against the best mutual-kNN
configuration's 0.827.

The choice was frozen and the held-out split opened once.

| Split | Precision | Recall | F₀.₅ | Coverage |
|---|---:|---:|---:|---:|
| Calibration | 1.000 | 0.587 | 0.877 | 0.329 |
| Held out, opened once | 1.000 [0.851, 1.000] | 0.524 | 0.846 | 0.329 |

Recall is the informative figure of the two. The held-out precision rests on the **five** negative
pairs in that split, so its interval reaches 0.851 and it cannot distinguish a precise method from a
merely lucky one. That limitation is the subject of §6.3.

The frozen clustering holds **140 clusters covering 7,160 crops**, 32.9% of the corpus. It is never
revised after this point. Sections 6.3 and 6.4 evaluate it; they do not re-make it.

### 6.3 What the Comparison Could Decide

`3_MethodSelection.ipynb` §7 asks what that comparison was capable of resolving, using only the
grids already computed and the labels already collected. The answer is very little, and three
findings carry it.

- **Neither eligibility constraint excluded a single configuration.** All 24 passed both gates.
- **The selected configuration would have needed seven false merges out of eleven** before the
  precision gate rejected it. A method misjudging six of eleven negative pairs would still have
  passed.
- **Twenty of the twenty-four configurations produced no false merge at all**, so their precision is
  identically 1.000 and their intervals overlap completely.

**Figure 2.1: Power of the precision gate**
(`figures/eligibility_gate_power.png`). The probability that the 0.90
precision gate rejects a configuration, as a function of the true error rate it would commit on
negative pairs, given the eleven negatives the calibration split holds. The gate reaches an 80%
chance of rejection only at a true error rate near 0.70. It is a filter against methods that are
wrong most of the time, not against methods that are wrong sometimes.

With precision constant across the grid, F₀.₅ varies almost entirely with recall. **A rule written
to weight precision four times as heavily as recall therefore selected on recall.** The arithmetic
is sound and the rule was applied exactly as written; what failed is the evidence available to it.

Two questions leave this notebook, and they are not the same question. The first concerns the
**sampling frame**: a catalogue depends on how many of the pairs a method *merged* are wrong, which
similarity-sampled pairs cannot answer at any sample size. The second concerns **statistical
power**. Because a question is only asked of a winner, §6.4 settles which method is right before
§6.6 asks what that method's output is worth.

### 6.4 The Methods Compared on Evidence That Can Discriminate

**Two weighting limitations bound everything in this section.** The recovery percentages here, and
the horizontal coordinates of Figure 2.2, weight each stratum by its count in the stochastic
candidate pool the labelled pairs were drawn from, rather than by the size of the complete
disagreement region. Exact enumeration over the same eligible crops gives a different composition,
under which the ordering of the leading descriptive recovery estimates reverses. Because the design
also caps pairs drawn per cluster, a corrected weighting needs an explicit estimand and
cluster-aware uncertainty, not a substitution of region sizes. These percentages are therefore
descriptive summaries of the labelled sample and establish no population ordering of reach.

Separately, the reported precision on unilateral merges weights each pair by the full size of its
cluster rather than by that cluster's eligible members. Restricting to eligible crops preserves the
direction of the comparison, connected components remaining the more precise method, but narrows the
gap. The labelled counts themselves depend on neither weighting.

**The excluded family is tested first.** At the 0.89 identity threshold, connected components places
**83% of the corpus in a single component**, exactly the chaining the protocol predicted, and is
rejected by the largest-cluster constraint alone. The chaining subsides above 0.93. At 0.95 the
method is eligible and, under the rule exactly as written, it **outscores the selected method on
every tie-breaker**: F₀.₅ 0.916 against 0.877, on higher recall and higher coverage. Notebook 3's
grid was incomplete.

That is not a reason to change the selection, because §6.3 established that this annotation cannot
rank methods at all. A ranking produced by an instrument with no resolution is not evidence either
way, so a better instrument was built.

**The disagreement benchmark.** Two methods that group a pair the same way say nothing about which
is better, so **250 pairs were drawn from the regions where the three candidates disagree**, at
fixed quotas of 45 per single-method stratum, 30 per two-of-three stratum and 25 all-three controls,
with at most three pairs from any one cluster. One judgement therefore scores all three methods at
once. Crops were restricted to ornament-like ones by image properties alone, aspect ratio at most 3,
minimum side at least 40 pixels, Otsu ink fraction between 0.08 and 0.45, and ink contact with the
crop border at most 0.25, reducing 21,750 crops to 10,354. The filter reads pixels only, so it is
blind to every clustering and independent of the catalogue. The annotation contains **35 genuine
mismatches**, against sixteen in the whole of the pair benchmark, and they sit where the methods
actually differ.

This instrument was designed after §6.3, not before it. It evaluates a frozen choice; it does not
re-make one.

**Reliability.** Sixty of the pairs were presented again, renumbered, reshuffled and with the two
crops exchanged. The second judgement agreed with the first on **60 of 60**, on both labels.

**Precision.** When a method merges a pair no other method will:

| | pairs judged | wrong | rate | 95% CI |
|---|---:|---:|---:|---|
| HDBSCAN | 40 | 16 | **40.0%** | [24.9%, 56.7%] |
| mutual-kNN | 44 | 13 | 29.5% | [16.8%, 45.2%] |
| connected components at 0.95 | 40 | 4 | **10.0%** | [2.8%, 23.7%] |

Fisher exact *p* = 0.0038 between the two extremes, and on per-pair correctness across all 232
identity pairs HDBSCAN loses to connected components with McNemar *p* = 0.025. Against mutual-kNN it
is not distinguishable (*p* = 0.50). The pool drew clusters uniformly rather than merged pairs, so
the rates were recomputed with each pair weighted by its cluster's pair count: HDBSCAN falls to
34.4% and connected components rises to 12.7%, which narrows the gap and leaves the ordering
unchanged. HDBSCAN's errors **rise with cluster size**, from 20% below 25 members to 33% between 25
and 100 and 57% above 100.

**Fragmentation.** Precision alone can always be improved by merging less, so the complementary
quantity is measured from the same labels. Weighted back to the composition of the disagreement
region, HDBSCAN reunites **79%** of the pairs judged to show the same design, connected components
66%, mutual-kNN 47%.

| | clusters | median size | spanning ≥ 2 works | median works |
|---|---:|---:|---:|---:|
| HDBSCAN | 140 | 22 | 85.7% | 4 |
| mutual-kNN | 781 | 7 | 85.0% | 3 |
| connected components at 0.95 | 1,141 | 2 | 39.8% | 1 |

**Figure 2.2: The trade the selection rule could not see**
(`figures/precision_reach_tradeoff.png`). Left, each method placed by
the share of judged matches it reunites against its precision on unilateral merges, with Wilson
intervals: the two axes order the three methods in opposite directions. Right, the error rate on
unilateral merges by the size of the cluster the pair came from.

**The conclusion is a stated trade rather than a score.** Connected components at 0.95 buys its
precision with 1,141 clusters of median size two, 60% of which never leave a single work and so
demonstrate no recurrence at all, the one property a catalogue of recurrent ornaments cannot do
without. Mutual-kNN reaches almost as far across works as HDBSCAN and is not distinguishable from it
on precision, but reunites 47% of judged matches against 79%, so it is dominated rather than
preferred. HDBSCAN is retained on that reading, and the reading rests on the reach comparison, to
which this section's opening weighting limitation applies: under exact region enumeration the
ordering of those descriptive estimates reverses, so the retention follows from the evidence as
weighted here and is not established independently of that weighting. The recorded rule is reported
as inadequate for the task, since it scores identity pairs and nothing in it penalises
fragmentation, so on adequate evidence it points at the method that merges least.

### 6.5 Stability, Reach, and the Review Shortlist

The frozen clustering holds 140 clusters and the stage reviews 55. Two conditions decide which, both
fixed before the shortlist was drawn, and neither looking at the crops themselves.

**Stability.** The method is refitted on 25 random subsamples of 80% of the corpus. Each cluster
scores the highest Jaccard overlap (Jaccard, 1912) it reaches with any cluster of a refit, over its
own members drawn into that subsample. Mean adjusted Rand index (Hubert & Arabie, 1985) across the
refits is **0.831** (sd 0.013), so
the structure survives resampling in the aggregate, but individual clusters vary and the variation
is almost entirely a matter of size: median Jaccard **0.53** below 30 members, **0.88** between 30
and 100, and **0.91** above 100. The cutoff of 0.75 follows Hennig's (2007) convention for a valid,
stable pattern.

**Reach.** Scans and books are counted from the shelfmark encoded in each filename, so no image
content and no human judgement enters. **122 of the 140 clusters (87.1%)** appear in more than one
book, with a median of six books among those. The shortlist condition is stated at volume level,
which is the weaker of the two available readings; §6.4 reports the same clustering at work level,
where 85.7% span two or more works, and §7 gives the effect on the shortlist itself.

| Step | Clusters | Crops |
|---|---:|---:|
| Frozen clustering | 140 | 7,160 |
| Stable at 0.75 or above | 63 | 5,260 |
| And present in more than one book | **55** | **5,017** |

The notebook reproduces the shortlist the rest of the stage uses exactly: the same 55 clusters, with
membership, scan and book counts equal cluster by cluster and stability scores agreeing to four
decimals.

**The cutoff decides less than its precision suggests.** Twenty of the 140 clusters sit within 0.05
of 0.75, so the notebook sweeps it with the multi-book condition held fixed. Moving it by 0.05 in
either direction gives 47 to 65 clusters but only 4,736 to 5,353 crops, a range of about a tenth of
the shortlist. The clusters that cross the line are the small ones, which follows from the size
dependence above: they are numerous and carry few crops between them.

**Figure 2.5: What the cutoff costs**
(`figures/stability_cutoff_sensitivity.png`). Left, mean Jaccard against cluster size with the 0.75
line drawn, multi-book clusters in colour. Right, crops remaining in the shortlist as the cutoff
moves.

What the filter costs is worth stating. The 85 clusters it removes are **not asserted to be wrong**,
only unreviewed. A cluster confined to one book may hold a real ornament that simply cannot
demonstrate recurrence, and instability under resampling is a property of the sample rather than
proof of error.

### 6.6 What the Selected Method's Output Is Worth

**150 pairs** were drawn from inside the 55-cluster shortlist, which holds 5,017 of the 7,160
assigned crops. The sample is stratified by **centrality**, the similarity between a crop and its
cluster's centroid, into pairs of two central members, one central and one peripheral, and two
peripheral members. No cluster contributes more than one pair per stratum.

The sampling design matters, in three ways. It gives equal quotas to the three centrality strata.
It gives nearly equal influence to clusters, by allowing at most one pair per cluster per stratum.
And central and peripheral are the top and bottom **terciles** of centrality, so a pair involving a
middle-tercile crop is outside the frame entirely. The three strata therefore cover four ninths of
the pairs the shortlist merges, and the design is balanced rather than proportional within that
coverage. The 85 excluded clusters are also not assumed to behave like the shortlist.

**Of the 103 usable identity judgements in this deliberately balanced audit, six are false merges:
5.8% [2.2%, 12.2%]** by Wilson. Reweighted from the equal quotas to the tercile-implied composition
of the region the design does cover, the figure is 6.1%, so the balance across centrality strata is
worth about two tenths of a point. What the pooled figure cannot be scaled to is the whole
shortlist, because 65% of its within-cluster pair mass sits in a single cluster that contributes at
most three audited pairs. The stratum-specific counts below are therefore the primary result.

It is also not the error rate of the catalogue, for a reason that has nothing to do with weighting.
Every crop in the catalogue was confirmed by eye afterwards, as §8 describes. The 5.8% is the error
of the **unreviewed** output, which is what the human step exists to remove, not a defect carried
into the product.

**The errors are structured, not diffuse.**

- **Centre to edge**: 0 of 37 central pairs, 2 of 34 mixed, 4 of 32 peripheral. Cochran-Armitage
  *p* = 0.027, permutation *p* = 0.007, cluster bootstrap gap 0.125 [0.031, 0.250]. The permutation
  shuffles labels without preserving clusters and the gap bootstrap does not preserve the matched
  cluster structure, so these support the ordering rather than establish it.
- **By cluster size**: 0 of 9 pairs below 25 crops, 3 of 67 between 25 and 100, 3 of 27 above 100.
  The bands are small and their intervals overlap, so this is a suggestive descriptive direction,
  not an independent replication.
- **Contamination runs the other way**: 65% of audited pairs in clusters below 25 crops involve
  material that is not a usable ornament, against 27% between 25 and 100 and 16% above 100, with
  31.3% over all 150 pairs. Small clusters are a usability problem and large clusters an identity
  problem, and the pooled figure hides both.
- **Five of the 55 audited clusters account for every observed false merge** (6, 107, 111, 120,
  128).

**Figure 2.3: Where the errors are**
(`figures/audit_error_structure.png`). Left, the false-merge rate by
centrality stratum with Wilson intervals. Right, the two error kinds plotted against cluster size on
the same axis, running in opposite directions.

The operating rule follows: historical claims are made from central cluster members, where no error
was observed in 37 pairs, with a Wilson upper bound of 9.5%, against 12.5% observed at the
periphery. The rule bounds the error where it has been measured rather than asserting that central
members are free of it.

### 6.7 The Descriptor

Everything above rests on a choice made before the stage began. Four representations are compared on
the 150 identity pairs of the pair benchmark (134 `same_fleuron`, 16 `different`), scored by the
area under the ROC curve of their cosine similarity against the human labels (Hanley & McNeil,
1982). Each crop is preprocessed exactly as in §2.1, so the comparison is between representations
and not between preprocessing pipelines. AUC is used because it is threshold free, and because §6.3
established that this annotation cannot support threshold-level comparisons.

| Representation | Dimensions | AUC | 95% CI |
|---|---:|---:|---|
| DINOv2 ViT-B/14 | 768 | **0.878** | [0.816, 0.929] |
| DINOv2 + PCA-50 | 50 | 0.872 | [0.790, 0.941] |
| Raw pixels 32 × 32 | 1,024 | 0.841 | [0.745, 0.923] |
| HOG, 9 orientations, 16 px cells | 6,084 | 0.816 | [0.744, 0.882] |

Comparing those intervals would be the wrong test, because all four are scored on the same pairs and
their errors are correlated. The difference is therefore resampled directly, and **every paired
interval contains zero**: 0.006 [-0.051, 0.068] against PCA-50, 0.060 [-0.016, 0.137] against HOG
(Dalal & Triggs, 2005), 0.036 [-0.056, 0.138] against raw pixels.

**Figure 2.4: No advantage is distinguishable from zero**
(`figures/representation_delta_auc.png`). The paired bootstrap AUC advantage of
DINOv2 over each alternative, with 95% intervals; all three cross zero.

**DINOv2 ranks first and cannot be shown to be better.** On this annotation a 32 × 32 downsampled
bitmap is not measurably worse than a pretrained vision transformer at deciding whether two crops
show the same design. That is an absence of demonstrated advantage rather than demonstrated
equivalence: the comparison rests on the same sixteen negative pairs that could not separate
clustering methods.

**The sample is not representation-neutral, and the direction of that bias matters.** The pairs were
stratified by DINO similarity, and the high bands draw on DINO neighbours, so the benchmark is built
in DINOv2's favour. A test rigged towards one competitor on which that competitor still cannot win
makes the negative result conservative rather than doubtful: the conclusion that a cheaper
descriptor was not shown to be worse survives the bias, because removing the bias could only narrow
the gap further. What the bias does undermine is any claim that DINOv2 is *better*, which this
chapter does not make.

**PCA to 50 dimensions shows little loss**, costing 0.006 of AUC. The same reasoning applies: the
sample favours the full 768-dimensional embedding over its own projection, so 0.006 is an upper
bound on the loss rather than an estimate of it. A crop-disjoint sample selected independently of
every representation would be required to put a number on either comparison.

Nothing is revised by this notebook. The descriptor is retained because it ranks first, is already
computed, and supports the retrieval stage, and the stage states plainly that a cheaper descriptor
was not shown to be worse.

## 7. Discussion and Limitations

**What this stage establishes.** A representation that separates designs at a calibrated threshold.
A method chosen under an explicitly encoded rule, and a measurement of how little that rule's evidence
could support. A comparison of three methods on 250 judgements drawn where they disagree, with a
reliability check that agreed 60 times out of 60. A shortlist whose two filters are measured rather
than asserted. Six false merges among 103 usable judgements in a balanced shortlist audit, with the
observed errors located at cluster edges and in five identifiable groups. A catalogue of 92 verified
fleuron identities.

Five sets of limitations qualify those conclusions.

**The reference is one reviewer's.** All 690 pair judgements come from one reviewer, so no
inter-annotator agreement exists and no claim of human-level ground truth can be made. The two
re-presented sets measure self-consistency instead, and separate the two judgements the annotation
asks for. **Identity is reproducible**: 60 of 60 on the disagreement benchmark, and 36 of 36 among
the pairs the 50-pair re-review put in that class twice. **Usability is not**: 9 of those 50 changed
label between passes, every one involving `non_fleuron`, in both directions. That matters because
31.3% of audited pairs carry that label and the small-cluster finding of §6.6 rests on it.
Renumbering and reshuffling limit recognition of a previously seen pair but cannot exclude it, so
these agreements bound the reviewer's independence from their own earlier judgement from above.

**No sample is a probability sample.** All three annotations are purposive and deliberately
unrepresentative, so no rate in §6 describes the corpus, and every figure is quoted with the event
it conditions on; the 40% of §6.4 and the 5.8% of §6.6 are not competing estimates of one quantity.
The method comparison is further restricted to the 10,354 crops passing §6.4's image-property
filter, so it compares the methods on ornament-like material and says nothing about the rest.

**Leakage in the pair benchmark.** The 180 pairs were split by pair rather than by crop, so five
crops appear on both sides, touching six of the sixty held-out pairs. All five sit in positive
pairs, leaving the negative set and therefore the precision estimate untouched. Removing the six
lowers §6.1's held-out recall from 0.881 to 0.865 and its held-out F₀.₅ from 0.974 to 0.970, so the
leakage flatters the result by 0.004 of F₀.₅. It is disclosed rather than repaired, since
re-splitting would mean relabelling a benchmark whose conclusions do not depend on the fix. Future
benchmarks should split on crops.

**Book counting.** Section 6.5 states the shortlist condition at volume level, so a cluster spanning
two volumes of one edition counts as multi-book while demonstrating no recurrence across works. The
two readings are close: 87.1% multi-book at volume level (§6.5) against 85.7% spanning two or more
works (§6.4). Re-scored with work identifiers, 54 of the 55 shortlisted clusters still qualify, the
exception being a cluster whose two books are two volumes of one edition.

**What the stage does not deliver.** Verified identities, not recurrence, which is Chapter 3's
subject. It runs on the earlier extraction rather than Chapter 1's frozen configuration (§2.2). Its
descriptor is retained without demonstrated advantage over a downsampled bitmap (§6.7). And the
catalogue of §8 was assembled by moving symlinks by hand, so the result is auditable while the
sequence of operations that produced it is not.

The principal substantive limitation is the one §6.4 identifies: the recorded rule scores
identity pairs and nothing in it penalises fragmentation, so applied to adequate evidence it selects
the method that merges least, which for this task is the least useful method. The rule was not
rewritten, because rewriting a criterion with knowledge of the answer would be result-driven. It is
reported as inadequate and the method is retained on a stated trade.

**Qualified answer to the research question.** The stage demonstrates that unsupervised grouping
can reduce 21,750 crop decisions to a feasible human review and contribute verified identities to a
catalogue. It does not establish whole-catalogue purity: the audit measures the benchmarked pass,
which contributed 217 of the 7,152 crops clustering placed in the catalogue, while the remaining
6,935 come from the earlier curation and carry no purity estimate. Nor does it yet support the
reported HDBSCAN precision–reach advantage, per §6.4. Recurrence across works is deliberately
completed by Chapter 3, which finds 76 of 93 final identities in at least two works.

## 8. From Clusters to the Catalogue

**A hand-curated catalogue already existed when this chapter began.** The same crops had been
grouped with mutual-kNN and sorted by hand into classes, every crop confirmed by eye. That
verification is the expensive part and no better clustering method regenerates it, so the
benchmarked pass of §6 extended and corrected the existing catalogue rather than replacing it: one
catalogue with two contributors, not two competing ones. The curation can play that role without
circularity because it was complete before the method comparison was designed and was consulted
nowhere in §6, and no result in this chapter is scored against it.

Assembly was done by hand, so this part has scripts rather than a notebook. A notebook would report
numbers read from CSV files and imply an executable record that does not exist.

**Matching the benchmarked clusters against the existing classes.** Each of the 55 shortlisted
clusters was reduced to the mean of its crops' vectors and compared with each curated class the same
way. **37 matched a class the catalogue already held**, confirming it. **18 did not** and were
opened and inspected: some held real fleurons the first pass had never produced and became new
classes, the rest held borders, rules and damaged impressions and were discarded. Inspection also
corrected a small number of curated classes holding more than one design.

**Duplicates were not retained.** Merging two independently built groupings can file one design
under two names, so every pair of classes was ranked by centroid similarity and those above the
identity threshold reviewed by eye. The final audit asserts the outcome directly: no crop is filed
under two identities and no link is broken. Both rankings only order candidates, since a class
centroid averages over impressions of very different quality and a high score is equally consistent
with a duplicate and with two related designs from one workshop. Every decision was taken by eye.

| Contribution | Crops |
|---|---:|
| Clustering, mutual-kNN pass | 6,935 |
| Clustering, HDBSCAN pass | 217 |
| **Catalogue at the end of this stage** | **7,152** |
| Retrieval (Chapter 3) | 698 |
| Detection (Chapter 4) | 702 |
| **Catalogue now** | **8,552** |

**93 fleurons over 8,552 crops**, no broken links, and no crop filed under two identities. The
benchmarked pass contributed 217 of those crops: **four fleurons the first pass had never
produced**, and further impressions to **nine classes it already held**. Retrieval and detection
later enlarged the evidence for the same 92 identities without introducing a new one. The
ninety-third, `Fleuron_80`, came from a correction rather than a new stage: a purity audit of the
largest class found `Fleuron_1` to be two designs merged in error, and split the second out with
sixteen of its crops and five more reached from the residual pool.

The split between the two clustering passes is derived from membership of the frozen curation. The
split between the stage totals is read from
`3_retrieval_outputs/occurrence_v1/catalogue_provenance.csv`, which records one row per catalogue
crop.

**Handoff chronology.** Retrieval's main corpus run used the interim 89-class catalogue and found
7,005 of its crops in the embedding matrix. A later extension searched six identities created after
that run; the consolidated catalogue now contains 93 classes. The current 93-class total is
therefore an end-state, not the input count of the main retrieval run.

The two totals reconcile exactly, and the arithmetic is worth stating because §9 uses the first and
§8 the second:

| | Crops |
|---|---:|
| Clustering-stage crops in the catalogue when retrieval began (§9) | 7,005 |
| Added to the clustering-stage tally after that run, all of them clustered by HDBSCAN | 139 |
| Present in the catalogue but not in the 21,750-row embedding manifest | 8 |
| **Clustering's contribution to the final catalogue (§8)** | **7,152** |

The 139 are the reason §9's assigned column reads 4,550 while the final catalogue carries 4,689
clustered crops; the rejected column, 2,455, is unchanged by the extension.

## 9. What This Stage Leaves Behind

The selected method assigned **7,160 of 21,750 crops**, a third of the corpus. Whether the remaining
two-thirds holds anything worth recovering is not a matter of judgement here, because a second
grouping method was run over the same vectors and its output was reviewed by the same person.

| | HDBSCAN clustered it | HDBSCAN rejected it | |
|---|---:|---:|---:|
| **In the catalogue** | 4,550 | **2,455** | 7,005 |
| **Not in the catalogue** | 2,610 | 12,135 | 14,745 |
| | 7,160 | 14,590 | 21,750 |

Two cells carry the argument.

**2,455 crops the selected method called noise are verified fleurons.** The mutual-kNN pass reached
them on identical vectors, and a reviewer confirmed each one. What this stage discards is therefore
neither empty nor uniformly junk, and the cost of the precision-weighted rule is visible at corpus
scale rather than only in the samples of §6.4. Recovering more of that remainder is the subject of
Chapter 3, which searches it against the verified identities delivered here.

**2,610 crops the method clustered were dropped by a reviewer.** That is the corpus-scale
counterpart of the 31.3% non-fleuron rate §6.6 measured on a sample: material the method grouped
correctly by appearance and which has no place in a catalogue.

The counts are of embedded crops, so they are taken over the 21,750 vectors of §2 and read against
the catalogue as it stood when Chapter 3 began.

## 10. Reproducibility

Each notebook asserts that it runs from this folder and writes to a directory suffixed by its
`RUN_TAG`, so a re-run cannot overwrite the frozen artifacts it reads. Execution order is the
reading order, 1 to 7. Notebook 5 is the slow one, because it refits the clustering 25 times. All
sampling and every model fit are seeded at 42.

This is why the run directory holds two directories for some stages. The untagged one is the frozen
artifact read as input: `method_comparison/` is the clustering of §6.2, and the other untagged
directories hold the human judgements. The `_rerun1` directory beside it is where the corresponding
notebook wrote its own output. Notebooks 4 to 7 all read the untagged `method_comparison/`, so the
clustering this chapter reports is the one every later notebook consumes, whether or not notebook 3
is executed again.

Every human judgement is read as a frozen input and no notebook regenerates one, because re-drawing
a sample would discard completed labels that cannot be recovered. `INDEX.md` names the run directory
and lists where each judgement, and every other artifact this chapter cites, actually lives.
`figures/` is the only place this folder keeps a copy of a run-directory artifact, refreshed from
the run directory by `sync_figures.sh`.

Two single-row summary tables sit beside the notebooks rather than in the run directory, because
each is a one-off check on an input to this stage rather than a product of the clustering:

| File | What it records | Written by |
|---|---|---|
| `segmentation_config_mismatch_check.csv` | the seven counts tabulated in §2.2, quantifying the inherited-extraction mismatch | `1_FeatureExtraction.ipynb` |
| `binarisation_confound_stats.csv` | the before-and-after cosine comparison behind §2.1's binarisation decision, reported in full in the methodology chapter | a diagnostic that is not part of this package |

The six one-off computations behind §6.4 and §8, and the figure scripts, are in `_tools/`. `INDEX.md`
lists each with what it writes.

Key package versions at execution time were Python 3.10.19, NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2 (which supplies `HDBSCAN` and `PCA`), scikit-image 0.25.2, OpenCV 4.13.0,
umap-learn 0.5.11, pandas 2.3.3 and PyTorch 2.5.1+cu121, with DINOv2 ViT-B/14 run on an
NVIDIA GeForce GTX 1080 Ti.

The datable record behind §5's provenance claim is `benchmark_protocol.json`, which fixes the
sampling design. As §5 states, the numerical gates were never written down before the run, so they
are reported as encoded rather than as pre-registered.

## References

Armitage, P. (1955). Tests for linear trends in proportions and frequencies. *Biometrics*, 11(3),
375–386.

Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on
hierarchical density estimates. *Advances in Knowledge Discovery and Data Mining (PAKDD)*, 160–172.

Cochran, W. G. (1954). Some methods for strengthening the common χ² tests. *Biometrics*, 10(4),
417–451.

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *Proceedings
of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 886–893.

Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical
Society: Series B*, 69(3), 369–390.

Fisher, R. A. (1935). *The Design of Experiments.* Oliver & Boyd.

Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating
characteristic (ROC) curve. *Radiology*, 143(1), 29–36.

Hennig, C. (2007). Cluster-wise assessment of cluster stability. *Computational Statistics & Data
Analysis*, 52(1), 258–271.

Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components.
*Journal of Educational Psychology*, 24(6), 417–441.

Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.

Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*, 11(2),
37–50.

McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering.
*Journal of Open Source Software*, 2(11), 205.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection
for dimension reduction. *arXiv:1802.03426*.

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or
percentages. *Psychometrika*, 12(2), 153–157.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., et al. (2024). DINOv2:
Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research*.

Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on
Systems, Man, and Cybernetics*, 9(1), 62–66.

van Rijsbergen, C. J. (1979). *Information Retrieval* (2nd ed.). Butterworths.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal
of the American Statistical Association*, 22(158), 209–212.
