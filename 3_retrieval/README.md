# Chapter 3: Retrieval Against a Verified Catalogue

This is the third of four stages in the ornament-identification pipeline:

```
   Candidate extraction  →  find candidate regions on each ornament scan
   Clustering            →  discover which candidates are the same fleuron
→  Retrieval             →  given a verified fleuron, find more of its impressions
   Detection             →  find impressions that were never extracted at all
```

**Where we are and where we are going:**

| Step | Plain-language answer |
|---|---|
| **Input** | Verified catalogue identities and 14,745 embedded crops outside the catalogue |
| **This chapter** | Rank those existing crops against known identities for human confirmation |
| **Output** | 696 reviewer-confirmed retrieval proposals, of which 693 remain in the consolidated catalogue |
| **Stopping point** | Retrieval cannot see an impression for which segmentation produced no usable crop |
| **Next** | Detection returns to all 613 scans and proposes locations independently of crop extraction; Notebook 3 then recounts the completed catalogue |

Terminology (*composite ornament*, *fleuron*, *scan*, *candidate*, *crop*, *impression*) is
defined once in the [root README](../README.md) §2 and used identically across all four stages.
Chapter 2 delivered a human-verified catalogue of ornament identities and, in its §9, a measured
limitation: the selected method assigned only 32.9% of candidates, and a second grouping method run
on the same vectors reached 2,455 crops it had rejected, every one of them later confirmed as a
genuine catalogue member. The remainder is therefore known to be productive. This chapter asks how much more
of it can be recovered once verified identities exist to search with.

**How the chapter is organised.** Sections 1–2 state the question and what the stage inherits.
Section 3 records an evaluation protocol fixed before any retrieval was run, which is the
methodological point of the chapter, together with the one change made to it and the reason. Section
4 reports the corpus run, including a prospectively specified criterion that was not met. Section 5
reports a closed-world self-consistency check. Section 6 records a follow-up search over new or
revised class definitions.
Section 7 delivers the occurrence evidence the pipeline exists to produce. Sections 8–10 qualify and
reproduce the result.

| Notebook | Question | Section |
|---|---|---|
| `1_BestMemberRetrieval` | Which uncatalogued crops are impressions of a known catalogue design? | §3, §4 |
| `2_HoldOutValidation` | Can known catalogue members be reassigned among competing classes? | §5 |
| `3_OccurrenceTables` | Which catalogue designs recur, across how many works, and what did retrieval add? | §7 |

Bibliographic identity follows the corpus-wide rule in `book_identifiers.py` at the project root:
occurrence counts are made over **works**, not digitised volumes.

## 1. Task and the Argument for the Stage

The question this chapter answers is:

> Given a catalogue of verified identities, can similarity ranking recover further impressions
> efficiently enough for human review, and does it meet the prospectively specified 0.90 precision
> criterion for autonomous acceptance?

Clustering and retrieval solve different problems, which is what licenses a more aggressive operating
point here. Clustering had no labels and had to infer both the number of identities and their
membership from density alone; the safe policy under that ignorance is to group where density is
unambiguous and refuse otherwise, and Chapter 2 §9 measures what that refusal cost. Retrieval begins
with the harder half already answered: each catalogue class is a verified exemplar set, so asking
whether a crop belongs to a *known* design is a question with a reference to check against.

That the remainder still holds real catalogue members is a count rather than a conjecture. Chapter 2
§9 reports 2,455 crops in the verified catalogue that the selected clustering rejected and the
mutual-kNN pass reached on identical vectors. The question here is how many more can be found
without inspecting all of them.

**Two kinds of missing impression require different remedies, and conflating them would misstate
what any recall figure means.**

| Kind | Where it is | Remedy |
|---|---|---|
| The crop exists but nothing grouped it | among the 14,745 unassigned crops | retrieval, this chapter |
| The fleuron was never cut from the scan | not in the candidate set at all; segmentation recall was 0.746 | detection, Chapter 4 |

This chapter addresses the first only.

### 1.1 Stage-transition audit

The **clustering→retrieval** handoff is logical and traceable: the main run used the interim
89-class verified catalogue (7,005 catalogue crops present in the feature manifest) as anchors and
searched the 14,745 residual embedded crops. Retrieval changes the question from unsupervised
discovery to assignment against known identities; it is not a second clustering pass.

The **retrieval→detection** boundary is useful but not perfectly disjoint. Of Chapter 4's 702
confirmed detector recoveries, 365 had no extracted region, 164 came from merged regions and 44 from
fragments, all clear localisation failures. But 71 had regions removed by the 24-pixel size filter
and 58 had adequate regions that were never curated, and those 129 fall partly inside blind spots
the declared small-crop retrieval variant or further curation could have addressed. Detection is
therefore a complementary recovery stage, not evidence that everything it found was intrinsically
unreachable by retrieval.

The proposed **detection→retrieval** composition has encouraging but limited evidence: similarity
matching re-identified all 103 known-overlap `Fleuron_74` detector confusions after excluding the
overlapped catalogue crop. It has not been tested on the 365 genuinely unextracted recoveries or on
all 702 detector-derived crops, and the detector notebook reads the frozen 103-row result without
containing the code that generated it. It supports the architecture on those confusion locations and
does not yet validate an end-to-end system on new ones.

## 2. What Retrieval Inherits

### 2.1 Four measured blind spots

Each is a quantified limitation of an earlier stage rather than a suspicion, and each suggests what
retrieval should be able to recover.

| Blind spot | Evidence | Why retrieval can address it |
|---|---|---|
| Small fleurons | `MIN_SIDE ≥ 24` discarded 9,054 of 30,804 candidates | the discarded set can be searched directly |
| Rotated impressions | asymmetric designs lose most rotated instances at the 0.90 operating threshold, one class falling to 0.00 at 180° | cosine can be max-pooled over dihedral transforms |
| Rare fleurons | `min_cluster_size = 10` makes designs appearing fewer than ten times undiscoverable | matching against an exemplar has no minimum |
| Degraded impressions | a single strict threshold misses genuine matches (Chapter 2 §6.1) | ranking, reviewed top-down, does not require a threshold |

The rotation item reaches beyond recall. The thesis problem statement claims fleurons recur across
orientations, and no stage of the pipeline currently handles rotation; the dihedral variant is
declared in §3 and remains unrun, which is recorded in §9 rather than left implicit.

### 2.2 The candidate pool is residual, and the composition says how much

The pool is every embedded crop not in the catalogue when this chapter ran: **14,745** of the
21,750. That is not the same set as the 14,590 crops clustering left unassigned, and the difference
is not bookkeeping. The catalogue is a hand-curated artefact drawn from two grouping passes, so some
crops the clustering assigned are outside it and some it rejected are inside it. Chapter 2 §9 gives
the four-way split; this chapter searches its bottom row.

| | crops | what it is |
|---|---:|---|
| never grouped | 12,135 | the clustering rejected it and no reviewer picked it up |
| grouped, then dropped | 2,610 | the clustering placed it in a cluster and a reviewer removed it |

The second row makes this a selected, difficult pool. A proposal drawn from it asks the reviewer to
overturn a judgement they have already made once, on a crop they have already seen. Its shortlist
precision cannot be generalised to an unfiltered population or treated as a formal lower bound,
because the unexamined remainder is not a random sample of the collection (Buckley & Voorhees,
2004). §5 therefore reports a separate constructed-population check rather than an adjustment.

### 2.3 Catalogue correctness precedes catalogue completeness

Searching with a corrupted anchor propagates the corruption: retrieval against a class that
secretly holds two designs will confirm impressions of both, and every count built on that class
inherits the error. Correctness is therefore a precondition of this chapter rather than a part of
it, and it belongs to Chapter 2, which owns the catalogue and reports the audit in its §8.

One finding from that pass bears directly on this chapter's method and is carried into §3.1: a
class centroid is a poor query for a class holding real variation in inking and wear.

## 3. Protocol, Prospectively Specified Before the Run

Chapter 2's principal methodological weakness was a benchmark designed after the method it
evaluated. This chapter fixes its evaluation first, and the protocol below is the record of that
commitment.

**Method.** Each catalogue class supplies exemplars in the existing L2-normalised DINOv2 space
(Oquab et al., 2024). Every crop not already in the catalogue is scored by cosine similarity against
the classes and assigned its best-matching class and score: vector-space retrieval in the standard
sense (Salton & McGill, 1983).

**Query variants**, declared in advance and to be reported separately: plain matching; dihedral
max-pooling over four transforms; and a search extended to the crops discarded by `MIN_SIDE ≥ 24`.

**Evaluation.** For each variant, sample across score bands, render blind review sheets showing the
candidate beside the class it matched, and label `same_fleuron`, `different`, or `non_fleuron`.
Precision at k is the headline (Manning, Raghavan & Schütze, 2008). Recall against the corpus is not
estimable, because the true number of impressions of any fleuron is unknown, and will not be
claimed.

**Acceptance threshold.** Chosen on a calibration split of the review labels and applied once to a
held-out split, split on crops rather than on pairs, which is the error identified in the Chapter 2
benchmark.

**Pre-declared success criterion.** The pass is worth including in the thesis if it recovers
impressions at precision ≥ 0.90 at its chosen operating point, whatever the yield. A low yield at
high precision is a valid result and will be reported as such.

**Reporting rule.** Recovered crops enter the catalogue only after human confirmation. Retrieval
proposes and the reviewer disposes, so the catalogue remains a human-verified artefact throughout.

**Implementation status.** The executed corpus run covered the plain best-member variant and
exported the top 40 globally assigned candidates per class above 0.80. It did **not** execute the
planned score-band sampling, three-way labels, calibration/held-out threshold selection, dihedral
variant, or small-crop variant: review was binary (kept/deleted), and cumulative thresholds were
examined on the same reviewed proposals. The 0.90 criterion is therefore a prospectively specified
target that the observed run did not demonstrate, not the result of the full planned threshold-
selection protocol. Calling this a formal pre-registration would require an externally timestamped
record; none is stored in this folder.

### 3.1 One change, made before the run and disclosed

The protocol above specified matching against class **centroids**. It was replaced with
**best-member matching**, in which a candidate's score is its maximum cosine against any single
member of the class.

The evidence was already in hand from the correctness pass. A class that legitimately contains
variation in inking, wear and contrast has a centroid that sits between its modes and close to
nothing in particular, so true matches can rank below false ones; the catalogue-correctness pass of
Chapter 2 recorded one confirmed match ranking ninth by centroid and first by best-member. The same
reasoning appears in Chapter 2 §8, where centroid comparison was
adequate for detecting near-duplicate classes and inadequate for larger and messier ones.

The change was made before any retrieval was run, which is why it is a protocol amendment rather
than a result-driven one, and it is recorded here for that reason. The output directory retains the
name `centroid_match_v1` from the original protocol; the method it holds is best-member matching
throughout.

## 4. The Corpus Run

**Retrieval is measured twice in this chapter, on two different populations.** The figures must not
be treated as estimates of one common quantity.

| | reported quantity | measured on | reading |
|---|---:|---|---|
| §4, the corpus run | **0.297** | 2,234 reviewed proposals from the residual pool | observed reviewer-confirmation rate for this shortlist |
| §5, the hold-out | **0.997** | withheld members of verified classes | optimistic closed-world assignment precision / self-consistency |

Neither is an external estimate of corpus-wide retrieval accuracy. The first establishes observed
review yield on the real shortlist; the second checks whether ordinary catalogue members can be
assigned among competing known classes.

`1_BestMemberRetrieval.ipynb` scored all **14,745** crops outside the catalogue. The 40
highest-scoring candidates per class above cosine 0.80 were exported as symlink folders for human
review (**2,234 candidates across 82 classes**) and reviewed by deleting what did not belong.

| Score band | Proposed | Confirmed | Precision | 95% Wilson |
|---|---:|---:|---:|---|
| 0.80–0.85 | 280 | 22 | 0.079 | [0.052, 0.116] |
| 0.85–0.88 | 252 | 30 | 0.119 | [0.085, 0.165] |
| 0.88–0.90 | 273 | 43 | 0.158 | [0.119, 0.205] |
| 0.90–0.92 | 552 | 138 | 0.250 | [0.216, 0.288] |
| 0.92–0.95 | 700 | 292 | 0.417 | [0.381, 0.454] |
| 0.95–1.00 | 177 | 139 | **0.785** | [0.719, 0.839] |
| **All** | **2,234** | **664** | **0.297** | [0.279, 0.317] |

Precision rises monotonically with score and the intervals of the top and bottom bands are disjoint.

**This table is confounded, and the confound is a deviation from the protocol.** §3 specified blind
review, but the implementation exported each candidate as a symlink named `001_sim0959_…`, so the
reviewer saw both rank and similarity score while deciding. Clear matches are less vulnerable to the
cue, but for ambiguous crops the score was visible information, and any tendency to resolve doubt in
its favour would produce exactly the monotone gradient reported. The correlation between score and
confirmation is therefore partly self-fulfilling and is not independent validation of the ranking.

§5 supplies separate evidence that the score controls retention of ordinary catalogue members with
no human in the loop, recall falling from 0.997 to 0.719 as the threshold rises from 0.80 to 0.95
while assignment precision stays flat. It does **not** independently calibrate confirmation
probability in the residual pool. The band table remains a description of observed review burden and
yield at each score range, which is what §4.2 uses it for. A future run should strip the score from
the exported filenames and randomise order within each class folder, which costs nothing.

### 4.1 The prospectively specified criterion was not met

§3 declared the pass worth reporting if it recovered impressions at precision ≥ 0.90 at its
operating point. Cumulative precision as the threshold tightens:

| Threshold | Proposed | Confirmed | Precision | 95% Wilson |
|---|---:|---:|---:|---|
| ≥ 0.90 | 1,429 | 569 | 0.398 | [0.373, 0.424] |
| ≥ 0.95 | 177 | 139 | 0.785 | [0.719, 0.839] |
| ≥ 0.955 | 92 | 78 | 0.848 | [0.761, 0.907] |
| ≥ 0.965 | 12 | 11 | 0.917 | [0.646, 0.985] |

These are in-sample cumulative summaries, because the planned calibration/held-out threshold split
was not executed. The bar is nominally reached only at 0.965, where 12 candidates survive, and there the Wilson
(1927) interval runs from 0.646 to 0.985, so twelve candidates cannot establish that the bar was
met at all. **As an automatic thresholded classifier, retrieval fails the declared criterion**, and
that is reported as the chapter's headline negative result rather than repaired by moving the bar.

### 4.2 What the run does support: review efficiency, measured on the real pool

Reviewing 2,234 ranked proposals, **15.2% of the 14,745-crop residual pool**, yielded 664 verified
recoveries: **one recovery per 3.4 proposals inspected**. Dividing the full pool by those same 664
recoveries gives 22.2 crops per recovery only under a fixed-yield counterfactual that assumes the
unreviewed pool contains no further positives; it is not an observed random-review baseline.

**This is the chapter's primary operational result on the pool retrieval actually faces.** It
measures the yield of the selected shortlist, not the causal gain over random or exhaustive review;
§5's hold-out answers a different question on a population that is not the residual pool.

For a pipeline whose design is that the machine proposes and the human disposes, review efficiency
is the property that matters operationally, and precision at a fixed threshold was the wrong
criterion to have declared. That judgement is post-hoc and reported as such.

This is the second occasion in this thesis where a prospectively specified criterion was not met and
a different justification was adopted afterwards, the first being Chapter 2's selection rule,
retained on a stated trade rather than on its score. Both are disclosed at the point of use, and the
pattern is worth naming: prospective specification here is doing its intended work, which is to make
the gap between expectation and outcome visible and arguable, not to guarantee that expectations
were correct.

### 4.3 Sixteen classes recovered nothing

Of the 82 classes searched, 16 had every proposal rejected. Whether these are designs the
representation cannot match, designs whose remaining impressions are too degraded to confirm, or
classes whose scores are systematically misleading is not established, and §9 records it as
outstanding.

## 5. Closed-World Self-Consistency Check

The 0.297 of §4 measures a method, a residual pool and a single-reviewer decision process.
`2_HoldOutValidation.ipynb` probes the matching rule on a constructed pool that is not residual;
it does not by itself identify which factor caused the corpus-run result.

**Design.** Using the final, expanded catalogue, 30% of the members of every class with at least five members are withheld at
random as the test queries; the remaining 70% forms the reference exemplar set; each withheld crop is scored by best-member matching
and assigned the class of its best match. Ground truth is known for every withheld crop, so no
review is required and the catalogue is read but never written. The split is repeated 20 times, and
all **75 eligible classes**, 7,798 crops, are used rather than a sample of classes, which removes
any question of how a sample was chosen.

| Threshold | Mean retrieved per split | Precision | Recall | Recall range over 20 splits |
|---|---:|---:|---:|---|
| 0.80 | 2,342 | 0.997 | 0.997 | [0.995, 0.998] |
| 0.85 | 2,338 | 0.997 | 0.995 | [0.993, 0.997] |
| 0.90 | 2,308 | 0.997 | 0.983 | [0.979, 0.986] |
| 0.92 | 2,231 | 0.998 | 0.950 | [0.938, 0.961] |
| 0.95 | 1,684 | 0.9999 | 0.719 | [0.699, 0.730] |
| 0.965 | 1,039 | 1.000 | 0.444 | [0.433, 0.459] |

**Precision does not move materially across the operating range: when best-member matching
retrieves a crop, it almost always assigns it to the right class.** The 0.95 row is reported to four
decimal places because rounding it to three would state a perfect result: two misassignments survive
there across the 33,677 crops retrieved over 20 repeats, and none at all above it. What the
threshold buys or spends is recall, 0.983 at 0.90, 0.950 at 0.92, and then a fall to 0.719 at 0.95,
where roughly a quarter of genuine members are left behind. The narrow spread across 20 splits shows
that the result is not peculiar to one crop-level partition of this catalogue.

**The two figures do not carry equal weight, because of how the classes were built.** A crop is in
its class because a similarity method in this same embedding space put it there and a reviewer kept
it, so the test population is selected by the criterion under test.

*Precision is more informative than recall here, but remains internal.* It answers whether a
withheld crop is closer to a member of its own class than to any of the other 74. The competing
classes were built the same way, so nothing in the construction settles that, and the 152
misassignments of §5.1 show it is not settled in practice. Catalogue selection and crop-level
splitting still make the figure optimistic, since same-scan and same-work siblings can land on
opposite sides of a split.

*Recall does not survive it.* An impression resembling nothing in its class could never have joined
it, and so can never be withheld from it. The recall figures describe recovery of easy members while
the residual pool is made of the other kind, and are reported for the shape of the threshold trade,
not as a prediction of corpus recovery.

Neither figure supersedes the other. Best-member matching assigns withheld known-class members at
0.997 here; 0.297 of the selected residual-pool proposals were confirmed. The difference is
consistent with strong pool and selection effects but estimates neither causally. The two panels of
`occurrence_v1/figures/method_vs_pool.png` place §4's band table and this one side by side with
their conditioning stated, and are the figure form of this argument.

**A second asymmetry compounds the first.** A withheld crop leaves its siblings in the reference
set, some of them other impressions of the same block and occasionally from the same scan.
Recovering it is easier than finding an impression no member of the class closely resembles, which
is precisely what the residual pool demands. A scan- or work-disjoint evaluation and macro-per-class
reporting are still needed for stronger validation.

### 5.1 The errors are confined to a small set of designs

152 misassignments occur across the 20 repeats, from roughly 2,340 withheld crops per repeat. They
involve **34 pairs of classes**, and the ten worst account for 69% of them:
`Fleuron_17`/`Fleuron_35`,
`Fleuron_1`/`Fleuron_4`, `Fleuron_4`/`Fleuron_79`, `Fleuron_1`/`Fleuron_3` and `Fleuron_16`/
`Fleuron_48` lead the list (`occurrence_v1/figures/retrieval_error_concentration.png`).

Retrieval does not fail diffusely; it fails on an enumerable set of designs the representation
cannot separate. Chapter 2 reaches the same limitation by a different route: its confusable-designs
figure shows crop pairs whose embeddings exceed the identity threshold and which a reviewer judged
to be different designs, differing in nothing but a lobe count or an interior element. Two methods,
evaluated by two different procedures on two different samples, fail on the same *kind* of design,
which locates the limitation in the shared DINOv2 representation rather than in either algorithm:
silhouette survives the embedding, interior detail and countable structure do not.

The consequence for this chapter is narrow and specific. Any historical claim resting on one of
these design pairs requires the two classes to be checked against each other by eye before a count
is published. Designs outside the list were not flagged by this particular test; they are not
thereby certified as confusion-free, especially for the 17 classes excluded by the minimum-size
rule.

## 6. Follow-up Search After Catalogue Revision

After the main run, three identities were added (`Fleuron_69`, `Fleuron_78`, `Fleuron_79`), two
labels were materially redefined (`Fleuron_35`, `Fleuron_9`), and `Fleuron_63` was searched again
because the global-best rule had surfaced no proposal for it above 0.80. The frozen follow-up holds
**147 proposal rows across 6 target classes (146 unique crops), of which 32 unique crops were
confirmed**, a row-level confirmation rate of 0.218.

This is not a strictly identical repeat of the main procedure. The main run assigns each crop to
one global best class; the follow-up is class-specific and one crop appears under two target
classes. Its candidate-generation code is no longer present in this folder, so the notebook reads
the frozen candidates and reconstructs the review labels but cannot regenerate the proposals. The
32 confirmations remain valid retrieval-assisted catalogue additions, but the 0.218 is descriptive
and not directly comparable as a controlled extension of the 0.297 corpus run.

**Confirmed recoveries across both passes: 696.**

One accounting detail matters for anyone recounting these figures from the directories. The review
folders hold 41 symlinks that were never proposed by either run, 39 in the main review directory
and 2 in the extension, placed there during curation from other sources. Every figure in this
chapter counts only candidates that these runs proposed and a reviewer then confirmed, so a naive
count of surviving symlinks overstates the yield by exactly those links.

## 7. Occurrence Across Works

`3_OccurrenceTables.ipynb` produces the summary evidence the pipeline exists to support, from
frozen artifacts only. Every catalogue crop is first assigned a provenance, 7,152 placed by
clustering and hand curation, 698 reached by retrieval, 702 recovered by
Chapter 4's detector, identified through the `confirmed` columns of the review tables rather
than by counting symlinks, for the reason given in §6. Two corrections to earlier accounting
emerged and are recorded here. Detector-derived crops occupy **four** classes, `Fleuron_72`
(494), `Fleuron_73` (121), `Fleuron_2` (54) and `Fleuron_74` (33), not `Fleuron_2` alone as
previously stated, so the occurrence counts exclude detection crops for every class rather than
special-casing one. And 693 of the 696 confirmed recoveries survive in the catalogue: the three
absentees are proposals for `Fleuron_64` whose links later curation replaced with better
impressions of the same design from the same scan, so no occurrence count is affected, but 696
should be read as confirmations at review time rather than as surviving members.

Occurrences are counted over bibliographic works, with volumes retained as a separate column
(`occurrence_v1/occurrence_by_class.csv`). A second long-form table,
`occurrence_v1/occurrence_by_class_work.csv`, exposes the work IDs behind each class count. These
IDs are reproducible filename-derived identifiers, not yet verified bibliographic titles.

All **93 classes are single ornamental designs** and are treated consistently as fleurons in this
thesis. Numeric and letter-based class labels are identifiers only; they do not define different
object types.

| Quantity | Value |
|---|---:|
| Fleurons spanning two or more works | **76 of 93** |
| Median works per catalogue class | 4 |
| Widest reach | `Fleuron_16`, 23 of the 51 works |
| Work-occurrences added by retrieval | 44 across 29 classes |
| Classes recurrent **only** because of retrieval | 6 classes |

This is the recurrence evidence Chapter 2 §1 deferred to this chapter, now measured on the
catalogue after all fold-ins rather than on clustering output alone
(`occurrence_v1/figures/occurrence_reach.png`). Retrieval's historical value is part of the
measurement: without this chapter, six designs' tables would show a single work and record no
reuse at all.

The primary reach column excludes detector-derived crops so all 93 classes are compared under the
same upstream pipeline. The sensitivity column including detection changes no class's recurrent /
non-recurrent status, but it does widen the observed reach of three detector targets:
`Fleuron_72` from 9 to 13 works, `Fleuron_73` from 14 to 16, and `Fleuron_74` from 12 to 13. This is
why detection adds catalogue completeness and, for selected targets, some reach even though it does
not change the 75-of-92 threshold count.

Four qualifications bound the table, all inherited and stated rather than new. The counts are
**lower bounds**: segmentation recall was 0.746, retrieval is not exhaustive, and Chapter 4
demonstrates impressions exist that were never extracted. Counts for classes carrying a
`confusable_with` flag (the 34 hold-out pairs of §5.1, plus six further pairs of close catalogue
classes noted while assembling Chapter 2's confusable-designs figure) are conditional on the flagged
pair being checked by eye. The three look-alike
identifier families are counted as separate works pending bibliographic confirmation, which can
overstate reach if a family proves to be one work. And no chronological reading is attached, because five volumes carry no
year and the filename years have not been confirmed against catalogue records; dates belong to
the per-design case studies, which remain outstanding (§9).

## 8. Discussion and Limitations

**The candidate pool is residual.** The 0.297 is the confirmation rate of a selected shortlist from
that pool, while §5 is an optimistic closed-world self-consistency check. Neither is a lower or upper
bound in a statistical sense, and neither alone characterises retrieval on unseen corpus material.

**Recall against the corpus is not estimable**, for the reason given in §3, so no corpus-wide recall
figure is claimed. §5 supplies recall for a closed-world sample only, where the denominator is known
by construction.

**The earlier manual curation has no metrics.** Substantial recovery happened before this stage was
recognised as one, performed as curation rather than as a measured experiment, with no acceptance
threshold recorded and no error rate. Four catalogue snapshots survive, and diffing them establishes
how every crop entered the catalogue, which Chapter 2 §9 reports. What they cannot recover is the
sequence of decisions inside the curation: which class boundaries were merged or split, and on what
evidence.

**Two of the three declared query variants were not run.** Dihedral max-pooling and the small-crop
search were declared in §3 and remain unexecuted, so the blind spots they address are documented but
not closed. The planned calibration/held-out threshold selection and three-way review labels were
also not executed. This part of the prospective protocol is incomplete, not merely unsuccessful.

**The review was not blind to the similarity score**, contrary to the protocol, because the exported
filenames encoded each candidate's rank and score. The consequence is confined but real: the
precision-by-band table of §4 cannot serve as independent evidence that the ranking is informative,
since the reviewer could see the ranking. It may also have affected borderline keep/delete decisions,
so catalogue membership from this pass is single-reviewer, score-visible verification rather than
blind ground truth. §5 avoids reviewer cueing but remains an internal self-consistency check.

**Best-member scores are class-size dependent unless controlled.** Taking a maximum over more
exemplars gives large classes more opportunities to produce a high score. The main shortlist spans
anchor classes from 1 to 1,929 crops, but this chapter does not report a capped-exemplar, medoid or
macro-per-class comparison. The claim that low-yield classes fail because they are small is therefore
a hypothesis, not an established mechanism.

**The Wilson intervals treat proposal decisions as independent.** Proposals are clustered within
classes and scans, and the same reviewer labels all of them, so the displayed intervals are
descriptive binomial intervals rather than cluster-robust uncertainty. A scan- or class-clustered
bootstrap is needed for population-level inference.

**Visual ambiguity is intrinsic rather than a review failure**, and two situations were handled
differently. Where a crop is not legible (partial fleurons, touching ink, heavy inking) it cannot
be judged and was excluded; that is the segmentation ceiling propagating forward rather than a
question about identity. Where crops are legible but the designs are nearly identical, image
evidence alone cannot separate the same physical block from a copy, a recast, or a different block
cut to a common pattern. This is why the label is defined throughout as *same visual design* and
never *same physical block*, why ambiguous cases were left out of classes rather than forced to a
decision, and why §5.1's list of confusable pairs is a precondition on historical claims.

**Claims this chapter supports:** that retrieval against verified anchors yields reviewer-confirmed
impressions from the residual crop pool; that the catalogue is more complete after retrieval than
before; that the matching rule reaches 0.997 precision with 0.983 recall at cosine 0.90 in a
crop-level closed-world self-consistency check; and that 76 of 93 fleurons recur across two or more
works under the counting rules of §7, six evidenced as recurrent only by this chapter's
recoveries. **Claims it does not
support:** any corpus-wide recall estimate; that recovered impressions come from the same physical
block; external predictive precision; a causal estimate of ranking efficiency; or that retrieval
found all remaining instances.

## 9. Further Work

One item determines how completely this chapter answers the historical question it sets.

**The "which books" result awaits bibliographic completion.** The long-form table of §7 exposes
class-to-work identifiers, but those identifiers are not yet resolved to verified titles and
printers, and no per-design case study with source-page evidence accompanies them. As it stands the
chapter establishes whether and how many designs recur, and identifies the works involved only at
the level of the corpus identifier rather than in a form a reader can use directly.

The remaining items bear on methodological validity.

The **unexecuted parts of the protocol**, listed in §8, would each close a measured blind spot. The
rotation variant additionally bears on a claim made in the thesis problem statement.

The **retrieval evaluation needs a stronger independent check**: blind re-review of a stratified
sample, scan- or work-disjoint hold-out, macro-per-class results, and an exemplar-count control for
best-member matching. The current review and hold-out are useful operational and internal checks,
not external validation.

The **16 classes that recovered nothing** have not been examined. Distinguishing designs the
representation cannot match from designs whose remaining impressions are illegible would sharpen
§4.3 from an observation into a finding.

The **curation decisions** have no surviving record. Crop-level provenance is settled by the
snapshots, but which class boundaries were merged or split during curation, and on what evidence, is
not recoverable from what remains. Only a re-audit of the class boundaries against the scans would
give that work a measured account.

For any per-design case study, restrict cluster-derived evidence to core members following the
false-merge gradient measured in Chapter 2 §6.6, check flagged confusable classes by eye, and verify
each cited book against catalogue records. Three identifier families may collapse to fewer works and
five volumes carry no year, so no chronological claim can rest on filenames alone.

## 10. Reproducibility

`_tools/make_class_work_matrix.py` rebuilds the class-by-work appendix table of the thesis directly
from `occurrence_v1/occurrence_by_class.csv` and `occurrence_by_class_work.csv`, asserting the two
against each other so the appendix cannot disagree with the summary it is drawn from. It runs from
the package root and needs nothing beyond the frozen tables.


Three notebooks record the main run, the self-consistency check and the final occurrence products.
Each resolves paths relative to this folder. Notebook 3 is conceptually a post-detection
consolidation step: it must run only after Chapter 4's reviewed crops have been folded into the final
catalogue, despite its location and number inside Chapter 3.

| # | Notebook | Input → output |
|---|---|---|
| 1 | `1_BestMemberRetrieval.ipynb` | catalogue + features → `3_retrieval_outputs/centroid_match_v1/` |
| 2 | `2_HoldOutValidation.ipynb` | catalogue + features → `3_retrieval_outputs/holdout_v1/` |
| 3 | `3_OccurrenceTables.ipynb` | post-detection catalogue + frozen review labels → `3_retrieval_outputs/occurrence_v1/` |

```
3_retrieval_outputs/centroid_match_v1/                  corpus run, 89 classes, 2026-08-01
├── retrieval_scores.csv        all 14,745 candidates with best class, score, matched exemplar
├── review_shortlist.csv        the 2,234 exported, with the confirmed column after review
├── review/                     82 class folders; surviving symlinks are confirmed recoveries
└── review_LABELS_BACKUP/       copy of the reviewed state

The two review folders are not part of the submitted package: their symlinks are absolute paths
into the crop store and would not resolve elsewhere. Their outcome is carried by the `confirmed`
column of `review_shortlist.csv`, which marks exactly the 664 surviving proposals.

3_retrieval_outputs/best_member_match_v1_gap_classes/   §6, frozen follow-up over new/revised classes
3_retrieval_outputs/holdout_v1/                         §5, repeated hold-out and confusion analysis
3_retrieval_outputs/occurrence_v1/                      §7, provenance, class and class-work tables, figures
```

**Figures.** Three figures carry the chapter.

| Figure | Section | File |
|---|---|---|
| The method against the pool it was measured on | §4–§5 | `occurrence_v1/figures/method_vs_pool.png` |
| Error concentration in confusable design pairs | §5.1 | `occurrence_v1/figures/retrieval_error_concentration.png` |
| Occurrence reach across works | §7 | `occurrence_v1/figures/occurrence_reach.png` |

**Human decisions are never overwritten.** The review folders carry their labels in their contents:
a candidate is confirmed if its symlink survived review. `1_BestMemberRetrieval.ipynb` therefore
refuses to export over a review directory that already exists alongside its shortlist, and prints
the instruction to raise `RUN_TAG` instead; re-scoring reads the surviving links rather than
regenerating them. `2_HoldOutValidation.ipynb` writes to a guarded output
directory and touches no catalogue file at all. `3_OccurrenceTables.ipynb` reads the frozen review
labels and the catalogue links, writes only `occurrence_v1/`, and derives provenance from the
`confirmed` columns rather than from surviving symlinks, so it is immune to the counting trap of §6.

The follow-up in §6 is only partially reproducible: its frozen `candidates.csv` and reviewed labels
are retained, but the code that generated its 147 proposal rows is not, so the reported figures can
be re-derived from the stored proposals while the proposals themselves cannot be regenerated.

The cross-stage 103/103 detection to retrieval result is likewise supported by a stored artifact
rather than by a calculation this package reproduces. That artifact and the detector stage that
produced it belong to Chapter 4 and are outside the scope of these notebooks; the result is
reported here as inherited evidence and is not recomputed.

The hold-out experiment of §5 is a fresh, seeded run rather than a reproduction of an earlier one.
An unrecorded version of this experiment over twelve sampled classes was carried out before the
analysis was written up, and its class sample and seed were not preserved; the version reported here
covers all 75 eligible classes across 20 seeded splits and supersedes it. It is seeded at `20260806`
rather than at the `42` used elsewhere in the thesis, deliberately, so that its splits are drawn
independently of any earlier run of the same experiment. Its figures are close to
the earlier ones on recall and slightly lower on precision, which is the expected consequence of
searching 75 classes rather than 12, since a larger catalogue offers more opportunities to confuse.

## References

Buckley, C., & Voorhees, E. M. (2004). Retrieval evaluation with incomplete information.
*Proceedings of the 27th Annual International ACM SIGIR Conference on Research and Development in
Information Retrieval*, 25–32.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.*
Cambridge University Press.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., et al. (2024).
DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning
Research*.

Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval.* McGraw-Hill.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal
of the American Statistical Association*, 22(158), 209–212.
