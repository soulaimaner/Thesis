# Recurrent Printers' Ornaments in Eighteenth-Century Printed Books

### An Unsupervised Discovery Pipeline for Catalogue Construction at Corpus Scale

Master's thesis, Ingénieur civil en informatique, École polytechnique de Bruxelles, Université
libre de Bruxelles.

This repository holds the computational work behind the thesis: the notebooks that produced every
reported result, a written account of each stage, and the scripts behind each reported figure.

## The problem

Eighteenth-century printers assembled the decorative bands at the head and foot of a page from
individual pieces of cast type, called fleurons. Because a printing house owned a finite stock of
these pieces and reused them across the books it produced, a fleuron that recurs in two books is
evidence connecting them. Identifying such recurrences is how bibliographers attribute undated or
falsely imprinted editions to a press.

Doing this by eye does not scale. A single ornament may contain dozens of fleurons, a corpus holds
hundreds of ornaments, and the same design appears at different inkings, orientations and states of
wear. The question this thesis asks is whether the identification can be made tractable: whether a
pipeline can propose which pieces are the same design, and reduce a human's task from judging every
region to confirming a manageable number of proposals.

The answer is a catalogue. Its construction, and the measurement of what each automated step costs
in error, is the work.

## The corpus

613 scans drawn from 57 digitised volumes representing 51 bibliographic works. Each scan is a
photograph of one composite ornament rather than a full book page. Volume and work identity are
decoded from the shelfmark filenames by `book_identifiers.py`, which every stage imports.
Recurrence is always counted over works, never over volumes, so that a design appearing in three
volumes of one multi-volume edition counts once.

## The four stages

Each stage exists because the stage before it has a measured limit. The chapter README in each
directory states the task, the method, the results and the limitations in full.

**1. Candidate-region extraction** ([`1_segmentation/`](1_segmentation/)) asks where on a scan a
fleuron might be. Three method families were calibrated on six volumes and the winner applied once
to a held-out split fixed in advance. Otsu thresholding with connected-component extraction was
selected and reached F₂ 0.708 on 30 held-out scans from 18 held-out volumes. The chapter makes no
claim that Otsu is significantly more accurate than Sauvola: six calibration volumes cannot resolve
that difference, and the case for retaining it rests on candidate count and cost as well as on F₂.

**2. Clustering** ([`2_clustering/`](2_clustering/)) asks which candidates are the same design.
21,750 crops were embedded with DINOv2 and grouped so that a reviewer could judge clusters rather
than individual crops, reducing the review task from 21,750 decisions to a 55-cluster shortlist.
The chapter measures what that reduction costs, using a labelled benchmark built on the region where
the candidate methods disagree.

**3. Retrieval** ([`3_retrieval/`](3_retrieval/)) asks where else a known design appears. Ranking
14,745 embedded crops outside the catalogue against verified identities produced 696 reviewer
confirmed proposals, of which 693 remain in the consolidated catalogue. A pre-registered success
criterion was set before the run and was not met; the chapter reports that outcome rather than
restating the criterion afterwards.

**4. Detection** ([`4_detection/`](4_detection/)) asks what extraction never found at all. Single
fleuron detectors were trained for four target designs and run over all 613 scans. Of 835 reviewed
candidates, 702 were confirmed, and 644 of those had been lost before any crop reached the
catalogue. This measures the recall ceiling that stage 1 imposes on everything downstream.

## What the work produced

A human-verified catalogue of **93 fleuron classes over 8,552 crops**, in which **76 of the 93
designs recur across two or more bibliographic works**. The catalogue is the thesis's central
artefact, and the recurrence evidence built on it is what answers the bibliographic question.

Two limitations qualify reported numbers and are stated here so they are not missed. In Chapter 2
§6.4, the recovery percentages weight each stratum by its count in the stochastic candidate pool the
labelled pairs were drawn from rather than by the size of the complete disagreement region, and
under exact enumeration the ordering of those descriptive estimates reverses. In Chapter 3 §9, the
occurrence tables identify works by reproducible filename-derived identifiers that have not been
resolved to verified bibliographic titles.

## Vocabulary

Used identically across all four chapters. The human review labels in the data (`same_fleuron`,
`non_fleuron`) follow the same convention.

| Term | Meaning |
|---|---|
| **composite ornament** | The assembled decorative unit: headpiece, tailpiece, bandeau. One scan in this corpus is one composite ornament. |
| **fleuron** / **element** | One individual piece of type composing it. One catalogue class is one fleuron. |
| **scan** | One corpus image. These are photographs of individual ornaments, not book pages; median 560 × 348 px. |
| **candidate** | A region the segmentation stage proposes as possibly containing a fleuron. |
| **crop** | A candidate cut out as an image file, and the unit the feature extractor embeds. |
| **impression** | One printed occurrence of a fleuron: one crop a human confirmed belongs to a catalogue class. |
| **ornament** (bare) | The research area and the decorative material in general. Never a catalogue unit. |

## How to read this repository

Read the chapters in order. Within a chapter, notebooks are numbered in execution order, and the
chapter README is the way in. Chapter 2 additionally carries an [`INDEX.md`](2_clustering/INDEX.md)
naming the exact artifact behind every claim it makes.

**The notebooks are stored with their outputs intact.** Every table, figure and printed result can
therefore be read here exactly as it was produced, without installing or executing anything.

What is not distributed here: the corpus scans, the candidate crops, the DINOv2 embeddings, the
fleuron catalogue, the trained detector weights, and the frozen result trees the four stages wrote.
These are source material and bulk derived data rather than the account of the work, and the
catalogue in particular is composed of crops cut from the source scans and so is deposited with the
thesis under restricted consultation instead. No reported number depends on their presence.

The consequence is that the scripts under each `_tools/` directory are reference rather than
runnable: most read a result tree that is not included. They are here because the code behind a
reported artifact belongs with the account of that artifact. `requirements.txt` records the pinned
environment, Python 3.10.19, that every result was produced under.
