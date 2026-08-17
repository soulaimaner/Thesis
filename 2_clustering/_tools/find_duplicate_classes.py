"""Rank pairs of catalogue classes by centroid similarity, to surface possible duplicate fleurons.

Two folders holding the same design would corrupt every recurrence count built on the catalogue.
This script computes a centroid per class and scores every pair, producing a queue for review.

It nominates and does not decide. A class centroid averages over impressions of very different
quality, so a high score is equally consistent with a duplicate and with two related designs from
one workshop. Every merge in the catalogue was confirmed by eye before being made.

Writes 2_clustering_outputs/<run>/catalogue_v1/class_pair_similarity.csv
"""

import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_RUN_ID = 'all_regions_v1_minside24_dinov2_vitb14_binarized_v1'
BENCHMARK_DIR = PROJECT_DIR / '2_clustering_outputs' / f'{FEATURE_RUN_ID}_clustering_benchmark_v1'
FEATURE_DIR = PROJECT_DIR / 'feature_extraction_outputs' / FEATURE_RUN_ID
CATALOGUE_DIR = PROJECT_DIR / 'Fleurons' / 'Fleurons_v2_plus_retrieval'
OUTPUT_DIR = BENCHMARK_DIR / 'catalogue_v1'

IDENTITY_THRESHOLD = 0.89
REPORT_TOP = 20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = np.load(FEATURE_DIR / 'dino_features_binarized.npy', mmap_mode='r')
    vectors = np.array(features, dtype=np.float32, copy=True)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    manifest = pd.read_csv(BENCHMARK_DIR / 'method_comparison' / 'selected_clusters.csv')
    row_of = {os.path.realpath(PROJECT_DIR / p): i
              for i, p in zip(manifest.feature_index, manifest.crop_path)}

    names, centroids, crop_counts = [], [], []
    for class_dir in sorted(p for p in CATALOGUE_DIR.iterdir() if p.is_dir()):
        rows = [row_of[os.path.realpath(f)] for f in class_dir.iterdir()
                if os.path.realpath(f) in row_of]
        if not rows:
            continue
        centre = vectors[rows].mean(axis=0)
        names.append(re.sub(r'\s*\(\d+\)$', '', class_dir.name))
        centroids.append(centre / (np.linalg.norm(centre) + 1e-9))
        crop_counts.append(len(rows))
    centroids = np.vstack(centroids)
    similarity = centroids @ centroids.T

    pairs = pd.DataFrame(
        [{'class_a': names[i], 'class_b': names[j],
          'crops_a': crop_counts[i], 'crops_b': crop_counts[j],
          'centroid_similarity': float(similarity[i, j])}
         for i, j in combinations(range(len(names)), 2)]
    ).sort_values('centroid_similarity', ascending=False)
    pairs.to_csv(OUTPUT_DIR / 'class_pair_similarity.csv', index=False)

    above = int((pairs.centroid_similarity >= IDENTITY_THRESHOLD).sum())
    print(f'{len(names)} classes, {len(pairs):,} pairs scored')
    print(f'  pairs above the {IDENTITY_THRESHOLD} identity threshold: {above}')
    print(f'  (a queue for review, not a list of duplicates)')
    print(pairs.head(REPORT_TOP).to_string(index=False))
    print(f'wrote {OUTPUT_DIR / "class_pair_similarity.csv"}')


if __name__ == '__main__':
    main()
