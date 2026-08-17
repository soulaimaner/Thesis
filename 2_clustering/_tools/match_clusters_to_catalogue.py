"""Compare each shortlisted HDBSCAN cluster with the hand-curated fleuron classes.

Both a cluster and a class are reduced to the mean of their crops' DINOv2 vectors, normalised, and
compared by cosine similarity. A cluster whose closest class reaches the identity threshold shows a
fleuron the catalogue already held; one below it is a candidate for a new class. The output ranks
the clusters, it does not decide anything: every cluster was adjudicated by eye.

Writes 2_clustering_outputs/<run>/catalogue_v1/cluster_to_class_matches.csv
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_RUN_ID = 'all_regions_v1_minside24_dinov2_vitb14_binarized_v1'
BENCHMARK_DIR = PROJECT_DIR / '2_clustering_outputs' / f'{FEATURE_RUN_ID}_clustering_benchmark_v1'
FEATURE_DIR = PROJECT_DIR / 'feature_extraction_outputs' / FEATURE_RUN_ID
CURATION_DIR = PROJECT_DIR / 'all_regions_outputs' / 'Fleurons'
OUTPUT_DIR = BENCHMARK_DIR / 'catalogue_v1'

IDENTITY_THRESHOLD = 0.89   # calibrated in notebook 3


def normalised_features():
    features = np.load(FEATURE_DIR / 'dino_features_binarized.npy', mmap_mode='r')
    vectors = np.array(features, dtype=np.float32, copy=True)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors


def centroid(vectors, rows):
    centre = vectors[rows].mean(axis=0)
    return centre / (np.linalg.norm(centre) + 1e-9)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vectors = normalised_features()
    manifest = pd.read_csv(BENCHMARK_DIR / 'method_comparison' / 'selected_clusters.csv')
    labels = np.load(BENCHMARK_DIR / 'method_comparison' / 'selected_cluster_labels.npy')
    shortlist = pd.read_csv(BENCHMARK_DIR / 'robustness_v1' / 'hdbscan_shortlist.csv')
    row_of = {os.path.realpath(PROJECT_DIR / p): i
              for i, p in zip(manifest.feature_index, manifest.crop_path)}

    names, class_centroids = [], []
    for class_dir in sorted(p for p in CURATION_DIR.iterdir() if p.is_dir()):
        rows = [row_of[os.path.realpath(f)] for f in class_dir.iterdir()
                if os.path.realpath(f) in row_of]
        if not rows:
            continue
        names.append(re.sub(r'\s*\(\d+\)$', '', class_dir.name))
        class_centroids.append(centroid(vectors, rows))
    class_centroids = np.vstack(class_centroids)

    sizes = pd.Series(labels[labels >= 0]).value_counts()
    cluster_ids = shortlist.cluster.to_numpy()
    cluster_centroids = np.vstack(
        [centroid(vectors, np.flatnonzero(labels == c)) for c in cluster_ids])
    similarity = cluster_centroids @ class_centroids.T

    matches = pd.DataFrame({
        'cluster_id': cluster_ids,
        'crops': sizes[cluster_ids].to_numpy(),
        'closest_class': [names[i] for i in similarity.argmax(axis=1)],
        'similarity': similarity.max(axis=1),
    }).sort_values('similarity', ascending=False)
    matches['already_in_the_catalogue'] = matches.similarity >= IDENTITY_THRESHOLD
    matches.to_csv(OUTPUT_DIR / 'cluster_to_class_matches.csv', index=False)

    print(f'{len(matches)} shortlisted clusters against {len(names)} curated classes')
    print(f'  already in the catalogue      : {int(matches.already_in_the_catalogue.sum())}')
    print(f'  reviewed as possible new ones : {int((~matches.already_in_the_catalogue).sum())}')
    print(f'wrote {OUTPUT_DIR / "cluster_to_class_matches.csv"}')


if __name__ == '__main__':
    main()
