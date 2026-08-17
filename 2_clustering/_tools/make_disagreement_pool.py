"""Build the disagreement benchmark that compares the three candidate clustering methods.

Notebook 3 could not rank methods: 20 of its 24 configurations tied at precision 1.000 because
only 11 negative pairs were available, and none of them sat in the similarity region where the
clustering makes its decisions. This task fixes that by labelling where the methods actually
differ.

The three candidates are fixed, and their labels are read from frozen artifacts:

    HDBSCAN            method_comparison/selected_cluster_labels.npy   (mcs 10, ms 5)
    mutual-kNN         method_comparison/graph_grid_labels.npz         (core 0.97, expand 0.90)
    connected comps    robustness_v1/cc095_labels.npy                  (cosine 0.95)

A pair is informative when at least one method merged it and at least one did not, because such a
pair separates the methods no matter how it is labelled. Pairs merged by all three are drawn as a
small control stratum, to estimate the base rate where the methods agree and to detect a pool that
is unrepresentative.

Sampling constraints:
  * both crops must pass a shape filter, so that the reviewer spends the session on ornaments
    rather than on the rules and borders that made 31% of the within-cluster audit non-fleuron;
  * at most MAX_PAIRS_PER_CLUSTER pairs share a cluster in any method, so that intervals can be
    made robust to clustering of the sample;
  * sheets are blind: a pair identifier and two crops, nothing else. Which methods merged the pair
    lives only in the key file, which is not needed for labelling.

Outputs, under 2_clustering_outputs/<run>/disagreement_v1/:
    disagreement_pool.csv    every candidate pair found, with its method pattern
    disagreement_key.csv     the 250 drawn pairs, their pattern and provenance
    disagreement_labels.csv  empty label column, to be filled in
    disagreement_sheets/     one PNG per pair, named by pair id only
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = TOOLS_DIR.parent
PROJECT_DIR = NOTEBOOK_DIR.parent

FEATURE_RUN_ID = 'all_regions_v1_minside24_dinov2_vitb14_binarized_v1'
BENCHMARK_RUN_ID = f'{FEATURE_RUN_ID}_clustering_benchmark_v1'
FEATURE_DIR = PROJECT_DIR / 'feature_extraction_outputs' / FEATURE_RUN_ID
BENCHMARK_DIR = PROJECT_DIR / '2_clustering_outputs' / BENCHMARK_RUN_ID
METHOD_DIR = BENCHMARK_DIR / 'method_comparison'
ROBUSTNESS_DIR = BENCHMARK_DIR / 'robustness_v1'
OUTPUT_DIR = BENCHMARK_DIR / 'disagreement_v1'
SHEETS_DIR = OUTPUT_DIR / 'disagreement_sheets'

RANDOM_SEED = 20260810
TARGET_PAIRS = 250
CANDIDATES_PER_METHOD = 400_000     # sampled merged pairs per method, before filtering
MAX_PAIRS_PER_CLUSTER = 3           # per method, to limit within-cluster dependence

# Crop filter, applied to both crops of a pair. Every criterion is a property of the image alone,
# never of a clustering, so the filter cannot favour one method over another.
MAX_ASPECT_RATIO = 3.0              # rules and borders are long thin bands
MIN_SIDE_PIXELS = 40                # below this a crop is a fragment, not a design
MIN_INK_FRACTION = 0.08             # blank or near-blank crops carry no design
MAX_INK_FRACTION = 0.45             # solid blobs and dense hatching are not ornaments
MAX_BORDER_INK_FRACTION = 0.25      # ink running off the edge means the crop cuts a larger region

# How the 250 are allocated. The three "only" strata carry the comparison; the "two of three"
# strata locate which method is the odd one out; agreement pairs are the control.
STRATUM_QUOTA = {
    'only_hdbscan': 45,
    'only_graph': 45,
    'only_cc': 45,
    'hdbscan_graph_not_cc': 30,
    'hdbscan_cc_not_graph': 30,
    'graph_cc_not_hdbscan': 30,
    'all_three': 25,
}

METHODS = ('hdbscan', 'graph', 'cc')


def load_labels():
    manifest = pd.read_csv(METHOD_DIR / 'selected_clusters.csv')
    hdbscan = np.load(METHOD_DIR / 'selected_cluster_labels.npy')
    with np.load(METHOD_DIR / 'graph_grid_labels.npz') as archive:
        graph = archive['graph_core0.97_expand0.90']
    cc = np.load(ROBUSTNESS_DIR / 'cc095_labels.npy')
    assert len(manifest) == len(hdbscan) == len(graph) == len(cc), 'label vectors misaligned'
    return manifest, {'hdbscan': hdbscan, 'graph': graph, 'cc': cc}


def shape_filter(manifest):
    longest = manifest[['width', 'height']].max(axis=1)
    shortest = manifest[['width', 'height']].min(axis=1)
    aspect = longest / shortest.clip(lower=1)
    return ((aspect <= MAX_ASPECT_RATIO) & (manifest.min_side >= MIN_SIDE_PIXELS)).to_numpy()


def ink_filter(manifest, shape_ok, cache_path):
    """Ink coverage and border-contact, measured on the Otsu binarisation of each crop.

    A fleuron is a line ornament sitting in white space. A crop cut out of hatching, a solid
    initial, or a large inked area either covers too much of its own box or runs ink off its
    edges. Both are measured on the image, so the filter is blind to every clustering.
    """
    if cache_path.exists():
        stats = pd.read_csv(cache_path)
    else:
        rows = []
        for index in np.flatnonzero(shape_ok):
            with Image.open(PROJECT_DIR / manifest.crop_path.iloc[index]) as image:
                grey = np.asarray(image.convert('L'), dtype=np.uint8)
            counts = np.bincount(grey.reshape(-1), minlength=256).astype(np.float64)
            weights = counts / counts.sum()
            levels = np.arange(256)
            omega = np.cumsum(weights)
            mu = np.cumsum(weights * levels)
            valid = (omega > 0) & (omega < 1)
            between = np.zeros(256)
            between[valid] = ((mu[-1] * omega[valid] - mu[valid]) ** 2
                              / (omega[valid] * (1 - omega[valid])))
            ink = grey <= int(np.argmax(between))
            border = np.concatenate([ink[0], ink[-1], ink[:, 0], ink[:, -1]])
            rows.append({'feature_index': int(index),
                         'ink_fraction': float(ink.mean()),
                         'border_ink_fraction': float(border.mean())})
        stats = pd.DataFrame(rows)
        stats.to_csv(cache_path, index=False)
    keep = np.zeros(len(manifest), dtype=bool)
    usable = stats[(stats.ink_fraction >= MIN_INK_FRACTION)
                   & (stats.ink_fraction <= MAX_INK_FRACTION)
                   & (stats.border_ink_fraction <= MAX_BORDER_INK_FRACTION)]
    keep[usable.feature_index.to_numpy()] = True
    return keep


def sample_merged_pairs(labels, eligible, rng, n_pairs):
    """Draw pairs that this method merged, uniformly over clusters then over members."""
    assigned = np.flatnonzero((labels >= 0) & eligible)
    if len(assigned) == 0:
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(labels[assigned], kind='stable')
    assigned = assigned[order]
    cluster_ids, starts = np.unique(labels[assigned], return_index=True)
    bounds = np.append(starts, len(assigned))
    sizes = np.diff(bounds)
    usable = np.flatnonzero(sizes >= 2)
    if len(usable) == 0:
        return np.empty((0, 2), dtype=np.int64)
    picks = rng.choice(usable, size=n_pairs, replace=True)
    left = np.empty(n_pairs, dtype=np.int64)
    right = np.empty(n_pairs, dtype=np.int64)
    for position, cluster_index in enumerate(picks):
        members = assigned[bounds[cluster_index]:bounds[cluster_index + 1]]
        a, b = rng.choice(members, size=2, replace=False)
        left[position], right[position] = (a, b) if a < b else (b, a)
    return np.stack([left, right], axis=1)


def stratum_of(pattern):
    merged = [name for name, flag in zip(METHODS, pattern) if flag]
    if len(merged) == 3:
        return 'all_three'
    if len(merged) == 1:
        return f'only_{merged[0]}'
    if len(merged) == 2:
        missing = [name for name in METHODS if name not in merged][0]
        return {'cc': 'hdbscan_graph_not_cc',
                'graph': 'hdbscan_cc_not_graph',
                'hdbscan': 'graph_cc_not_hdbscan'}[missing]
    return None


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    manifest, labels = load_labels()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shape_ok = shape_filter(manifest)
    print(f'crops passing the shape filter : {shape_ok.sum():,} of {len(manifest):,}')
    eligible = shape_ok & ink_filter(manifest, shape_ok, OUTPUT_DIR / 'crop_ink_stats.csv')
    print(f'crops passing the ink filter   : {eligible.sum():,}')

    pairs = []
    for name in METHODS:
        drawn = sample_merged_pairs(labels[name], eligible, rng, CANDIDATES_PER_METHOD)
        print(f'  sampled from {name:9s}: {len(drawn):,} merged pairs')
        pairs.append(drawn)
    pairs = np.unique(np.concatenate(pairs, axis=0), axis=0)
    print(f'distinct candidate pairs       : {len(pairs):,}')

    left, right = pairs[:, 0], pairs[:, 1]
    pattern = {name: (labels[name][left] >= 0) & (labels[name][left] == labels[name][right])
               for name in METHODS}
    pool = pd.DataFrame({
        'left_index': left, 'right_index': right,
        **{f'merged_{name}': pattern[name] for name in METHODS},
        **{f'cluster_{name}': labels[name][left] for name in METHODS},
    })
    pool['n_methods_merging'] = pool[[f'merged_{n}' for n in METHODS]].sum(axis=1)
    pool['stratum'] = [stratum_of(row) for row in
                       pool[[f'merged_{n}' for n in METHODS]].to_numpy()]
    pool = pool[pool.stratum.notna()].reset_index(drop=True)

    features = np.load(FEATURE_DIR / 'dino_features_binarized.npy', mmap_mode='r')
    normalised = np.array(features, dtype=np.float32, copy=True)
    normalised /= np.linalg.norm(normalised, axis=1, keepdims=True) + 1e-9
    pool['similarity'] = np.einsum('ij,ij->i', normalised[pool.left_index.to_numpy()],
                                   normalised[pool.right_index.to_numpy()])

    print('\ncandidate pairs by stratum')
    print(pool.stratum.value_counts().to_string())

    # Draw the benchmark, capping how many pairs any one cluster may contribute.
    drawn_rows = []
    for stratum, quota in STRATUM_QUOTA.items():
        available = pool[pool.stratum == stratum].sample(frac=1.0, random_state=RANDOM_SEED)
        used = {name: {} for name in METHODS}
        for row in available.itertuples(index=False):
            keys = [(name, getattr(row, f'cluster_{name}')) for name in METHODS
                    if getattr(row, f'merged_{name}')]
            if any(used[name].get(cluster, 0) >= MAX_PAIRS_PER_CLUSTER for name, cluster in keys):
                continue
            for name, cluster in keys:
                used[name][cluster] = used[name].get(cluster, 0) + 1
            drawn_rows.append(row._asdict())
            if sum(entry['stratum'] == stratum for entry in drawn_rows) >= quota:
                break
    drawn = pd.DataFrame(drawn_rows)
    print(f'\ndrawn pairs                    : {len(drawn)} of {TARGET_PAIRS} requested')
    print(drawn.stratum.value_counts().to_string())

    # Blind the benchmark: shuffle pair order, and randomise which crop is shown on the left.
    drawn = drawn.sample(frac=1.0, random_state=RANDOM_SEED + 1).reset_index(drop=True)
    drawn.insert(0, 'pair_id', [f'D{i + 1:03d}' for i in range(len(drawn))])
    flip = rng.random(len(drawn)) < 0.5
    shown_left = np.where(flip, drawn.right_index, drawn.left_index)
    shown_right = np.where(flip, drawn.left_index, drawn.right_index)
    drawn['shown_left_index'] = shown_left
    drawn['shown_right_index'] = shown_right

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUTPUT_DIR / 'disagreement_pool.csv', index=False)
    drawn.to_csv(OUTPUT_DIR / 'disagreement_key.csv', index=False)
    pd.DataFrame({'pair_id': drawn.pair_id, 'label': '', 'notes': ''}).to_csv(
        OUTPUT_DIR / 'disagreement_labels.csv', index=False)

    crop_path = manifest.crop_path.to_numpy()
    for row in drawn.itertuples(index=False):
        figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))
        for axis, index in zip(axes, (row.shown_left_index, row.shown_right_index)):
            with Image.open(PROJECT_DIR / crop_path[index]) as image:
                axis.imshow(image.convert('RGB'))
            axis.axis('off')
        figure.suptitle(row.pair_id, fontsize=13)
        figure.tight_layout()
        figure.savefig(SHEETS_DIR / f'{row.pair_id}.png', dpi=120, bbox_inches='tight')
        plt.close(figure)
    print(f'\nwrote {len(drawn)} sheets to {SHEETS_DIR}')


if __name__ == '__main__':
    sys.exit(main())
