"""Four pairs of crops the descriptor scores as one design and a reviewer judged as two.

The figure is an illustration, not a measurement. Each panel shows two crops side by side
with the cosine similarity of their DINOv2 embeddings, computed from the two crops alone.
All four sit above the 0.89 identity threshold calibrated in `3_MethodSelection.ipynb`,
and all four were labelled `different` by the reviewer.

No rate is computed here and none should be: the pairs were located rather than sampled.

Paths resolve relative to this file, so it can be invoked from anywhere:
    python 2_clustering/_tools/make_confusable_examples_figure.py

Unlike the other figure scripts in this folder, this one does NOT run from the submission
package alone: it opens the crop images themselves, under `all_regions_outputs/crops/`,
which are excluded for size (root README §6). It fails on the first missing crop.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_RUN_ID = 'all_regions_v1_minside24_dinov2_vitb14_binarized_v1'
BENCHMARK_DIR = PROJECT_DIR / '2_clustering_outputs' / f'{FEATURE_RUN_ID}_clustering_benchmark_v1'
SOURCE_DIR = BENCHMARK_DIR / 'validation_v1'
REPORT_FIGURES = PROJECT_DIR / 'report' / 'figures'

# Chosen for legibility and for showing four distinct failures rather than one repeated:
# lobe count at 4 against 3, at 7 against 6, an interior square present or absent, and
# two variants of one motif family.
PAIR_IDS = ['H083', 'H071', 'H008', 'H095']
IDENTITY_THRESHOLD = 0.89
CANVAS = 220


def square(crop_path, size=CANVAS):
    """Pad to a square canvas rather than stretching, as the feature stage does."""
    with Image.open(PROJECT_DIR / crop_path) as image:
        grayscale = image.convert('L')
        width, height = grayscale.size
        side = max(width, height)
        canvas = Image.new('L', (side, side), 255)
        canvas.paste(grayscale, ((side - width) // 2, (side - height) // 2))
        return np.asarray(canvas.resize((size, size), Image.Resampling.LANCZOS))


def main():
    key = pd.read_csv(SOURCE_DIR / 'hard_negative_key.csv')
    labels = pd.read_csv(SOURCE_DIR / 'hard_negative_labels.csv')
    pairs = key.merge(labels[['pair_id', 'label']], on='pair_id').set_index('pair_id')

    selected = pairs.loc[PAIR_IDS]
    assert (selected.label == 'different').all(), 'a chosen pair is not labelled different'
    assert (selected.similarity >= IDENTITY_THRESHOLD).all(), \
        'a chosen pair falls below the identity threshold, which is the point of the figure'

    figure = plt.figure(figsize=(11, 6.2))
    outer = figure.add_gridspec(2, 2, wspace=0.14, hspace=0.24)
    for position, pair_id in enumerate(PAIR_IDS):
        row = selected.loc[pair_id]
        inner = outer[position // 2, position % 2].subgridspec(1, 2, wspace=0.04)
        for column, side in enumerate(['left', 'right']):
            axis = figure.add_subplot(inner[0, column])
            axis.imshow(square(row[f'{side}_crop_path']), cmap='gray', vmin=0, vmax=255)
            axis.axis('off')
            if column == 0:
                axis.text(1.05, 1.06, f'cosine {row.similarity:.2f}', transform=axis.transAxes,
                          ha='center', va='bottom', fontsize=14)

    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_FIGURES / 'fig_clu_confusable.png'
    figure.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'wrote {output_path}')
    print(selected[['similarity', 'label']].round(3).to_string())


if __name__ == '__main__':
    main()
