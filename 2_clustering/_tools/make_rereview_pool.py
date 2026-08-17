"""Blind re-review of the disagreement benchmark, to measure intra-rater agreement.

The 250-pair benchmark was labelled in one session, and the rate at which pairs were called
`different` rose over its second half. Whether that is a real property of the material or reviewer
drift cannot be settled by argument, so a subset is presented again, disguised, and the two
judgements of the same pair are compared.

Disguise: pairs are renumbered R001 onwards, drawn in a fresh random order, and the left and right
crops are swapped relative to the presentation the reviewer saw the first time. The images
themselves are never altered, because mirroring or rotating a crop would change the design being
judged.

Composition is deliberately enriched with the pairs that carry the result: every pair originally
called `different`, plus a sample originally called `same_fleuron` drawn from the same strata.
Enrichment inflates the apparent prevalence of disagreement, so agreement is reported separately
for each original label rather than as a single prevalence-sensitive coefficient.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TOOLS_DIR.parent.parent
BENCHMARK_DIR = (PROJECT_DIR / '2_clustering_outputs'
                 / 'all_regions_v1_minside24_dinov2_vitb14_binarized_v1_clustering_benchmark_v1')
SOURCE_DIR = BENCHMARK_DIR / 'disagreement_v1'
OUTPUT_DIR = SOURCE_DIR / 'rereview_v1'
BATCH_DIR = OUTPUT_DIR / 'rereview_batches'

RANDOM_SEED = 20260811
POSITIVES_TO_DRAW = 25
PAIRS_PER_PAGE = 10
PAIRS_PER_ROW = 2


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    key = pd.read_csv(SOURCE_DIR / 'disagreement_key.csv')
    labels = pd.read_csv(SOURCE_DIR / 'disagreement_labels.csv', keep_default_na=False)
    original = key.merge(labels[['pair_id', 'label']], on='pair_id')

    negatives = original[original.label == 'different']
    positive_pool = original[(original.label == 'same_fleuron')
                             & original.stratum.isin(negatives.stratum.unique())]
    positives = positive_pool.sample(n=POSITIVES_TO_DRAW, random_state=RANDOM_SEED)
    drawn = pd.concat([negatives, positives]).sample(frac=1.0, random_state=RANDOM_SEED + 1)
    drawn = drawn.reset_index(drop=True)
    drawn.insert(0, 'rereview_id', [f'R{i + 1:03d}' for i in range(len(drawn))])

    # Swap the sides relative to the first presentation, so the layout is not recognisable.
    drawn['rereview_left_index'] = drawn.shown_right_index
    drawn['rereview_right_index'] = drawn.shown_left_index

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    drawn.to_csv(OUTPUT_DIR / 'rereview_key.csv', index=False)
    pd.DataFrame({'rereview_id': drawn.rereview_id, 'label': '', 'notes': ''}).to_csv(
        OUTPUT_DIR / 'rereview_labels.csv', index=False)

    manifest = pd.read_csv(BENCHMARK_DIR / 'method_comparison' / 'selected_clusters.csv')
    crop_path = manifest.crop_path.to_numpy()
    rows_per_page = PAIRS_PER_PAGE // PAIRS_PER_ROW
    pages = int(np.ceil(len(drawn) / PAIRS_PER_PAGE))
    for page_start in range(0, len(drawn), PAIRS_PER_PAGE):
        page = drawn.iloc[page_start:page_start + PAIRS_PER_PAGE]
        figure, axes = plt.subplots(rows_per_page, PAIRS_PER_ROW * 2,
                                    figsize=(13.5, 2.6 * rows_per_page))
        for axis in axes.flat:
            axis.axis('off')
        for position, row in enumerate(page.itertuples(index=False)):
            grid_row, grid_column = divmod(position, PAIRS_PER_ROW)
            for offset, index in enumerate((row.rereview_left_index, row.rereview_right_index)):
                axis = axes[grid_row, grid_column * 2 + offset]
                with Image.open(PROJECT_DIR / crop_path[index]) as image:
                    axis.imshow(image.convert('RGB'))
                axis.axis('off')
            axes[grid_row, grid_column * 2].set_title(row.rereview_id, fontsize=13,
                                                      loc='left', pad=6)
        page_number = page_start // PAIRS_PER_PAGE + 1
        figure.suptitle(f'Re-review page {page_number:02d} of {pages:02d}'
                        f'  ({page.rereview_id.iloc[0]} to {page.rereview_id.iloc[-1]})',
                        fontsize=12)
        figure.tight_layout()
        figure.savefig(BATCH_DIR / f'page_{page_number:02d}.png', dpi=115, bbox_inches='tight')
        plt.close(figure)

    print(f'drew {len(drawn)} pairs '
          f'({len(negatives)} originally different, {len(positives)} originally same)')
    print(f'wrote {pages} pages to {BATCH_DIR}')


if __name__ == '__main__':
    main()
