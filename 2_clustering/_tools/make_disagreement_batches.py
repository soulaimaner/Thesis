"""Render the 250 disagreement pairs as batch sheets, ten pairs to a page.

Same blind material as the per-pair sheets, laid out so a reviewer can label a page at a time.
Each cell shows one pair, captioned with its identifier only. Nothing about which methods merged
the pair appears anywhere on the page.
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
OUTPUT_DIR = BENCHMARK_DIR / 'disagreement_v1'
BATCH_DIR = OUTPUT_DIR / 'disagreement_batches'

PAIRS_PER_PAGE = 10
PAIRS_PER_ROW = 2


def main():
    key = pd.read_csv(OUTPUT_DIR / 'disagreement_key.csv')
    manifest = pd.read_csv(BENCHMARK_DIR / 'method_comparison' / 'selected_clusters.csv')
    crop_path = manifest.crop_path.to_numpy()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    rows_per_page = PAIRS_PER_PAGE // PAIRS_PER_ROW
    for page_start in range(0, len(key), PAIRS_PER_PAGE):
        page = key.iloc[page_start:page_start + PAIRS_PER_PAGE]
        figure, axes = plt.subplots(rows_per_page, PAIRS_PER_ROW * 2,
                                    figsize=(13.5, 2.6 * rows_per_page))
        for axis in axes.flat:
            axis.axis('off')
        for position, row in enumerate(page.itertuples(index=False)):
            grid_row, grid_column = divmod(position, PAIRS_PER_ROW)
            for offset, index in enumerate((row.shown_left_index, row.shown_right_index)):
                axis = axes[grid_row, grid_column * 2 + offset]
                with Image.open(PROJECT_DIR / crop_path[index]) as image:
                    axis.imshow(image.convert('RGB'))
                axis.axis('off')
            centre = axes[grid_row, grid_column * 2]
            centre.set_title(row.pair_id, fontsize=13, loc='left', pad=6)
        page_number = page_start // PAIRS_PER_PAGE + 1
        figure.suptitle(f'Page {page_number:02d} of {int(np.ceil(len(key) / PAIRS_PER_PAGE)):02d}'
                        f'  ({page.pair_id.iloc[0]} to {page.pair_id.iloc[-1]})', fontsize=12)
        figure.tight_layout()
        figure.savefig(BATCH_DIR / f'page_{page_number:02d}.png', dpi=115, bbox_inches='tight')
        plt.close(figure)
    print(f'wrote {int(np.ceil(len(key) / PAIRS_PER_PAGE))} pages to {BATCH_DIR}')


if __name__ == '__main__':
    main()
