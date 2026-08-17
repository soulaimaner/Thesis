"""Render the Otsu calibration trade-off figure from the frozen calibration grid.

The figure is the visual form of the selection rule: every configuration in the
27-point grid is placed in (candidates per scan, recall) space, so the reader can
see the frontier the pre-declared F2 rule walks along and where it stops. It reads
only `annotations/benchmark/otsu_calibration_grid.csv`, so it never re-segments a
scan and cannot disturb any frozen output.

The same code is embedded in `1_SegmentationBenchmark.ipynb` §10; this script
exists so the figure can be regenerated without executing the benchmark notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CHAPTER_DIR = Path(__file__).resolve().parent.parent
assert CHAPTER_DIR.name == '1_segmentation', f'Unexpected chapter folder: {CHAPTER_DIR}'
PROJECT_DIR = CHAPTER_DIR.parent
BENCHMARK_DIR = PROJECT_DIR / 'annotations' / 'benchmark'

SELECTED = {'gaussian_size': 3, 'minimum_area': 40, 'padding': 3}
BASELINE = {'gaussian_size': 5, 'minimum_area': 80, 'padding': 3}

# one ordinal blue ramp: minimum area is an ordered quantity, not a set of
# categories, and a single hue separated by lightness survives greyscale printing
AREA_COLOURS = {40: '#86b6ef', 80: '#2a78d6', 160: '#104281'}
ACCENT = '#eb6834'
INK = '#0b0b0b'
MUTED = '#52514e'


def render(grid_results: pd.DataFrame, output_path: Path) -> Path:
    def locate(setting):
        mask = ((grid_results.gaussian_size == setting['gaussian_size'])
                & (grid_results.minimum_area == setting['minimum_area'])
                & (grid_results.padding == setting['padding']))
        return grid_results.loc[mask].iloc[0]

    selected_row = locate(SELECTED)
    baseline_row = locate(BASELINE)

    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    })
    # built at its printed size so nothing is scaled down in the thesis
    figure, axis = plt.subplots(figsize=(4.4, 2.75))

    for minimum_area, area_group in grid_results.groupby('minimum_area'):
        axis.scatter(
            area_group.candidates_per_scan, area_group.recall_iou50,
            s=26, color=AREA_COLOURS[int(minimum_area)], edgecolor='white',
            linewidth=0.5, zorder=3, label=f'min. area {int(minimum_area)}',
        )

    axis.scatter(baseline_row.candidates_per_scan, baseline_row.recall_iou50,
                 marker='X', s=85, color=INK, edgecolor='white', linewidth=0.7,
                 zorder=5, label='exploratory baseline')
    axis.scatter(selected_row.candidates_per_scan, selected_row.recall_iou50,
                 marker='*', s=200, color=ACCENT, edgecolor='white', linewidth=0.7,
                 zorder=5, label='calibration-selected')

    # only the two decision-relevant points are labelled; the other 25 are context
    axis.annotate('5 / 80 / 3', (baseline_row.candidates_per_scan, baseline_row.recall_iou50),
                  textcoords='offset points', xytext=(-8, 4), ha='right',
                  fontsize=7, color=INK)
    axis.annotate('3 / 40 / 3', (selected_row.candidates_per_scan, selected_row.recall_iou50),
                  textcoords='offset points', xytext=(-10, 2), ha='right',
                  fontsize=7, color=ACCENT)

    axis.set_xlabel('Candidate boxes per calibration scan')
    axis.set_ylabel('Glyph recall at IoU 0.50')
    axis.margins(x=0.05, y=0.09)
    axis.grid(alpha=0.18, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(0.7)
    axis.tick_params(color=MUTED, labelcolor=MUTED, width=0.7)
    axis.legend(loc='lower right', frameon=False, handletextpad=0.35,
                borderaxespad=0.2, labelspacing=0.3)

    figure.tight_layout(pad=0.3)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return output_path


if __name__ == '__main__':
    grid = pd.read_csv(BENCHMARK_DIR / 'otsu_calibration_grid.csv')
    assert len(grid) == 27, f'Expected the 27-point Otsu grid, found {len(grid)}.'
    written = render(grid, BENCHMARK_DIR / 'otsu_calibration_tradeoff.png')
    print('Wrote', written)

    # the thesis keeps its own copy so that the report folder stays self-contained
    report_figures = PROJECT_DIR / 'report' / 'figures'
    if report_figures.is_dir():
        copied = report_figures / 'fig_seg_tradeoff.png'
        copied.write_bytes(written.read_bytes())
        print('Wrote', copied)
