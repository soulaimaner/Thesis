"""Render all three candidate-generation grids in one comparison space.

The appendix previously reproduced the Otsu, Sauvola and SAM-B grids as three
tables totalling sixty-six rows. Read as tables they answer only which row won.
Placed together in (candidates per scan, recall) space they answer the question the
chapter actually asks: whether the three families occupy different regions, and
whether the selected setting of the winning family is separated from the others by
more than grid noise.

Each family keeps its own marker and hue; the configuration selected on the
calibration split is starred. Reads the three frozen calibration grids and the
three selected-configuration records, and writes nothing back to them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CHAPTER_DIR = Path(__file__).resolve().parent.parent
assert CHAPTER_DIR.name == '1_segmentation', f'Unexpected chapter folder: {CHAPTER_DIR}'
PROJECT_DIR = CHAPTER_DIR.parent
BENCHMARK_DIR = PROJECT_DIR / 'annotations' / 'benchmark'

# one hue per family, separated in lightness so the figure survives greyscale
FAMILIES = {
    'Otsu':    {'grid': 'otsu_calibration_grid.csv',    'colour': '#eb6834', 'marker': 'o',
                'rows': 27, 'offset': (-6, 8),  'align': 'right'},
    'Sauvola': {'grid': 'sauvola_calibration_grid.csv', 'colour': '#2a78d6', 'marker': 's',
                'rows': 27, 'offset': (7, -11), 'align': 'left'},
    'SAM-B':   {'grid': 'sam_calibration_grid.csv',     'colour': '#104281', 'marker': '^',
                'rows': 12, 'offset': (7, 4),   'align': 'left'},
}
INK = '#0b0b0b'
MUTED = '#52514e'


def selected_row(grid: pd.DataFrame) -> pd.Series:
    """The configuration the pre-declared rule picked: highest micro F2 at IoU 0.50."""
    return grid.loc[grid.f2_iou50.idxmax()]


def render(grids: dict, output_path: Path) -> Path:
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    })
    figure, axis = plt.subplots(figsize=(4.6, 3.0))

    for name, grid in grids.items():
        style = FAMILIES[name]
        axis.scatter(grid.candidates_per_scan, grid.recall_iou50,
                     s=20, marker=style['marker'], color=style['colour'],
                     edgecolor='white', linewidth=0.45, alpha=0.75, zorder=3,
                     label=f"{name} grid ({len(grid)})")
        best = selected_row(grid)
        axis.scatter(best.candidates_per_scan, best.recall_iou50,
                     marker='*', s=175, color=style['colour'],
                     edgecolor='white', linewidth=0.7, zorder=5)
        axis.annotate(f"{name}  $F_2$={best.f2_iou50:.3f}",
                      (best.candidates_per_scan, best.recall_iou50),
                      textcoords='offset points', xytext=style['offset'],
                      ha=style['align'], fontsize=6.8, color=style['colour'])

    axis.set_xlabel('Candidate boxes per calibration scan')
    axis.set_ylabel('Glyph recall at IoU 0.50')
    axis.margins(x=0.10, y=0.12)
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
    loaded = {}
    for name, spec in FAMILIES.items():
        grid = pd.read_csv(BENCHMARK_DIR / spec['grid'])
        assert len(grid) == spec['rows'], \
            f"Expected {spec['rows']} rows in the {name} grid, found {len(grid)}."
        loaded[name] = grid

    written = render(loaded, BENCHMARK_DIR / 'three_method_calibration_grids.png')
    print('Wrote', written)
    for name, grid in loaded.items():
        best = selected_row(grid)
        print(f'  {name:8s} best F2 {best.f2_iou50:.3f} at '
              f'{best.candidates_per_scan:.1f} candidates/scan, '
              f'recall {best.recall_iou50:.3f}')

    report_figures = PROJECT_DIR / 'report' / 'figures'
    if report_figures.is_dir():
        copied = report_figures / 'fig_seg_method_grids.png'
        copied.write_bytes(written.read_bytes())
        print('Wrote', copied)
