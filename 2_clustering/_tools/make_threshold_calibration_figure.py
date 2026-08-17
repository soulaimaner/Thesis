"""Render the cosine-threshold calibration sweep for the thesis appendix.

The appendix previously carried this as a thirteen-row table showing every
hundredth of the sweep. The sweep itself runs from 0.80 to 0.99 in steps of 0.005,
so the table displayed a third of what was measured and none of its shape. The
curve shows the whole sweep, and shows the thing the table cannot: precision is
already near its ceiling well before the selected threshold, so the choice trades
recall away for very little purity above 0.89.

The selected threshold is marked, and the precision band carries its Wilson
interval so the reader can see that the apparent gains above 0.89 are inside the
noise of 120 calibration pairs.

Reads only `method_comparison_rerun1/similarity_threshold_calibration.csv`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CHAPTER_DIR = Path(__file__).resolve().parent.parent
assert CHAPTER_DIR.name == '2_clustering', f'Unexpected chapter folder: {CHAPTER_DIR}'
PROJECT_DIR = CHAPTER_DIR.parent
RUN_DIR = (PROJECT_DIR / '2_clustering_outputs'
           / 'all_regions_v1_minside24_dinov2_vitb14_binarized_v1_clustering_benchmark_v1'
           / 'method_comparison_rerun1')

SELECTED_THRESHOLD = 0.89

# precision is the constrained quantity and takes the accent; recall and F0.5 are
# the quantities it is traded against and share the ordinal blue ramp
PRECISION = '#eb6834'
RECALL = '#86b6ef'
F_SCORE = '#104281'
INK = '#0b0b0b'
MUTED = '#52514e'


def render(sweep: pd.DataFrame, output_path: Path) -> Path:
    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    })
    figure, axis = plt.subplots(figsize=(4.6, 2.9))

    axis.fill_between(sweep.threshold, sweep.precision_ci95_lower,
                      sweep.precision_ci95_upper, color=PRECISION, alpha=0.13,
                      linewidth=0, zorder=1)
    axis.plot(sweep.threshold, sweep.precision, color=PRECISION, linewidth=1.4,
              zorder=4, label='precision (95% Wilson band)')
    axis.plot(sweep.threshold, sweep.recall, color=RECALL, linewidth=1.4,
              zorder=3, label='recall')
    axis.plot(sweep.threshold, sweep.f0_5, color=F_SCORE, linewidth=1.4,
              linestyle=(0, (4, 1.6)), zorder=3, label='$F_{0.5}$')

    axis.axvline(SELECTED_THRESHOLD, color=INK, linewidth=0.9,
                 linestyle=(0, (2, 2)), zorder=5)
    axis.annotate(f'selected {SELECTED_THRESHOLD:.2f}', (SELECTED_THRESHOLD, 0.045),
                  textcoords='offset points', xytext=(-5, 0), ha='right',
                  fontsize=7, color=INK)

    axis.set_xlabel('Cosine similarity threshold')
    axis.set_ylabel('Calibration score')
    axis.set_ylim(0.0, 1.04)
    axis.margins(x=0.01)
    axis.grid(alpha=0.18, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(0.7)
    axis.tick_params(color=MUTED, labelcolor=MUTED, width=0.7)
    axis.legend(loc='lower left', frameon=False, handletextpad=0.45,
                borderaxespad=0.3, labelspacing=0.3)

    figure.tight_layout(pad=0.3)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return output_path


if __name__ == '__main__':
    calibration = pd.read_csv(RUN_DIR / 'similarity_threshold_calibration.csv')
    assert len(calibration) == 39, \
        f'Expected the 39-step 0.80-0.99 sweep, found {len(calibration)} rows.'
    assert (calibration.threshold == SELECTED_THRESHOLD).any(), \
        'The selected threshold is not present in the sweep.'

    written = render(calibration, CHAPTER_DIR / 'figures' / 'similarity_threshold_calibration.png')
    print('Wrote', written)
    chosen = calibration.loc[calibration.threshold == SELECTED_THRESHOLD].iloc[0]
    print(f'  at {SELECTED_THRESHOLD}: precision {chosen.precision:.3f} '
          f'[{chosen.precision_ci95_lower:.3f}, {chosen.precision_ci95_upper:.3f}], '
          f'recall {chosen.recall:.3f}, F0.5 {chosen.f0_5:.3f}')

    report_figures = PROJECT_DIR / 'report' / 'figures'
    if report_figures.is_dir():
        copied = report_figures / 'fig_clu_threshold.png'
        copied.write_bytes(written.read_bytes())
        print('Wrote', copied)
