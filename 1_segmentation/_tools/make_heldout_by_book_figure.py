"""Render the per-book held-out result from the frozen evaluation tables.

The appendix previously carried this as a nineteen-row table. A table states the
eighteen book-level scores but hides the one property the reader needs: how widely
they disperse around the pooled value. That dispersion is the reason every held-out
interval in Chapter 1 is bootstrapped over books rather than over scans, so the
figure shows the spread directly, with the pooled estimate and its book-clustered
interval behind the points.

Point area encodes how many evaluation scans a book contributes, because a book
represented by one scan and a book represented by six are not equally informative
and the eye should not weigh them alike.

Reads only `annotations/benchmark/otsu_final_evaluation_by_book.csv` and
`otsu_final_evaluation_book_bootstrap_ci.csv`, so it cannot disturb any frozen
output or reopen the held-out split.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CHAPTER_DIR = Path(__file__).resolve().parent.parent
assert CHAPTER_DIR.name == '1_segmentation', f'Unexpected chapter folder: {CHAPTER_DIR}'
PROJECT_DIR = CHAPTER_DIR.parent
BENCHMARK_DIR = PROJECT_DIR / 'annotations' / 'benchmark'

# the same ink used by the calibration trade-off figure, so the two read as one pair
BOOK_BLUE = '#2a78d6'
ACCENT = '#eb6834'
INK = '#0b0b0b'
MUTED = '#52514e'


def render(by_book: pd.DataFrame, pooled: pd.Series, output_path: Path) -> Path:
    ordered = by_book.sort_values('f2_iou50').reset_index(drop=True)

    plt.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    })
    figure, axis = plt.subplots(figsize=(4.6, 3.4))

    # pooled estimate and its book-clustered interval sit behind the books
    axis.axvspan(pooled.lower_95, pooled.upper_95, color=ACCENT, alpha=0.13,
                 zorder=0, label='pooled 95% CI (book bootstrap)')
    axis.axvline(pooled.estimate, color=ACCENT, linewidth=1.1, zorder=1,
                 label=f'pooled $F_2$ = {pooled.estimate:.3f}')

    positions = range(len(ordered))
    axis.scatter(ordered.f2_iou50, positions,
                 s=18 + 13 * ordered.scans, color=BOOK_BLUE, edgecolor='white',
                 linewidth=0.6, zorder=3, label='book (area = scans)')

    axis.set_yticks(list(positions))
    axis.set_yticklabels(ordered.book, fontsize=6.4)
    axis.set_xlabel('Held-out $F_2$ at IoU 0.50, by book')
    axis.set_xlim(0.0, 1.0)
    axis.margins(y=0.03)
    axis.grid(axis='x', alpha=0.18, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(0.7)
    axis.tick_params(color=MUTED, labelcolor=MUTED, width=0.7)
    # the books sort low-left to high-right, so the upper left is the only clear corner
    axis.legend(loc='upper left', frameon=False, handletextpad=0.35,
                borderaxespad=0.3, labelspacing=0.35)

    figure.tight_layout(pad=0.3)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return output_path


if __name__ == '__main__':
    books = pd.read_csv(BENCHMARK_DIR / 'otsu_final_evaluation_by_book.csv')
    intervals = pd.read_csv(BENCHMARK_DIR / 'otsu_final_evaluation_book_bootstrap_ci.csv')
    pooled_f2 = intervals.set_index('metric').loc['f2_iou50']

    assert len(books) == 18, f'Expected 18 held-out books, found {len(books)}.'
    assert pooled_f2.bootstrap_unit == 'book', 'Interval is not clustered by book.'

    written = render(books, pooled_f2, BENCHMARK_DIR / 'otsu_final_evaluation_by_book.png')
    print('Wrote', written)
    print(f'  {len(books)} books, F2 from {books.f2_iou50.min():.3f} to '
          f'{books.f2_iou50.max():.3f}, pooled {pooled_f2.estimate:.3f} '
          f'[{pooled_f2.lower_95:.3f}, {pooled_f2.upper_95:.3f}]')

    report_figures = PROJECT_DIR / 'report' / 'figures'
    if report_figures.is_dir():
        copied = report_figures / 'fig_seg_heldout_by_book.png'
        copied.write_bytes(written.read_bytes())
        print('Wrote', copied)
