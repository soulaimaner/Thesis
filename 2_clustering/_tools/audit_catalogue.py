"""Audit the fleuron catalogue and report which stage contributed each crop.

The merge that produced the catalogue was carried out by hand, so it has no executable record. What
can be checked is the result, and this script does that on demand: the class and crop counts, the
absence of broken links, and the absence of any crop filed under two fleurons.

It also reads the catalogue back by contributor. The stage that added each crop is taken from
`3_retrieval_outputs/occurrence_v1/catalogue_provenance.csv`, which carries one row per catalogue
crop, and the pre-retrieval crops are split between the two clustering passes by whether they are
present in the frozen mutual-kNN curation. Link names are not used to infer provenance: retrieval
added some crops under the plain naming convention, so a name-based rule misattributes them.

Writes 2_clustering_outputs/<run>/catalogue_v1/catalogue_contributions.{csv,png}
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_RUN_ID = 'all_regions_v1_minside24_dinov2_vitb14_binarized_v1'
BENCHMARK_DIR = PROJECT_DIR / '2_clustering_outputs' / f'{FEATURE_RUN_ID}_clustering_benchmark_v1'
CURATION_DIR = PROJECT_DIR / 'all_regions_outputs' / 'Fleurons'
CATALOGUE_DIR = PROJECT_DIR / 'Fleurons' / 'Fleurons_v2_plus_retrieval'
PROVENANCE_PATH = (PROJECT_DIR / '3_retrieval_outputs' / 'occurrence_v1'
                   / 'catalogue_provenance.csv')
OUTPUT_DIR = BENCHMARK_DIR / 'catalogue_v1'

EXPECTED_FLEURONS = 93
EXPECTED_CROPS = 8552
ORDER = ['mutual-kNN pass', 'HDBSCAN pass', 'retrieval', 'detection']


def read_classes(root):
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = re.sub(r'\s*\(\d+\)$', '', class_dir.name)
        for entry in class_dir.iterdir():
            rows.append({'class_name': name, 'class_dir': class_dir.name,
                         'link_name': entry.name, 'target': os.path.realpath(entry),
                         'broken': entry.is_symlink() and not entry.resolve().exists()})
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curation = read_classes(CURATION_DIR)
    catalogue = read_classes(CATALOGUE_DIR)
    curated_targets = set(curation.target)

    # Provenance is read from the record the retrieval stage wrote, one row per catalogue crop,
    # rather than inferred from link names. Inference is unreliable here: retrieval added some
    # crops under the plain naming convention, and guessing from the name misattributes those.
    provenance = pd.read_csv(PROVENANCE_PATH)
    catalogue = catalogue.merge(provenance[['class_dir', 'link_name', 'origin']],
                                on=['class_dir', 'link_name'], how='left', validate='one_to_one')
    assert catalogue.origin.notna().all(), 'the provenance record does not cover every crop'

    curated_names = {os.path.basename(target) for target in curated_targets}
    catalogue['contributor'] = np.where(
        catalogue.origin != 'pre-retrieval', catalogue.origin,
        np.where(catalogue.link_name.map(lambda n: n.split('__')[-1]).isin(curated_names)
                 | catalogue.target.map(lambda t: t in curated_targets),
                 'mutual-kNN pass', 'HDBSCAN pass'))

    assert catalogue.class_name.nunique() == EXPECTED_FLEURONS, 'fleuron count changed'
    assert len(catalogue) == EXPECTED_CROPS, 'crop count changed'
    assert not catalogue.broken.any(), 'the catalogue holds a broken link'
    assert not catalogue.duplicated('target').any(), 'a crop is filed under two fleurons'

    counts = catalogue.contributor.value_counts()
    table = pd.DataFrame({'crops': [
        counts['mutual-kNN pass'], counts['HDBSCAN pass'],
        counts['mutual-kNN pass'] + counts['HDBSCAN pass'],
        counts['retrieval'], counts['detection'], len(catalogue)]},
        index=['clustering, mutual-kNN pass', 'clustering, HDBSCAN pass',
               'catalogue at the end of the clustering chapter',
               'retrieval', 'detection', 'catalogue now'])
    table.to_csv(OUTPUT_DIR / 'catalogue_contributions.csv')

    from_hdbscan = set(catalogue[catalogue.contributor == 'HDBSCAN pass'].class_name)
    curated_classes = set(curation.class_name)

    print(f'{catalogue.class_name.nunique()} fleurons, {len(catalogue):,} crops, '
          f'no broken links, no crop filed twice\n')
    print(table.to_string())
    print(f'\nnew fleurons found by the HDBSCAN pass : {len(from_hdbscan - curated_classes)}  '
          f'{sorted(from_hdbscan - curated_classes)}')
    print(f'earlier fleurons it added crops to     : {len(from_hdbscan & curated_classes)}')

    figure, axis = plt.subplots(figsize=(6, 4.2))
    bottom = 0
    for name, colour in zip(ORDER, ['#b9c6d4', '#1f4e79', '#eb6834', '#f0a882']):
        axis.bar(0, counts[name], bottom=bottom, color=colour, width=0.55, label=name)
        axis.text(0, bottom + counts[name] / 2, f'{counts[name]:,}', ha='center', va='center',
                  fontsize=9, color='white' if name == 'HDBSCAN pass' else 'black')
        bottom += counts[name]
    chapter_total = counts['mutual-kNN pass'] + counts['HDBSCAN pass']
    axis.axhline(chapter_total, color='#333333', linestyle='--', linewidth=1)
    axis.annotate(f'end of the clustering chapter, {chapter_total:,} crops',
                  (0.30, chapter_total), fontsize=9, va='bottom')
    axis.set(xticks=[], ylabel='Crops in the catalogue', xlim=(-0.45, 1.15),
             title='92 fleurons: the two clustering passes, then later chapters')
    axis.legend(fontsize=8, loc='lower right')
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / 'catalogue_contributions.png', dpi=180, bbox_inches='tight')
    print(f'\nwrote {OUTPUT_DIR / "catalogue_contributions.csv"} and .png')


if __name__ == '__main__':
    main()
