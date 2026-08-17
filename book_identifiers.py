"""Canonical bibliographic identifiers for the ornament corpus.

Every scan filename in this corpus follows one shelfmark convention:

    <work abbreviation>[<volume digit>]<place or publisher code><year>.<page>.<ext>

    dicolosn73.A3.t1.1.jpg      ->  work `dicolo`, place `sn`, year 1773
    vaoe2lhgo1761_017.jpg       ->  work `vaoe`,   volume 2, place `lhgo`, year 1761
    camoeuli[liba]59.nu.1.bmp   ->  work `camoeuli`, year 1759, `[liba]` = copy annotation

Two levels of identity follow from that convention and they are not the same thing:

`volume_id`
    One physical volume. This is what the earlier analyses called `book_id`.
`work_id`
    One bibliographic work. Volumes of a multi-volume set share it, so a fleuron
    appearing in volumes 1 and 3 of the same edition counts once, not twice.

Cross-book recurrence claims must be made at work level: the volume level inflates
reach for exactly the multi-volume sets whose volumes were printed together, in the
same shop, from the same ornament stock, which is the least interesting kind of
"recurrence" the corpus contains.

The volume-to-work grouping is deliberately conservative. Two volume identifiers are
merged only when they share both an alphabetic work abbreviation and a place code and
differ in an interior digit, the pattern a multi-volume set produces. Identifiers that
merely begin alike (`fororovi68` / `foropapr68`) are left separate, because a shared
first syllable is not evidence of a shared work. Everything the rule merges is listed
in `MULTIVOLUME_SETS` after the fact and can be inspected in one glance.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

__all__ = [
    'volume_id', 'legacy_book_id', 'parse_volume_id',
    'build_work_map', 'identifier_table', 'UNIDENTIFIED',
]

# Copy annotations such as `[liba]` sit between the work abbreviation and the year and
# carry no bibliographic identity of their own.
_BRACKETED = re.compile(r'\[[^\]]*\]')

# Photographs taken in the reading room rather than scanned from a shelfmarked file.
_PHOTOGRAPH_PREFIX = 'IMG_HDI'

# One volume whose page filenames vary in punctuation below the volume level
# (`vochroou70.ph.101`, `vochroou70ph155`), so the delimiter split alone would
# scatter it across several identifiers.
_IRREGULAR_PREFIXES = ('vochroou70',)

# `prefix` + one volume digit + `place` + a two- or four-digit year.
_VOLUME_PATTERN = re.compile(
    r'^(?P<prefix>[^\W\d_]+?)(?P<volume>\d)(?P<place>[^\W\d_]+)(?P<year>\d{2,4})$',
    re.UNICODE,
)

# Identifiers carrying no year in the filename: either explicitly undated (`sd`,
# sine dato) or photographed without a shelfmark.
UNIDENTIFIED = frozenset({'img_hdi', 'pd3', 'montesquieu', 'oxpelava', 'chcolonosd'})


def volume_id(filename: str) -> str:
    """Return the identifier of the physical volume a scan belongs to."""
    stem = unicodedata.normalize('NFC', Path(filename).stem)
    stem = _BRACKETED.sub('', stem)
    if stem.upper().startswith(_PHOTOGRAPH_PREFIX):
        return 'img_hdi'
    for prefix in _IRREGULAR_PREFIXES:
        if stem.lower().startswith(prefix):
            return prefix
    token = re.split(r'[.,_ ]', stem)[0].lower()
    # `...bis` marks a second scanning batch of a volume already in the corpus
    # (`tomœamre48bis` alongside `tomœamre48`), not a second volume.
    return re.sub(r'bis$', '', token) or token


def legacy_book_id(filename: str) -> str:
    """The identifier rule used by notebooks 2-5, kept so the frozen analyses can be
    reproduced and the effect of correcting them can be measured.

    It differs from :func:`volume_id` in two ways: it splits on `[`, which truncates
    `carepany[liba]65` to `carepany` and discards the year, and it treats a `bis`
    scanning batch as a separate book.
    """
    stem = unicodedata.normalize('NFC', Path(filename).stem)
    if stem.upper().startswith(_PHOTOGRAPH_PREFIX):
        return 'img_hdi'
    if stem.lower().startswith('vochroou70'):
        return 'vochroou70'
    return re.split(r'[.,_ \[]', stem)[0].lower()


def parse_volume_id(vid: str) -> dict | None:
    """Decompose a volume identifier into work abbreviation, volume, place and year.

    Returns None for identifiers that do not carry an interior volume digit, which is
    the majority: a single-volume work has no volume number to record.
    """
    match = _VOLUME_PATTERN.match(vid)
    return match.groupdict() if match else None


def build_work_map(volume_ids) -> tuple[dict, dict]:
    """Group volume identifiers into works.

    Returns ``(work_of, sets)`` where ``work_of`` maps every volume identifier to its
    work identifier, and ``sets`` maps each multi-volume work identifier to its
    member volumes. Volumes with no detected sibling map to themselves, so the work
    level is never coarser than the evidence supports.
    """
    families = defaultdict(list)
    for vid in volume_ids:
        parsed = parse_volume_id(vid)
        if parsed:
            families[(parsed['prefix'], parsed['place'])].append(vid)

    work_of, sets = {}, {}
    for (prefix, place), members in families.items():
        if len(members) < 2:
            continue                      # a lone volume digit is not a set
        members = sorted(members)
        earliest = min(int(parse_volume_id(v)['year']) for v in members)
        wid = f'{prefix}{place}{earliest}'
        sets[wid] = members
        for vid in members:
            work_of[vid] = wid

    for vid in volume_ids:
        work_of.setdefault(vid, vid)
    return work_of, dict(sorted(sets.items()))


def identifier_table(filenames):
    """Build the auditable scan -> volume -> work table as a DataFrame."""
    import pandas as pd

    frame = pd.DataFrame({'filename': list(filenames)})
    frame['volume_id'] = frame.filename.map(volume_id)
    frame['legacy_book_id'] = frame.filename.map(legacy_book_id)

    work_of, sets = build_work_map(sorted(frame.volume_id.unique()))
    frame['work_id'] = frame.volume_id.map(work_of)

    parsed = frame.volume_id.map(lambda v: parse_volume_id(v) or {})
    frame['volume_no'] = parsed.map(lambda p: p.get('volume'))
    frame['place_code'] = parsed.map(lambda p: p.get('place'))
    frame['year'] = parsed.map(lambda p: p.get('year'))
    frame['multivolume'] = frame.work_id.isin(sets)
    frame['dated'] = ~frame.volume_id.isin(UNIDENTIFIED)
    return frame, sets
