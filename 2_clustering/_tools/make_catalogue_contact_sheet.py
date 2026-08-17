"""Contact sheet of the catalogue: one representative impression per class.

Run from the root of this package:

    python 2_clustering/_tools/make_catalogue_contact_sheet.py

**Why this exists.** The inventory of Appendix A7 lists every class by identifier and
count, which tells a reader how large each class is and nothing at all about what it
looks like. A catalogue of visual designs that never shows a design is a table of
numbers. This sheet pairs each identifier with one impression so the inventory can be
read.

**Which impression is shown.** By default, the crop with the largest shorter side, that
is, the highest-resolution surviving impression of the design. Resolution is a proxy for
legibility and not always a good one: for several classes the largest impression is also
the most heavily inked, and the sheet showed a blot where the class holds readable
impressions a few ranks down. A reviewer may therefore override the default class by
class, using `make_exemplar_picker.py`; this script picks up those choices from the
newest `catalogue_exemplars_*` directory and records, in a CSV beside the figure, which
crop each class printed and whether it was chosen or defaulted. Whichever route a class
took, the impression printed is one the catalogue actually holds, and it is the clearest
or the largest instance rather than a typical one: worn or partial impressions of the
same design will look poorer than what is shown here.

Crops are binarised for display exactly as the descriptor stage binarises them
(\\cref{sec:meth-repr-binarisation}), so the sheet shows what the pipeline compares
rather than what the scanner captured.
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT = Path(".").resolve()
assert (PROJECT / "book_identifiers.py").exists(), "run from the project root"
sys.path.insert(0, str(Path(__file__).resolve().parent))   # `_catalogue_render` sits here
from _catalogue_render import (  # noqa: E402
    class_name, class_sort_key, crop_size, label_font, render, resolve)

CAT = PROJECT / "Fleurons" / "Fleurons_v2_plus_retrieval"
OUT = PROJECT / "report" / "figures" / "fig_catalogue_contact_sheet.png"
CHOICES = OUT.with_name("fig_catalogue_contact_sheet_choices.csv")

COLS = 8
CELL = 190           # px per cell, square
PAD = 8              # px of white inside each cell
LABEL_H = 24         # px reserved under each thumbnail
BG = 255


def picker_dir():
    """The newest `catalogue_exemplars_*` directory, or None if no review has been run."""
    dirs = sorted(d for d in PROJECT.glob("catalogue_exemplars_*") if d.is_dir())
    return dirs[-1] if dirs else None


def chosen_exemplar(picker: Path, name: str):
    """The reviewer's pick: the lowest-ranked link left in the class folder."""
    folder = picker / "classes" / name
    if not folder.is_dir():
        return None
    surviving = sorted(folder.iterdir(), key=lambda f: f.name)
    for link in surviving:
        real = resolve(link)
        if crop_size(real) is not None:
            return real, link.name
    return None


def default_exemplar(folder: Path):
    """The crop with the largest shorter side, i.e. the best-resolution impression."""
    sized = []
    for f in folder.iterdir():
        real = resolve(f)
        size = crop_size(real)
        if size is not None:
            sized.append((min(size), real))
    if not sized:
        return None
    # The crop name breaks ties so the default does not depend on directory order.
    return min(sized, key=lambda t: (-t[0], t[1].name))[1]


classes = sorted([d for d in CAT.iterdir() if d.is_dir()], key=class_sort_key)
picker = picker_dir()

rows = -(-len(classes) // COLS)
sheet = Image.new("L", (COLS * CELL, rows * (CELL + LABEL_H)), BG)
draw = ImageDraw.Draw(sheet)
font = label_font(15)

missing, records = [], []
for i, folder in enumerate(classes):
    name = class_name(folder)
    r, c = divmod(i, COLS)
    x0, y0 = c * CELL, r * (CELL + LABEL_H)

    pick = chosen_exemplar(picker, name) if picker else None
    if pick is not None:
        src, how, link = pick[0], "chosen", pick[1]
        # Rank 01 surviving means the reviewer looked and kept the default.
        if link.startswith("01_"):
            how = "default kept"
    else:
        src, how, link = default_exemplar(folder), "default", ""

    if src is None:
        missing.append(name)
    else:
        im = render(src, CELL - 2 * PAD)
        sheet.paste(im, (x0 + (CELL - im.width) // 2, y0 + (CELL - im.height) // 2))
        w, h = crop_size(src)
        records.append(dict(class_name=name, selection=how, link_name=link,
                            crop_name=src.name, width=w, height=h, target=str(src)))

    w = draw.textlength(name, font=font)
    draw.text((x0 + (CELL - w) / 2, y0 + CELL + 3), name, fill=90, font=font)

assert not missing, f"no readable exemplar for: {missing}"

OUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUT, optimize=True)

with CHOICES.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

by_hand = sum(1 for r in records if r["selection"] == "chosen")
print(f"{len(classes)} classes on a {COLS}x{rows} sheet, {sheet.size[0]}x{sheet.size[1]} px")
print(f"exemplars: {by_hand} chosen by hand, {len(records) - by_hand} by the default rule"
      + (f" (choices read from {picker.name}/)" if picker else " (no picker directory)"))
if by_hand:
    print("NOTE: the appendix caption still describes selection by a fixed mechanical "
          "rule; update it in report/report_structure.tex.")
print(f"written to {OUT.relative_to(PROJECT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
print(f"choices recorded in {CHOICES.relative_to(PROJECT)}")
