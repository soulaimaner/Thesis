"""Build the material for choosing, by hand, which impression represents each class.

Run from the root of this package:

    python 2_clustering/_tools/make_exemplar_picker.py

**Why this exists.** The appendix contact sheet showed one impression per class, chosen
mechanically as the crop with the largest shorter side. Resolution is not legibility: for
several classes the largest surviving impression is also the most heavily inked one, so
the sheet showed a blot where the class holds perfectly readable impressions a few ranks
down. Choosing by hand fixes that, and the choice has to be recorded rather than made in
an image editor, or the figure stops being reproducible from the catalogue.

**What is built.** A dated directory holding, for each class, the `CANDIDATES` crops with
the largest shorter side as ranked symlinks, and a contact sheet of the same candidates
drawn by the same renderer the appendix figure uses. Rank 01 is the impression the old
mechanical rule chose, so a directory nobody touches reproduces the previous figure.

**How to choose.** In `classes/<class>/`, delete the ranks you do not want. The figure
uses the lowest-numbered survivor. Curating a class therefore means deleting from the top
until the first remaining thumbnail is the one to print, and a class that already looks
right needs no action at all. To use a crop outside the top `CANDIDATES`, symlink it into
the class folder under a name beginning `00_`.

Deleting a link here never touches the catalogue or the crop store; it records a verdict
about which impression to print. The directory is dated and is never regenerated in
place, because regenerating it would silently restore links a reviewer had removed.
"""

import csv
import datetime as dt
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT = Path(".").resolve()
assert (PROJECT / "book_identifiers.py").exists(), "run from the project root"
sys.path.insert(0, str(Path(__file__).resolve().parent))   # `_catalogue_render` sits here
from _catalogue_render import (  # noqa: E402
    class_name, class_sort_key, crop_size, label_font, render, resolve)

CAT = PROJECT / "Fleurons" / "Fleurons_v2_plus_retrieval"
OUT = PROJECT / f"catalogue_exemplars_{dt.date.today():%Y%m%d}"

CANDIDATES = 24      # three rows of eight per class
COLS = 8
CELL = 190           # px per cell, matching the appendix figure
PAD = 8
LABEL_H = 24

if OUT.exists():
    sys.exit(f"{OUT.name} already exists; it is never regenerated in place, because that "
             f"would restore links a reviewer had deleted. Remove it deliberately, or "
             f"let the figure script read the choices already recorded in it.")

classes = sorted([d for d in CAT.iterdir() if d.is_dir()], key=class_sort_key)
(OUT / "classes").mkdir(parents=True)
(OUT / "sheets").mkdir(parents=True)

font = label_font(15)
small = label_font(13)
key = []

for folder in classes:
    name = class_name(folder)

    sized = []
    for f in folder.iterdir():
        real = resolve(f)
        size = crop_size(real)
        if size is not None:
            sized.append((min(size), size, real))
    assert sized, f"no readable crop in {folder.name}"
    # Largest shorter side first; the crop name breaks ties so the ranking is stable
    # across runs and across filesystems that iterate in different orders.
    sized.sort(key=lambda t: (-t[0], t[2].name))
    top = sized[:CANDIDATES]

    links = OUT / "classes" / name
    links.mkdir()
    for i, (side, (w, h), real) in enumerate(top, start=1):
        link = links / f"{i:02d}_{w}x{h}_{real.name}"
        link.symlink_to(real)
        key.append(dict(class_name=name, rank=i, link_name=link.name, target=str(real),
                        crop_name=real.name, width=w, height=h, shorter_side=side))

    rows = -(-len(top) // COLS)
    sheet = Image.new("L", (COLS * CELL, rows * (CELL + LABEL_H)), 255)
    draw = ImageDraw.Draw(sheet)
    for i, (side, (w, h), real) in enumerate(top):
        r, c = divmod(i, COLS)
        x0, y0 = c * CELL, r * (CELL + LABEL_H)
        im = render(real, CELL - 2 * PAD)
        sheet.paste(im, (x0 + (CELL - im.width) // 2, y0 + (CELL - im.height) // 2))
        cap = f"{i + 1:02d}   {w}x{h}"
        draw.text((x0 + (CELL - draw.textlength(cap, font=small)) / 2, y0 + CELL + 4),
                  cap, fill=90, font=small)
    title = f"{name}   ({len(sized)} crops, {len(top)} shown)"
    band = Image.new("L", (sheet.width, 30), 255)
    ImageDraw.Draw(band).text((6, 7), title, fill=0, font=font)
    page = Image.new("L", (sheet.width, sheet.height + 30), 255)
    page.paste(band, (0, 0))
    page.paste(sheet, (0, 30))
    page.save(OUT / "sheets" / f"{name}.png", optimize=True)

with (OUT / "key.csv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(key[0]))
    writer.writeheader()
    writer.writerows(key)

(OUT / "README.md").write_text(f"""\
# Catalogue exemplar choice — {dt.date.today():%d %B %Y}

Which impression represents each class in the appendix contact sheet
(`report/figures/fig_catalogue_contact_sheet.png`, Appendix A7).

Built from `Fleurons/Fleurons_v2_plus_retrieval/` by
`2_clustering/_tools/make_exemplar_picker.py`: {len(classes)} classes, up to {CANDIDATES}
candidates each.

## What is here

| | |
|---|---|
| `sheets/<class>.png` | the candidates for one class, captioned `rank  width x height` |
| `classes/<class>/` | the same candidates as ranked symlinks, for curation by deletion |
| `key.csv` | one row per candidate: class, rank, link name, target, pixel size |

Candidates are the crops with the largest shorter side, largest first. Rank 01 is what
the old mechanical rule printed, so an untouched directory reproduces the previous
figure exactly.

## How to choose

Open `sheets/<class>.png`, decide which rank should represent the class, then in
`classes/<class>/` delete every link above it. **The figure uses the lowest-numbered
surviving link.** A class whose rank 01 already reads well needs no action.

To print a crop outside the top {CANDIDATES}, symlink it into the class folder under a
name starting `00_`; it then outranks everything else.

Deleting a link here does not touch the catalogue or the crop store. It records a
verdict about which impression to print, nothing more.

## Then

    python 2_clustering/_tools/make_catalogue_contact_sheet.py

It finds this directory automatically, rebuilds the figure, and writes
`report/figures/fig_catalogue_contact_sheet_choices.csv` recording the crop printed for
every class and whether it was chosen or left to the default. If any class was chosen by
hand, the figure caption in `report/report_structure.tex` must say so: the caption
currently claims selection by a fixed mechanical rule, and that claim stops being true
the moment a link is deleted.
""")

print(f"{len(classes)} classes, {len(key)} candidate links, "
      f"{sum(1 for _ in (OUT / 'sheets').iterdir())} sheets")
print(f"written to {OUT.relative_to(PROJECT)}/")
print(f"next: browse {OUT.name}/sheets/, delete links in {OUT.name}/classes/<class>/, "
      f"then rerun make_catalogue_contact_sheet.py")
