"""Generate the catalogue class inventory table for the appendix.

Run from the root of this package:

    python 2_clustering/_tools/make_catalogue_inventory.py

Writes `report/tab_catalogue_inventory.tex`, a longtable body listing every catalogue
class with its crop, scan and work counts. The chapter text is written by hand; only the
rows are generated, so the table cannot drift from the catalogue on disk.

Scans and works are resolved through the feature manifest and the corpus-wide identity
rule in `book_identifiers.py`, so the work column counts bibliographic works rather than
digitised volumes. A small number of catalogue crops have no manifest entry and are
counted in the crop column but contribute no scan or work; the count is asserted below so
that it cannot grow silently.
"""

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(".").resolve()
assert (PROJECT / "book_identifiers.py").exists(), "run from the project root"
sys.path.insert(0, str(PROJECT))
from book_identifiers import build_work_map, volume_id  # noqa: E402

CAT = PROJECT / "Fleurons" / "Fleurons_v2_plus_retrieval"
OUT = PROJECT / "report" / "tab_catalogue_inventory.tex"
UNRESOLVED_EXPECTED = 8

m = pd.read_csv(PROJECT / "all_regions_outputs" / "features" / "features_manifest.csv")
m["crop_name"] = m.crop_path.map(lambda p: Path(p).name)
m["volume"] = m.rel_path.map(volume_id)
work_of, _ = build_work_map(sorted(m.volume.unique()))
m["work"] = m.volume.map(work_of)
by_name = m.drop_duplicates("crop_name").set_index("crop_name")

rows = []
for d in sorted(CAT.iterdir()):
    if not d.is_dir():
        continue
    names = [(Path(os.readlink(f)) if f.is_symlink() else f).name for f in d.iterdir()]
    sub = by_name.reindex([n for n in names if n in by_name.index])
    rows.append(dict(cls=d.name.split(" (")[0], crops=len(names), matched=len(sub),
                     scans=sub.stem.nunique(), works=sub.work.nunique()))

df = pd.DataFrame(rows)
df["sort"] = df.cls.str.extract(r"(\d+)")[0].fillna("9999").astype(int)
df = df.sort_values(["sort", "cls"]).reset_index(drop=True)

unresolved = int(df.crops.sum() - df.matched.sum())
assert unresolved == UNRESOLVED_EXPECTED, (
    f"{unresolved} catalogue crops have no manifest entry, expected {UNRESOLVED_EXPECTED}")

half = (len(df) + 1) // 2
left, right = df.iloc[:half].reset_index(drop=True), df.iloc[half:].reset_index(drop=True)

def cell(r):
    if r is None:
        return " & & & "
    return (f"\\texttt{{{r.cls.replace('_', chr(92) + '_')}}} & {r.crops} "
            f"& {r.scans} & {r.works}")

rows_tex = []
for i in range(len(left)):
    a = left.iloc[i]
    b = right.iloc[i] if i < len(right) else None
    rows_tex.append(f"{cell(a)} & {cell(b)} \\\\")

# The whole environment is written here, not just the rows. \input inside an alignment
# breaks the row scanner, so the appendix inputs a complete longtable instead of a body.
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    "{\\footnotesize\\setlength{\\tabcolsep}{4.5pt}\n"
    "\\begin{longtable}{lrrr@{\\hskip 1.8em}lrrr}\n"
    "\\toprule\n"
    "Class & Crops & Scans & Works & Class & Crops & Scans & Works \\\\\n"
    "\\midrule\n"
    "\\endfirsthead\n"
    "\\toprule\n"
    "Class & Crops & Scans & Works & Class & Crops & Scans & Works \\\\\n"
    "\\midrule\n"
    "\\endhead\n"
    "\\bottomrule\n"
    "\\endfoot\n"
    + "\n".join(rows_tex) + "\n"
    "\\end{longtable}}\n")

print(f"{len(df)} classes, {int(df.crops.sum()):,} crops, "
      f"{unresolved} without a manifest entry")
print(f"spanning >= 2 works: {int((df.works >= 2).sum())}, "
      f"median {df.works.median():.0f}, max {int(df.works.max())} "
      f"({df.loc[df.works.idxmax(), 'cls']})")
print(f"written to {OUT.relative_to(PROJECT)}")
