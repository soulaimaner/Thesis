"""Generate the class-to-work occurrence table for the appendix.

Run from the root of this package:

    python 3_retrieval/_tools/make_class_work_matrix.py

Writes `report/tab_class_work_matrix.tex`, a longtable body naming, for every catalogue
class, the bibliographic works its verified impressions occur in. This is the class-level
form of the recurrence result: the inventory table gives how many works a class reaches,
this one gives which. The chapter text is written by hand; only the rows are generated,
so the table cannot drift from the occurrence artefacts on disk.

Detector-derived crops are excluded, matching the primary reach rule of the recurrence
section, because detection was run for four adaptively selected classes only. The
`included_in_primary_reach` column of the source table already encodes that rule and is
the only filter applied here.

Counts are asserted against the summary table so that the two appendices cannot disagree.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(".").resolve()
assert (PROJECT / "book_identifiers.py").exists(), "run from the project root"

SRC = PROJECT / "3_retrieval_outputs" / "occurrence_v1"
BY_CLASS_WORK = SRC / "occurrence_by_class_work.csv"
BY_CLASS = SRC / "occurrence_by_class.csv"
OUT = PROJECT / "report" / "tab_class_work_matrix.tex"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main() -> None:
    cw = pd.read_csv(BY_CLASS_WORK)
    summary = pd.read_csv(BY_CLASS)

    primary = cw[cw["included_in_primary_reach"].astype(bool)]

    works = (
        primary.groupby("class_name")["work_id"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )

    # The work counts here must reproduce the `works` column of the inventory exactly.
    expected = dict(zip(summary["class_name"], summary["works"]))
    for cls, ws in works.items():
        assert len(ws) == expected[cls], (
            f"{cls}: {len(ws)} works listed but summary says {expected[cls]}"
        )
    assert len(works) == len(expected), "class set differs between the two tables"

    def sort_key(cls: str):
        stem = cls.split("_", 1)[-1]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    caption = (
        "Every catalogue class and the bibliographic works its verified impressions "
        "occur in, excluding detector-derived crops. The work count reproduces the "
        r"\emph{Works} column of \cref{app:catalogue}. Source: "
        r"\texttt{occurrence\_by\_class\_work.csv}."
    )
    short = "Catalogue classes by the works they occur in"

    lines = [
        r"{\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}lr>{\raggedright\arraybackslash}p{9.4cm}@{}}",
        r"\caption[%s]{%s}\label{tab:app-class-work}\\" % (short, caption),
        r"\toprule",
        r"Class & Works & Work identifiers \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{\emph{continued from the previous page}}\\",
        r"\toprule",
        r"Class & Works & Work identifiers \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for cls in sorted(works, key=sort_key):
        ids = ", ".join(r"\texttt{%s}" % tex_escape(w) for w in works[cls])
        lines.append(
            r"\texttt{%s} & %d & %s \\" % (tex_escape(cls), len(works[cls]), ids)
        )
    lines += [r"\end{longtable}}", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    total = sum(len(v) for v in works.values())
    recurrent = sum(1 for v in works.values() if len(v) >= 2)
    print(
        f"wrote {OUT} : {len(works)} classes, {total} class-work occurrences, "
        f"{recurrent} classes reaching two or more works"
    )


if __name__ == "__main__":
    main()
