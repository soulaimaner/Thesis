"""Training curves for the two `Fleuron_2` augmentation settings.

Run from the root of this package, the directory holding `book_identifiers.py`:

    python 4_detection/_tools/make_training_curves_figure.py

**What this shows, and what it does not.** Both runs come from the original
train/validation split, which \\cref{sec:res-det-validation} later found not to be
work-disjoint. Their absolute values are withdrawn and are not evidence of
generalisation. The figure is retained for one purpose only: the two runs differ in a
single parameter, `degrees`, so the gap between them isolates the effect of arbitrary
rotation augmentation on an axis-aligned detector. The gap is largest on mAP50-95, the
metric most sensitive to box tightness, which is the signature the correction predicted.

Reads `4_detection_outputs/runs/fleuron_2_v1/results.csv` (degrees=180) and
`.../fleuron_2_flipsonly/results.csv` (degrees=0), both written by Ultralytics at
training time.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(".").resolve()
assert (PROJECT / "book_identifiers.py").exists(), "run from the project root"
RUNS = PROJECT / "4_detection_outputs" / "runs"
OUT = PROJECT / "report" / "figures" / "fig_det_training_curves.png"

NAVY, ORANGE, GREY = "#1f4e79", "#e8703a", "#b0b7bd"
RUNS_SPEC = [("fleuron_2_v1", "arbitrary rotation (\\texttt{degrees=180})", ORANGE, "--"),
             ("fleuron_2_flipsonly", "flips only (\\texttt{degrees=0})", NAVY, "-")]

data = {}
for name, _, _, _ in RUNS_SPEC:
    df = pd.read_csv(RUNS / name / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    data[name] = df

panels = [("metrics/mAP50-95(B)", "mAP50--95", "most sensitive to box tightness"),
          ("metrics/mAP50(B)", "mAP50", "localisation at a single threshold"),
          ("metrics/precision(B)", "precision", "against incomplete catalogue labels")]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
for ax, (col, title, sub) in zip(axes, panels):
    for name, label, colour, style in RUNS_SPEC:
        df = data[name]
        ax.plot(df.epoch, df[col], style, color=colour, lw=1.5,
                label=label.replace("\\texttt{", "").replace("}", ""))
        best = df[col].max()
        ax.axhline(best, color=colour, lw=0.7, ls=":", alpha=0.55)
        # Best-value labels sit at opposite ends so they cannot collide with either
        # curve: the rotation run stops at epoch 43, so its label goes on the right.
        at_right = name == "fleuron_2_v1"
        ax.annotate(f"{best:.3f}", xy=(76 if at_right else 1, best),
                    xytext=(-2 if at_right else 2, 3), textcoords="offset points",
                    fontsize=7, color=colour, ha="right" if at_right else "left")
    ax.set_title(f"{title}\n{sub}", fontsize=8.5)
    ax.set_xlabel("epoch", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7.5)
axes[0].set_ylabel("validation metric", fontsize=8)
axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")

for name, label, _, _ in RUNS_SPEC:
    df = data[name]
    print(f"{name:22} {len(df):>3} epochs  "
          f"best mAP50-95 {df['metrics/mAP50-95(B)'].max():.3f}  "
          f"mAP50 {df['metrics/mAP50(B)'].max():.3f}")
print(f"written to {OUT.relative_to(PROJECT)}")
