"""Show what the detection experiments have finished and what is still running.

Run from the root of this package (the directory holding `book_identifiers.py`):

    python 4_detection/_tools/progress.py

Reads the output directories on disk rather than any log file, so it reports the true state
after a disconnect, a crash, or a restart.
"""

import json
import subprocess
from pathlib import Path

PROJECT_DIR = Path(".").resolve()
assert (PROJECT_DIR / "book_identifiers.py").exists(), "run from the project root"
OUTPUTS = PROJECT_DIR / "4_detection_outputs"

EXPECTED_FOLDS = {"fleuron_2": 17, "fleuron_74": 12, "fleuron_73": 14, "fleuron_72": 9}

PLANNED = [
    ("capacity: yolo11s", ["fleuron_73_lowo_s", "fleuron_2_lowo_s",
                           "fleuron_74_lowo_s", "fleuron_72_lowo_s"]),
    ("stability: seed 1", ["fleuron_74_lowo_seed1", "fleuron_2_lowo_seed1",
                           "fleuron_73_lowo_seed1", "fleuron_72_lowo_seed1"]),
    ("stability: seed 2", ["fleuron_74_lowo_seed2", "fleuron_2_lowo_seed2",
                           "fleuron_73_lowo_seed2", "fleuron_72_lowo_seed2"]),
    ("baseline: template matching", ["fleuron_2_tm", "fleuron_74_tm",
                                     "fleuron_73_tm", "fleuron_72_tm"]),
    ("cross-validation: yolo11n", ["fleuron_2_lowo", "fleuron_74_lowo",
                                   "fleuron_73_lowo", "fleuron_72_lowo"]),
]


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def running():
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=10)
        jobs = [ln for ln in out.stdout.splitlines()
                if "leave_one_work_out.py" in ln or "template_matching_baseline.py" in ln]
        names = set()
        for j in jobs:
            parts = j.split()
            for k, tok in enumerate(parts):
                if tok.endswith(".py") and k + 1 < len(parts):
                    names.add(f"{Path(tok).stem} {parts[k+1]}")
        return sorted(names)
    except Exception:
        return []


def state(name):
    d = OUTPUTS / name
    if not d.exists():
        return "not started", ""
    if (d / "pooled_predictions.csv").exists() or (d / "template_detections.csv").exists():
        return "done", ""
    fr = d / "_fold_results"
    if fr.exists():
        done = len([f for f in fr.glob("*.csv") if not f.name.endswith("__truth.csv")])
        target = name.split("_lowo")[0]
        total = EXPECTED_FOLDS.get(target, "?")
        return "running", f"{done}/{total} folds"
    return "started", ""


print("=" * 66)
print("DETECTION EXPERIMENTS")
print("=" * 66)
live = running()
print(f"GPU: {gpu()}")
print(f"processes: {', '.join(live) if live else 'none running'}")

total_done = total_all = 0
for label, runs in PLANNED:
    print(f"\n{label}")
    for r in runs:
        st, extra = state(r)
        total_all += 1
        total_done += (st == "done")
        mark = {"done": "[x]", "running": "[~]", "not started": "[ ]", "started": "[?]"}[st]
        print(f"  {mark} {r:<28} {st}{'  ' + extra if extra else ''}")

print(f"\n{total_done} of {total_all} runs complete")

# Headline numbers for whatever has finished.
print("\n" + "=" * 66)
print("RESULTS SO FAR (precision / recall, one prediction per impression, IoU 0.50)")
print("=" * 66)
for label, runs in PLANNED:
    shown = False
    for r in runs:
        d = OUTPUTS / r
        s = d / "pooled_summary.csv"
        if s.exists():
            import pandas as pd
            row = pd.read_csv(s).iloc[0]
            proto = json.loads((d / "protocol.json").read_text()) if (d / "protocol.json").exists() else {}
            if not shown:
                print(f"\n{label}")
                shown = True
            print(f"  {r:<28} P {row.precision:.3f}  R {row.recall:.3f}   "
                  f"({proto.get('impressions', '?')} impressions, "
                  f"{proto.get('model', 'yolo11n.pt')}, seed {proto.get('seed', 42)})")
print()
