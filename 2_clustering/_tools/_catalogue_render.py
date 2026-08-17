"""Shared rendering for the catalogue contact sheet and its exemplar picker.

Both scripts import this so that a candidate shown in the picker is drawn by exactly the
same code path as the figure that ends up in the appendix. If the two diverge, a reviewer
chooses one image and the thesis prints a different one.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

# Fit each impression to its cell. Catalogue crops are small: the median class exemplar
# has a shorter side of 78 px against a 174 px cell, so drawing at native resolution left
# most classes occupying a third of their cell. Binarisation still happens at the source
# resolution, exactly as the descriptor stage does it; the resize is a display step
# applied to the binary silhouette afterwards. Set False to restore native-size drawing.
FIT_CELL = True


def resolve(link: Path) -> Path:
    """The real crop behind a catalogue symlink."""
    return Path(os.readlink(link)) if link.is_symlink() else link


def class_name(folder: Path) -> str:
    """`Fleuron_13 (33)` -> `Fleuron_13`. Folder names carry their crop count."""
    return folder.name.split(" (")[0]


def class_sort_key(folder: Path):
    """Numeric classes in numeric order, lettered classes after them, alphabetically."""
    digits = "".join(c for c in class_name(folder) if c.isdigit())
    return (int(digits) if digits else 9999, folder.name)


def crop_size(path: Path):
    """(width, height), or None if the file is unreadable or a dangling link."""
    if not path.exists():
        return None
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def render(src: Path, box: int) -> Image.Image:
    """One impression, binarised as the descriptor stage binarises it, fitted to `box`."""
    im = Image.open(src).convert("L")
    arr = np.array(im)
    thr = arr.mean()          # crops are bitonal or near-bitonal; Otsu agrees on these
    out = Image.fromarray(np.where(arr > thr, 255, 0).astype(np.uint8))
    if FIT_CELL:
        s = box / max(out.size)
        out = out.resize((max(1, round(out.width * s)), max(1, round(out.height * s))),
                         Image.LANCZOS)
        out = out.point(lambda v: 255 if v > 128 else 0)   # LANCZOS greys back to binary
    else:
        out.thumbnail((box, box), Image.LANCZOS)
    return out


def label_font(size: int = 15):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()
