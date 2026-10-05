#!/usr/bin/env python3
"""
Regenerate the Hypernym braille logo marks used by the nebula status-line badge (../brand.py) from the logo PNG —
so the marks never depend on anyone remembering how they were made. See README.md in this directory.

    <python> generate_marks.py            # print MID (3 rows, the default) and FULL (4 rows) as Python constants
    <python> generate_marks.py --check    # assert they are byte-identical to brand.MID / brand.MARK (exit 1 if not)

Method (identical for both marks, only the grid size and threshold differ):
  1. source/hypernym-logo.png (256×256 RGBA). INK = alpha > 128 AND min(r,g,b) < 235 (drops the white fringe).
     Crop to the ink bounding box [46:210, 46:210] (164×164).
  2. DOTS: an N×N grid over the crop; each dot = mean ink coverage of its (164/N)² block; dot ON if coverage > THR.
     N = 4 × text rows, because a braille character is 2 dots wide × 4 dots tall and a terminal cell is ~1:2 (w:h),
     so dots are square and an N×N logo is N/2 characters wide × N/4 rows tall.
  3. BRAILLE: dots packed per 2×4 cell with the Unicode bit order [[0x01,0x08],[0x02,0x10],[0x04,0x20],[0x40,0x80]]
     onto U+2800.
  4. COLOUR (one per character — a terminal glyph has one foreground colour): the MEDIAN RGB of the ink pixels under
     that character's whole region, then saturation-lifted so the gradient survives a dark background:
     HLS lightness clamped to [0.55, 0.72], saturation × 1.35 + 0.10 (capped at 1). A cell with no ink, or whose
     braille is blank, has colour None (drawn as U+2800, nothing).
NO .get(). Imports at top.
"""
import argparse
import colorsys
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import brand                                                     # noqa: E402

LOGO = HERE / 'source' / 'hypernym-logo.png'
CROP = (46, 210)
BITS = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]
MARKS = {                                    # name: (text rows, dot threshold, brand.py constant)
    'MID': (3, 0.50, 'MID'),                 # 12×12 dots — the chosen default (pilot: "I like this size the best")
    'FULL': (4, 0.45, 'MARK'),               # 16×16 dots — the larger one (pilot: "the older larger one was nice too")
}


def load():
    a = np.array(Image.open(LOGO).convert('RGBA')).astype(float)
    inkm = (a[..., 3] > 128) & (a[..., :3].min(axis=2) < 235)
    lo, hi = CROP
    return inkm[lo:hi, lo:hi], a[lo:hi, lo:hi, :3]


def lift(rgb01):
    hh, ll, ss = colorsys.rgb_to_hls(*rgb01)
    r, g, b = colorsys.hls_to_rgb(hh, min(0.72, max(0.55, ll)), min(1.0, ss * 1.35 + 0.1))
    return (int(r * 255), int(g * 255), int(b * 255))


def generate(rows: int, thr: float):
    km, rgb = load()
    ink = km.astype(float)
    h, w = ink.shape
    n = rows * 4
    cells = rows * 2
    g = np.array([[ink[int(r * h / n):int((r + 1) * h / n), int(c * w / n):int((c + 1) * w / n)].mean() > thr
                   for c in range(n)] for r in range(n)])
    out = []
    for R in range(rows):
        row = []
        for C in range(cells):
            v = 0x2800
            for dr in range(4):
                for dc in range(2):
                    if g[R * 4 + dr, C * 2 + dc]:
                        v |= BITS[dr][dc]
            y0, y1 = int(R * h / rows), int((R + 1) * h / rows)
            x0, x1 = int(C * w / cells), int((C + 1) * w / cells)
            px = rgb[y0:y1, x0:x1][km[y0:y1, x0:x1]]
            if v == 0x2800 or len(px) == 0:
                row.append((chr(v), None))
            else:
                row.append((chr(v), lift(np.median(px, axis=0) / 255)))
        out.append(tuple(row))
    return tuple(out), g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    ok = True
    for name, (rows, thr, const) in MARKS.items():
        mark, g = generate(rows, thr)
        shipped = getattr(brand, const)
        same = mark == shipped
        ok &= same
        sym = bool((g == g[::-1, ::-1]).all())
        print(f"{name} ({rows} rows, {rows * 4}×{rows * 4} dots, thr {thr}): "
              f"{'IDENTICAL to' if same else 'DIFFERS from'} brand.{const}; 180° symmetric dots: {sym}")
        for row in mark:
            print('   ', ''.join(ch for ch, _ in row))
        if not args.check:
            print(f"{const} = {mark!r}\n")
    if args.check and not ok:
        sys.exit("FAIL: a shipped mark no longer matches its generator — regenerate or explain the edit in README.md")


if __name__ == '__main__':
    main()
