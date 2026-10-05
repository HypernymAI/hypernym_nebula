#!/usr/bin/env python3
"""
The WINGS contact sheet (S32c): every candidate braille wing icon, numbered — the sheet users pick from
(/nebula:shield-wings, brand.wings in ~/.nebula/settings.json).

    <python> make_wings_sheet.py            # writes wings_candidates.json + wings_contact_sheet.png beside this file
    <python> make_wings_sheet.py --check    # rebuild in memory; exit 1 unless identical to the shipped JSON

Generation (deterministic):
  LEFT WING = every 3×3 dot pattern (512) that is 4–7 dots, has its OUTER TIP high (top-left dot), is 8-connected, reaches
  toward the body (a dot in its inner column), and has a feathered lower edge (bottom row #.# / .#. / #.. / ..#, or
  middle row #.#); not a solid top two rows. → 80 wing shapes, in bit order.
  PAIR = the left wing, a gap of 0 or 1 dot columns, the mirror image (right wing) → 160 candidates, numbered in order
  (shape i → #2i gap 0, #2i+1 gap 1). Width padded to whole braille characters; the bottom braille row is blank.
NO .get(). Imports at top.
"""
import json
import sys
from itertools import product
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BITS = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]
LEFT, RIGHT = (205, 222, 255), (255, 210, 222)
FONT = '/System/Library/Fonts/Menlo.ttc'


def connected(W) -> bool:
    pts = [(r, c) for r in range(3) for c in range(3) if W[r][c]]
    seen, stack = {pts[0]}, [pts[0]]
    while stack:
        r, c = stack.pop()
        for dr, dc in product((-1, 0, 1), repeat=2):
            q = (r + dr, c + dc)
            if q in pts and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(pts)


def candidates() -> list:
    wings = []
    for bits in range(512):
        W = [[(bits >> (r * 3 + c)) & 1 for c in range(3)] for r in range(3)]
        n = sum(map(sum, W))
        if not (4 <= n <= 7) or not W[0][0]:
            continue
        if not connected(W) or not any(W[r][2] for r in range(3)):
            continue
        if W[2] not in ([1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]) and W[1] not in ([1, 0, 1],):
            continue
        if W[0] == [1, 1, 1] and W[1] == [1, 1, 1]:
            continue
        wings.append(W)
    out = []
    for W in wings:
        for gap in (0, 1):
            rows = []
            for r in range(3):
                left = ''.join('#' if x else '.' for x in W[r])
                rows.append(left + '.' * gap + left[::-1])
            width = len(rows[0])
            pad = (-width) % 2
            rows = [r + '.' * pad for r in rows] + ['.' * (width + pad)]
            k = (width + pad) // 2
            glyph = ''.join(chr(0x2800 | sum(BITS[r][dc] for r in range(4) for dc in range(2)
                                             if rows[r][c * 2 + dc] == '#')) for c in range(k))
            out.append({'rows': rows, 'gap': gap, 'glyph': glyph, 'width': width})
    return out


def sheet(cands: list, path: Path) -> None:
    per_row, TW, TH = 10, 230, 150
    img = Image.new('RGB', (per_row * TW, ((len(cands) + per_row - 1) // per_row) * TH), (98, 7, 44))
    d = ImageDraw.Draw(img)
    fw, fl = ImageFont.truetype(FONT, 11), ImageFont.truetype(FONT, 13)
    for i, cnd in enumerate(cands):
        x0, y0 = (i % per_row) * TW + 12, (i // per_row) * TH + 10
        w = cnd['width']
        for scale, yoff in ((0.8, 8), (0.4, 88)):
            CW, px, dot = 34 * scale, 14 * scale, 10 * scale
            for r in range(3):
                for c in range(len(cnd['rows'][0])):
                    if cnd['rows'][r][c] == '#':
                        cx, cy = x0 + (c // 2) * CW + (c % 2) * px, y0 + yoff + r * px * 1.1
                        d.ellipse([cx, cy, cx + dot, cy + dot], fill=LEFT if c < w / 2 else RIGHT)
        d.text((x0 + len(cnd['rows'][0]) // 2 * 14 + 6, y0 + 86), "HYPERNYM.ai", font=fw, fill=(215, 215, 215))
        d.text((x0, y0 + 112), f"#{i}  gap {cnd['gap']}", font=fl, fill=(235, 235, 235))
    img.save(path)


def main() -> None:
    cands = candidates()
    if sys.argv[1:] == ['--check']:
        shipped = json.loads((HERE / 'wings_candidates.json').read_text())
        ok = shipped == cands
        print(f"{len(cands)} candidates — {'IDENTICAL to' if ok else 'DIFFER from'} wings_candidates.json")
        sys.exit(0 if ok else 1)
    (HERE / 'wings_candidates.json').write_text(json.dumps(cands, ensure_ascii=False))
    sheet(cands, HERE / 'wings_contact_sheet.png')
    print(f"{len(cands)} candidates → wings_candidates.json, wings_contact_sheet.png")


if __name__ == '__main__':
    main()
