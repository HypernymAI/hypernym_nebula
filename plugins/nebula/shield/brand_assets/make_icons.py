#!/usr/bin/env python3
"""
Hypernym-caliber SHIELD and ZEPHYR icons in braille — same trace method as the logo (generate_marks.py):
source PNG → N×N square-pitch dot grid (area coverage > THR) → 2×4-dot braille cells → one colour per cell (median of
the source colour under it, saturation-lifted).

    <python> make_icons.py            # writes source/*.png, prints the marks, renders previews/icons_preview.png

SHIELD  (source/shield.png, drawn here): a heater shield filled with the Hypernym logo's four-corner gradient
        (TL blue, TR coral, BL purple, BR green — the CLAUDE.md H-logo map), with the Hypernym NEGATIVE-SPACE H (two
        stems + crossbar) cut out of it.
ZEPHYR  (source/zephyr_emblem.png): the winged-diamond emblem of the ZEPHYR anti-gravity racing craft, traced from
        the Zephyr design sheet (design_racer.png, env HYPERNYM_ZEPHYR_SHEET) (crop 148,128–258,168; the title text excluded), recoloured in the Zephyr livery pink.
NO .get(). Imports at top.
"""
import colorsys
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from collections import deque

HERE = Path(__file__).resolve().parent
SRC = HERE / 'source'
BITS = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]
TL, TR, BL, BR = (79, 133, 206), (255, 154, 139), (139, 145, 228), (148, 218, 151)
ZEPHYR_SHEET = Path(os.environ['HYPERNYM_ZEPHYR_SHEET']) if 'HYPERNYM_ZEPHYR_SHEET' in os.environ else None  # the pink design_racer.png (dev only)
PINK_HI, PINK, PINK_LO = (255, 92, 170), (232, 24, 128), (170, 30, 150)


def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def draw_shield(size=512):
    """Heater shield, logo-gradient fill, negative-space H cut out. RGBA."""
    s = size
    grad = np.zeros((s, s, 3))
    for y in range(s):
        v = y / (s - 1)
        for x in range(0, s, 4):
            u = x / (s - 1)
            grad[y, x:x + 4] = lerp(lerp(TL, TR, u), lerp(BL, BR, u), v)
    mask = Image.new('L', (s, s), 0)
    d = ImageDraw.Draw(mask)
    m = 0.06 * s
    top, shoulder, tip = m, 0.52 * s, s - m
    pts = [(m, top), (s - m, top), (s - m, shoulder)]
    for i in range(1, 25):                                   # right flank curving to the tip
        t = i / 24
        pts.append((s - m - (s / 2 - m) * t ** 1.6, shoulder + (tip - shoulder) * t))
    for i in range(24, 0, -1):                               # left flank back up
        t = i / 24
        pts.append((m + (s / 2 - m) * t ** 1.6, shoulder + (tip - shoulder) * t))
    pts.append((m, shoulder))
    d.polygon(pts, fill=255)
    # negative-space H: two stems + crossbar, proportioned like the Hypernym mark's inner H
    sw, gap_top, gap_bot = 0.10 * s, 0.22 * s, 0.68 * s
    xl, xr = 0.31 * s, 0.69 * s - sw
    d.rectangle([xl, gap_top, xl + sw, gap_bot], fill=0)
    d.rectangle([xr, gap_top, xr + sw, gap_bot], fill=0)
    d.rectangle([xl, 0.41 * s, xr + sw, 0.41 * s + sw], fill=0)
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    rgba = np.dstack([grad, np.array(mask)]).astype(np.uint8)
    return Image.fromarray(rgba, 'RGBA')


def trace_zephyr():
    """The emblem from the Zephyr design sheet: dark ink → alpha; recoloured pink (hi on the wings' leading edge)."""
    if ZEPHYR_SHEET is None:
        raise SystemExit('set HYPERNYM_ZEPHYR_SHEET to the Zephyr design sheet (design_racer.png) to regenerate')
    im = Image.open(ZEPHYR_SHEET).convert('RGB').crop((148, 128, 258, 168)).resize((440, 160), Image.LANCZOS)
    a = np.array(im).astype(float)
    lum = a.mean(axis=2)
    alpha = np.clip((215 - lum) / 120, 0, 1)                 # the paper is ~235; the emblem is dark
    h, w = lum.shape
    out = np.zeros((h, w, 4))
    for y in range(h):
        for x in range(w):
            t = abs(x - w / 2) / (w / 2)                        # centre (diamond) deep → wing tips bright
            out[y, x, :3] = lerp(PINK_LO, PINK_HI, t) if t > 0.18 else PINK
    out[..., 3] = alpha * 255
    return Image.fromarray(out.astype(np.uint8), 'RGBA')


def trace_zephyr_ship():
    """The ZEPHYR craft itself (the pink hero render of design_racer.png): hull pink + canopy/intake dark, gaps closed
    (Max/Min 9), largest connected blob kept; the inset photo (bottom-right) masked out. RGBA, original colours."""
    if ZEPHYR_SHEET is None:
        raise SystemExit('set HYPERNYM_ZEPHYR_SHEET to the Zephyr design sheet (design_racer.png) to regenerate')
    im = Image.open(ZEPHYR_SHEET).convert('RGB').crop((215, 230, 1195, 800))
    a = np.array(im).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    ink = ((r > 150) & (g < 130) & (r - g > 90) & (b > 60)) | (r + g + b < 170)
    ink[510:, 780:] = False                                  # the inset photo: sheet x >= 995, y >= 740
    m = Image.fromarray((ink * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))
    arr = np.array(m) > 128
    h, w = arr.shape
    lab = np.zeros((h, w), dtype=np.int32)
    best, n = (0, 0), 0
    for sy in range(0, h, 3):
        for sx in range(0, w, 3):
            if arr[sy, sx] and not lab[sy, sx]:
                n += 1
                q, c = deque([(sy, sx)]), 0
                lab[sy, sx] = n
                while q:
                    y, x = q.popleft()
                    c += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and arr[yy, xx] and not lab[yy, xx]:
                            lab[yy, xx] = n
                            q.append((yy, xx))
                if c > best[0]:
                    best = (c, n)
    ship = lab == best[1]
    return Image.fromarray(np.dstack([a, ship * 255]).astype(np.uint8), 'RGBA')


def lift(rgb01):
    hh, ll, ss = colorsys.rgb_to_hls(*rgb01)
    r, g, b = colorsys.hls_to_rgb(hh, min(0.72, max(0.55, ll)), min(1.0, ss * 1.35 + 0.1))
    return (int(r * 255), int(g * 255), int(b * 255))


def trace(img: Image.Image, rows: int, cols: int, thr: float):
    """rows × cols braille cells = (4·rows) × (2·cols) square-pitch dots over the image's alpha bounding box."""
    a = np.array(img).astype(float)
    ink = a[..., 3] / 255.0
    ys, xs = np.where(ink > 0.5)
    ink, rgb = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1], a[ys.min():ys.max() + 1, xs.min():xs.max() + 1, :3]
    km = ink > 0.5
    h, w = ink.shape
    ny, nx = rows * 4, cols * 2
    g = np.array([[ink[int(r * h / ny):int((r + 1) * h / ny), int(c * w / nx):int((c + 1) * w / nx)].mean() > thr
                   for c in range(nx)] for r in range(ny)])
    out = []
    for R in range(rows):
        row = []
        for C in range(cols):
            v = 0x2800
            for dr in range(4):
                for dc in range(2):
                    if g[R * 4 + dr, C * 2 + dc]:
                        v |= BITS[dr][dc]
            y0, y1, x0, x1 = int(R * h / rows), int((R + 1) * h / rows), int(C * w / cols), int((C + 1) * w / cols)
            px = rgb[y0:y1, x0:x1][km[y0:y1, x0:x1]]
            row.append((chr(v), None) if (v == 0x2800 or len(px) == 0) else (chr(v), lift(np.median(px, axis=0) / 255)))
        out.append(tuple(row))
    return tuple(out), g


def preview(items, path: Path, bg=(98, 7, 44)):
    """Each item: (label, source image, mark, dot grid) → source beside its braille render (square dots, cell gaps)."""
    W = 40 + sum(260 + 60 + len(m[0]) * 30 + 60 for _, _, m, _ in items)
    img = Image.new('RGB', (W, 330), bg)
    d = ImageDraw.Draw(img)
    x = 20
    for label, src, mark, g in items:
        sw, sh = src.size
        k = 240 / max(sw, sh)
        th = src.resize((int(sw * k), int(sh * k)), Image.LANCZOS)
        img.paste(th, (x, 45), th)
        x += 260 + 40
        for R, row in enumerate(mark):
            for C, (ch, col) in enumerate(row):
                for dr in range(4):
                    for dc in range(2):
                        if g[R * 4 + dr, C * 2 + dc]:
                            cx, cy = x + C * 30 + 4 + dc * 12, 50 + R * 64 + 6 + dr * 14
                            d.ellipse([cx, cy, cx + 8, cy + 8], fill=col)
        d.text((x, 300), label, fill=(230, 230, 230))
        x += len(mark[0]) * 30 + 80
    img.save(path)


def main() -> None:
    SRC.mkdir(exist_ok=True)
    shield = draw_shield()
    shield.save(SRC / 'shield.png')
    zeph = trace_zephyr()
    zeph.save(SRC / 'zephyr_emblem.png')
    ship = trace_zephyr_ship()
    ship.save(SRC / 'zephyr_ship.png')
    marks = {}
    for name, img, rows, cols, thr in (('SHIELD_3x1', shield, 1, 3, 0.5), ('ZEPHYR_3x1', ship, 1, 3, 0.4),
                                       ('SHIELD', shield, 3, 6, 0.5), ('ZEPHYR_SHIP_3ROW', ship, 3, 9, 0.4),
                                       ('ZEPHYR_EMBLEM', zeph, 3, 16, 0.25)):
        mark, g = trace(img, rows, cols, thr)
        marks[name] = (mark, g, img)
        print(f"{name} ({rows} rows × {cols} cells = {rows * 4}×{cols * 2} dots):")
        for row in mark:
            print('   ', ''.join(ch for ch, _ in row))
        print(f"{name} = {mark!r}\n")
    preview([(n, marks[n][2], marks[n][0], marks[n][1]) for n in marks], HERE / 'previews' / 'icons_preview.png')
    print(HERE / 'previews' / 'icons_preview.png')


if __name__ == '__main__':
    sys.exit(main())
