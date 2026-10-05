"""
Hypernym brand cameo for the status-line badge (build doc S26).

LOGO: the real logo PNG (portable_hypernym_processor/tools/runtime_code_compression_website/public/hypernym-logo.png,
256×256; ink = alpha > 128 and not white; ink bbox 46..209) sampled onto a 16×16 dot grid (area coverage > 0.45 = dot)
and packed into braille: 8 cells × 4 rows (each braille cell = 2×4 dots). Each cell's colour = the median colour of
the logo pixels under it, saturation-lifted so the gradient survives on a dark background (None = no ink in the cell).
Generator: build doc S26c. WORDMARK = the per-letter colours of hypernym.ai's footer brand text (…/public/index.html:1154).

SCHEDULE ('cameo' mode): every 21–59 s for APPEAR_S seconds (deterministic per SLOT_S slot, hash-jittered — every window
agrees, no state). Most appearances TRANSLUCENT (blended ALPHA toward the terminal background; terminals have no real
alpha); 1 in SHINY_ONE_IN OPAQUE + bold with a holographic sweep (hue rotation + a slanted white glint crossing the logo
and the wordmark). 'always' mode = a continuous LOOP_S shiny loop (preview). Hidden = same-width blanks: nothing moves.
'gentle' mode (pilot, S26d): the logo rests STILL in its true colours; one shiny flash (FLASH_S) every 10–15 min at a
deterministic random time (GENTLE_SLOT_S slots, start jittered by < GENTLE_JITTER_S → consecutive flashes 600–900 s
apart); 1 flash in DOUBLE_ONE_IN is a DOUBLE flash (two sweeps back to back).
Pure functions of (now, bg, mode) — no I/O — EXCEPT icon_flash (below), which keeps one tiny state file per icon key.
UPDATE FLASH (pilot, S51): any braille icon (any width, any number of rows) flashes shiny when the data behind it
changes: a random 2.4–2.9 s = 3 status-line frames at Claude Code's 1-per-second redraw, the glint crossing the icon: icon_flash(mark, key, signature, now). NO .get().
"""
import colorsys
import json
import hashlib
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

WORD = (('H', (0x4f, 0x85, 0xce)), ('Y', (0x6d, 0xb3, 0xd4)), ('P', (0xb0, 0xd8, 0x93)), ('E', (0xdb, 0xd0, 0x6f)),
        ('R', (0xff, 0x9a, 0x8b)), ('N', (0x8b, 0x91, 0xe4)), ('Y', (0xe3, 0x8c, 0xc4)), ('M', (0x94, 0xda, 0x97)))
MARK = (
    (('⢀', (54, 132, 226)), ('⣸', (55, 175, 255)), ('⠿', (65, 208, 255)), ('⢀', (69, 226, 255)),
     ('⣠', (255, 196, 100)), ('⠿', (255, 126, 112)), ('⠇', (255, 143, 111)), ('⣀', (78, 200, 201))),
    (('⣿', (81, 107, 237)), ('⡇', (95, 121, 231)), ('⠀', None), ('⠛', (89, 240, 175)),
     ('⠛', (180, 236, 92)), ('⠀', None), ('⢸', (79, 189, 200)), ('⣿', (68, 215, 145))),
    (('⣿', (227, 104, 219)), ('⡇', (226, 109, 183)), ('⠀', None), ('⣤', (213, 109, 223)),
     ('⣤', (255, 98, 150)), ('⠀', None), ('⢸', (128, 213, 120)), ('⣿', (236, 233, 69))),
    (('⠉', (255, 112, 167)), ('⢡', (107, 115, 255)), ('⣶', (154, 112, 254)), ('⠋', (244, 107, 214)),
     ('⠁', (240, 110, 239)), ('⣶', (239, 119, 228)), ('⡏', (251, 143, 115)), ('⠁', (255, 200, 70))),
)
MARK_W = len(MARK[0])
# S26k COMPACT (one line, 3 braille cells = 6×4 dots): the logo is a NEGATIVE-SPACE H — the empty stems and crossbar
# are the mark, the four pieces frame them (S26j's filled-piece pair '⢞⡵' missed that):
#   ●·●●·●   left bar · top block · right bar
#   ●····●   crossbar (empty)
#   ●····●
#   ●·●●·●   left bar · bottom block · right bar      → '⡇⣉⢸'
# 4 dot rows cannot also hold the tabs (tabs + block + crossbar + block + tabs = 5 bands); the 'full' mark (16×16) has
# them. Cell colours = logo map: left blue, centre green, right amber.
# S26l MID (pilot: 3 rows; a character cell is ~1:2, a braille cell 2×4 dots → square dots): 3 lines × 6 cells =
# 12×12 dots, the logo PNG area-sampled at 12×12 (> 0.5 = dot; 180°-symmetric, asserted when generated): tabs,
# diagonal joins, bars, centre blobs, and the NEGATIVE-SPACE H (empty stems + crossbar) all present. Per-cell colour =
# median logo colour under the cell, saturation-lifted. Checked visually vs the PNG (scratchpad h12_preview.png).
MID = ((('⣠', (57, 136, 230)), ('⠟', (62, 183, 255)), ('⢃', (71, 229, 254)), ('⡼', (254, 177, 108)),
        ('⠋', (255, 138, 112)), ('⣤', (80, 198, 200))),
       (('⣿', (154, 105, 229)), ('⠀', None), ('⢈', (114, 230, 203)), ('⡁', (206, 225, 122)), ('⠀', None),
        ('⣿', (99, 221, 121))),
       (('⠛', (255, 108, 173)), ('⣠', (134, 112, 255)), ('⡞', (228, 108, 231)), ('⢡', (240, 112, 218)),
        ('⣴', (249, 117, 185)), ('⠋', (255, 206, 67))))
# S26m HOUSE (pilot: "use ClaudeStorm's own logo tech + half blocks"): the logical 5×5 grid of ux/hypernym_logo.py's
# LOGO_ROWS (bars at 5 offsets; pairs of rows = one logical row) — the canonical ClaudeStorm H:
#   ·X·X·   top tabs
#   X·X·X
#   X···X   ← empty stems + crossbar = the negative-space H
#   X·X·X
#   ·X·X·   bottom tabs
# drawn with half blocks (▀ fg = top pixel, bg = bottom pixel): square pixels, 3 text rows × 5 chars (6th pixel row
# empty). Per-pixel colour = ux/hypernym_logo._base_rgb (bilinear TL/TR/BL/BR corners, pulled to MID on the crossbar),
# copied so the plugin does not import ClaudeStorm.
HOUSE = ('.X.X.', 'X.X.X', 'X...X', 'X.X.X', '.X.X.')
H_TL, H_TR, H_BL, H_BR, H_MID = (79, 133, 206), (255, 154, 139), (139, 145, 228), (148, 218, 151), (176, 216, 147)
# Per-pixel colours used for the badge: the MEDIAN logo-PNG colour in a ±12% window around each pixel's centre (the
# house bilinear grade washes the middle row to one green; this keeps the real gradient — checked vs the PNG,
# scratchpad house_preview2.png).
HOUSE_RGB = {(0, 1): (85, 191, 232), (0, 3): (239, 151, 133), (1, 0): (97, 134, 207), (1, 2): (163, 212, 139),
             (1, 4): (79, 155, 149), (2, 0): (159, 128, 207), (2, 4): (120, 196, 141), (3, 0): (225, 130, 182),
             (3, 2): (227, 127, 178), (3, 4): (219, 198, 106), (4, 1): (163, 135, 231), (4, 3): (223, 142, 204)}


def _lerp3(a, b, f):
    return tuple(a[i] + (b[i] - a[i]) * f for i in range(3))


def house_rgb(u: float, v: float) -> Tuple[int, int, int]:
    """= ux/hypernym_logo._base_rgb."""
    base = _lerp3(_lerp3(H_TL, H_TR, u), _lerp3(H_BL, H_BR, u), v)
    return tuple(int(x) for x in _lerp3(H_MID, base, abs(v - 0.5) * 2.0))


def _px(c, kind: str, p: float, pos: float, row: int, bg: Tuple[int, int, int], alpha: float):
    """Final RGB of one pixel under the current envelope (None = empty)."""
    if c is None:
        return None
    if kind == 'solid':
        return _blend(bg, c, alpha)
    if kind == 'alpha':
        return _blend(bg, c, ALPHA * math.sin(math.pi * p))
    return _holo(c, pos, row, p)


def house_rows(kind: str, p: float, bg: Tuple[int, int, int], alpha: float) -> List[str]:
    n = len(HOUSE)
    grid = [[HOUSE_RGB[(r, c)] if HOUSE[r][c] == 'X' else None for c in range(n)] for r in range(n)]
    grid.append([None] * n)                                     # 6th pixel row: 3 whole text rows
    out = []
    for R in range(3):
        cells = []
        for c in range(n):
            top = _px(grid[2 * R][c], kind, p, c * 1.6, 2 * R, bg, alpha)
            bot = _px(grid[2 * R + 1][c], kind, p, c * 1.6, 2 * R + 1, bg, alpha)
            b = BOLD if kind == 'shiny' else ''
            if top and bot:
                cells.append(f'{b}\x1b[38;2;{top[0]};{top[1]};{top[2]}m\x1b[48;2;{bot[0]};{bot[1]};{bot[2]}m▀{RESET}')
            elif top:
                cells.append(f'{b}\x1b[38;2;{top[0]};{top[1]};{top[2]}m▀{RESET}')
            elif bot:
                cells.append(f'{b}\x1b[38;2;{bot[0]};{bot[1]};{bot[2]}m▄{RESET}')
            else:
                cells.append(' ')
        out.append(''.join(cells))
    return out


COMPACT = ((('⡇', (84, 151, 219)), ('⣉', (176, 216, 147)), ('⢸', (246, 187, 128))),)
# S29 — 3×1 ICONS (one line, 3 braille cells = 6×4 dots), HAND-BUILT from each shape's structure (auto-traces blur at
# this size — brand_assets/make_icons.py + README §8). S30: shown on the badge when brand.icons is true.
#   SHIELD  ######  heater shield: square shoulders tapering to a point; the two holes = the Hypernym H's stems;
#           #.##.#  logo-map colours left→right: blue · green · coral
#           .####.
#           ..##..
#   ZEPHYR  .....#  the ZEPHYR racer (images/zephyr/pink/design_racer.png) in profile: pointed nose ←, canopy block,
#           ..####  tall swept rear fin; livery hot pink → pink → thruster magenta
#           ######
#           .####.
SHIELD_ICON = ((('⠫', (84, 151, 219)), ('⣿', (176, 216, 147)), ('⠝', (255, 154, 139))),)
#   MINI H  .#|#.|#.  (S31f, pilot) LEFT BAR = right dot column of character 1; MIDDLE (one-dot top + bottom
#           .#|..|#.  blocks) = left column of character 2, with its right column EMPTY (the extra space, pilot);
#           .#|#.|#.  RIGHT BAR = left column of character 3. Empty gaps + empty middle row = the negative-space H.
#           ..|..|..  Bottom braille row BLANK. Blue · green · coral. (History: S31 ⢎⠶⡱ … S31e ⠸⠽.)
MINI_H_ICON = ((('⠸', (84, 151, 219)), ('⠅', (176, 216, 147)), ('⠇', (255, 154, 139))),)
#   WINGS   ##....##  (S32b, pilot: 'fluffy … with braille') each wing sweeps up and OUT to a high tip, then steps
#           .##..##.  down in scalloped feather tiers (#.# = the fluff); the empty middle = the body. 4 characters:
#           #.#..#.#  feathers need the width (3 cells gave blocks — S32 ⠻⠀⠟ 'bad'). Bottom braille row BLANK.
#           ........  Soft blue-white (left wing) · soft pink (right wing). Rejected: ε-3 scallops ⠺⠅⠨⠗ (read as
#                     puffy brackets), ʚɞ loops ⠪⠆⠰⠕ (tiny hearts). previews/wings2_tiers_CHOSEN.png
WINGS_ICON = ((('⠝', (205, 222, 255)), ('⠆', (205, 222, 255)), ('⠰', (255, 210, 222)), ('⠫', (255, 210, 222))),)
ZEPHYR_ICON = ((('⢤', (255, 92, 170)), ('⣶', (232, 24, 128)), ('⡾', (190, 60, 200))),)
GRAPHITE = (0x80, 0x84, 0x8a)                                  # '.ai' suffix colour (pilot, S26i)
SLOT_S, JITTER_S, APPEAR_S = 40, 20, 6
SHINY_ONE_IN = 5
ALPHA = 0.35
LOOP_S = 12                                                    # 'always' mode: one slow holo loop (smooth at 1 fps)
GENTLE_SLOT_S, GENTLE_JITTER_S, FLASH_S, DOUBLE_ONE_IN = 750, 150, 8, 10
RAMP_S = FLASH_S / 2                                           # opacity ramp at twice the sweep's rate
SWEEP_W = MARK_W + 1 + len(WORD)                               # the glint crosses the logo then the wordmark
RESET, BOLD = '\x1b[0m', '\x1b[1m'


def _h(slot: int) -> int:
    return int.from_bytes(hashlib.sha256(f'hypernym-cameo|{slot}'.encode()).digest()[:8], 'big')


def phase(now: float) -> Tuple[str, float]:
    """('hidden' | 'alpha' | 'shiny', progress 0..1 through the appearance)."""
    for slot in (int(now // SLOT_S), int(now // SLOT_S) - 1):  # an appearance may spill into the next slot
        h = _h(slot)
        start = slot * SLOT_S + h % JITTER_S
        if start <= now < start + APPEAR_S:
            return ('shiny' if (h >> 20) % SHINY_ONE_IN == 0 else 'alpha'), (now - start) / APPEAR_S
    return 'hidden', 0.0


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def gentle_phase(now: float, rest_alpha: float) -> Tuple[str, float, float]:
    """('solid' | 'shiny', sweep progress 0..1, opacity). Resting opacity = rest_alpha; it eases up to 1.0 over RAMP_S
    before a flash (twice the sweep's rate), stays OPAQUE for the whole flash, and eases back down over RAMP_S."""
    base = int(now // GENTLE_SLOT_S)
    for slot in (base - 1, base, base + 1):                    # a ramp can spill across slot edges
        h = int.from_bytes(hashlib.sha256(f'hypernym-flash|{slot}'.encode()).digest()[:8], 'big')
        start = slot * GENTLE_SLOT_S + h % GENTLE_JITTER_S
        end = start + (2 if (h >> 24) % DOUBLE_ONE_IN == 0 else 1) * FLASH_S
        if start <= now < end:
            return 'shiny', ((now - start) / FLASH_S) % 1.0, 1.0
        if start - RAMP_S <= now < start:
            return 'solid', 0.0, rest_alpha + (1 - rest_alpha) * _smooth((now - (start - RAMP_S)) / RAMP_S)
        if end <= now < end + RAMP_S:
            return 'solid', 0.0, rest_alpha + (1 - rest_alpha) * _smooth(1 - (now - end) / RAMP_S)
    return 'solid', 0.0, rest_alpha


def _blend(c: Tuple[int, int, int], d: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c, d))


def _holo(c: Tuple[int, int, int], pos: float, row: int, p: float, width: int = SWEEP_W) -> Tuple[int, int, int]:
    """Shiny-card foil: hue rotates with the sweep, then a white glint where the (slanted) sweep crosses this glyph.
    `width` = how many cells the glint travels across (the logo + wordmark by default; an icon's own width)."""
    hh, ll, ss = colorsys.rgb_to_hls(*(x / 255 for x in c))
    hh = (hh + 0.35 * math.sin(2 * math.pi * (p * 1.5 + pos * 0.12 + row * 0.08))) % 1.0
    rr, gg, bb = colorsys.hls_to_rgb(hh, min(0.85, ll * 1.05), min(1.0, ss * 1.2))
    sweep = -3 + p * (width + 6) - row * 0.6                   # slanted glint travels past both edges
    glint = math.exp(-((pos - sweep) ** 2) / 1.6)
    return _blend((int(rr * 255), int(gg * 255), int(bb * 255)), (255, 255, 255), 0.75 * glint)


def _glyph(ch: str, c, kind: str, p: float, pos: float, row: int, bg: Tuple[int, int, int], rest_alpha: float) -> str:
    if c is None:
        return ch
    if kind == 'solid':                                        # resting: rest_alpha over the REAL background (S26e)
        r, g, b = _blend(bg, c, rest_alpha)
        return f'\x1b[38;2;{r};{g};{b}m{ch}{RESET}'
    if kind == 'alpha':
        r, g, b = _blend(bg, c, ALPHA * math.sin(math.pi * p))  # eases in and out of the translucent level
        return f'\x1b[38;2;{r};{g};{b}m{ch}{RESET}'
    r, g, b = _holo(c, pos, row, p)
    return f'{BOLD}\x1b[38;2;{r};{g};{b}m{ch}{RESET}'


def logo(now: float, bg: Tuple[int, int, int], mode: str, rest_alpha: float, mark: str) -> Tuple[List[str], str]:
    """(4 styled logo rows, MARK_W columns each; styled wordmark, len(WORD) columns); same-width blanks when hidden."""
    if mode == 'always':
        kind, p = 'shiny', (now % LOOP_S) / LOOP_S
    elif mode == 'cameo':
        kind, p = phase(now)
    elif mode == 'gentle':
        kind, p, rest_alpha = gentle_phase(now, rest_alpha)     # opacity now follows the flash envelope
    else:
        raise ValueError(f"brand mode must be 'gentle', 'cameo' or 'always', got {mode!r}")
    if mark == 'full':
        shape = MARK
    elif mark == 'compact':
        shape = COMPACT
    elif mark == 'mid':
        shape = MID
    elif mark == 'house':
        shape = None
    else:
        raise ValueError(f"brand mark must be 'house', 'full', 'mid' or 'compact', got {mark!r}")
    if mark == 'house':
        rows = [' ' * len(HOUSE)] * 3 if kind == 'hidden' else house_rows(kind, p, bg, rest_alpha)
    elif kind == 'hidden':
        return [' ' * len(shape[0]) for _ in shape], ' ' * len(WORD)
    else:
        rows = [''.join(_glyph(ch, c, kind, p, i, r, bg, rest_alpha) for i, (ch, c) in enumerate(row)) for r, row in enumerate(shape)]
    # S26g (pilot): the HYPERNYM letters are ALWAYS 100% opaque in their own brand colours — no alpha, no holo
    word = ''.join(f'\x1b[38;2;{c[0]};{c[1]};{c[2]}m{ch}{RESET}' for ch, c in WORD) \
        + f'\x1b[38;2;{GRAPHITE[0]};{GRAPHITE[1]};{GRAPHITE[2]}m.ai{RESET}'
    return rows, word


def icon(mark) -> str:
    """A 1-row icon (SHIELD_ICON / ZEPHYR_ICON) in its own colours, always opaque (S30)."""
    return ''.join(ch if c is None else f'\x1b[38;2;{c[0]};{c[1]};{c[2]}m{ch}{RESET}' for ch, c in mark[0])


# ------------------------------------------------------------------------------------------------ update flash (S51)
FLASH_DIR = Path.home() / '.hypernym-nebula' / 'flash'
FLASH_MIN_S, FLASH_MAX_S = 2.4, 2.9                           # > 2 s and < 3 s → exactly 3 frames at 1 fps (pilot: '1')


def icon_rows(mark, p=None) -> List[str]:
    """ANY braille icon mark (tuple of rows of (char, rgb|None)), one styled string per row. p None = its own colours,
    opaque (= icon() for a 1-row mark); p in 0..1 = the shiny frame at that point of the sweep: the logo's holographic
    foil + a slanted white glint travelling across the icon's OWN width, so it fits any size."""
    width = max(len(r) for r in mark)
    out = []
    for r, row in enumerate(mark):
        cells = []
        for i, (ch, c) in enumerate(row):
            if c is None:
                cells.append(ch)
            elif p is None:
                cells.append(f'\x1b[38;2;{c[0]};{c[1]};{c[2]}m{ch}{RESET}')
            else:
                rr, gg, bb = _holo(c, i, r, p, width)
                cells.append(f'\x1b[38;2;{rr};{gg};{bb}m{ch}{RESET}')   # colour only: bold braille can
        out.append(''.join(cells))                                     # fall back to a font drawing ALL 8 dots
    return out


def flash_progress(key: str, signature: str, now: float, state_dir: Path = FLASH_DIR):
    """Sweep progress (0..1) while `key`'s update flash runs, else None. A flash starts when `signature` (whatever means
    'the data behind this icon changed') differs from the last one seen for `key`, and lasts a random FLASH_MIN_S..
    FLASH_MAX_S. The first sighting of a key only records it (no flash when a window opens). State: one small JSON per
    key in state_dir holding a HASH of the signature (never its content), the start time and the duration."""
    kid = hashlib.sha256(key.encode()).hexdigest()[:24]
    sig = hashlib.sha256(signature.encode()).hexdigest()
    f = state_dir / f'{kid}.json'
    st = json.loads(f.read_text()) if f.exists() else None
    if st is None or st['sig'] != sig:
        st = {'sig': sig, 'at': now if st is not None else 0.0, 'dur': random.uniform(FLASH_MIN_S, FLASH_MAX_S)}
        state_dir.mkdir(parents=True, exist_ok=True)
        tmp = state_dir / f'{kid}.{os.getpid()}.tmp'
        tmp.write_text(json.dumps(st))
        tmp.replace(f)
    t = now - st['at']
    return (0.3 + 0.45 * t / st['dur']) if 0.0 <= t < st['dur'] else None   # the glint enters left, leaves right


def icon_flash(mark, key: str, signature: str, now: float, state_dir: Path = FLASH_DIR) -> str:
    """The unified 'this just updated' flash for ANY icon: its rows (joined by newline for a multi-row mark), shiny
    while `key`'s flash runs (see flash_progress), its own colours otherwise. The status line repaints at most once a
    second at idle (Claude Code's minimum refreshInterval), more often while replies stream (300 ms debounce) — so the
    flash is 3 frames at idle (the glint visibly crossing) and a smoother sweep when the bar repaints faster."""
    return '\n'.join(icon_rows(mark, flash_progress(key, signature, now, state_dir)))


WINGS_CANDIDATES = Path(__file__).resolve().parent / 'brand_assets' / 'wings_candidates.json'
WING_LEFT, WING_RIGHT, WING_MID = (205, 222, 255), (255, 210, 222), (230, 216, 238)


def wings_icon(n: int):
    """Candidate #n of the WINGS contact sheet (brand_assets/wings_contact_sheet.png, S32c) as an icon mark.
    Colour per character: its dots all in the left half → soft blue-white; all in the right half → soft pink;
    straddling the middle → a blend. The pilot's pick is #76 (the default)."""
    cands = json.loads(WINGS_CANDIDATES.read_text())
    if not 0 <= n < len(cands):
        raise ValueError(f"brand.wings must be 0..{len(cands) - 1} (see brand_assets/wings_contact_sheet.png), got {n}")
    c = cands[n]
    w, half = c['width'], c['width'] / 2
    cells = []
    for k, ch in enumerate(c['glyph']):
        on = [col for col in (2 * k, 2 * k + 1) if col < len(c['rows'][0]) and any(r[col] == '#' for r in c['rows'])]
        if not on:
            cells.append((ch, None))
        elif all(col < half for col in on):
            cells.append((ch, WING_LEFT))
        elif all(col >= half for col in on):
            cells.append((ch, WING_RIGHT))
        else:
            cells.append((ch, WING_MID))
    return (tuple(cells),)
