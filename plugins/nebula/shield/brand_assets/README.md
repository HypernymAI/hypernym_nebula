# Hypernym logo on the nebula status-line badge

Everything needed to understand, reproduce and change the Hypernym logo that sits on the nebula judgement badge
in Claude Code's status line. Build-log entries: `__production_steps/2026-09-27_dfc_solid_state_shield_classifier_build.md`
S26 → S26o.

```
⣠⠟⢃⡼⠋⣤
⣿⠀⢈⡁⠀⣿  🛡 confe ▱▱▱▱▱ 0.1  handb ▰▱▱▱▱ 0.4  claim ▰▰▱▱▱ 0.9  repet ▰▱▱▱▱ 0.6  HYPERNYM.ai
⠛⣠⡞⢡⣴⠋
```

---

## 1. Requirements (the pilot's, in order given)

| # | requirement | where it lives |
|---|---|---|
| R1 | Hypernym logo next to the badge; the image + the coloured HYPERNYM letters from the website | `brand.py` `MID`, `WORD` |
| R2 | Mostly translucent, sometimes opaque, **shiny-trading-card** flash | `brand._holo`, `gentle_phase` |
| R3 | Higher frame rate, smooth | status line `refreshInterval: 1` (the max Claude Code repaints) + slow easing |
| R4 | The logo must be **the actual logo**, properly scaled — "it's a negative-space H" | `MID` (12×12 braille traced from the PNG) |
| R5 | Gentler: resting still; one shiny flash every **10–15 min at random**; **1 in 10 is a double flash** | `GENTLE_*`, `DOUBLE_ONE_IN` |
| R6 | Blend with **whatever the user's real background is**, not just "lighter text" | `termbg.py` |
| R7 | Resting opacity **10%**; eases to opaque at **twice the sweep's rate**; **opaque for the whole flash** | `rest_alpha 0.10`, `RAMP_S = FLASH_S / 2` |
| R8 | The **HYPERNYM letters always 100% opaque** in their own brand colours (only the mark fades/flashes) | `logo()` wordmark line |
| R9 | Wordmark on the **shield line**, right after `repet`; `.ai` in **graphite** | `shield_badge.py`, `GRAPHITE` |
| R10 | **3 rows** is the size ("I like this one size the best"); the **larger 4-row one was nice too** → keep both | `mark: mid` (default) / `full` |
| R11 | Nothing that rendered the logo may slip away: on disk, reproducible, documented | this directory |

## 2. What renders it (files)

| file | role |
|---|---|
| `../brand.py` | pure functions of `(now, bg, mode, rest_alpha, mark)` → styled logo rows + wordmark. Holds the marks as constants. No I/O. |
| `../termbg.py` | the REAL background colour of the Terminal tab the status line is drawn in (Apple Terminal, AppleScript by tty; cached per tab 300 s in `~/.nebula/termbg.json`). Failure is SHOWN on the badge, never replaced by a guess. |
| `../shield_badge.py` | the status-line command: judgement bars + logo rows at the LEFT (column-aligned whatever width the terminal gives 🛡) + wordmark on the middle row. |
| `../build_config.py` → `../config.json` | `brand: {enabled, bg: 'auto' \| [r,g,b], rest_alpha, mark, mode}` |
| `generate_marks.py` | rebuilds `MID` and `MARK` (FULL) from `source/hypernym-logo.png`; `--check` proves byte-identity (also test 9 of `../test_claude_integration.py`). |
| `source/hypernym-logo.png` | the logo, copied from `hypernym/portable_hypernym_processor/tools/runtime_code_compression_website/public/hypernym-logo.png` (sha256 `10db0081…6395c53`, identical). |
| `previews/` | every rendering tried, drawn next to the real logo (see §6). |

`~/.claude/settings.json` → `statusLine: {type: command, command: <venv python> …/shield_badge.py, refreshInterval: 1}`.

## 3. The mark: how the logo becomes text

**Geometry.** A terminal cell is ~1 wide : 2 tall. A braille character is 2 dots wide × 4 dots tall, so braille dots
are SQUARE. An N×N-dot logo is therefore N/2 characters wide × N/4 rows tall:

| mark | rows | chars | dots | status |
|---|---:|---:|---|---|
| `mid` | 3 | 6 | 12×12 | **default — chosen** |
| `full` | 4 | 8 | 16×16 | kept (the "older larger one") |
| `compact` | 1 | 3 | 6×4 | kept, hand-built negative-space H without tabs |
| `house` | 3 | 5 | 5×5 half-block pixels | kept, rejected by the pilot |

**Tracing (`generate_marks.py`, exact).**
1. INK = alpha > 128 and min(r,g,b) < 235; crop to the ink bounding box `[46:210, 46:210]` (164×164).
2. Each dot = mean ink coverage of its block; ON if > 0.50 (`mid`) / 0.45 (`full`).
3. Pack 2×4 dots per character, Unicode braille bit order `[[01,08],[02,10],[04,20],[40,80]]` + U+2800.
4. One colour per character (a glyph has one foreground): the median colour of the ink under it, saturation-lifted
   (HLS lightness clamped 0.55–0.72, saturation ×1.35 + 0.10) so the gradient survives a dark background.

The `mid` dot grid is exactly 180°-symmetric (like the logo): the tabs, the diagonal joins, both bars, the two centre
blobs and the **negative-space H** (empty stems + crossbar) are all present:

```
··●●●··●●●··     top tabs
··●●●··●●···
·●●···●●··●●     diagonal joins
●●···●●···●●     bar · upper blob · bar
●●···●●···●●
●●········●●     crossbar (empty)  ← the H is the empty space
●●········●●
●●···●●···●●     bar · lower blob · bar
●●···●●···●●
●●··●●···●●·     diagonal joins
···●●··●●●··     bottom tabs
··●●●··●●●··
```

Known imperfection: the logo's centreline lands exactly on a character border, so the centre blobs show a hairline
split. Inherent at 12 dots. (`full`'s 16×16 grid is not exactly symmetric at threshold 0.45.)

**Wordmark.** `HYPERNYM` per-letter colours from hypernym.ai's footer brand text
(`…/public/index.html:1154`): H `#4f85ce` Y `#6db3d4` P `#b0d893` E `#dbd06f` R `#ff9a8b` N `#8b91e4` Y `#e38cc4`
M `#94da97`, then `.ai` in graphite `#80848a`. Always opaque, never animated (R8).

## 4. Motion and alpha

Terminals draw glyphs opaque — there is no per-glyph alpha. "Alpha" = the colour blended toward the **actual**
background behind it: `out = bg + (logo − bg) × opacity`, with `bg` read live from the Terminal tab (R6). This tab:
profile "Grass" `rgb(98, 7, 44)`; the default profile "Ocean" is blue — a fixed guess would be wrong per window.

**`gentle` mode (default).**
- Resting: the mark still, at `rest_alpha` = 10%.
- A flash every 10–15 min: time is cut into 750-s slots; each slot's flash starts at `slot + (sha256(slot) mod 150)`
  → consecutive flashes 600–900 s apart. Deterministic: every window flashes at the same moment, no shared state.
- 1 flash in 10 is a DOUBLE (two 8-s sweeps back to back). Simulated week: 806 flashes, gaps 10.1–14.8 min, 81 doubles.
- Envelope: smoothstep ease 10% → 100% over 4 s before the flash (twice the sweep's rate), 100% for the whole flash,
  4 s back down. At 1 fps: `0.10 0.24 0.55 0.86 [1.00 × 8 s or 16 s] 0.86 0.55 0.24 0.10`.
- The flash ("shiny card"): per glyph, hue rotated by `0.35·sin(2π(1.5p + 0.12·col + 0.08·row))`, lightness ×1.05,
  saturation ×1.2, then a white glint `exp(−(col − sweep)²/1.6) × 0.75` on a sweep slanted by 0.6 col per row that
  crosses the whole logo. Bold.

Other modes: `cameo` (appears every 21–59 s for 6 s; mostly translucent, 1 in 5 shiny), `always` (continuous 12-s
shiny loop — the preview mode used while tuning).

## 5. Change it

```
python3 build_config.py                          # defaults: mark mid, mode gentle, rest_alpha 0.10, bg auto
python3 build_config.py --brand-mark full        # the 4-row 16×16 logo
python3 build_config.py --brand-mode always      # continuous shiny (preview)
python3 brand_assets/generate_marks.py           # print the regenerated constants
python3 brand_assets/generate_marks.py --check   # prove the shipped marks match the PNG
```
`bg` can be a fixed `[r, g, b]` in `build_config.py` for terminals other than Apple Terminal (the badge says so in red
until then). Changing a mark: edit the generator (or its thresholds), regenerate, paste into `brand.py`, re-run
`--check` and the test suite, and log it in the build doc.

## 6. What was tried and why it was rejected (previews/)

| attempt | result |
|---|---|
| `▞▚` two block glyphs (S26) | not the logo |
| `⢎⣍⡱` 1-row braille from the PNG (S26b) | 4 dots of height cannot show the shape |
| 4-row 16×16 braille (S26c) | correct shape; kept as `full` ("the larger one was nice too") |
| `⢞⡵` 4×4 structural pair (S26j) — `rejected_pair_4x4.png` | drew the pieces, missed the negative-space H |
| averaging / exhaustive MSE search over all 65,536 4×4 patterns | a ring / a checkerboard dither |
| `⡇⣉⢸` 6×4 negative-space H (S26k) — `compact_6x4_negative_H.png` | right idea, no room for tabs (needs 5 vertical bands); kept as `compact` |
| half-block pixels 6/8/10 (S26m) — `rejected_halfblock_6_8_10.png` | blurry at this size |
| ClaudeStorm house H (`ux/hypernym_logo.py` 5×5) in half blocks — `house_*.png` | "looks like shit"; kept as `house` |
| every ClaudeStorm Image-Bay art mode (`tools/zephyr_renderers.py`: 120 glyph-matching + braille dithers + rich-pixels + ansi, 136 renders) — `zephyr_all_modes_*.png` | "some not bad, nothing better than present" |
| **12×12 braille, 3 rows (S26l)** — `mid_12x12_braille_CHOSEN.png` | **chosen**: "looks fucking phenomenal" |

Note: in the zephyr contact sheets the braille/luminance tiles show boxes — the sheet was drawn in Menlo, which has
no braille glyphs; the engines themselves were fine.

## 7. Limits (stated, not hidden)
- Frame rate = the status line's `refreshInterval` (1 s). Motion is eased so 1-s steps read smoothly.
- Apple Terminal on macOS 14 has no 24-bit colour: truecolour is mapped to its 256-colour palette, so gradients and
  fades land on palette steps (smooth in iTerm2 / Ghostty).
- A translucent Terminal window shows the desktop behind it; the blend uses the profile colour, not the wallpaper.
- Background auto-detection is Apple-Terminal-only today.

## 8. Icons: shield and Zephyr (S29)

Requested 3×1 (one line, three braille characters = 6×4 square dots). Constants: `SHIELD_ICON`, `ZEPHYR_ICON` in
`../brand.py`; preview `previews/icons_3x1_CHOSEN.png`. Not yet placed on the badge.

| icon | glyphs | dots | design |
|---|---|---|---|
| shield | `⠫⣿⠝` | `######` `#.##.#` `.####.` `..##..` | heater shield; the two holes are the Hypernym H's stems; blue · green · coral (logo map) |
| Zephyr | `⢤⣶⡾` | `.....#` `..####` `######` `.####.` | the ZEPHYR racer (`images/zephyr/pink/design_racer.png`, the pink F-Zero-style craft, Hypernym's own design) in profile: pointed nose, canopy, tall rear fin; hot pink → livery pink → thruster magenta |

Built by hand from structure, like the chosen logo: auto-traces at 6×4 averaged into smears. `make_icons.py` keeps the
larger traced versions and their sources: `source/shield.png` (drawn: heater shield in the logo gradient with the
negative-space H), `source/zephyr_ship.png` (the craft cut from the sheet by hull colour; its 3-row trace still picks up
the dark ground shadow — needs a shadow mask before use), `source/zephyr_emblem.png` (the winged-diamond badge from the
same sheet; 3-row trace `ZEPHYR_EMBLEM` reads well). Preview of those: `previews/icons_preview.png`.

## 9. Choose your wings — the contact sheet (S32c)

The icon beside HYPERNYM.ai is any one of **160 braille wing designs**, all on one numbered sheet:
`wings_contact_sheet.png` (large + at badge size beside the wordmark). Default **#76** (the author's pick):
`##..##` / `.#..#.` / `..##..` → `⠙⠤⠋`.

**For users**
1. `/nebula:shield-wings` — opens the contact sheet in your image viewer and shows your current wings.
2. Pick a number. `/nebula:shield-wings 91` — saved in `~/.nebula/settings.json` (`"brand": {"wings": 91}`);
   the badge changes within a second.
3. `/nebula:shield-wings off` / `on` — all badge icons off (the plain 🛡) / on.
Even numbers: the wings touch (3 characters). Odd: one-dot gap (4 characters).

**How the sheet is made** — `make_wings_sheet.py`: every 3×3 left-wing dot pattern that is 4–7 dots, tip high at the
outer top, connected, reaching toward the body, with a feathered (gapped) lower edge (80 shapes), mirrored for the right
wing, with a 0- or 1-dot gap (160). `make_wings_sheet.py --check` proves the shipped `wings_candidates.json` is exactly
what the generator produces. Colours: left wing soft blue-white, right soft pink, a character straddling the middle a
blend (`brand.wings_icon`).
