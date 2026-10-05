#!/usr/bin/env python3
"""
Claude Code STATUS LINE: a red-gradient judgement badge for the latest reply of this session (build doc S23).

stdin: the status-line JSON Claude Code passes (session_id, …). Reads ~/.nebula/nebula.db READ-ONLY (the Stop hook
writes it) — no scoring here, so it costs ~ms. Per behavior class (confession / handback / claim / repetition): level =
max over its heads of score / threshold. Colour: grey → amber → deep red as level rises past 1.0 (flagged);
certified detectors BLINK when flagged. Truecolor ANSI.

SELF-HEALING (S24): the Stop hook is the normal writer, but if it never runs (plugin hooks not loaded in that Claude
Code process — observed 07:08–07:56 on 2026-09-27: the badge showed one stale reply for 50 minutes) the badge would
lie by staleness. Each refresh therefore checks, cheaply: (1) has the last turn ENDED (last user/assistant/system record
is system turn_duration / stop_hook_summary), and (2) does the transcript hold user/assistant records after the DB's
checkpoint for it? Both → start sync_one.py detached (one lock per session) and SAY so on the badge; a crashed sync is
shown as failed (traceback in <db dir>/sync.log), never hidden. While a turn is running the badge marks the shown
score as the previous reply's (⋯).

    echo '{"session_id": "…"}' | <python> shield_badge.py [--config config.json] [--row]
--row: ONLY the judgement row (shield icon + four bars + note). The shared hypernym-nebula status line
(nebula_statusline.py) draws the Hypernym logo, its shine and the wordmark around it — for either plugin (S39).
NO .get().
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from brand import SHIELD_ICON, icon, icon_flash, logo, wings_icon
from settings import excluded, resolve_config
from termbg import background
from transcript_state import records, turn_ended

HERE = Path(__file__).resolve().parent
CLASS_OF = {'H08_S2': 'confession', 'H08_S6': 'confession', 'H15_S2': 'handback', 'H15_S6': 'handback',
            'H06_S2': 'claim', 'H13_S6': 'repetition'}
ORDER = ('confession', 'handback', 'claim', 'repetition')
# the badge's words (pilot, 2026-09-29: "confe makes no immediate sense. confess. handback. claim. repeat.")
LABEL = {'confession': 'confess', 'handback': 'handback', 'claim': 'claim', 'repetition': 'repeat'}
RESET, BLINK, BOLD, DIM = '\x1b[0m', '\x1b[5m', '\x1b[1m', '\x1b[2m'
DELTA_MAX = 4 * 1024 * 1024                            # a larger unrecorded tail is unrecorded without parsing it
SYNC_RUNNING_S = 120                                   # a lock younger than this = a sync in flight
SYNC_RETRY_S = 300                                     # a lock older than this = failed sync; retried after this


def unrecorded(c: sqlite3.Connection, jsonl: Path) -> bool:
    """True iff the transcript has user/assistant records past the DB's checkpoint for it (or no checkpoint)."""
    row = c.execute("SELECT size FROM checkpoints WHERE source_path=?", (str(jsonl),)).fetchone()
    if row is None:
        return True
    size = jsonl.stat().st_size
    if size - row[0] > DELTA_MAX:
        return True
    with open(jsonl, 'rb') as f:
        f.seek(row[0])
        recs = records(f.read())
    return any(isinstance(r, dict) and 'type' in r and r['type'] in ('user', 'assistant') for r in recs)


def has_tracker(db: Path) -> bool:
    """The DB file can exist WITHOUT the tracker's tables: the decision log (shield_core.DecisionLog — MCP server, hooks)
    creates nebula.db first, the tracker tables appear only at the first sync (seen 2026-09-29 on the pilot's second Mac:
    'no such table: checkpoints'). Such a DB is treated exactly like no DB: nothing recorded yet."""
    if not db.exists():
        return False
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracker_schema'").fetchone() is not None
    finally:
        c.close()


def catch_up(config: str, jsonl: Path, state: Path) -> str:
    """'started' | 'running' | 'failed' — never silent."""
    lock = state / f'sync_{jsonl.stem}.lock'
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < SYNC_RUNNING_S:
            return 'running'
        if age < SYNC_RETRY_S:
            return 'failed'
        lock.unlink()
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"{time.time()}\n".encode())
    os.close(fd)
    log = open(state / 'sync.log', 'a')
    subprocess.Popen([sys.executable, str(HERE / 'sync_one.py'), '--config', config, '--transcript', str(jsonl),
                      '--lock', str(lock)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                     start_new_session=True, cwd=str(HERE))
    log.close()
    return 'started'


def rgb(level: float):
    """grey (level 0) → amber (1.0) → deep red (≥ 3.0)."""
    if level < 1.0:
        t = max(0.0, level)
        return (int(110 + (255 - 110) * t), int(110 + (170 - 110) * t), int(110 - 110 * t))
    t = min(1.0, (level - 1.0) / 2.0)
    return (255, int(170 - 150 * t), int(40 * (1 - t)))


def bar(level: float, cells: int = 5) -> str:
    n = int(round(min(level / 3.0, 1.0) * cells))
    return '▰' * n + '▱' * (cells - n)


def render(levels: dict, icons: bool, lead_icon: str = '') -> str:
    parts = []
    for k in ORDER:
        lv, cert = levels[k]
        r, g, b = rgb(lv)
        style = (BLINK + BOLD if (lv >= 1.0 and cert) else '') + f'\x1b[38;2;{r};{g};{b}m'
        parts.append(f"{style}{LABEL[k]} {bar(lv)} {min(lv, 9.9):.1f}{RESET}")
    worst = max(v[0] for v in levels.values())
    r, g, b = rgb(worst)
    lead = (lead_icon or icon(SHIELD_ICON)) if icons else f"\x1b[38;2;{r};{g};{b}m🛡{RESET}"   # S30; 🛡 = backup
    return lead + ' ' + '  '.join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--row', action='store_true')
    args = ap.parse_args()
    event = json.load(sys.stdin)
    cfg_path = resolve_config(args.config)
    cfg = json.loads(cfg_path.read_text())
    mark = icon(SHIELD_ICON) if cfg['brand']['icons'] is True else '🛡'   # the braille shield, never the emoji
    if cfg['enabled'] is not True:
        print(f'{mark} off')
        return
    if excluded(cfg, event['transcript_path'], event['cwd'] if 'cwd' in event else None):
        print(f'{mark} not tracked here (excluded project)')
        return
    db = Path(cfg['db_path']).expanduser()
    jsonl = Path(event['transcript_path'])
    if not jsonl.exists():                          # session just started: Claude Code has not written it yet —
        print(f'{mark} waiting for the first reply')     # an explicit wait state, not fake data (CLAUDE.md Example 4)
        return
    ended = turn_ended(jsonl)
    if not has_tracker(db):                        # no DB, or a DB the tracker has not initialised yet (S57)
        if ended:                                   # nothing ever recorded — the hook is not writing; catch up
            db.parent.mkdir(parents=True, exist_ok=True)
            st = catch_up(str(cfg_path), jsonl, db.parent)
            print(f'{mark} ⟳ recording replies the Stop hook missed' if st in ('started', 'running') else
                  f"{mark} ✖ catch-up sync failed — {db.parent / 'sync.log'}")
        else:
            print(f'{mark} no data yet')
        return
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    note = ''
    if ended and unrecorded(c, jsonl):
        st = catch_up(str(cfg_path), jsonl, db.parent)
        note = (f"  {DIM}⟳ recording replies the Stop hook missed{RESET}" if st in ('started', 'running') else
                f"  \x1b[38;2;255;60;60m✖ catch-up sync failed — {db.parent / 'sync.log'}{RESET}")
    elif not ended:
        note = f"  {DIM}⋯ previous reply{RESET}"
    row = c.execute("SELECT live_scores_json, exchange_index FROM replies WHERE session_id=? AND live_scores_json IS NOT "
                    "NULL ORDER BY exchange_index DESC LIMIT 1", (event['session_id'],)).fetchone()
    if row is None:
        print(f'{mark} waiting for the first judged reply' + note)
        return
    levels = {}
    for h, s in json.loads(row[0]).items():
        k = CLASS_OF[h]
        lv = s / cfg['stop'][h]['threshold']
        cert = cfg['stop'][h]['certified'] is True
        if k not in levels or lv > levels[k][0]:
            levels[k] = (lv, cert)
    brand = cfg['brand']
    # S51 (pilot): the shield flashes shiny (3 frames, 2.4–2.9 s) whenever a NEW judgement lands (brand.icon_flash)
    lead = icon_flash(SHIELD_ICON, f"shield:{event['session_id']}", f"{row[1]}|{row[0]}", time.time())
    if args.row:
        print(render(levels, brand['icons'] is True, lead) + note)
        return
    if brand['enabled'] is not True:
        print(render(levels, False) + note)
        return
    # 4-row logo at the LEFT (so its rows stay column-aligned whatever width the terminal gives 🛡); badge on row 2,
    # wordmark after repeat on row 2 (S26h)
    alpha, mode = brand['rest_alpha'], brand['mode']
    if brand['bg'] == 'auto':
        try:
            bg = background(db.parent / 'termbg.json')           # the REAL background of this tab (S26e)
        except RuntimeError:                                     # unreadable (not Apple Terminal, Linux …): OPAQUE —
            bg, alpha = (0, 0, 0), 1.0                           # at alpha 1 nothing is blended, no colour guessed
            mode = 'gentle' if mode == 'cameo' else mode
    else:
        bg = tuple(brand['bg'])
    rows, word = logo(time.time(), bg, mode, alpha, brand['mark'])
    # logo rows at the LEFT; the shield line (badge + wordmark after `repeat`) rides the middle logo row
    k = (len(rows) - 1) // 2                                     # full 4 rows → 1 · mid 3 → 1 · compact 1 → 0
    lines = list(rows)
    if brand['icons'] is True:
        word = icon(wings_icon(brand['wings'])) + ' ' + word      # S32c: the chosen wings (contact sheet #)
    lines[k] = rows[k] + '  ' + render(levels, brand['icons'] is True, lead) + '  ' + word + note
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
