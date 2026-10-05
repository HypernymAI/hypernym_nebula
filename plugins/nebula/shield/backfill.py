#!/usr/bin/env python3
"""
Backfill every Claude Code session on disk into the central tracker (build doc Part XX). Gentle by design:
nice(10), one session at a time, a short pause between sessions, unchanged transcripts skipped (checkpoints),
sessions modified in the last LIVE_S seconds left to the Stop hook (never records a half-written turn).
Resumable: rerun at any time; every row is keyed by message uuid.

    <python> backfill.py [--config config.json] [--pause 0.05]
NO .get().
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import argparse
import os
import sys
import time
from pathlib import Path

from settings import excluded, resolve_config
from shield_core import Shield
from tracker import Tracker
from transcript_state import turn_ended

PROJECTS = Path.home() / '.claude' / 'projects'
LIVE_S = 300
FINAL_S = 86400                    # untouched for a day: the transcript is final — record what is there (S38)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--pause', type=float, default=0.05)
    ap.add_argument('--rebuild-replies', action='store_true',
                    help='re-derive every reply AND pilot turn (texts kept, deduplicated) — needed when the reader changes '
                         'exchange boundaries (S25)')
    args = ap.parse_args()
    os.nice(10)
    sh = Shield(resolve_config(args.config))
    if not sh.enabled:
        sys.exit("nebula disabled (kill switch) — nothing backfilled")
    tr = Tracker(Path(sh.cfg['db_path']), sh)
    if args.rebuild_replies:
        c = tr._conn()
        c.execute("DELETE FROM replies")
        c.execute("DELETE FROM pilot_turns")
        c.execute("DELETE FROM checkpoints")
        c.commit()
        c.close()
        print("rebuild: replies + pilot turns + checkpoints cleared (texts kept)")
    files = sorted((p for p in PROJECTS.glob('*/*.jsonl') if not p.name.startswith('agent-')),
                   key=lambda p: p.stat().st_mtime)
    t0 = time.time()
    done = skipped = live = excl = 0
    tot_p = tot_r = 0
    for f in files:
        if excluded(sh.cfg, str(f), None):
            excl += 1
            continue
        if time.time() - f.stat().st_mtime < LIVE_S:
            live += 1
            continue
        if tr.unchanged(f):
            skipped += 1
            continue
        final = time.time() - f.stat().st_mtime > FINAL_S
        r = tr.sync_session(f, last_turn_complete=final or turn_ended(f))
        tot_p += r['new_pilot_turns']
        tot_r += r['new_replies']
        done += 1
        time.sleep(args.pause)
    print(f"backfill: {len(files)} transcripts — synced {done}, unchanged {skipped}, live (left to hook) {live}, excluded {excl}; "
          f"new pilot turns {tot_p}, new replies {tot_r}; {time.time() - t0:.1f}s; db {tr.db_path}")


if __name__ == '__main__':
    main()
