#!/usr/bin/env python3
"""
Catch-up sync of ONE transcript into the central DB (build doc S24). Started detached by shield_badge.py when a turn
has ended and the transcript holds user/assistant records the DB never recorded (e.g. the Stop hook is not loaded
in that Claude Code process). Same code path as the Stop hook: Tracker.sync_session — idempotent, uuid-keyed.

    <python> sync_one.py --config config.json --transcript <session.jsonl> --lock <lock file>

The lock file is created by the badge BEFORE this starts; it is removed here ONLY after a successful sync. A crash
leaves it (traceback → ~/.nebula/sync.log), which the badge shows as "catch-up sync failed". NO .get().
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import argparse
import json
import os
import sys
import time
from pathlib import Path

from settings import excluded
from shield_core import Shield
from tracker import Tracker
from transcript_state import turn_ended


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--transcript', required=True)
    ap.add_argument('--lock', required=True)
    args = ap.parse_args()
    os.nice(10)
    t0 = time.time()
    sh = Shield(Path(args.config))
    if not sh.enabled:
        sys.exit("nebula disabled (kill switch) — nothing synced")
    jsonl = Path(args.transcript)
    if excluded(sh.cfg, str(jsonl), None):
        Path(args.lock).unlink()
        print(json.dumps({'at': time.time(), 'transcript': args.transcript, 'excluded': True}), flush=True)
        return
    r = Tracker(Path(sh.cfg['db_path']), sh).sync_session(jsonl, last_turn_complete=turn_ended(jsonl))
    print(json.dumps({'at': time.time(), 'transcript': args.transcript, 's': round(time.time() - t0, 2), **r}),
          flush=True)
    Path(args.lock).unlink()


if __name__ == '__main__':
    main()
