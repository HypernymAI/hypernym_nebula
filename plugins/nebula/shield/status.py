#!/usr/bin/env python3
"""
Shield status for the /shield-status command (build doc Part XVIII): kill switch, loaded heads + thresholds,
and the most recent decisions from the log. Read-only.

    <python> status.py [--config path] [--last 15]
NO .get().
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

import nebula_statusline
from settings import resolve_config

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--last', type=int, default=15)
    args = ap.parse_args()
    cfg = json.loads(resolve_config(args.config).read_text())
    print(f"nebula: {'ENABLED' if cfg['enabled'] is True else 'DISABLED (kill switch)'}  mode: {cfg['mode']}  "
          f"db: {cfg['db_path']}")
    print(nebula_statusline.peers('nebula'))
    print(f"models: combined {cfg['combined_dir']} (research: {cfg['model_dir']})   max rewrites: {cfg['max_rewrites']}")
    for ev in ('user_prompt', 'stop'):
        heads = cfg[ev]
        print(f"  {ev:<12} " + (', '.join(f"{h} ({hc['model']}, {'certified' if hc['certified'] else 'working'}, "
                                          f"thr {hc['threshold']:.3f})" for h, hc in heads.items())
                                if heads else 'no heads'))
    db = Path(cfg['db_path']).expanduser()
    if not db.exists():
        print("central db: not created yet")
        return
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ('sessions', 'pilot_turns', 'replies', 'texts')}
    print(f"central db {db} ({db.stat().st_size / 2**20:.1f} MB): sessions {n['sessions']}, pilot turns "
          f"{n['pilot_turns']}, replies {n['replies']}, texts {n['texts']}")
    print("by model:")
    for m, rep, fl, rate, s2, s6 in c.execute("SELECT * FROM v_model_summary ORDER BY replies DESC"):
        print(f"  {m:<34} replies {rep:>6}  flagged {fl:>5} ({rate:.1%})")
    # every configured detector — rates from the RECORDED flags (the same models the hooks use)
    heads = list(cfg['stop'])
    by = {}
    for m, fj in c.execute("SELECT model, fired_json FROM replies"):
        f = json.loads(fj)
        if m not in by:
            by[m] = {'n': 0, **{h: 0 for h in heads}}
        by[m]['n'] += 1
        for h in heads:
            by[m][h] += int(h in f)
    print("detectors — share of replies flagged (C = certified: PASS on every judged codebase; w = working):")
    print("  " + ' ' * 26 + ''.join(f"{h + ('C' if cfg['stop'][h]['certified'] else 'w'):>9}" for h in heads))
    for m in sorted(by, key=lambda x: -by[x]['n']):
        print(f"  {m[:20]:<20} {by[m]['n']:>5}" + ''.join(f"{by[m][h] / by[m]['n']:>9.1%}" for h in heads))
    print("rageometry, last 7 days:")
    for day, turns, caps, anger, celeb, prof, rep in c.execute(
            "SELECT * FROM v_rage_daily ORDER BY day DESC LIMIT 7"):
        print(f"  {day}  turns {turns:>4}  caps {caps:>3}  anger onsets {anger:>3}  celebrations {celeb:>2}  "
              f"profanity {prof:>4}  repeats {rep:>3}")
    print(f"last {args.last} flagged replies:")
    for sid, ei, model, s2, s6 in c.execute(
            "SELECT session_id, exchange_index, model, h08_s2, h08_s6 FROM replies WHERE flagged=1 "
            "ORDER BY created_at DESC, exchange_index DESC LIMIT ?", (args.last,)):
        print(f"  {sid[:8]} #{ei:<4} {model:<28} confession text {s2:.2f} reply {s6:.2f}")


if __name__ == '__main__':
    main()
