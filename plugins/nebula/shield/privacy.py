#!/usr/bin/env python3
"""
nebula privacy controls — everything the plugin keeps is in ~/.nebula on THIS machine; this tool shows it, exports
it, deletes it, and changes what is kept.

    privacy.py report                     what is stored, where, how much, what is excluded, what is retained
    privacy.py export                     write everything to ~/.nebula/export-<time>.json (texts decoded if kept)
    privacy.py purge                      show what WOULD be deleted
    privacy.py purge yes                  delete the database, caches, logs and locks (settings.json is kept)
    privacy.py retention none|excerpt|full
    privacy.py exclude <glob>             never read sessions whose project folder or working directory matches
    privacy.py include <glob>             remove an exclusion
NO .get(). Imports at top.
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import json
import os
import sqlite3
import sys
import time
import zlib
from pathlib import Path

from settings import RETENTION, USER_DIR, USER_SETTINGS, resolve_config, user_settings, _validate_user

TABLES = ('sessions', 'pilot_turns', 'replies', 'texts', 'checkpoints', 'decisions')
GUARANTEE = """  • No network: every nebula process installs a Python audit hook that blocks all network connections and DNS
    lookups (noegress.py). There is no telemetry, no upload, no update check. Nothing leaves this machine.
  • No training: the models are fixed files shipped with the plugin and verified by sha256 every time they load.
    Nothing you write is used to train, tune or change them — here or anywhere else.
  • Your data stays in ~/.nebula, readable only by your user account. Delete it any time: /nebula:shield-purge"""


def db_path() -> Path:
    return Path(json.loads(resolve_config(None).read_text())['db_path'])


def write_settings(u: dict) -> None:
    _validate_user(u)
    USER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = USER_SETTINGS.with_name(f'{USER_SETTINGS.name}.{os.getpid()}.tmp')   # own temp file: no write race
    tmp.write_text(json.dumps(u, indent=1, sort_keys=True) + '\n')
    tmp.replace(USER_SETTINGS)


def report() -> None:
    cfg = json.loads(resolve_config(None).read_text())
    db = Path(cfg['db_path'])
    print("nebula — what is kept, and where\n")
    print(GUARANTEE + "\n")
    print(f"  enabled:          {cfg['enabled']}   mode: {cfg['mode']}")
    print(f"  text retention:   {cfg['text_retention']}   "
          + {'none': '(only scores and counts — no transcript text is stored)',
             'excerpt': '(the first 300 characters of each message are stored)',
             'full': '(full message text is stored, compressed)'}[cfg['text_retention']])
    print(f"  excluded:         {cfg['exclude_projects'] if cfg['exclude_projects'] else 'nothing'}")
    print(f"  your settings:    {USER_SETTINGS if USER_SETTINGS.exists() else '(none — using the defaults)'}")
    print(f"  database:         {db}")
    if not db.exists():
        print("                    not created yet — nothing stored")
        return
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        have = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in TABLES:
            if t in have:
                print(f"    {t:<12} {c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>8} rows")
        if 'texts' in have:
            kept = c.execute("SELECT retention, COUNT(*), SUM(LENGTH(body)) FROM texts GROUP BY retention").fetchall()
            for ret, n, size in kept:
                print(f"    texts kept as '{ret}': {n} ({(size if size else 0) / 1024:.0f} KB compressed)")
    finally:
        c.close()
    files = sorted(p for p in USER_DIR.iterdir() if p.is_file())
    print(f"  files in {USER_DIR}: " + ', '.join(f"{p.name} ({p.stat().st_size / 1024:.0f} KB)" for p in files))


def export() -> None:
    db = db_path()
    if not db.exists():
        sys.exit("nothing to export — the database does not exist")
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {'exported_at': time.time(), 'database': str(db), 'tables': {}}
    try:
        have = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in TABLES:
            if t not in have:
                continue
            cur = c.execute(f"SELECT * FROM {t}")
            cols = [d[0] for d in cur.description]
            rows = []
            for r in cur.fetchall():
                row = dict(zip(cols, r))
                for k, v in list(row.items()):
                    if isinstance(v, bytes):
                        row[k] = zlib.decompress(v).decode('utf-8') if (t == 'texts' and k == 'body') else v.hex()
                rows.append(row)
            out['tables'][t] = rows
    finally:
        c.close()
    dst = USER_DIR / f"export-{time.strftime('%Y%m%d_%H%M%S')}.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"exported {sum(len(v) for v in out['tables'].values())} rows → {dst}")


def purge(confirm: bool) -> None:
    targets = sorted(p for p in USER_DIR.iterdir() if p.is_file() and p != USER_SETTINGS) if USER_DIR.exists() else []
    if not targets:
        print("nothing to delete")
        return
    total = sum(p.stat().st_size for p in targets)
    print(("deleting" if confirm else "would delete") + f" {len(targets)} files ({total / 2**20:.1f} MB) in {USER_DIR}:")
    for p in targets:
        print(f"  {p.name}")
    if not confirm:
        print("\nnothing deleted. To delete: /nebula:shield-purge yes   (your settings.json is kept)")
        return
    for p in targets:
        p.unlink()
    print("done — the shield starts from empty; history is NOT re-read unless you run /nebula:shield-backfill")


def main() -> None:
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == 'report' and len(a) == 1:
        report()
    elif cmd == 'export' and len(a) == 1:
        export()
    elif cmd == 'purge' and len(a) in (1, 2):
        if len(a) == 2 and a[1] != 'yes':
            sys.exit("purge takes no argument, or exactly 'yes'")
        purge(len(a) == 2)
    elif cmd == 'retention' and len(a) == 2:
        if a[1] not in RETENTION:
            sys.exit(f"retention must be one of {RETENTION}")
        u = user_settings()
        u['text_retention'] = a[1]
        write_settings(u)
        print(f"text retention → {a[1]} (applies to messages stored from now on; earlier texts stay as they were — "
              f"purge to remove them)")
    elif cmd in ('exclude', 'include') and len(a) == 2:
        u = user_settings()
        cur = u['exclude_projects'] if 'exclude_projects' in u else []
        if cmd == 'exclude':
            if a[1] in cur:
                sys.exit(f"{a[1]!r} is already excluded")
            cur = cur + [a[1]]
        else:
            if a[1] not in cur:
                sys.exit(f"{a[1]!r} is not in the exclusions: {cur}")
            cur = [g for g in cur if g != a[1]]
        u['exclude_projects'] = cur
        write_settings(u)
        print(f"excluded projects → {cur if cur else 'nothing'} (already-stored data from them is kept — purge to remove)")
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
