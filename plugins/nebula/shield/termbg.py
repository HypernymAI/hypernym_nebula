"""
The REAL background colour of the terminal tab the status line is drawn in (build doc S26e) — so the Hypernym logo can be
alpha-blended against what is actually behind it (terminals draw text opaque; there is no per-glyph alpha).

Apple Terminal only (TERM_PROGRAM == Apple_Terminal): walk up from this process to the first ancestor with a tty (Claude
Code's), then AppleScript: the tab of Terminal whose tty is that one → `background color` (16-bit RGB) + profile name.
Cached per tty in <cache> for MAX_AGE_S (the AppleScript round trip is ~0.75 s; the status line refreshes every 1 s).
Anything else → RuntimeError with the reason; the status line then draws the logo fully OPAQUE (alpha 1: nothing is
blended, so no colour is ever guessed).
NO .get(). Imports at top.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

MAX_AGE_S = 300
OSA_TIMEOUT_S = 5
SCRIPT = '''tell application "Terminal"
repeat with w in windows
repeat with t in tabs of w
if tty of t is "{tty}" then return {{background color of t, name of current settings of t}}
end repeat
end repeat
return "no-tab"
end tell'''


def controlling_tty() -> str:
    pid = os.getpid()
    for _ in range(12):
        out = subprocess.run(['ps', '-o', 'tty=,ppid=', '-p', str(pid)], capture_output=True, text=True, timeout=2)
        parts = out.stdout.split()
        if len(parts) != 2:
            raise RuntimeError(f"ps gave no tty/ppid for pid {pid}")
        tty, ppid = parts
        if tty not in ('??', '-'):
            return f'/dev/{tty}'
        pid = int(ppid)
        if pid <= 1:
            break
    raise RuntimeError("no ancestor process has a terminal")


def background(cache: Path) -> Tuple[int, int, int]:
    prog = os.environ['TERM_PROGRAM'] if 'TERM_PROGRAM' in os.environ else '(unset)'
    if prog != 'Apple_Terminal':
        raise RuntimeError(f"auto background supports Apple Terminal only (TERM_PROGRAM={prog}); set brand.bg to [r,g,b]")
    tty = controlling_tty()
    data = _load(cache)
    if tty in data and time.time() - data[tty]['at'] < MAX_AGE_S:
        return tuple(data[tty]['rgb'])
    r = subprocess.run(['osascript', '-e', SCRIPT.format(tty=tty)], capture_output=True, text=True,
                       timeout=OSA_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"Terminal did not answer (osascript exit {r.returncode}): {r.stderr.strip()[:120]}")
    out = r.stdout.strip()
    if out == 'no-tab':
        raise RuntimeError(f"no Terminal tab has tty {tty}")
    vals = [v.strip() for v in out.split(',')]
    rgb = tuple(int(round(int(v) / 257)) for v in vals[:3])
    data[tty] = {'rgb': list(rgb), 'profile': ','.join(vals[3:]), 'at': time.time()}
    _save(cache, data)
    return rgb


def _load(cache: Path) -> dict:
    """The per-tty cache. A corrupt cache (it happened: two windows once interleaved their writes) is DELETED and reported
    — rebuilt on the next refresh — never trusted, never allowed to take the status line down."""
    if not cache.exists():
        return {}
    try:
        return json.loads(cache.read_text())
    except json.JSONDecodeError as e:
        cache.unlink()
        raise RuntimeError(f"background cache was corrupt ({e.msg}); it is rebuilt on the next refresh")


def _save(cache: Path, data: dict) -> None:
    """Atomic and race-free: every process writes its OWN temp file (pid in the name), then renames it over the cache."""
    tmp = cache.with_name(f'{cache.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(cache)
