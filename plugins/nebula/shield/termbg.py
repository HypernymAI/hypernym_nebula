"""
The REAL background colour of the terminal tab the status line is drawn in (build doc S26e) — so the Hypernym logo can be
alpha-blended against what is actually behind it (terminals draw text opaque; there is no per-glyph alpha).

Apple Terminal only (TERM_PROGRAM == Apple_Terminal): walk up from this process to the first ancestor with a tty (Claude
Code's), then ask Terminal by AppleScript for the tab with that tty → `background color` (16-bit RGB) + profile name.
Anything else → RuntimeError with the reason; the status line then draws the logo fully OPAQUE (alpha 1: nothing is
blended, so no colour is ever guessed).

GENTLE WITH TERMINAL (2026-10-05, after Apple Terminal 2.14 hard-crashed four times — EXC_BAD_ACCESS on its main thread
in aeProcessAppleEvent → _AppleEventsLookupApplicationPort, i.e. while looking up the SENDER of an incoming AppleEvent;
the old code asked every ~1 s from every session, never cached a failure, and killed osascript at 5 s mid-request):
  - ASK RARELY: an answer is kept OK_AGE_S, a failure FAIL_AGE_S (failures are cached too), per tty, shared by every
    process through <cache>.
  - NEVER KILL A SENDER MID-REQUEST, NEVER WAIT: osascript is started DETACHED (its own session) writing its answer to
    a file beside the cache; this refresh returns at once (the last known colour, or opaque) and a later refresh reads
    the answer. At most one question per tty is in flight; one unanswered after ASK_WAIT_S counts as a failure.
  - ONE BAD WINDOW CANNOT SINK THE LOOKUP: every window is read inside its own `try` (a Terminal window that cannot
    be queried used to fail the whole walk, which was then retried forever).
NO .get(). Imports at top.
"""
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

OK_AGE_S = 1800                 # a known colour is reused for 30 min (a profile change shows within that)
FAIL_AGE_S = 600                # a failed lookup is not retried for 10 min
ASK_WAIT_S = 120                # a question unanswered this long counts as a failure
SCRIPT = '''tell application "Terminal"
repeat with w in windows
try
repeat with t in tabs of w
if tty of t is "{tty}" then return {{background color of t, name of current settings of t}}
end repeat
end try
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
    now = time.time()
    data = _load(cache)
    entry = data[tty] if tty in data else {}
    answer = _answer_path(cache, tty)
    if 'asked' in entry:                                        # a question is out: has Terminal answered?
        got = _collect(answer, entry['pid'])
        if got is not None:
            entry = _record(got, now, entry)
            data[tty] = entry
            _save(cache, data)
        elif now - entry['asked'] >= ASK_WAIT_S:
            entry = dict(entry, failed=f"Terminal did not answer within {ASK_WAIT_S} s", at=now)
            del entry['asked'], entry['pid']
            data[tty] = entry
            _save(cache, data)
    fresh_ok = 'rgb' in entry and 'failed' not in entry and now - entry['at'] < OK_AGE_S
    fresh_fail = 'failed' in entry and now - entry['at'] < FAIL_AGE_S
    if 'asked' not in entry and not fresh_ok and not fresh_fail:
        entry = dict(entry, asked=now, pid=_ask(tty, answer))   # detached; answered on a later refresh
        data[tty] = entry
        _save(cache, data)
    if 'rgb' in entry and 'failed' not in entry:
        return tuple(entry['rgb'])                              # known colour (also while a refresh is out)
    if 'failed' in entry:
        raise RuntimeError(entry['failed'])
    raise RuntimeError("asking Terminal for this tab's background (answer on a later refresh)")


def _answer_path(cache: Path, tty: str) -> Path:
    return cache.with_name(f"{cache.name}.{tty.replace('/dev/', '')}.answer")


def _ask(tty: str, answer: Path) -> int:
    """Start osascript DETACHED (own session, never waited on, never killed) → its answer lands in `answer`."""
    for p in (answer, answer.with_suffix('.err')):
        if p.exists():
            p.unlink()
    with open(answer, 'w') as out, open(answer.with_suffix('.err'), 'w') as err:
        proc = subprocess.Popen(['osascript', '-e', SCRIPT.format(tty=tty)], stdin=subprocess.DEVNULL, stdout=out,
                                stderr=err, start_new_session=True)
    return proc.pid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _collect(answer: Path, pid: int):
    """('ok', stdout) / ('err', stderr) once osascript has finished, else None (still asking)."""
    out = answer.read_text() if answer.exists() else ''
    if out.strip() and out.endswith('\n'):
        done = ('ok', out.strip())
    elif _alive(pid):
        return None
    else:
        err = answer.with_suffix('.err')
        done = ('err', err.read_text().strip() if err.exists() else 'no answer')
    for p in (answer, answer.with_suffix('.err')):
        if p.exists():
            p.unlink()
    return done


def _record(got, now: float, old: dict) -> dict:
    kind, text = got
    if kind == 'err':
        return {'failed': f"Terminal did not answer: {text[:120]}", 'at': now}
    if text == 'no-tab':
        return {'failed': "no Terminal tab has this tty", 'at': now}
    vals = [v.strip() for v in text.split(',')]
    if len(vals) < 3 or not all(v.isdigit() for v in vals[:3]):
        return {'failed': f"unexpected answer from Terminal: {text[:80]}", 'at': now}
    return {'rgb': [int(round(int(v) / 257)) for v in vals[:3]], 'profile': ','.join(vals[3:]), 'at': now}


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
