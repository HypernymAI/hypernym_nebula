"""
Integrity of the bundled native scorer (2026-10-21 security update).

The scorer is native code — the macOS module `_native.cpython-313-darwin.so` (loaded into the Python process) and the
static Linux programs `static/ctxshield-scorer-linux-<cpu>` (run as a child process). Python's audit hook cannot see
inside native code, so the code itself must be the code that was built: every file is checked against the sha256
pinned in `native/PLATFORMS.json` (written by native/static_build/update_platforms.py at build time) BEFORE it is
loaded or run. A missing pin or a mismatch raises ImportError — there is no fallback and no "run it anyway".

    verify(path)    raises unless sha256(path) == the pin for that file
NO .get(). Imports at top.
"""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PLATFORMS = PACKAGE.parent / 'native' / 'PLATFORMS.json'


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pinned(path: Path) -> str:
    """The sha256 PLATFORMS.json pins for this bundled file; ImportError when the file is not one of ours."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(PACKAGE).as_posix()
    except ValueError:
        raise ImportError(f"ctxshield: {p} is not part of this plugin's scorer — refusing to load it") from None
    modules = json.loads(PLATFORMS.read_text())['modules']
    if rel not in modules:
        raise ImportError(f"ctxshield: {rel} has no pinned sha256 in {PLATFORMS} — refusing to load it")
    return modules[rel]['sha256']


def verify(path: Path) -> None:
    want = pinned(path)
    got = sha256_of(Path(path))
    if got != want:
        raise ImportError(f"ctxshield: {Path(path).name} does not match its pinned sha256 (have {got[:16]}…, "
                          f"pinned {want[:16]}…) — refusing to load it; reinstall the plugin")
