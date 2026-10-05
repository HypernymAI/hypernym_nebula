#!/usr/bin/env python3
"""
/nebula:statusline — install or remove the SHARED hypernym-nebula status line (nebula_statusline.py; the hypernym
plugin's /hypernym-statusline runs the same code). Plugins cannot register a status line themselves, so this edits ONLY the
"statusLine" key of Claude Code's settings.json, after a timestamped backup, and never replaces a foreign status line
unless --force.

    install_statusline.py              install (also: turns automatic set-up on session start back on)
    install_statusline.py --force      install over someone else's status line (backed up first)
    install_statusline.py --remove     remove ours; it is not re-added automatically until installed again
NO .get(). Imports at top.
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import sys

import nebula_statusline


def main() -> int:
    a = sys.argv[1:]
    if a == ['--remove']:
        return nebula_statusline.remove()
    if a in ([], ['--force']):
        return nebula_statusline.install(a == ['--force'])
    sys.exit(__doc__)


if __name__ == '__main__':
    sys.exit(main())
