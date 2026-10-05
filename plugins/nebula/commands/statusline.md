---
description: The hypernym-nebula status line (shield badge + Hypernym logo, and the Zephyr line when the hypernym plugin is installed). Set up automatically on session start; "remove" takes it out, "force" replaces someone else's status line (backed up first).
argument-hint: [remove | force]
allowed-tools: Bash(*)
---

!`R="${CLAUDE_PLUGIN_ROOT}/bin/nebula-run"; case "$ARGUMENTS" in remove) "$R" install_statusline.py --remove;; force) "$R" install_statusline.py --force;; *) "$R" install_statusline.py;; esac`

Show the output above verbatim. Do not run anything else.
