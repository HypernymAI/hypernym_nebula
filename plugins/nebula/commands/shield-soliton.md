---
description: Soliton (beta) — when a reply peaks orange on the shield, flag it with a vivid banner and have Claude rephrase it once. "on", "off", or nothing to see the current state.
argument-hint: [on | off]
allowed-tools: Bash(*)
---

!`R="${CLAUDE_PLUGIN_ROOT}/bin/nebula-run"; if [ -z "$ARGUMENTS" ]; then "$R" customize.py soliton; else "$R" customize.py soliton $ARGUMENTS; fi`

Show the output above verbatim. Do not run anything else.
