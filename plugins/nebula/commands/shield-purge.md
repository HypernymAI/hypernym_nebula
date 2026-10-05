---
description: Delete everything nebula has stored (database, caches, logs). Without "yes" it only shows what would be deleted.
argument-hint: [yes]
allowed-tools: Bash(*)
---

!`"${CLAUDE_PLUGIN_ROOT}/bin/nebula-run" privacy.py purge $ARGUMENTS`

Show the output above verbatim. Do not run anything else, and never re-run this with "yes" on your own — only the user types it.
