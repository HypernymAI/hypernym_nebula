---
description: nebula privacy report — exactly what is stored on this machine, where, how much, and what is excluded. Nothing leaves your machine.
allowed-tools: Bash(*)
---

!`"${CLAUDE_PLUGIN_ROOT}/bin/nebula-run" privacy.py report`

Show the report above verbatim in a code block. Then list the controls: /nebula:shield-retention none|excerpt|full, /nebula:shield-exclude <glob>, /nebula:shield-include <glob>, /nebula:shield-export, /nebula:shield-purge. Do not run anything else.
