---
description: Choose what transcript text nebula keeps — none (default: scores and counts only), excerpt (first 300 chars), or full.
argument-hint: none|excerpt|full
allowed-tools: mcp__plugin_nebula_nebula__shield_retention
---

The user's argument to this command: `$ARGUMENTS`
- `none`, `excerpt` or `full` → call the nebula MCP tool `shield_retention` with that `level`
- anything else (or nothing) → say this command takes `none`, `excerpt` or `full`, and call nothing.
Show the tool's output verbatim. Do not run anything else.
