---
description: Remove a project exclusion.
argument-hint: <glob>
allowed-tools: mcp__plugin_nebula_nebula__shield_include
---

The user's argument to this command: `$ARGUMENTS`
- a glob → call the nebula MCP tool `shield_include` with `pattern` set to exactly that text
- nothing → say this command needs the excluded glob to remove, and call nothing.
Show the tool's output verbatim. Do not run anything else.
