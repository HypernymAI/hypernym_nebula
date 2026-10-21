---
description: Never read, score or store sessions from matching projects (glob on the project folder name or working directory).
argument-hint: <glob, e.g. *client-x*>
allowed-tools: mcp__plugin_nebula_nebula__shield_exclude
---

The user's argument to this command: `$ARGUMENTS`
- a glob (e.g. `*client-x*`) → call the nebula MCP tool `shield_exclude` with `pattern` set to exactly that text
- nothing → say this command needs a glob on the project folder name or working directory, and call nothing.
Show the tool's output verbatim. Do not run anything else.
