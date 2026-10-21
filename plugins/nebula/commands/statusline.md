---
description: The hypernym-nebula status line (shield badge + Hypernym logo, and the Zephyr line when the hypernym plugin is installed). Set up automatically on session start; "remove" takes it out, "force" replaces someone else's status line (backed up first).
argument-hint: [remove | force]
allowed-tools: mcp__plugin_nebula_nebula__shield_statusline
---

The user's argument to this command: `$ARGUMENTS`
- empty → call the nebula MCP tool `shield_statusline` with `action: "install"`
- `remove` or `force` → call `shield_statusline` with that `action`
- anything else → say this command takes `remove`, `force` or nothing, and call nothing.
Show the tool's output verbatim. Do not run anything else.
