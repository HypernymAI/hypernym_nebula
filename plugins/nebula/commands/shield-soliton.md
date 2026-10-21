---
description: Soliton (beta) — when a reply peaks orange on the shield, flag it with a vivid banner and have Claude rephrase it once. "on", "off", or nothing to see the current state.
argument-hint: [on | off]
allowed-tools: mcp__plugin_nebula_nebula__shield_soliton
---

The user's argument to this command: `$ARGUMENTS`
- empty → call the nebula MCP tool `shield_soliton` with `state: "status"`
- `on` or `off` → call `shield_soliton` with that `state`
- anything else → say this command takes `on`, `off` or nothing, and call nothing.
Show the tool's output verbatim. Do not run anything else.
