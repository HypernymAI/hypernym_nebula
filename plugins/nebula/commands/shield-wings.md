---
description: Choose the wings beside HYPERNYM.ai on the nebula badge — opens the numbered contact sheet; give a number to switch (e.g. 91), or on/off for all badge icons.
argument-hint: [number | on | off]
allowed-tools: mcp__plugin_nebula_nebula__shield_wings
---

The user's argument to this command: `$ARGUMENTS`
- empty → call the nebula MCP tool `shield_wings` with `choice: "show"` (it opens the numbered contact sheet); then
  tell them to pick a number from it and run /nebula:shield-wings <number>
- a number, `on` or `off` → call `shield_wings` with that `choice`
- anything else → say this command takes a number, `on`, `off` or nothing, and call nothing.
Show the tool's output verbatim. Do not run anything else.
