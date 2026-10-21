---
description: Delete everything nebula has stored (database, caches, logs). Without "yes" it only shows what would be deleted.
argument-hint: [yes]
allowed-tools: mcp__plugin_nebula_nebula__shield_purge
---

The user's argument to this command: `$ARGUMENTS`
- empty → call the nebula MCP tool `shield_purge` with no arguments: it only shows what would be deleted
- exactly `yes` → call `shield_purge` with `confirm: true`: a dialog asks the user, and only their Accept deletes
- anything else → say this command takes `yes` or nothing, and call nothing.
Show the tool's output verbatim. Do not run anything else, and never use `confirm: true` unless the argument above is
exactly `yes`.
