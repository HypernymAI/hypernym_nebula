---
name: hypernym-nebula
description: When the user says "hypernym nebula", switch on the Nebula shield's soliton mode — a reply that peaks orange gets a vivid SHIELDED banner and is rephrased once. "hypernym nebula off" switches it off (pass "off" as the argument).
argument-hint: [on | off]
allowed-tools: Bash(*)
---

!`A="$ARGUMENTS"; [ -z "$A" ] && A=on; "${CLAUDE_PLUGIN_ROOT}/bin/nebula-run" customize.py soliton "$A"`

Show the output above verbatim and stop there: no question, no offer, no next step. Do not run anything else.
If no command output appears above (Codex runs no `!` line), call the nebula MCP tool `shield_soliton` with state "on"
— or "off" when the user said off — instead, and show its returned text verbatim, the same way.

(Soliton, beta: when a reply peaks orange on the shield — confession, handback or claim at 1.0× or more — a bright
SHIELDED banner appears under it, the spinner shows "shielding", and Claude rephrases it once, keeping anything the user
asked for on purpose. Switching it on also allows nebula's own local scoring tools, so no permission prompt interrupts a
rephrase; the output lists exactly what it allowed. Nothing leaves the computer.)
