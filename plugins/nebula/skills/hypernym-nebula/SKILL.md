---
name: hypernym-nebula
description: When the user says "hypernym nebula", switch on the Nebula shield's soliton mode — a reply that peaks orange gets a vivid SHIELDED banner and is rephrased once. "hypernym nebula off" switches it off (pass "off" as the argument).
argument-hint: [on | off]
allowed-tools: mcp__plugin_nebula_nebula__shield_soliton
---

Call the nebula MCP tool `shield_soliton` with `state: "off"` when the user said off (for example "hypernym nebula
off"), otherwise with `state: "on"`. Show its returned text verbatim and stop there: no question, no offer, no next
step. Do not run anything else.

(Soliton: when a reply peaks orange on the shield — confession, handback or claim at 1.0× or more — a bright
SHIELDED banner appears under it, the spinner shows "shielding", and Claude rephrases it once, keeping anything the user
asked for on purpose. Nothing leaves the computer.)
