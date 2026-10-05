---
description: Choose the wings beside HYPERNYM.ai on the nebula badge — opens the numbered contact sheet; give a number to switch (e.g. 91), or on/off for all badge icons.
argument-hint: [number | on | off]
allowed-tools: Bash(*)
---

!`A="$ARGUMENTS"; R="${CLAUDE_PLUGIN_ROOT}/bin/nebula-run"; if [ -z "$A" ]; then open "${CLAUDE_PLUGIN_ROOT}/shield/brand_assets/wings_contact_sheet.png" 2>/dev/null; "$R" customize.py wings; elif [ "$A" = on ] || [ "$A" = off ]; then "$R" customize.py icons "$A"; else "$R" customize.py wings "$A"; fi`

Show the output above verbatim. If no number was given, the contact sheet has just been opened in the user's image viewer — tell them to pick a number from it and run /nebula:shield-wings <number>. Do not run anything else.
