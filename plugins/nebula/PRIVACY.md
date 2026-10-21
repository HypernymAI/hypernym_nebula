# nebula — privacy

**nebula never sends your data anywhere, and never trains on it.** Everything below is enforced by the plugin's code
and checked by its test suite before every release — not just promised.

| guarantee | how it is enforced | proof |
|---|---|---|
| **No network. Nothing leaves your machine.** No telemetry, no uploads, no update checks. | Every nebula process loads `shield/noegress.py` first. It installs a Python audit hook — which cannot be removed once installed — that blocks every network connection, DNS lookup and HTTP request made by Python code, and blocks launching any program except: `ps` and `osascript` (local only — reading your terminal's background colour, and opening the wings preview in Preview); nebula's own Python running nebula's own scripts (the settings changes behind its commands, and catching up replies the Stop hook missed); and the plugin's own scorer. The test suite checks every launch in the code against this list. The scorer is pinned by content: the macOS scorer module and the Linux scorer program run only when they match the sha256 published in `ctxshield/native/PLATFORMS.json`, checked before the module loads and before the program starts. The Linux scorer is built reproducibly from the C++ source in `ctxshield/native/`: rebuilding it gives the identical file, and its code reads stdin and writes stdout, nothing else. The MCP server talks to Claude Code over stdin/stdout, not a socket. | N1 every entry point imports it first · N2 no network library anywhere in the code · N3 inside all 10 entry-point processes, a real socket, DNS lookup, HTTP request and `curl` launch are all blocked |
| **No training on your data.** | The models are fixed files shipped with the plugin. Every load verifies them against a sha256 manifest; nothing in the plugin writes to them. | D5 one flipped bit in a weight file → the plugin refuses to load |
| **Your text is not stored by default.** | `text_retention` defaults to `none`: the local database keeps scores, flags and counts (caps ratio, profanity count…), not what you or Claude wrote. Opt in with `/nebula:shield-retention excerpt` or `full`. | D3 after a full hook run, the secret prompt and reply text appear nowhere in the database files |
| **Your data stays in `~/.nebula`, readable only by you.** | The directory is created and kept at mode 0700. | D3 |
| **You can see all of it.** | `/nebula:shield-privacy` reports exactly what is stored, where and how much. `/nebula:shield-export` writes all of it to a readable JSON file. | D4 |
| **You can delete all of it.** | `/nebula:shield-purge` shows what would be deleted; `/nebula:shield-purge yes` deletes the database, caches and logs (your settings are kept). | D4 |
| **You can keep projects out entirely.** | `/nebula:shield-exclude <glob>` — matching sessions are never opened, scored or stored. | D2 the transcript file is never even opened (proved with a file-open audit hook) |
| **Off means off.** | `"enabled": false` in `~/.nebula/settings.json` — the hooks read nothing and create nothing. | D1 |
| **Your history is not read unless you ask.** | Only the current session is recorded, as you work. Reading past sessions is opt-in: `/nebula:shield-backfill`. | — |
| **Your commands never run as shell.** | Every `/nebula:` command and the "hypernym nebula" skill call nebula's own tool with checked values (on/off, a number, a glob passed as data); nothing you type is pasted into a shell line, and no command grants shell access. `/nebula:shield-purge yes` asks you in a dialog before deleting. | — |
| **Your permissions are yours.** | nebula never adds a permission rule to Claude Code's settings. Claude Code asks you the first time a nebula tool runs, and the choice is yours. | — |
| **Your status line is yours.** | On session start the shared hypernym-nebula status line is set ONLY when you have none (or an older one of ours), and only by a copy of the plugin that Claude Code itself installed (a Codex session never touches Claude Code's settings): `~/.claude/settings.json` is backed up first and only its `statusLine` entry changes — anything else is refused by the code. A status line that isn't ours is never replaced (unless you run `/nebula:statusline force`). `/nebula:statusline remove` takes out only ours, and it is not re-added until you install it again. The status-line program only runs the two plugins' own status programs; it has no network code. | D6 |

## What is downloaded, once

On the first session after install or update — from the plugin's session-start hook, never from the status line (which
only asks for that one restart) — `uv` creates the plugin's private Python environment (Python 3.13 + numpy, pinned by
`uv.lock` with package hashes). That is a download of **packages**, from PyPI — it sends nothing about you. After that
the plugin runs offline.

## The models were trained on the author's sessions — and gated

The detectors were trained on the plugin author's own labeled Claude Code sessions, never on yours. Before release the
weights passed a **leakage gate**: every feature seen in fewer than 3 of the author's sessions was removed and the
models retrained without them (39–58% of feature columns per detector), then re-judged against the same held-out
codebases — every verdict unchanged. Project, company and personal names were removed from the weights. A second check
compares how far a phrase from a single training session moves a score against an ordinary phrase: for four of the six
detectors there is no difference; the claim detector and one handback detector still lean slightly on session-specific
wording, which is why no name or identifier is left in any of the weights.

## Files

`~/.nebula/settings.json` (yours; updates never touch it) · `config.json` (derived) · `nebula.db` (the record) ·
`termbg.json` (terminal colour cache) · `sync.log`,
`export-*.json` (only if you export). Shared with the hypernym plugin, no data: `~/.hypernym-nebula/bin/` (the status-line
launcher + `nebula_statusline.py`) and `statusline_off` (only after you remove the status line).
