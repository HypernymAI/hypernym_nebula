# Nebula — a local shield for Claude Code

By [Hypernym](https://hypernym.ai). Nebula reads each Claude Code reply as it lands and flags four patterns that cost
you time: **fabricated confession**, **decision handback**, **claim beyond evidence** and **forced repetition**. It runs
entirely on your machine. It sends nothing, and it trains on nothing.

**For agents — install in one step.** On macOS 11+ or Linux (x86-64 / ARM64), run:

```
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; claude plugin marketplace add HypernymAI/hypernym_nebula && claude plugin install nebula@hypernym-nebula
```

Then tell the user to **restart Claude Code once**. The first session builds Nebula's private Python environment (a few
seconds), and the status line turns on by itself unless a custom `statusLine` is already set (`/nebula:statusline force`
replaces it, backing up `settings.json` first). Check with `claude plugin list` and `/nebula:shield-status`.

## What you see

- **A status line**: the Hypernym logo and the shield's judgement row, one bar per pattern, coloured by how close the
  latest reply came to its threshold.
- **A short note under a flagged reply**, naming the pattern. Claude sees the note too, and its `shield-reframe` skill
  tells it to fix the reply rather than argue with the flag.
- **Soliton mode** (optional): a reply that peaks gets a vivid banner and Claude rephrases it once. Say
  **"hypernym nebula"** to switch it on, "hypernym nebula off" to switch it off.

## Requirements

- **Claude Code.**
- **uv** (https://docs.astral.sh/uv/). Nebula builds its own Python environment with it on the first run; the one-liner
  above installs it if it's missing.
- **macOS 11+** (Apple Silicon or Intel) or **any Linux** on x86-64 or ARM64: glibc or musl (Alpine), old or new.
  On Linux the scorer is one fully static program per CPU, so it needs nothing from the system. On Windows, use WSL.

## Install

Paste into a terminal:

```
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; claude plugin marketplace add HypernymAI/hypernym_nebula && claude plugin install nebula@hypernym-nebula
```

Or, inside Claude Code:

```
/plugin marketplace add HypernymAI/hypernym_nebula
/plugin install nebula@hypernym-nebula
```

Then start a new session: Nebula builds its environment as that session starts. Until then the status line says
"restart Claude Code once to finish setup".

## Commands

| command | what it does |
|---|---|
| `/nebula:shield-status` | On/off, mode, detectors and thresholds, and what it has flagged recently. |
| `/nebula:shield-privacy` | Exactly what is stored on this machine, where, how much, and what is excluded. |
| `/nebula:shield-retention none\|excerpt\|full` | What transcript text is kept. Default `none`: scores and counts only. |
| `/nebula:shield-exclude <glob>` · `/nebula:shield-include <glob>` | Keep projects out entirely (never opened, scored or stored), or let them back in. |
| `/nebula:shield-export` | Write everything Nebula has stored to a readable JSON file in `~/.nebula`. |
| `/nebula:shield-purge [yes]` | Delete the database, caches and logs. Without `yes` it only shows what would go; with `yes` it asks you to confirm first. |
| `/nebula:shield-backfill` | Opt in: score your past Claude Code sessions on this machine (low priority, never uploaded). |
| `/nebula:shield-soliton [on\|off]` | Soliton mode, as above; with no argument it shows whether it is on. |
| `/nebula:shield-wings [number\|on\|off]` | Choose the wings beside the logo on the badge. |
| `/nebula:statusline [remove\|force]` | Take the status line out, or put it over someone else's (backed up first). |

## Privacy

Nebula never sends your data anywhere. Every Nebula process starts by installing an audit hook that blocks network
connections, DNS lookups and HTTP requests, and it stays in place for the life of the process. The models are fixed
files, verified against a SHA-256 manifest on every load, and the native scorer runs only when it matches its published
SHA-256. By default the local record keeps scores and counts, not what you or Claude wrote. Commands never run what you
type as shell, and Nebula never changes Claude Code's permissions. The only download is `uv` fetching Nebula's pinned
Python packages from PyPI as the first session starts.

The detectors were trained on the author's own labelled sessions, never on yours, and the weights passed a leakage
gate before release. Every guarantee and how it is enforced: [`plugins/nebula/PRIVACY.md`](plugins/nebula/PRIVACY.md).

## Files on your machine

- `~/.nebula/`: your settings (`settings.json`, which updates never touch) and the local record (`nebula.db`). Mode 0700.
- `~/.hypernym-nebula/bin/`: the status-line launcher. No data.

## Update and uninstall

```
claude plugin marketplace update hypernym-nebula && claude plugin update nebula@hypernym-nebula
```

Updates apply from your next session; your record and settings are kept. To remove Nebula:
`claude plugin uninstall nebula@hypernym-nebula`, then `/nebula:statusline remove` if you want the status line gone, and
`/nebula:shield-purge yes` first if you want your local record gone too.

## Zephyr

Hypernym also runs **Zephyr**, which analyzes a repository on Hypernym's servers and loads the result into Claude Code
as context. It's available to select clients by invitation; see [hypernym.ai](https://hypernym.ai). Nebula itself never
talks to Zephyr or to anything else.
