"""
Turn state of a Claude Code transcript (build doc S24/S25/S38) — shared by the badge (must stay light: no numpy, no
models) and the tracker's callers. Rows are write-once, so a turn must never be stored while it is still being written.

A turn has ENDED when, walking back from the end past AMBIENT system records (away_summary, informational,
local_command … — Claude Code writes those after a turn, or between turns):
  - a system `turn_duration` / `stop_hook_summary` is found (what Claude Code writes after a turn)       → ended
  - a user record that is NOT only tool results is found (a new prompt / command: the turn before it is over) → ended
  - an assistant record that stopped normally (stop_reason end_turn / stop_sequence) is found           → ended
  - anything else (an assistant tool call, a tool result, an unknown stop)                               → NOT ended
S38: the previous rule looked only at the LAST record, so every transcript ending in an ambient system record (most of
them) read as "not ended" and backfill dropped its final reply.
NO .get(). Imports at top.
"""
import json
from pathlib import Path

TURN_END = ('turn_duration', 'stop_hook_summary')
DONE_STOPS = ('end_turn', 'stop_sequence')
TAIL_BYTES = 262144


def records(data: bytes) -> list:
    """JSON records of a byte span; a line cut by a seek (first/last) is not a record and is skipped."""
    out = []
    for line in data.split(b'\n'):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _only_tool_results(content) -> bool:
    return (isinstance(content, list) and len(content) > 0
            and all(isinstance(b, dict) and 'type' in b and b['type'] == 'tool_result' for b in content))


def turn_ended(jsonl: Path) -> bool:
    with open(jsonl, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - TAIL_BYTES))
        recs = records(f.read())
    for r in reversed(recs):
        if not (isinstance(r, dict) and 'type' in r and r['type'] in ('user', 'assistant', 'system')):
            continue
        if r['type'] == 'system':
            if 'subtype' in r and r['subtype'] in TURN_END:
                return True
            continue                                                  # ambient system record
        m = r['message'] if 'message' in r and isinstance(r['message'], dict) else {}
        if r['type'] == 'user':
            return not _only_tool_results(m['content'] if 'content' in m else None)
        return 'stop_reason' in m and m['stop_reason'] in DONE_STOPS
    return False
