#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit hook — local shield on the PILOT's new turn (build doc Part XVIII). NOT registered.

stdin: the event JSON (session_id, transcript_path, prompt, …). Scores the new turn with the configured
user_prompt heads (e.g. H13_S1 forced repetition: the pilot is having to repeat an earlier ask). Any head ≥
threshold → additionalContext injected before the assistant replies, telling it to reframe (re-read the
repeated ask; do exactly that; no hedging, no hand-back). Nothing is blocked — the pilot's message always goes
through. Disabled → exit 0. Every decision logged. Errors raise.

    <python> hook_user_prompt.py [--config path] < event.json
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import argparse
import json
import sys
import time
from pathlib import Path

from settings import excluded, resolve_config
from shield_core import Shield, sha
from shield_reader import load_session_jsonl          # S25: image-bearing pilot turns kept


def run(event: dict, config: Path) -> tuple:
    t0 = time.time()
    sh = Shield(config)
    if not sh.enabled:                       # kill switch: nothing is read, scored or written
        return 0, '', ''
    if excluded(sh.cfg, event['transcript_path'], event['cwd'] if 'cwd' in event else None):
        return 0, '', ''
    prompt = event['prompt']
    base = {'event': 'user_prompt', 'session_id': event['session_id'], 'pilot_sha': sha(prompt),
            'attempt': 0, 'text': prompt, 'config_sha': sh.config_sha}
    if sh.cfg['mode'] == 'shadow':
        return 0, '', ''                     # SHADOW: the pilot turn is recorded by the Stop hook's sync
    if not sh.is_real_pilot(prompt):
        return 0, '', ''
    exchanges = load_session_jsonl(Path(event['transcript_path']))
    scores, fired = sh.score_heads('user_prompt', {'S1': [sh.pilot_turn_fields(prompt, exchanges)]})
    out = ''
    if fired:
        out = json.dumps({'hookSpecificOutput': {
            'hookEventName': 'UserPromptSubmit',
            'additionalContext': "SHIELD (local): the pilot's message shows this pattern — reframe BEFORE replying "
                                 "(use the shield-reframe skill):\n" + sh.reframe_text(fired, 'user_prompt')
                                 + "\nThe pilot is likely repeating an earlier ask because it was not done. Find that "
                                   "ask in their recent turns and do exactly it now."}})
    sh.log.put(dict(base, scores_json=json.dumps(scores), fired_json=json.dumps(fired),
                    decision='context' if fired else 'allow', ms=(time.time() - t0) * 1000))
    return 0, out, ''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    code, out, err = run(json.load(sys.stdin), resolve_config(args.config))
    if out:
        print(out)
    sys.exit(code)


if __name__ == '__main__':
    main()
