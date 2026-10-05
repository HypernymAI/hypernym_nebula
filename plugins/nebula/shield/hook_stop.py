#!/usr/bin/env python3
"""
Claude Code Stop hook — local shield on the assistant's reply (build doc Part XVIII). NOT registered.

stdin: the Stop event JSON (session_id, transcript_path, …). Scores every assistant text block of this
turn with the configured Stop heads (max per head). Any head ≥ its threshold → {"decision": "block",
"reason": …} so the reply is reframed before it reaches the pilot. After `max_rewrites` blocks for the same
pilot message → allowed with a visible systemMessage (never silent, never endless). Disabled → exit 0.
Every decision is logged. Errors raise (exit 1, stderr) — no silent pass.

    <python> hook_stop.py [--config path] < event.json
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import argparse
import json
import sys
import time
from pathlib import Path

from brand import SHIELD_ICON
from settings import excluded, resolve_config, soliton_on
from shield_core import Shield, sha, transcript_turn
from tracker import Tracker


CLASS_OF = {'H08_S2': 'confession', 'H08_S6': 'confession', 'H15_S2': 'handback', 'H15_S6': 'handback',
            'H06_S2': 'claim', 'H13_S6': 'repetition'}
ORDER = ('confession', 'handback', 'claim', 'repetition')
ORANGE = '\x1b[1;38;5;208m'                        # bold orange — 256-colour, so Apple Terminal shows it too
RESET = '\x1b[0m'
# SOLITON (beta): the pilot's ONE message, verbatim (2026-09-29); only the peaked scores and his less/fewer phrase
# per class are swapped in — nothing added (pilot, 2026-09-30):
# "Hi there, your handback score and your claim score were a little bit too high for that one. Can you rephrase that
#  exactly with less handback to the user and fewer bold claims that you can't substantiate?"
ASK = {'confession': 'fewer confessions',
       'handback': 'less handback to the user',
       'claim': "fewer bold claims that you can't substantiate"}
# What can trigger a rephrase. NOT repetition: it measures the PILOT's side (did he have to ask again) — a rewrite of
# the reply cannot change it, and asking for one produced nonsense (pilot, 2026-09-30: "2 … obvi"). It still shows on
# the badge and in the judgement line.
TRIGGERS = ('confession', 'handback', 'claim')


def class_levels(tr: Tracker, sh: Shield, session_id: str):
    """Latest reply of the session: ({class: (score / threshold, certified)}, live scores) — or (None, None) when
    the tracker holds no judged reply yet. ≥ 1.0 = flagged (the badge's orange)."""
    c = tr._conn()
    try:
        row = c.execute("SELECT live_scores_json FROM replies WHERE session_id=? ORDER BY exchange_index DESC "
                        "LIMIT 1", (session_id,)).fetchone()
    finally:
        c.close()
    if row is None or row[0] is None:
        return None, None
    live = json.loads(row[0])
    level = {}
    for h, s in live.items():
        k = CLASS_OF[h]
        lv = s / sh.cfg['stop'][h]['threshold']
        cert = sh.cfg['stop'][h]['certified'] is True
        if k not in level or lv > level[k][0]:
            level[k] = (lv, cert)
    return level, live


def and_list(items: list) -> str:
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def soliton_ask(flagged: list) -> str:
    scores = and_list([f"your {k} score" for k in flagged])
    were = 'was' if len(flagged) == 1 else 'were'
    return (f"Hi there, {scores} {were} a little bit too high for that one. Can you rephrase that exactly with "
            f"{and_list([ASK[k] for k in flagged])}?")


def soliton_banner(flagged: list) -> str:
    shield = ''.join(ch for ch, _ in SHIELD_ICON[0])
    return (f"{ORANGE}{shield} SHIELDED — don't read the reply above: it's being rephrased now "
            f"({', '.join(flagged)}){RESET}")


def judgement_line(tr: Tracker, sh: Shield, session_id: str) -> str:
    """Latest reply of the session: per class, the max over its heads of score / threshold (≥ 1 = flagged)."""
    level, _ = class_levels(tr, sh, session_id)
    if level is None:
        return ''
    parts = []
    for k in ORDER:
        lv, cert = level[k]
        mark = '●' if lv >= 1 else '○'
        parts.append(f"{mark} {k} {min(lv, 9.9):.1f}×{'' if cert else '·w'}")
    worst = max(v[0] for v in level.values())
    shield = ''.join(ch for ch, _ in SHIELD_ICON[0])        # the braille shield ⠫⣿⠝ (pilot: no 🛡 emoji)
    head = f'{shield} SHIELD' if worst < 1 else f'{shield} SHIELD ⚠'
    return f"{head}  " + '   '.join(parts) + "   (×threshold; ≥1.0 flagged; ·w = working, not certified)"


def run(event: dict, config: Path) -> tuple:
    """→ (exit_code, stdout, stderr)"""
    t0 = time.time()
    sh = Shield(config)
    if not sh.enabled:                       # kill switch: nothing is read, scored or written
        return 0, '', ''
    if excluded(sh.cfg, event['transcript_path'], event['cwd'] if 'cwd' in event else None):
        return 0, '', ''                     # excluded project: the transcript is never opened
    session_id = event['session_id']
    pilot, blocks, _ = transcript_turn(event['transcript_path'])
    reply = '\n'.join(blocks)
    base = {'event': 'stop', 'session_id': session_id, 'pilot_sha': sha(pilot), 'text': reply,
            'config_sha': sh.config_sha}
    # central shadow tracker: every exchange of this session → ~/.nebula (both modes; local only)
    tr = Tracker(Path(sh.cfg['db_path']), sh)
    tr.sync_session(Path(event['transcript_path']), last_turn_complete=True)   # Stop = this turn's reply is final
    if soliton_on(sh.cfg):
        return soliton(event, tr, sh, base, pilot, blocks, t0)
    if sh.cfg['mode'] == 'shadow':
        if sh.cfg['show'] is True:           # SHADOW + SHOW: a one-line judgement under the reply, never a block
            note = judgement_line(tr, sh, Path(event['transcript_path']).stem)   # the tracker's session key
            return 0, (json.dumps({'systemMessage': note}) if note else ''), ''
        return 0, '', ''                     # SHADOW: recorded above; Claude Code sees nothing
    if not pilot.strip() or not blocks:
        return 0, '', ''
    attempt = sh.log.blocks_so_far(session_id, sha(pilot)) + 1
    scores, all_fired = sh.score_heads('stop', sh.stop_docs(pilot, blocks))
    fired = sh.certified('stop', all_fired)          # enforcement: certified heads only
    if not fired:
        decision, out = 'allow', ''
    elif attempt > sh.cfg['max_rewrites']:
        decision = 'cap_allow'
        out = json.dumps({'systemMessage': f"SHIELD: reply still flagged after {sh.cfg['max_rewrites']} "
                                           f"reframes — shown as-is; logged.\n" + sh.reframe_text(fired, 'stop')})
    else:
        decision = 'block'
        out = json.dumps({'decision': 'block', 'reason':
                          "SHIELD (local) flagged this reply before it reached the pilot:\n"
                          + sh.reframe_text(fired, 'stop')
                          + "\nUse the shield-reframe skill: rewrite the WHOLE reply doing the asked work; do not "
                            "disguise the same behavior in new wording."})
    sh.log.put(dict(base, attempt=attempt, scores_json=json.dumps(scores), fired_json=json.dumps(all_fired),
                    decision=decision, ms=(time.time() - t0) * 1000))
    return 0, out, ''


def soliton(event: dict, tr: Tracker, sh: Shield, base: dict, pilot: str, blocks: list, t0: float) -> tuple:
    """SOLITON (beta). A reply with any class at ≥ 1.0× → a vivid banner right under it (Claude Code shows the reply
    before any Stop hook runs, and a finished reply cannot be hidden — measured 2026-09-30) + ONE rephrase, asked in
    the pilot's words. The rephrased reply (stop_hook_active true) is never blocked again: exactly one reroll per hit."""
    key = Path(event['transcript_path']).stem                # the tracker's session key
    level, live = class_levels(tr, sh, key)
    flagged = [] if level is None else [k for k in TRIGGERS if level[k][0] >= 1.0]
    rerolled = event['stop_hook_active'] is True
    if flagged and not rerolled and pilot.strip() and blocks:
        decision = 'soliton_block'
        out = json.dumps({'decision': 'block', 'systemMessage': soliton_banner(flagged), 'reason': soliton_ask(flagged)})
    else:
        decision = 'soliton_allow' if not flagged else 'soliton_shown'   # shown = still flagged after its one reroll
        note = judgement_line(tr, sh, key) if sh.cfg['show'] is True else ''
        out = json.dumps({'systemMessage': note}) if note else ''
    sh.log.put(dict(base, attempt=2 if rerolled else 1, scores_json=json.dumps(live if live is not None else {}),
                    fired_json=json.dumps(flagged), decision=decision, ms=(time.time() - t0) * 1000))
    return 0, out, ''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    code, out, err = run(json.load(sys.stdin), resolve_config(args.config))
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    sys.exit(code)


if __name__ == '__main__':
    main()
