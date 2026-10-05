"""
The shield's own transcript reader (build doc S25). A copy of
tools/alliance_labeling/extract_exchanges.load_session_jsonl (ClaudeStorm proper — READ-ONLY for this build) with ONE
change: the skip-prefix set is a parameter, and the shield's default keeps a pilot turn that starts with a pasted-image
placeholder ("[Image #2] look at this …").

Why: the original SKIP_PREFIX contains '[Image', written to drop Claude Code's auto-generated "[Image: source: <path>]"
records — but it also drops every REAL pilot message that begins with a pasted screenshot, so that turn and the reply to
it were never recorded (observed 2026-09-27 08:02: the reply to a screenshot turn got the previous reply's judgement).
SHIELD_SKIP_PREFIX = the original set with '[Image' narrowed to '[Image:' (Claude Code's own records: '[Image: source: …]',
'[Image: original WxH, displayed at …]'; a pasted placeholder is '[Image #N]').

Second change: ABSORB_PREFIX — '[Image:' user records inside a turn are skipped instead of ending the turn.

Third change: FEEDBACK_PREFIX — Claude Code records a Stop hook's block reason as a USER record starting
"Stop hook feedback:" (seen 2026-09-30 with soliton). It is not the pilot: such an exchange keeps its own reply (the
rephrase is judged as a reply of its own) but its 'user' is the pilot's last REAL message, and 'hook_feedback' is True
so the tracker never stores it as a pilot turn. Before this, the rephrase was scored against the hook's own words
("Can you rephrase that…" read as the pilot repeating himself: repetition 0.09 → 0.38) and the feedback was stored in
pilot_turns as his message.
With skip_prefix=extract_exchanges.SKIP_PREFIX and absorb_prefix=() this function returns EXACTLY what the original returns — asserted over
every transcript on disk by test_shield_reader.py. NO .get() in new code (the copied body is byte-identical apart from
the parameter). Imports at top.
"""
import json
from pathlib import Path
from typing import Dict, List

from vendor_exchange import RES_TOTAL, SKIP_PREFIX, TOOL_ARG_HEAD, _result_text

SHIELD_SKIP_PREFIX = tuple('[Image:' if p == '[Image' else p for p in SKIP_PREFIX)
assert SHIELD_SKIP_PREFIX != SKIP_PREFIX and len(SHIELD_SKIP_PREFIX) == len(SKIP_PREFIX)
# Records INSIDE a turn that must not end it: a pasted image is written as the pilot's message + a separate
# '[Image: source: …]' user record; the original reader ends the turn there, so the reply is never attached.
ABSORB_PREFIX = ('[Image:',)
FEEDBACK_PREFIX = ('Stop hook feedback:',)


def load_session_jsonl(jsonl_path: Path, skip_prefix: tuple = SHIELD_SKIP_PREFIX,
                       absorb_prefix: tuple = ABSORB_PREFIX, feedback_prefix: tuple = FEEDBACK_PREFIX) -> List[Dict]:
    """Three-part exchanges directly from a live Claude Code session JSONL.

    THE SOURCE OF TRUTH. Audited 2026-09-06: the corpus DBs were all snapshotted
    2026-08-03, so every session that continued past that date is TRUNCATED there —
    including ed15059e, whose Aug 22-23 night (the one the surrender ranking was
    claimed on) was entirely absent from the DB it was "measured" in. A session must
    contain its own ending or every trajectory read on it is fiction.

    Produces the same exchange dicts as load_session(), same caps on tool_args
    (head of each tool's JSON only — payload contents make the referent test vacuous).
    """
    msgs = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if 'type' not in d or d['type'] not in ('user', 'assistant'):
                continue
            if 'message' not in d:
                continue
            m = d['message']
            if 'content' not in m:
                continue
            c = m['content']
            blocks = []   # ordered assistant blocks: ('thinking'|'text'|'tool', str)
            if isinstance(c, str):
                texts, tools, errs, results, res_txt = [c], [], 0, 0, []
                blocks = [('text', c)]
            else:
                for b in c:
                    if not isinstance(b, dict) or 'type' not in b:
                        continue
                    if b['type'] == 'thinking' and 'thinking' in b and b['thinking']:
                        blocks.append(('thinking', b['thinking']))
                    elif b['type'] == 'text' and 'text' in b and b['text'].strip():
                        blocks.append(('text', b['text']))
                    elif b['type'] == 'tool_use':
                        blocks.append(('tool', f"{b['name'] if 'name' in b else '?'} "
                                       + json.dumps(b['input'] if 'input' in b else {})[:TOOL_ARG_HEAD]))
                texts = [b['text'] for b in c
                         if isinstance(b, dict) and b['type'] == 'text' and 'text' in b]
                tools = [b for b in c if isinstance(b, dict) and b['type'] == 'tool_use']
                res = [b for b in c if isinstance(b, dict) and b['type'] == 'tool_result']
                errs = sum(1 for b in res if 'is_error' in b and b['is_error'])
                results = len(res)
                res_txt = [('[ERROR] ' if ('is_error' in b and b['is_error']) else '')
                           + _result_text(b['content'] if 'content' in b else '')
                           for b in res]
            uuid = d['uuid'] if 'uuid' in d else ''
            ts = d['timestamp'] if 'timestamp' in d else ''
            sid = d['sessionId'] if 'sessionId' in d else jsonl_path.stem
            msgs.append({'type': d['type'], 'text': '\n'.join(texts), 'tools': tools,
                         'errs': errs, 'results': results, 'res_txt': res_txt,
                         'blocks': blocks, 'uuid': uuid, 'ts': ts, 'sid': sid})

    exchanges = []
    last_pilot = ''
    i = 0
    while i < len(msgs):
        t = msgs[i]['text'].strip()
        if (msgs[i]['type'] != 'user' or not t or t.startswith(skip_prefix)
                or msgs[i]['results'] > 0):
            i += 1
            continue
        j = i + 1
        parts, names, args, n_t, n_e = [], [], [], 0, 0
        res_all = []
        # ORDERED events inside the turn: ('text', str) / ('result', str). Needed to pair a
        # claim with the results that IMMEDIATELY PRECEDE it — the last-N-lines-of-the-turn
        # pairing judged claims against output from unrelated calls (rev. c defect 1).
        events = []
        interrupted = 0
        asst_uuid = ''
        while j < len(msgs):
            if msgs[j]['type'] == 'assistant':
                events.extend(msgs[j]['blocks'])
                if msgs[j]['text'].strip():
                    parts.append(msgs[j]['text'])
                    if not asst_uuid:
                        asst_uuid = msgs[j]['uuid']
                for tu in msgs[j]['tools']:
                    n_t += 1
                    names.append(tu['name'] if 'name' in tu else '?')
                    args.append(json.dumps(tu['input'] if 'input' in tu else {})[:300])
                j += 1
                continue
            if msgs[j]['type'] == 'user' and msgs[j]['results'] > 0:
                n_e += msgs[j]['errs']
                res_all.extend(msgs[j]['res_txt'])
                for rt in msgs[j]['res_txt']:
                    events.append(('result', rt))
                j += 1
                continue
            if absorb_prefix and msgs[j]['type'] == 'user' and msgs[j]['text'].strip().startswith(absorb_prefix):
                j += 1                   # S25: Claude Code's own image record belongs to the turn; it does not end it
                continue
            if msgs[j]['text'].strip().startswith('[Request interrupted'):
                interrupted = 1
                j += 1
                continue
            break
        feedback = bool(feedback_prefix) and t.startswith(feedback_prefix)
        if not feedback:
            last_pilot = msgs[i]['text']
        if parts:
            exchanges.append({
                'session_id': msgs[i]['sid'],
                'user_uuid': msgs[i]['uuid'],
                'assistant_uuid': asst_uuid,
                'user': last_pilot if feedback else msgs[i]['text'],
                'user_raw': msgs[i]['text'],
                'hook_feedback': feedback,
                'assistant': '\n'.join(parts),
                'tool_names': ','.join(names),
                'tool_args': ' '.join(args)[:2500],
                'tool_results': '\n---\n'.join(res_all)[:RES_TOTAL],
                'events': events,
                'tools': n_t,
                'errors': n_e,
                'interrupted': interrupted,
                'ts': msgs[i]['ts'][:10],
            })
        i = j if j > i else i + 1

    for idx, ex in enumerate(exchanges):
        ex['exchange_index'] = idx
        nxt = [x for x in exchanges[idx + 1:] if not x['hook_feedback']]
        ex['next_user'] = nxt[0]['user'] if nxt else ''
    return exchanges
