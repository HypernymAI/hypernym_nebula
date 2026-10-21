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
every transcript on disk by test_shield_reader.py.

Fourth change (2026-10-19): CODEX. A Codex CLI session file (~/.codex/sessions/…/rollout-*.jsonl, first record
`session_meta`) is read by codex_msgs() into the SAME message records the Claude Code reader builds, then paired by the
same loop — so scoring, the tracker and the judgement line are unchanged. Codex documents the transcript as "not a stable
interface", so codex_msgs reads only Codex's typed thread items (`event_msg` / `item_completed`: UserMessage, AgentMessage,
HookPrompt, Reasoning, CommandExecution, McpToolCall, FileChange, Extension) and `turn_aborted`, and RAISES when a rollout
with turns holds none of them (a changed format must never be read as a silent empty session). A Stop hook's
continuation (HookPrompt) becomes a FEEDBACK_PREFIX record, exactly like Claude Code's "Stop hook feedback:". NO .get() in new code (the copied body is byte-identical apart from
the parameter). Imports at top.
"""
import json
import re
from pathlib import Path
from typing import Dict, List

from vendor_exchange import RES_TOTAL, SKIP_PREFIX, TOOL_ARG_HEAD, _result_text

SHIELD_SKIP_PREFIX = tuple('[Image:' if p == '[Image' else p for p in SKIP_PREFIX)
assert SHIELD_SKIP_PREFIX != SKIP_PREFIX and len(SHIELD_SKIP_PREFIX) == len(SKIP_PREFIX)
# Records INSIDE a turn that must not end it: a pasted image is written as the pilot's message + a separate
# '[Image: source: …]' user record; the original reader ends the turn there, so the reply is never attached.
ABSORB_PREFIX = ('[Image:',)
FEEDBACK_PREFIX = ('Stop hook feedback:',)


def jsonl_lines(jsonl_path: Path, start: int):
    """(byte offset, line) of every line from `start` on. Binary lines split on b'\\n' — the same lines text mode yields
    for a JSONL file (JSON never holds a raw CR or LF) — decoded strictly as UTF-8, as the text-mode readers did."""
    with open(jsonl_path, 'rb') as f:
        f.seek(start)
        pos = start
        for raw in f:
            yield pos, raw.decode('utf-8')
            pos += len(raw)


# ---------------------------------------------------------------------------------------------------- Codex (2026-10-19)
CODEX_TOOL_ITEMS = ('CommandExecution', 'McpToolCall', 'FileChange', 'Extension')


def is_codex_rollout(jsonl_path: Path) -> bool:
    """A Codex CLI session file: its first record is `session_meta` (measured on every rollout on disk; no Claude Code
    transcript starts with one)."""
    with open(jsonl_path, encoding='utf-8') as f:
        first = f.readline()
    try:
        d = json.loads(first)
    except ValueError:
        return False
    if not isinstance(d, dict):
        return False
    if 'type' not in d:
        return False
    return d['type'] == 'session_meta'


def codex_project(jsonl_path: Path) -> str:
    """The rollout's working directory in Claude Code's project-folder form (every non-alphanumeric → '-'), so a repo's
    Codex sessions sit under the same project name as its Claude Code sessions."""
    with open(jsonl_path, encoding='utf-8') as f:
        meta = json.loads(f.readline())
    return re.sub(r'[^A-Za-z0-9]', '-', meta['payload']['cwd'])


def codex_models_by_item(jsonl_path: Path, start: int = 0) -> Dict[str, str]:
    """Each Codex reply (AgentMessage item id) → the model that wrote it: the `model` of the turn's `turn_context`
    record, which Codex writes at the start of every turn (before the turn's UserMessage — a window starts there)."""
    out, model = {}, 'unknown'
    for _, line in jsonl_lines(jsonl_path, start):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        if 'type' not in d:
            continue
        if 'payload' not in d:
            continue
        p = d['payload']
        if not isinstance(p, dict):
            continue
        if d['type'] == 'turn_context':
            if 'model' in p:
                model = p['model']
            continue
        if d['type'] != 'event_msg':
            continue
        if 'type' not in p:
            continue
        if p['type'] != 'item_completed':
            continue
        it = p['item']
        if it['type'] == 'AgentMessage':
            if 'id' in it:
                out[it['id']] = model
    return out


def _codex_record(kind: str, text: str, ts: str, uuid: str, sid: str) -> Dict:
    return {'type': kind, 'text': text, 'tools': [], 'errs': 0, 'results': 0, 'res_txt': [], 'blocks': [],
            'uuid': uuid, 'ts': ts, 'sid': sid}


def _codex_texts(content: list) -> str:
    """The text parts of a UserMessage / AgentMessage item (images and other parts carry no text)."""
    parts = []
    for c in content:
        if isinstance(c, dict):
            if 'text' in c:
                parts.append(c['text'])
    return '\n'.join(parts)


def _codex_tool(it: Dict) -> tuple:
    """(name, input, output text, failed) of one Codex tool item — the tool_use + tool_result pair of a Claude turn."""
    kind = it['type']
    failed = False
    if 'status' in it:
        if it['status'] != 'completed':
            failed = True
    if kind == 'CommandExecution':
        cmd = it['command']
        if isinstance(cmd, list):
            cmd = cmd[-1]
        out = ''
        if 'aggregated_output' in it:
            if it['aggregated_output'] is not None:
                out = it['aggregated_output']
        if 'exit_code' in it:
            if it['exit_code'] is not None:
                if it['exit_code'] != 0:
                    failed = True
        return 'exec_command', {'command': cmd}, out, failed
    if kind == 'McpToolCall':
        out = ''
        if 'result' in it:
            if it['result'] is not None:
                out = it['result']
                if not isinstance(out, str):
                    out = json.dumps(out)
        if 'error' in it:
            if it['error'] is not None:
                out = json.dumps(it['error'])
        args = {}
        if 'arguments' in it:
            args = it['arguments']
        return f"mcp__{it['server']}__{it['tool']}", args, out, failed
    if kind == 'FileChange':
        files = []
        if 'changes' in it:
            if isinstance(it['changes'], dict):
                files = sorted(it['changes'])
        out = ''
        for k in ('stdout', 'stderr'):
            if k in it:
                if it[k]:
                    out += it[k]
        return 'apply_patch', {'files': files}, out, failed
    query = ''
    if 'query' in it:
        if it['query'] is not None:
            query = it['query']
    out = ''
    if 'results' in it:
        if it['results'] is not None:
            out = json.dumps(it['results'])
    return it['kind'], {'query': query}, out, failed


def codex_msgs(jsonl_path: Path, start: int = 0, offsets: list = None) -> List[Dict]:
    """The message records _claude_msgs builds, from a Codex rollout's typed thread items, in order. A tool item becomes
    an assistant record carrying the call + a user record carrying its result (the shape of a Claude Code turn); a
    HookPrompt becomes a FEEDBACK_PREFIX record; turn_aborted becomes '[Request interrupted by user]'.
    Older Codex (0.101, before typed thread items — and before plugin hooks) wrote `user_message` / `agent_message`
    events instead: such a rollout is read from those, text only (they carry no tools). Raises when a rollout with turns
    holds neither: a format Codex changed must not read as an empty session.
    `start` / `offsets` (2026-10-21, incremental reads): read from byte `start`; when `offsets` is a list, a
    (read_from, line) pair is appended for every returned record: `line` is the record's own line; `read_from` is where
    a read must start to also see the model in effect for it — for a UserMessage the latest `turn_context` before it
    (a dialog answer comes as a UserMessage with no turn_context of its own, so that is an EARLIER turn's: a window
    reads from read_from but pairs only from `line`)."""
    msgs, items, turns = [], 0, 0
    legacy = []
    msg_offs, legacy_offs = [], []
    ctx_off = None
    sid = jsonl_path.stem
    for off, line in jsonl_lines(jsonl_path, start):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        if 'type' not in d:
            continue
        if d['type'] == 'turn_context':
            ctx_off = off
        if d['type'] != 'event_msg':
            continue
        if 'payload' not in d:
            continue
        p = d['payload']
        if not isinstance(p, dict):
            continue
        if 'type' not in p:
            continue
        ts = ''
        if 'timestamp' in d:
            ts = d['timestamp']
        if p['type'] == 'task_started':
            turns += 1
            continue
        if p['type'] == 'turn_aborted':
            msgs.append(_codex_record('user', '[Request interrupted by user]', ts, '', sid))
            legacy.append(_codex_record('user', '[Request interrupted by user]', ts, '', sid))
            msg_offs.append((off, off))
            legacy_offs.append((off, off))
            continue
        if p['type'] == 'user_message':
            legacy.append(_codex_record('user', p['message'], ts, '', sid))
            legacy_offs.append((off, off))
            continue
        if p['type'] == 'agent_message':
            r = _codex_record('assistant', p['message'], ts, '', sid)
            if r['text'].strip():
                r['blocks'] = [('text', r['text'])]
            legacy.append(r)
            legacy_offs.append((off, off))
            continue
        if p['type'] != 'item_completed':
            continue
        items += 1
        it = p['item']
        uid = ''
        if 'id' in it:
            uid = it['id']
        kind = it['type']
        if kind == 'UserMessage':
            msgs.append(_codex_record('user', _codex_texts(it['content']), ts, uid, sid))
            if ctx_off is None:
                msg_offs.append((off, off))
            else:
                msg_offs.append((ctx_off, off))
        elif kind == 'HookPrompt':
            said = '\n'.join(fr['text'] for fr in it['fragments'])
            msgs.append(_codex_record('user', f"{FEEDBACK_PREFIX[0]} {said}", ts, uid, sid))
            msg_offs.append((off, off))
        elif kind == 'AgentMessage':
            r = _codex_record('assistant', _codex_texts(it['content']), ts, uid, sid)
            if r['text'].strip():
                r['blocks'] = [('text', r['text'])]
            msgs.append(r)
            msg_offs.append((off, off))
        elif kind == 'Reasoning':
            thought = []
            for k in ('summary_text', 'raw_content'):
                if k in it:
                    if isinstance(it[k], list):
                        for x in it[k]:
                            if isinstance(x, str):
                                if x.strip():
                                    thought.append(x)
            if thought:
                r = _codex_record('assistant', '', ts, uid, sid)
                r['blocks'] = [('thinking', '\n'.join(thought))]
                msgs.append(r)
                msg_offs.append((off, off))
        elif kind in CODEX_TOOL_ITEMS:
            name, args, out, failed = _codex_tool(it)
            call = _codex_record('assistant', '', ts, uid, sid)
            call['tools'] = [{'name': name, 'input': args}]
            call['blocks'] = [('tool', f"{name} " + json.dumps(args)[:TOOL_ARG_HEAD])]
            res = _codex_record('user', '', ts, '', sid)
            res['results'] = 1
            mark = ''
            if failed:
                res['errs'] = 1
                mark = '[ERROR] '
            res['res_txt'] = [mark + _result_text(out)]
            msgs.extend([call, res])
            msg_offs.extend([(off, off), (off, off)])
    if items == 0:
        if len(legacy) > 0:
            if offsets is not None:
                offsets.extend(legacy_offs)
            return legacy
        if turns > 0:
            raise ValueError(f"nebula: unrecognized Codex transcript {jsonl_path.name} — {turns} turns but no "
                             f"item_completed or user_message records (Codex changed its session format; nebula "
                             f"needs an update)")
    if offsets is not None:
        offsets.extend(msg_offs)
    return msgs


def _claude_msgs(jsonl_path: Path, start: int = 0, offsets: list = None) -> List[Dict]:
    """Message records from a Claude Code session JSONL (the original reader's first pass, unchanged). `start` /
    `offsets` (2026-10-21): read from byte `start`; append (line offset, line offset) for each returned record."""
    msgs = []
    for off, line in jsonl_lines(jsonl_path, start):
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
        if offsets is not None:
            offsets.append((off, off))
    return msgs


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
    A Codex rollout is read by codex_msgs() and paired by the same loop below.
    """
    if is_codex_rollout(jsonl_path):
        msgs = codex_msgs(jsonl_path)
    else:
        msgs = _claude_msgs(jsonl_path)
    return pair_msgs(msgs, skip_prefix, absorb_prefix, feedback_prefix)[0]


def read_msgs(jsonl_path: Path, start: int) -> tuple:
    """(message records from byte `start`, the (read_from, line) offsets of each record, is this a Codex rollout)."""
    offsets = []
    codex = is_codex_rollout(jsonl_path)
    if codex:
        msgs = codex_msgs(jsonl_path, start, offsets)
    else:
        msgs = _claude_msgs(jsonl_path, start, offsets)
    return msgs, offsets, codex


def pair_msgs(msgs: List[Dict], skip_prefix: tuple = SHIELD_SKIP_PREFIX, absorb_prefix: tuple = ABSORB_PREFIX,
              feedback_prefix: tuple = FEEDBACK_PREFIX, offsets: list = None) -> tuple:
    """The original reader's pairing loop (unchanged) → (exchanges, starts): starts[k] = the (read_from, line) offsets of
    exchange k's user record (None when no offsets are given). Pairing the records from that line on (read from
    read_from) yields exchanges k… identical to the full read, for a NON-feedback exchange k: an exchange depends only on
    the records from its own user record on, and a feedback exchange also needs the real pilot message before it."""
    starts = []
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
            if offsets is None:
                starts.append(None)
            else:
                starts.append(offsets[i])
        i = j if j > i else i + 1

    for idx, ex in enumerate(exchanges):
        ex['exchange_index'] = idx
        nxt = [x for x in exchanges[idx + 1:] if not x['hook_feedback']]
        ex['next_user'] = nxt[0]['user'] if nxt else ''
    return exchanges, starts
