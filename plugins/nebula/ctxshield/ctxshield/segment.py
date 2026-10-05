"""
Exchange segmentation + head text fields, built from the DFC's own `messages` list.

Input: the list `compute_dimensional_field()` receives — dicts with at least
    'type'  ('user' | 'assistant' | anything else — other roles are invisible here)
    'text'  (str; canonical text = text blocks, AND thinking blocks, joined by '\n')
    'is_thinking_only' (bool)
    'tool_results' (present only when the line carried tool_result blocks)
This is the shape ClaudeStorm builds per JSONL line; ctxshield never imports ClaudeStorm, it only
reads these keys.

Output reproduces the frozen benchmark units BYTE-EXACTLY where the input allows it:
  segmentation rules  = the benchmark's live-JSONL exchange reader (pilot turn = a user line with
                        non-blank text, not a SKIP_PREFIX wrapper, not a tool-result line; the turn
                        absorbs assistant lines, tool-result lines and '[Request interrupted' lines;
                        any other user line ends it; kept only with >= 1 assistant text line;
                        exchange_index counts KEPT exchanges from 0)
  payloads            = S1 {pilot, previous_reply, earlier_asks}, S2 {pilot_ask, text},
                        S6 {pilot, reply} with the exact caps + marker text (ctxshield.shape)
  attachment (§4.2)   = S1 -> the pilot message; S2 -> that assistant message;
                        S6 -> the LAST assistant text message of the exchange

Known, measured limits (reported by the bridge T5 test, never hidden):
  - A line holding thinking AND text has canonical text = thinking + '\n' + text (build doc F16);
    only thinking-ONLY lines can be excluded here (is_thinking_only). Mixed lines mismatch.
  - The benchmark emitted one S2 unit per non-blank TEXT BLOCK; here one per assistant message.
    Identical for one-block lines (the Claude Code norm); a multi-text-block line differs.
  - The benchmark read the JSONL linearly (dead branches included, F13). Whatever list is passed in
    is segmented as given; the DFC passes the active branch only, so exchange_index and neighbour
    context (previous_reply, earlier_asks) can differ from the frozen unit next to a dead branch.
  - The benchmark dropped whole sessions with < 25 exchanges (a corpus-selection rule). Not applied
    at runtime: a live session is scored from its first exchange.

NO .get(). Imports at top. No ClaudeStorm imports.
"""
from typing import Dict, List

from ctxshield import shape

# Byte-exact copy of tools/alliance_labeling/extract_exchanges.py SKIP_PREFIX (asserted equal by
# tools/shield_bridge/test_segment_t5.py). Harness wrappers, not the pilot speaking.
SKIP_PREFIX = (
    '<local-command', '<command-', '<system-reminder', '[Request interrupted',
    '[Image', 'Caveat: The messages below', '<task-notification',
    'This session is being continued from a previous conversation',
)
INTERRUPTED = '[Request interrupted'
TURN_ROLES = ('user', 'assistant')

# Section -> {feature field name: payload key}. Must equal tools/shield_bridge/heads.py for every
# head of these sections (asserted by the bridge T5 test).
SECTION_FIELDS = {
    'S1': {'pilot': 'pilot', 'prev': 'previous_reply', 'earlier': 'earlier_asks'},
    'S2': {'ask': 'pilot_ask', 'text': 'text'},
    'S6': {'pilot': 'pilot', 'reply': 'reply'},
}
SECTIONS = tuple(sorted(SECTION_FIELDS))


def visible_text(m: Dict) -> str:
    """Text the benchmark saw for this line: a thinking-only assistant line contributes nothing."""
    if m['type'] == 'assistant' and m['is_thinking_only']:
        return ''
    return m['text']


def is_tool_result_line(m: Dict) -> bool:
    return 'tool_results' in m


def segment(messages: List[Dict]) -> List[Dict]:
    """Exchanges in order: {exchange_index, pilot_idx, reply_idxs, reply}.

    pilot_idx / reply_idxs are indices into `messages` (the DFC's message_idx space).
    reply = the assistant text lines of the turn joined by '\\n' (the benchmark's 'assistant').
    """
    seq = [i for i, m in enumerate(messages) if m['type'] in TURN_ROLES]
    exchanges = []
    p = 0
    while p < len(seq):
        i = seq[p]
        m = messages[i]
        t = m['text'].strip()
        if m['type'] != 'user' or not t or t.startswith(SKIP_PREFIX) or is_tool_result_line(m):
            p += 1
            continue
        q = p + 1
        reply_idxs = []
        while q < len(seq):
            mj = messages[seq[q]]
            if mj['type'] == 'assistant':
                if visible_text(mj).strip():
                    reply_idxs.append(seq[q])
                q += 1
                continue
            if is_tool_result_line(mj):
                q += 1
                continue
            if mj['text'].strip().startswith(INTERRUPTED):
                q += 1
                continue
            break
        if reply_idxs:
            exchanges.append({
                'exchange_index': len(exchanges),
                'pilot_idx': i,
                'reply_idxs': reply_idxs,
                'reply': '\n'.join(visible_text(messages[j]) for j in reply_idxs),
            })
        p = q if q > p else p + 1
    return exchanges


def build_units(messages: List[Dict]) -> Dict[str, List[Dict]]:
    """Per section, the units the benchmark would have frozen from this message list.

    Each unit: {section, local_key, exchange_index, message_idx, payload, sources}
      local_key   = '<xi>' (S1, S6) or '<xi>#t<n>' (S2) — the benchmark key minus '<session_id>#'
      message_idx = the DFC message the score attaches to (§4.2)
      payload     = the frozen payload shape (same keys, same caps, same marker text)
      sources     = every message_idx whose text entered the payload (for mismatch diagnosis)
    """
    ex = segment(messages)
    out = {s: [] for s in SECTIONS}
    for k, e in enumerate(ex):
        xi = e['exchange_index']
        pilot_text = messages[e['pilot_idx']]['text']
        ask = shape.head(pilot_text, shape.CAP_ASK)
        prev_reply = shape.tail(ex[k - 1]['reply'], shape.CAP_PREV_TAIL) if k > 0 else ''
        prior_range = range(max(0, k - 2), k)
        prior = [shape.head(messages[ex[j]['pilot_idx']]['text'], shape.CAP_EARLIER)
                 for j in prior_range]
        s1_sources = ([e['pilot_idx']] + [ex[j]['pilot_idx'] for j in prior_range]
                      + (ex[k - 1]['reply_idxs'] if k > 0 else []))
        out['S1'].append({
            'section': 'S1', 'local_key': f"{xi}", 'exchange_index': xi,
            'message_idx': e['pilot_idx'],
            'payload': {'pilot': shape.head(pilot_text, shape.CAP_PILOT),
                        'previous_reply': prev_reply,
                        'earlier_asks': prior},
            'sources': sorted(set(s1_sources)),
        })
        out['S6'].append({
            'section': 'S6', 'local_key': f"{xi}", 'exchange_index': xi,
            'message_idx': e['reply_idxs'][-1],
            'payload': {'pilot': shape.head(pilot_text, shape.CAP_PILOT),
                        'reply': shape.headtail(e['reply'], shape.CAP_REPLY_HEAD,
                                                shape.CAP_REPLY_TAIL)},
            'sources': [e['pilot_idx']] + list(e['reply_idxs']),
        })
        for n, j in enumerate(e['reply_idxs'], 1):
            out['S2'].append({
                'section': 'S2', 'local_key': f"{xi}#t{n}", 'exchange_index': xi,
                'message_idx': j,
                'payload': {'pilot_ask': ask,
                            'text': shape.head(visible_text(messages[j]), shape.CAP_BLOCK)},
                'sources': [e['pilot_idx'], j],
            })
    return out


def unit_fields(section: str, payload: Dict) -> Dict[str, str]:
    """Payload -> the feature fields a head of `section` is scored on (list fields joined by
    shape.LIST_JOIN — the same rule the bridge used to write the training rows)."""
    out = {}
    for name, key in SECTION_FIELDS[section].items():
        v = payload[key]
        out[name] = shape.join_list(v) if isinstance(v, list) else v
    return out
