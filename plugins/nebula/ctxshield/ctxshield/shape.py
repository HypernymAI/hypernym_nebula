"""
Payload shape helpers — BYTE-EXACT copies of the caps the benchmark labels were made on.

The labels judged these exact strings, marker text included (build doc F15, C6):
    head cut      text[:n] + ' …[+N]'
    tail cut      '[N]… ' + text[-n:]
    head + tail   text[:h] + ' …[N chars]… ' + text[-t:]
    tool result   head 300 + ' …[N chars]… ' + tail 200 (list content flattened to text blocks)

Sources copied from (ClaudeStorm): tools/jev/shield/units.py `_head/_tail/_headtail` and
tools/alliance_labeling/extract_exchanges.py `_result_text`. The bridge test
tools/shield_bridge/test_shape_parity.py asserts equality with those originals; a change here
without the same change there is a spec break.

NO .get(). No ClaudeStorm imports.
"""
from typing import List

CAP_PILOT = 1200
CAP_ASK = 600
CAP_BLOCK = 1500
CAP_REPLY_HEAD = 1200
CAP_REPLY_TAIL = 600
CAP_PREV_TAIL = 600
CAP_EARLIER = 300
CAP_BEFORE_TAIL = 300
TOOL_ARG_HEAD = 300
RES_HEAD = 300
RES_TAIL = 200

# Frozen join rule for list-valued payload fields (S1 `earlier_asks`) → one text field.
LIST_JOIN = '\n'


def head(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f' …[+{len(s) - n}]'


def tail(s: str, n: int) -> str:
    return s if len(s) <= n else f'[{len(s) - n}]… ' + s[-n:]


def headtail(s: str, h: int, t: int) -> str:
    return s if len(s) <= h + t else s[:h] + f' …[{len(s) - h - t} chars]… ' + s[-t:]


def result_text(content) -> str:
    """One tool_result's content (str or list of blocks) → head+tail text."""
    if isinstance(content, str):
        s = content
    elif isinstance(content, list):
        s = '\n'.join(b['text'] for b in content
                      if isinstance(b, dict) and 'type' in b and b['type'] == 'text'
                      and 'text' in b)
    else:
        s = ''
    if len(s) <= RES_HEAD + RES_TAIL:
        return s
    return s[:RES_HEAD] + f' …[{len(s) - RES_HEAD - RES_TAIL} chars]… ' + s[-RES_TAIL:]


def join_list(items: List[str]) -> str:
    return LIST_JOIN.join(items)
