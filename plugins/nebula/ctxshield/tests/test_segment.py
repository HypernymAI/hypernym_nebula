#!/usr/bin/env python3
"""
ctxshield.segment + dfc_adapter.finalize on a hand-built message list (no ClaudeStorm, no model).

    python3 ctxshield/tests/test_segment.py

Checks the F14 rules one by one: SKIP_PREFIX wrapper is not a pilot turn; tool-result lines and
'[Request interrupted' lines are absorbed; any other user line ends the turn; thinking-only lines
contribute no text; an exchange without assistant text is dropped and does not consume an
exchange_index; other roles are invisible; §4.2 attachment; disabled → None.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctxshield import dfc_adapter as DA   # noqa: E402
from ctxshield import segment as SEG      # noqa: E402
from ctxshield import shape               # noqa: E402


def m(t, text, thinking_only=False, tool_results=False):
    d = {'type': t, 'text': text, 'is_thinking_only': thinking_only}
    if tool_results:
        d['tool_results'] = [{'tool_use_id': 'x', 'is_error': False, 'user_rejected': False,
                              'result_preview': 'ok'}]
    return d


MSGS = [
    m('user', '<system-reminder>wrapper</system-reminder>'),   # 0 skipped wrapper
    m('user', 'first ask'),                                    # 1 pilot, exchange 0
    m('assistant', 'thinking about it', thinking_only=True),   # 2 thinking-only → no text
    m('assistant', ''),                                        # 3 tool call line, no text
    m('user', '', tool_results=True),                          # 4 tool result absorbed
    m('system', 'sys'),                                        # 5 invisible role
    m('assistant', 'reply one'),                               # 6 text
    m('user', '[Request interrupted by user]'),                # 7 absorbed
    m('assistant', 'reply two'),                               # 8 text (last of exchange 0)
    m('user', 'second ask'),                                   # 9 pilot, NO assistant text → dropped
    m('assistant', 'only thoughts', thinking_only=True),       # 10
    m('user', 'third ask ' + 'x' * 1300),                      # 11 pilot, exchange 1
    m('assistant', 'answer'),                                  # 12
]


def test_segment():
    ex = SEG.segment(MSGS)
    assert [(e['exchange_index'], e['pilot_idx'], e['reply_idxs']) for e in ex] == [
        (0, 1, [6, 8]), (1, 11, [12])], ex
    assert ex[0]['reply'] == 'reply one\nreply two'
    print('  ✓ segmentation: wrapper skipped, tool-result/interrupt absorbed, thinking-only silent,'
          ' text-less exchange dropped without an index')


def test_units():
    u = SEG.build_units(MSGS)
    assert [x['message_idx'] for x in u['S1']] == [1, 11]
    assert [x['message_idx'] for x in u['S6']] == [8, 12]
    assert [(x['local_key'], x['message_idx']) for x in u['S2']] == [('0#t1', 6), ('0#t2', 8),
                                                                    ('1#t1', 12)]
    s1 = u['S1'][1]['payload']
    assert s1['pilot'] == shape.head(MSGS[11]['text'], 1200) and s1['pilot'].endswith(' …[+110]')
    assert s1['previous_reply'] == 'reply one\nreply two'
    assert s1['earlier_asks'] == ['first ask']
    assert u['S1'][0]['payload']['previous_reply'] == '' and u['S1'][0]['payload']['earlier_asks'] == []
    assert u['S2'][2]['payload'] == {'pilot_ask': shape.head(MSGS[11]['text'], 600), 'text': 'answer'}
    f = SEG.unit_fields('S1', s1)
    assert f == {'pilot': s1['pilot'], 'prev': 'reply one\nreply two', 'earlier': 'first ask'}
    print('  ✓ units: §4.2 attachment (S1 pilot, S2 each text line, S6 last text line), caps, fields')


class FakeModel:
    version = 'fake00000000'
    kernel = 'fake'
    heads = {'H13_S1': {'section': 'S1', 'r90': 0.5}, 'H08_S2': {'section': 'S2', 'r90': 0.5},
             'H08_S6': {'section': 'S6', 'r90': 0.5}}

    def score(self, head, docs):
        return np.array([0.9 if i % 2 == 0 else 0.1 for i in range(len(docs))])


def test_finalize():
    res = DA.score_session(MSGS, FakeModel())
    signals = [{'message_idx': i} for i in (1, 6, 11, 12)]     # 8 has no signal → unattached
    top = DA.finalize(res, signals)
    assert signals[0]['shield'] == {'H13_S1': 0.9} and signals[0]['shield_flags'] == ['H13_S1']
    assert signals[1]['shield'] == {'H08_S2': 0.9}
    assert signals[3]['shield'] == {'H08_S2': 0.9, 'H08_S6': 0.1} and signals[3]['shield_flags'] == ['H08_S2']
    assert [e['message_idx'] for e in top['shield_unattached']] == [8]
    assert top['shield_noise_skipped'] == 1
    assert top['shield_eligible'] == {'H08_S2': 3, 'H08_S6': 2, 'H13_S1': 2}
    assert top['shield_counts'] == {'H08_S2': 2, 'H08_S6': 1, 'H13_S1': 1}
    s2 = [{'message_idx': 1}]
    top2 = DA.finalize(DA.score_session(MSGS, None), s2)
    assert all(v is None for v in top2.values()) and s2[0]['shield'] is None
    print('  ✓ finalize: attach by message_idx, unattached list, counts/eligible; disabled → None')


if __name__ == '__main__':
    test_segment()
    test_units()
    test_finalize()
    print('ALL PASS')
