#!/usr/bin/env python3
"""
nebula MCP server (stdio, JSON-RPC 2.0, newline-delimited) — build doc Part XVIII. NOT registered.

Implements the MCP handshake directly (no SDK dependency): initialize → notifications/initialized →
tools/list → tools/call. Tools (all local, no network):
  shield_heads            which heads are loaded, their classes, thresholds, specs
  shield_score_reply      score assistant text block(s) against the pilot's ask (Stop heads)
  shield_score_turn       score a pilot turn given the previous reply and earlier asks (user_prompt heads)
  shield_score_transcript score the LAST turn of a Claude Code transcript JSONL (both sides)
Every call is logged to the same decision log as the hooks (event 'mcp').

    <python> mcp_server.py [--config path]
NO .get() on data dicts (JSON-RPC params are validated explicitly). Errors → JSON-RPC error responses.
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

from settings import excluded, resolve_config
from shield_core import Shield, sha, transcript_turn

PROTOCOL = '2025-06-18'
TOOLS = [
    {'name': 'shield_heads', 'description': 'List the loaded shield heads (class, threshold, spec).',
     'inputSchema': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
    {'name': 'shield_score_reply',
     'description': "Score assistant text block(s) for the Stop heads (e.g. claim beyond evidence) "
                    "given the pilot's ask. Returns per-head max score, fired heads, reframe guidance.",
     'inputSchema': {'type': 'object', 'required': ['pilot_ask', 'blocks'],
                     'properties': {'pilot_ask': {'type': 'string'},
                                    'blocks': {'type': 'array', 'items': {'type': 'string'}}}}},
    {'name': 'shield_score_turn',
     'description': "Score a pilot turn (e.g. forced repetition) given the previous assistant reply and "
                    "up to two earlier pilot asks.",
     'inputSchema': {'type': 'object', 'required': ['pilot', 'previous_reply', 'earlier_asks'],
                     'properties': {'pilot': {'type': 'string'}, 'previous_reply': {'type': 'string'},
                                    'earlier_asks': {'type': 'array', 'items': {'type': 'string'}}}}},
    {'name': 'shield_score_transcript',
     'description': 'Score the last turn of a Claude Code transcript JSONL: the pilot turn and the reply.',
     'inputSchema': {'type': 'object', 'required': ['transcript_path'],
                     'properties': {'transcript_path': {'type': 'string'}}}},
]


def need(args: Dict, key: str, typ):
    if key not in args or not isinstance(args[key], typ):
        raise ValueError(f"argument '{key}' ({typ.__name__}) is required")
    return args[key]


def call(sh: Shield, name: str, args: Dict) -> Dict:
    t0 = time.time()
    if not sh.enabled:
        return {'enabled': False}
    if name == 'shield_heads':
        out = {ev: {h: dict(hc, class_id=sh.head_models[h].heads[h]['class_id'])
                    for h, hc in sh.cfg[ev].items()} for ev in ('user_prompt', 'stop')}
        return {'enabled': True, 'heads': out, 'config_sha': sh.config_sha}
    if name == 'shield_score_reply':
        ask = need(args, 'pilot_ask', str)
        blocks = need(args, 'blocks', list)
        scores, fired = sh.score_heads('stop', sh.stop_docs(ask, [str(b) for b in blocks]))
        text = '\n'.join(str(b) for b in blocks)
        guidance = sh.reframe_text(fired, 'stop') if fired else ''
    elif name == 'shield_score_turn':
        pilot = need(args, 'pilot', str)
        prev = need(args, 'previous_reply', str)
        earlier = [str(a) for a in need(args, 'earlier_asks', list)][-2:]
        ex = [{'user': a, 'assistant': ''} for a in earlier]
        if ex:
            ex[-1]['assistant'] = prev
        else:
            ex = [{'user': '', 'assistant': prev}]
        f = sh.pilot_turn_fields(pilot, ex)
        if not earlier:
            f['earlier'] = ''
        scores, fired = sh.score_heads('user_prompt', {'S1': [f]})
        text = pilot
        guidance = sh.reframe_text(fired, 'user_prompt') if fired else ''
    elif name == 'shield_score_transcript':
        path = need(args, 'transcript_path', str)
        if excluded(sh.cfg, path, None):
            raise ValueError(f"{path} is in an excluded project (settings: exclude_projects) — not read")
        pilot, blocks, ex = transcript_turn(path)
        prior = ex[:-1] if ex and blocks else ex
        s1, f1 = sh.score_heads('user_prompt', {'S1': [sh.pilot_turn_fields(pilot, prior)]})
        s2, f2 = sh.score_heads('stop', sh.stop_docs(pilot, blocks)) if blocks else ({}, {})
        scores, fired = {**s1, **s2}, {**f1, **f2}
        text = pilot + '\n---\n' + '\n'.join(blocks)
        guidance = '\n'.join(x for x in (sh.reframe_text(f1, 'user_prompt') if f1 else '',
                                          sh.reframe_text(f2, 'stop') if f2 else '') if x)
    else:
        raise ValueError(f"unknown tool {name}")
    sh.log.put({'event': 'mcp', 'session_id': 'mcp', 'pilot_sha': sha(text), 'attempt': 0, 'text': text,
                'scores_json': json.dumps(scores), 'fired_json': json.dumps(fired),
                'decision': 'flag' if fired else 'allow', 'config_sha': sh.config_sha,
                'ms': (time.time() - t0) * 1000})
    return {'enabled': True, 'scores': scores, 'fired': fired, 'reframe': guidance,
            'ms': (time.time() - t0) * 1000}


def respond(rid, result=None, error=None) -> None:
    msg = {'jsonrpc': '2.0', 'id': rid}
    if error is not None:
        msg['error'] = error
    else:
        msg['result'] = result
    sys.stdout.write(json.dumps(msg) + '\n')
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    sh = Shield(resolve_config(args.config))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg['method'] if 'method' in msg else None
        rid = msg['id'] if 'id' in msg else None
        if rid is None:                       # notification (e.g. notifications/initialized)
            continue
        try:
            if method == 'initialize':
                respond(rid, {'protocolVersion': PROTOCOL, 'capabilities': {'tools': {}},
                              'serverInfo': {'name': 'nebula', 'version': '0.0.1'}})
            elif method == 'tools/list':
                respond(rid, {'tools': TOOLS})
            elif method == 'tools/call':
                p = msg['params']
                res = call(sh, p['name'], p['arguments'] if 'arguments' in p else {})
                respond(rid, {'content': [{'type': 'text', 'text': json.dumps(res, indent=1)}],
                              'isError': False})
            elif method == 'ping':
                respond(rid, {})
            else:
                respond(rid, error={'code': -32601, 'message': f'method not found: {method}'})
        except (ValueError, KeyError) as e:
            respond(rid, {'content': [{'type': 'text', 'text': f'error: {e}'}], 'isError': True})


if __name__ == '__main__':
    main()
