#!/usr/bin/env python3
"""
nebula MCP server (stdio, JSON-RPC 2.0, newline-delimited) — build doc Part XVIII. NOT registered.

Implements the MCP handshake directly (no SDK dependency): initialize → notifications/initialized →
tools/list → tools/call. Tools (all local, no network):
  shield_heads            which heads are loaded, their classes, thresholds, specs
  shield_score_reply      score assistant text block(s) against the pilot's ask (Stop heads)
  shield_score_turn       score a pilot turn given the previous reply and earlier asks (user_prompt heads)
  shield_score_transcript score the LAST turn of a Claude Code transcript JSONL (both sides)
Settings (2026-10-21 security update — the slash commands and skills call these instead of putting the user's words
into a shell line, where Claude Code pastes them as text and any `;` would run as a command):
  shield_soliton          soliton (beta) on / off / status                       customize.py soliton
  shield_retention        none | excerpt | full                                  privacy.py retention
  shield_exclude          add a project exclusion (a glob, passed as data)       privacy.py exclude
  shield_include          remove a project exclusion                             privacy.py include
  shield_wings            show the contact sheet | a number | on | off           customize.py wings / icons
  shield_statusline       install | force | remove                               install_statusline.py
  shield_status           on/off, mode, detectors and thresholds, recent flags   status.py
  shield_privacy_report   what is stored, where, how much, what is excluded      privacy.py report
  shield_export           everything stored, as a JSON file in ~/.nebula         privacy.py export
  shield_backfill         read all past sessions into the local record (long)    backfill.py
  shield_purge            preview what would be deleted; with confirm, the PERSON's own dialog decides (elicitation):
                          Accept deletes, anything else deletes nothing. A client without dialogs cannot purge here.
Every argument is checked against its allowed values / shape, then handed to the same script as one argv element —
never through a shell. Every scoring call is logged to the same decision log as the hooks (event 'mcp').

    <python> mcp_server.py [--config path]
NO .get() on data dicts (JSON-RPC params are validated explicitly). Errors → JSON-RPC error responses.
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

from settings import excluded, resolve_config
from shield_core import Shield, sha, transcript_turn

PROTOCOL = '2025-06-18'
HERE = Path(__file__).resolve().parent
SHEET = HERE / 'brand_assets' / 'wings_contact_sheet.png'
RETENTION_LEVELS = ('none', 'excerpt', 'full')
STATUSLINE_ACTIONS = {'install': [], 'force': ['--force'], 'remove': ['--remove']}
MAX_PATTERN = 200
BACKFILL_TIMEOUT_S = 6 * 3600
CLIENT = {'elicitation': False}                   # set from the client's capabilities at initialize
BACKLOG = []                                      # messages that arrived while a dialog was open
ELICIT_SEQ = [0]
NO_PROPS = {'type': 'object', 'properties': {}, 'additionalProperties': False}
TOOLS = [
    {'name': 'shield_heads', 'description': 'List the loaded shield heads (class, threshold, spec).',
     'inputSchema': NO_PROPS},
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
    {'name': 'shield_soliton',
     'description': "Soliton (beta): when a reply peaks orange (confession, handback or claim at 1.0x or more) a SHIELDED "
                    "banner appears under it and the reply is rephrased once. state 'on' or 'off' switches it (one "
                    "switch for every session on this computer); 'status' shows it. Show the returned text verbatim.",
     'inputSchema': {'type': 'object', 'required': ['state'],
                     'properties': {'state': {'type': 'string', 'enum': ['on', 'off', 'status']}},
                     'additionalProperties': False}},
    {'name': 'shield_retention',
     'description': "What transcript text nebula keeps: none (scores and counts only), excerpt (first 300 chars) or "
                    "full. Show the returned text verbatim.",
     'inputSchema': {'type': 'object', 'required': ['level'],
                     'properties': {'level': {'type': 'string', 'enum': list(RETENTION_LEVELS)}},
                     'additionalProperties': False}},
    {'name': 'shield_exclude',
     'description': "Never read, score or store sessions from matching projects: a glob on the project folder name or "
                    "working directory (e.g. *client-x*). Show the returned text verbatim.",
     'inputSchema': {'type': 'object', 'required': ['pattern'], 'properties': {'pattern': {'type': 'string'}},
                     'additionalProperties': False}},
    {'name': 'shield_include',
     'description': "Remove a project exclusion (the same glob given to shield_exclude). Show the returned text verbatim.",
     'inputSchema': {'type': 'object', 'required': ['pattern'], 'properties': {'pattern': {'type': 'string'}},
                     'additionalProperties': False}},
    {'name': 'shield_wings',
     'description': "The wings beside HYPERNYM.ai on the nebula badge: 'show' opens the numbered contact sheet and lists "
                    "the current choice; a number switches to that candidate; 'on' / 'off' turns all badge icons on or "
                    "off. Show the returned text verbatim.",
     'inputSchema': {'type': 'object', 'required': ['choice'], 'properties': {'choice': {'type': 'string'}},
                     'additionalProperties': False}},
    {'name': 'shield_statusline',
     'description': "The shared hypernym-nebula status line: install it, force it over someone else's (backed up "
                    "first), or remove it. Show the returned text verbatim.",
     'inputSchema': {'type': 'object', 'required': ['action'],
                     'properties': {'action': {'type': 'string', 'enum': list(STATUSLINE_ACTIONS)}},
                     'additionalProperties': False}},
    {'name': 'shield_status',
     'description': "nebula status: on/off, mode, detectors and thresholds, what it has flagged recently, per model.",
     'inputSchema': NO_PROPS},
    {'name': 'shield_privacy_report',
     'description': "Exactly what nebula stores on this computer, where, how much, and what is excluded.",
     'inputSchema': NO_PROPS},
    {'name': 'shield_export',
     'description': "Export everything nebula has stored to a JSON file in ~/.nebula; returns where it was written.",
     'inputSchema': NO_PROPS},
    {'name': 'shield_backfill',
     'description': "Opt in: read ALL past Claude Code sessions on this computer into the local nebula record (never "
                    "uploaded). Low priority; can take a long time on a big history.",
     'inputSchema': NO_PROPS},
    {'name': 'shield_purge',
     'description': "Delete everything nebula has stored on this computer (database, caches, logs). Without confirm it "
                    "only shows what would be deleted. With confirm=true the person is asked in a dialog; only their "
                    "Accept deletes anything. Use confirm only when the user asked to purge (\"yes\").",
     'inputSchema': {'type': 'object', 'properties': {'confirm': {'type': 'boolean'}},
                     'additionalProperties': False}},
]


def run_script(script: str, args: list, timeout: int = 120) -> str:
    """One of this plugin's own scripts under this interpreter, args as argv elements (no shell); its output verbatim.
    Raises ValueError with the script's own message when it refuses or fails."""
    r = subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        why = r.stderr.strip()
        if not why:
            why = r.stdout.strip()
        raise ValueError(why[-600:])
    return r.stdout.strip()


def need(args: Dict, key: str, typ):
    if key not in args or not isinstance(args[key], typ):
        raise ValueError(f"argument '{key}' ({typ.__name__}) is required")
    return args[key]


def one_of(args: Dict, key: str, allowed) -> str:
    v = need(args, key, str)
    if v not in allowed:
        raise ValueError(f"argument '{key}' must be one of {list(allowed)}")
    return v


def pattern_of(args: Dict) -> str:
    """A project glob: plain text on one line, stored as data in nebula's settings."""
    p = need(args, 'pattern', str).strip()
    if not p:
        raise ValueError("argument 'pattern' is empty")
    if len(p) > MAX_PATTERN:
        raise ValueError(f"argument 'pattern' is longer than {MAX_PATTERN} characters")
    for ch in p:
        if ord(ch) < 32:
            raise ValueError("argument 'pattern' must be one line of plain text")
    return p


def wings(args: Dict) -> str:
    c = need(args, 'choice', str).strip()
    if c in ('on', 'off'):
        return run_script('customize.py', ['icons', c])
    if c.isdigit():
        if len(c) > 4:
            raise ValueError("argument 'choice' is not a wings number")
        return run_script('customize.py', ['wings', c])
    if c != 'show':
        raise ValueError("argument 'choice' must be 'show', a number, 'on' or 'off'")
    out = run_script('customize.py', ['wings'])
    if sys.platform == 'darwin':
        subprocess.run(['osascript', '-e', f'tell application "Preview" to open POSIX file "{SHEET}"', '-e',
                        'tell application "Preview" to activate'], capture_output=True, timeout=20)
        out += '\n\n(The contact sheet is open in Preview.)'
    return out


def purge(args: Dict) -> str:
    confirm = False
    if 'confirm' in args:
        if not isinstance(args['confirm'], bool):
            raise ValueError("argument 'confirm' must be true or false")
        confirm = args['confirm']
    preview = run_script('privacy.py', ['purge'])
    if not confirm:
        return preview
    if not CLIENT['elicitation']:
        return ("NOT DELETED — this client cannot show a confirmation dialog, so nothing is purged from here. What "
                "would be deleted:\n\n" + preview)
    message = ("Delete everything nebula has stored on this computer? " + ' '.join(preview.split()) +
               " Accept deletes it; Decline or closing this keeps everything.")
    try:
        answer = elicit(message, {'type': 'object', 'properties': {}})
    except EOFError:
        return "NOT DELETED — the client closed the connection before answering; nothing was deleted."
    if 'error' in answer:
        return f"NOT DELETED — the confirmation dialog failed ({answer['error']['message']}); nothing was deleted."
    if answer['result']['action'] != 'accept':
        return "Not deleted — you kept everything."
    return run_script('privacy.py', ['purge', 'yes'])


def settings_tool(name: str, args: Dict) -> str:
    if name == 'shield_soliton':
        state = one_of(args, 'state', ('on', 'off', 'status'))
        if state == 'status':
            return run_script('customize.py', ['soliton'])
        return run_script('customize.py', ['soliton', state])
    if name == 'shield_retention':
        return run_script('privacy.py', ['retention', one_of(args, 'level', RETENTION_LEVELS)])
    if name == 'shield_exclude':
        return run_script('privacy.py', ['exclude', pattern_of(args)])
    if name == 'shield_include':
        return run_script('privacy.py', ['include', pattern_of(args)])
    if name == 'shield_wings':
        return wings(args)
    if name == 'shield_statusline':
        return run_script('install_statusline.py', STATUSLINE_ACTIONS[one_of(args, 'action', STATUSLINE_ACTIONS)])
    if name == 'shield_purge':
        return purge(args)
    if name == 'shield_status':
        return run_script('status.py', [])
    if name == 'shield_privacy_report':
        return run_script('privacy.py', ['report'])
    if name == 'shield_export':
        return run_script('privacy.py', ['export'])
    if name == 'shield_backfill':
        return run_script('backfill.py', [], timeout=BACKFILL_TIMEOUT_S)
    raise KeyError(name)


SETTINGS_TOOLS = ('shield_soliton', 'shield_retention', 'shield_exclude', 'shield_include', 'shield_wings',
                  'shield_statusline', 'shield_purge', 'shield_status', 'shield_privacy_report', 'shield_export',
                  'shield_backfill')


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


def next_message():
    """The next client message: first any that arrived while a dialog was open, then stdin. None at end of input."""
    if BACKLOG:
        return BACKLOG.pop(0)
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        if line.strip():
            return json.loads(line)


def elicit(message: str, schema: dict) -> dict:
    """Send elicitation/create and wait for THE answer. Pings are answered meanwhile; anything else is kept."""
    ELICIT_SEQ[0] += 1
    rid = f"nebula-confirm-{ELICIT_SEQ[0]}"
    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': 'elicitation/create',
                                 'params': {'message': message, 'requestedSchema': schema}}) + '\n')
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        if not line.strip():
            continue
        msg = json.loads(line)
        if 'method' not in msg and 'id' in msg and msg['id'] == rid:
            return msg
        if 'method' in msg and msg['method'] == 'ping' and 'id' in msg:
            respond(msg['id'], {})
            continue
        BACKLOG.append(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    sh = Shield(resolve_config(args.config))
    while True:
        msg = next_message()
        if msg is None:
            return
        method = msg['method'] if 'method' in msg else None
        rid = msg['id'] if 'id' in msg else None
        if rid is None:                       # notification (e.g. notifications/initialized)
            continue
        if method is None:                    # a stray response (to a dialog that already ended)
            continue
        try:
            if method == 'initialize':
                p = msg['params'] if 'params' in msg else {}
                CLIENT['elicitation'] = 'capabilities' in p and 'elicitation' in p['capabilities']
                respond(rid, {'protocolVersion': PROTOCOL, 'capabilities': {'tools': {}},
                              'serverInfo': {'name': 'nebula', 'version': '0.0.1'}})
            elif method == 'tools/list':
                respond(rid, {'tools': TOOLS})
            elif method == 'tools/call':
                p = msg['params']
                arguments = p['arguments'] if 'arguments' in p else {}
                if p['name'] in SETTINGS_TOOLS:
                    text = settings_tool(p['name'], arguments)
                else:
                    text = json.dumps(call(sh, p['name'], arguments), indent=1)
                respond(rid, {'content': [{'type': 'text', 'text': text}], 'isError': False})
            elif method == 'ping':
                respond(rid, {})
            else:
                respond(rid, error={'code': -32601, 'message': f'method not found: {method}'})
        except (ValueError, KeyError) as e:
            respond(rid, {'content': [{'type': 'text', 'text': f'error: {e}'}], 'isError': True})


if __name__ == '__main__':
    main()
