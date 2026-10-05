#!/usr/bin/env python3
"""
nebula look-and-feel customization (writes ~/.nebula/settings.json; validated; updates never touch it).

    customize.py wings            show how to choose, the current wings, and where the contact sheet is
    customize.py wings <N>        use contact-sheet wings #N (0–159) beside HYPERNYM.ai
    customize.py icons on|off     the badge icons (shield, wings) — off = the plain 🛡 backup
    customize.py soliton          is soliton (beta) on?
    customize.py soliton on|off   soliton: a flagged reply gets a vivid banner under it and is rephrased once;
                                  off = the normal mode again (its own switch file, ~/.nebula/soliton.json)
NO .get(). Imports at top.
"""
import noegress                                   # noqa: F401 — FIRST: the no-network guarantee (noegress.py)
import json
import sys

import nebula_statusline
from brand import wings_icon
from settings import (PLUGIN_ROOT, USER_DIR, USER_SETTINGS, _validate_user, resolve_config, set_soliton, soliton_file,
                      soliton_on, user_settings, wings_count)

SHEET = PLUGIN_ROOT / 'shield' / 'brand_assets' / 'wings_contact_sheet.png'
# Soliton's rephrase can call nebula's own scoring tools (local, offline) to check a draft — a permission prompt there
# stopped the pilot mid-rephrase (2026-09-30). Turning soliton on grants exactly these, once, visibly; off takes back
# only what soliton itself added (recorded in soliton.json), never the person's own rules.
SOLITON_TOOLS = ['mcp__plugin_nebula_nebula__shield_heads', 'mcp__plugin_nebula_nebula__shield_score_reply',
                 'mcp__plugin_nebula_nebula__shield_score_transcript', 'mcp__plugin_nebula_nebula__shield_score_turn']


def soliton_permissions(grant: bool, granted_before: list) -> list:
    """Add (grant) or remove (the ones soliton added) the scoring tools in Claude Code's user settings.json
    (backed up first by write_settings). Returns the rules soliton now owns."""
    s = nebula_statusline.load_json(nebula_statusline.settings_path())
    perms = s['permissions'] if 'permissions' in s else {}
    allow = list(perms['allow']) if 'allow' in perms else []
    if grant:
        added = [t for t in SOLITON_TOOLS if t not in allow]
        owned = granted_before + added
        allow += added
    else:
        added = []
        owned = []
        allow = [t for t in allow if t not in granted_before]
    if added or (not grant and granted_before):
        perms['allow'] = allow
        s['permissions'] = perms
        note = nebula_statusline.write_settings(s)
        print(f"{'allowed' if grant else 'removed'}: {', '.join(added if grant else granted_before)}  "
              f"({nebula_statusline.settings_path()}; {note})")
    return owned


def save(u: dict) -> None:
    _validate_user(u)
    USER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = USER_SETTINGS.with_suffix('.tmp')
    tmp.write_text(json.dumps(u, indent=1, sort_keys=True) + '\n')
    tmp.replace(USER_SETTINGS)


def glyph(n: int) -> str:
    return ''.join(ch for ch, _ in wings_icon(n)[0])


def main() -> None:
    a = sys.argv[1:]
    if a == ['wings']:
        cur = json.loads(resolve_config(None).read_text())['brand']['wings']
        print(f"""Wings beside HYPERNYM.ai — current: #{cur}  {glyph(cur)}

How to choose:
  1. The contact sheet shows all {wings_count()} candidates, numbered, large and at badge size beside HYPERNYM.ai:
       {SHEET}
     (/nebula:shield-wings opens it for you.)
  2. Pick a number, then run   /nebula:shield-wings <number>     e.g. /nebula:shield-wings 91
  3. The badge updates within a second. Change your mind any time; nothing else is affected.
Even numbers: the two wings touch (3 characters). Odd numbers: a one-dot gap between them (4 characters).
Turn all badge icons off/on:  /nebula:shield-wings off   ·   /nebula:shield-wings on""")
    elif len(a) == 2 and a[0] == 'wings' and a[1].isdigit():
        n = int(a[1])
        wings_icon(n)                                       # raises with the valid range if out of bounds
        u = user_settings()
        u['brand'] = {**(u['brand'] if 'brand' in u else {}), 'wings': n}
        save(u)
        print(f"wings → #{n}  {glyph(n)}   (saved in {USER_SETTINGS})")
    elif len(a) == 2 and a[0] == 'icons' and a[1] in ('on', 'off'):
        u = user_settings()
        u['brand'] = {**(u['brand'] if 'brand' in u else {}), 'icons': a[1] == 'on'}
        save(u)
        print(f"badge icons {a[1]}")
    elif a == ['soliton']:
        cfg = json.loads(resolve_config(None).read_text())
        print(f"soliton (beta): {'on' if soliton_on(cfg) else 'off'}  ({soliton_file(cfg)})")
    elif len(a) == 2 and a[0] == 'soliton' and a[1] in ('on', 'off'):
        cfg = json.loads(resolve_config(None).read_text())
        f = soliton_file(cfg)
        before = json.loads(f.read_text())['granted'] if f.exists() and 'granted' in json.loads(f.read_text()) else []
        owned = soliton_permissions(a[1] == 'on', before)
        set_soliton(cfg, a[1] == 'on', owned)
        if a[1] == 'on':
            print("soliton (beta) on — when a reply peaks orange (any class ≥ 1.0×), a bright SHIELDED banner appears "
                  "under it and Claude rephrases it once, asked in plain words; the spinner shows \"shielding\" "
                  "meanwhile. Takes effect on the next reply.")
        else:
            print("soliton off — replies are scored and the judgement shown as before; nothing is rephrased.")
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
