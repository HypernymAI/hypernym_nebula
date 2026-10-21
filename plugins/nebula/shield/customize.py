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
import os
import sys

import nebula_statusline
from brand import wings_icon
from settings import (PLUGIN_ROOT, USER_DIR, USER_SETTINGS, _validate_user, resolve_config, set_soliton, soliton_file,
                      soliton_on, user_settings, wings_count)

SHEET = PLUGIN_ROOT / 'shield' / 'brand_assets' / 'wings_contact_sheet.png'
# Soliton used to grant nebula's scoring tools in Claude Code's settings when switched on; since 2026-10-21 it grants
# nothing (soliton_permissions) — Claude Code asks the person once, and "always allow" is their own rule.


def soliton_permissions(grant: bool, granted_before: list) -> list:
    """Soliton no longer grants anything (2026-10-21 security update): a plugin must not edit Claude Code's own
    permission rules. Turning soliton OFF still removes the rules an older version added (recorded in soliton.json) —
    removal only ever lowers what is allowed. Returns the rules soliton still owns."""
    if grant:
        return granted_before
    if not granted_before:
        return []
    s = nebula_statusline.load_json(nebula_statusline.settings_path())
    perms = s['permissions'] if 'permissions' in s else {}
    allow = list(perms['allow']) if 'allow' in perms else []
    perms['allow'] = [t for t in allow if t not in granted_before]
    s['permissions'] = perms
    note = nebula_statusline.write_settings(s)
    print(f"removed: {', '.join(granted_before)}  ({nebula_statusline.settings_path()}; {note})")
    return []


def save(u: dict) -> None:
    _validate_user(u)
    USER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = USER_SETTINGS.with_name(f'{USER_SETTINGS.name}.{os.getpid()}.tmp')   # own temp file: no write race
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
