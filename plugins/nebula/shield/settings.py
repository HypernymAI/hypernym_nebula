"""
nebula settings: SHIPPED DEFAULTS (plugin/shield/defaults.json, replaced on every plugin update) ⊕ the USER's
SETTINGS (~/.nebula/settings.json, never touched by an update) → the EFFECTIVE config (~/.nebula/config.json,
absolute paths into the current plugin version). JSON only — the status-line badge imports this, so it stays light.

User-settable keys (anything else raises — no silent ignore):
  enabled           true | false                       kill switch
  mode              "shadow" | "enforce"               shadow = score + record, never interferes
SOLITON (beta) is NOT a mode in settings.json: it is its own switch, soliton.json {"on": true|false} beside the
config's own database (~/.nebula/soliton.json for the real one; a test's temporary config has its own, so the pilot's
switch never leaks into a test — seen 2026-09-30)
(/nebula:shield-soliton). Reason (2026-09-30): settings.json is shared with every running session, and an older
version rejects a mode it does not know — "mode": "soliton" there made every session still on the old version fail
its Stop hook after every reply. Older versions never read soliton.json, so switching it can break nothing.
  show              true | false                       one-line judgement note under each reply
  text_retention    "none" | "excerpt" | "full"        what of the transcript TEXT is kept in the local DB
                                                        (default none: scores + numbers only)
  exclude_projects  ["*client-x*", ...]                fnmatch against the transcript's project folder AND the session
                                                        cwd: an excluded session is never read, scored or stored
  brand             {enabled, bg ("auto" | [r,g,b]), rest_alpha, mark, mode}
The effective config is rewritten when the defaults, the user settings or the plugin root change.
NO .get(). Imports at top.
"""
import fnmatch
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = PLUGIN_ROOT / 'shield' / 'defaults.json'
USER_DIR = Path.home() / '.nebula'
USER_SETTINGS = USER_DIR / 'settings.json'
EFFECTIVE = USER_DIR / 'config.json'

RETENTION = ('none', 'excerpt', 'full')
MODES = ('shadow', 'enforce')
BRAND_KEYS = {'enabled': bool, 'bg': (str, list), 'rest_alpha': (int, float), 'mark': str, 'mode': str,
              'icons': bool, 'wings': int}
BRAND_MARKS = ('mid', 'full', 'compact', 'house')
BRAND_MODES = ('gentle', 'cameo', 'always')


def _validate_user(u: Dict) -> None:
    allowed = {'enabled', 'mode', 'show', 'text_retention', 'exclude_projects', 'brand'}
    unknown = set(u) - allowed
    if unknown:
        raise ValueError(f"{USER_SETTINGS}: unknown setting(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    if 'enabled' in u and not isinstance(u['enabled'], bool):
        raise ValueError("enabled must be true or false")
    if 'show' in u and not isinstance(u['show'], bool):
        raise ValueError("show must be true or false")
    if 'mode' in u and u['mode'] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if 'text_retention' in u and u['text_retention'] not in RETENTION:
        raise ValueError(f"text_retention must be one of {RETENTION}")
    if 'exclude_projects' in u and not (isinstance(u['exclude_projects'], list)
                                        and all(isinstance(g, str) and g for g in u['exclude_projects'])):
        raise ValueError("exclude_projects must be a list of non-empty glob strings")
    if 'brand' in u:
        b = u['brand']
        if not isinstance(b, dict) or set(b) - set(BRAND_KEYS):
            raise ValueError(f"brand must be an object with keys from {sorted(BRAND_KEYS)}")
        for k, v in b.items():
            if not isinstance(v, BRAND_KEYS[k]):
                raise ValueError(f"brand.{k} has the wrong type")
        if 'bg' in b and b['bg'] != 'auto' and not (isinstance(b['bg'], list) and len(b['bg']) == 3
                                                    and all(isinstance(x, int) and 0 <= x <= 255 for x in b['bg'])):
            raise ValueError('brand.bg must be "auto" or [r, g, b] (0-255)')
        if 'rest_alpha' in b and not 0.0 <= b['rest_alpha'] <= 1.0:
            raise ValueError("brand.rest_alpha must be between 0 and 1")
        if 'mark' in b and b['mark'] not in BRAND_MARKS:
            raise ValueError(f"brand.mark must be one of {BRAND_MARKS}")
        if 'mode' in b and b['mode'] not in BRAND_MODES:
            raise ValueError(f"brand.mode must be one of {BRAND_MODES}")
        if 'wings' in b and not 0 <= b['wings'] < wings_count():
            raise ValueError(f"brand.wings must be 0..{wings_count() - 1} — pick from the contact sheet "
                             f"(/nebula:shield-wings)")


def wings_count() -> int:
    return len(json.loads((PLUGIN_ROOT / 'shield' / 'brand_assets' / 'wings_candidates.json').read_text()))


def user_settings() -> Dict:
    if not USER_SETTINGS.exists():
        return {}
    u = json.loads(USER_SETTINGS.read_text())
    if not isinstance(u, dict):
        raise ValueError(f"{USER_SETTINGS} must hold a JSON object")
    _validate_user(u)
    return u


def build_effective() -> Dict:
    cfg = json.loads(DEFAULTS.read_text())
    u = user_settings()
    for k, v in u.items():
        if k == 'brand':
            cfg['brand'] = {**cfg['brand'], **v}
        else:
            cfg[k] = v
    cfg['model_dir'] = str(PLUGIN_ROOT / cfg['model_dir'])
    cfg['combined_dir'] = str(PLUGIN_ROOT / cfg['combined_dir'])
    cfg['db_path'] = str(Path(cfg['db_path']).expanduser())
    cfg['log_db'] = str(Path(cfg['log_db']).expanduser())
    cfg['plugin_root'] = str(PLUGIN_ROOT)
    cfg['sources_sha'] = _sources_sha()
    return cfg


def _sources_sha() -> str:
    h = hashlib.sha256(str(PLUGIN_ROOT).encode())
    h.update(DEFAULTS.read_bytes())
    if USER_SETTINGS.exists():
        h.update(USER_SETTINGS.read_bytes())
    return h.hexdigest()


def resolve_config(explicit: Optional[str]) -> Path:
    """An explicit --config path is used as-is (tests). Otherwise the effective config, rebuilt if stale."""
    if explicit is not None:
        return Path(explicit)
    private_dir()
    if EFFECTIVE.exists():
        cur = json.loads(EFFECTIVE.read_text())
        if 'sources_sha' in cur and cur['sources_sha'] == _sources_sha():
            return EFFECTIVE
    cfg = build_effective()
    tmp = EFFECTIVE.with_name(f'{EFFECTIVE.name}.{os.getpid()}.tmp')   # own temp file: no write race
    tmp.write_text(json.dumps(cfg, indent=1, sort_keys=True) + '\n')
    tmp.replace(EFFECTIVE)
    return EFFECTIVE


def soliton_file(cfg: Dict) -> Path:
    """Soliton's switch for THIS config: beside its database (see the module docstring)."""
    return Path(cfg['db_path']).expanduser().parent / 'soliton.json'


def soliton_on(cfg: Dict) -> bool:
    f = soliton_file(cfg)
    return f.exists() and json.loads(f.read_text())['on'] is True


def set_soliton(cfg: Dict, on: bool, granted: list) -> None:
    """granted = the permission rules soliton itself added (customize.py), so turning it off removes exactly those."""
    f = soliton_file(cfg)
    f.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = f.with_name(f'{f.name}.{os.getpid()}.tmp')   # own temp file: no write race
    tmp.write_text(json.dumps({'on': on, 'granted': granted}) + '\n')
    tmp.replace(f)


def private_dir() -> Path:
    """~/.nebula exists and is readable by this user account only (0700)."""
    USER_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if USER_DIR.stat().st_mode & 0o077:
        os.chmod(USER_DIR, 0o700)
    return USER_DIR


def excluded(cfg: Dict, transcript_path: Optional[str], cwd: Optional[str]) -> bool:
    """True when the session matches an exclude_projects glob (by project folder name or by working directory)."""
    globs = cfg['exclude_projects']
    names = []
    if transcript_path:
        names.append(Path(transcript_path).parent.name)
    if cwd:
        names.append(cwd)
    return any(fnmatch.fnmatch(n, g) for n in names for g in globs)
