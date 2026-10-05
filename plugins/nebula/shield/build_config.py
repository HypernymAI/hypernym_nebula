#!/usr/bin/env python3
"""
DEVELOPER tool: write the SHIPPED defaults (defaults.json) from the heads' model cards. Users never run this — they
change behaviour through ~/.nebula/settings.json (see settings.py). Paths are RELATIVE to the plugin root so the
file is valid wherever Claude Code installs the plugin; settings.resolve_config makes them absolute at run time.

Heads: CERTIFIED = PASS against the Jev bar on every judged codebase (fabricated confession); WORKING = cross-codebase
AUC ≥ 0.80 but not certified — they flag and are recorded, never enforce. Thresholds = codebase-out-of-fold R90.

    python3 build_config.py
NO .get().
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_ROOT = HERE.parent
COMBINED_REL = 'models/combined_v1'
MODELS_REL = 'models/v1'
PASSED = {'H08_S2': ('stop', 'V2'), 'H08_S6': ('stop', 'V2')}
WORKING = {'H15_S2': ('stop', 'V2'), 'H15_S6': ('stop', 'V2'), 'H06_S2': ('stop', 'V2'), 'H13_S6': ('stop', 'V2')}


def main() -> None:
    cfg = {
        # ── user-settable (settings.py validates overrides) ──
        'enabled': True,
        'mode': 'shadow',                 # score + record locally; never blocks, never injects
        'show': True,                     # one-line judgement note under each reply
        'text_retention': 'none',         # PRIVACY DEFAULT: store scores + numbers, not the transcript text
        'exclude_projects': [],
        'brand': {'enabled': True, 'bg': 'auto', 'rest_alpha': 0.55, 'mark': 'mid', 'mode': 'gentle',
                  'icons': True, 'wings': 76},   # wings = contact-sheet number (pilot: #76)
        # ── fixed by the release ──
        'db_path': '~/.nebula/nebula.db',
        'log_db': '~/.nebula/nebula.db',
        'model_dir': MODELS_REL,
        'combined_dir': COMBINED_REL,
        'max_rewrites': 3,
        'user_prompt': {}, 'stop': {},
        'provenance': 'build doc S21/S22; certified = PASS on every judged codebase; models = combined_v1; '
                      'threshold = codebase-OOF R90',
    }
    for group, certified in ((PASSED, True), (WORKING, False)):
        for head, (event, spec) in group.items():
            info = json.loads((PLUGIN_ROOT / COMBINED_REL / f'{head}.json').read_text())
            cfg[event][head] = {'model': 'combined', 'spec': spec, 'threshold': info['r90'], 'certified': certified}
    (HERE / 'defaults.json').write_text(json.dumps(cfg, indent=1, sort_keys=True) + '\n')
    print(json.dumps(cfg, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
