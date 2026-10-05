#!/usr/bin/env python3
"""
ctxshield must stay separable: no source file may import ClaudeStorm.

    python3 ctxshield/tests/test_boundary.py
"""
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
FORBIDDEN = ('core', 'workers', 'tools', 'ux', 'claudestorm', 'ingestion', 'logging_config')
IMPORT = re.compile(r'^\s*(?:from|import)\s+([A-Za-z_][\w.]*)', re.M)


def test_no_claudestorm_imports():
    bad = []
    files = [p for p in PKG.rglob('*.py') if '.backup_' not in p.name]
    for p in files:
        for mod in IMPORT.findall(p.read_text(encoding='utf-8')):
            if mod.split('.')[0] in FORBIDDEN:
                bad.append(f"{p.relative_to(PKG)}: {mod}")
    assert not bad, "ctxshield imports ClaudeStorm:\n  " + '\n  '.join(bad)
    print(f"  ✓ boundary: {len(files)} source files, no ClaudeStorm imports")


def test_no_private_data_files():
    bad = [str(p.relative_to(PKG)) for p in PKG.rglob('*')
           if p.suffix in ('.db', '.sqlite', '.jsonl', '.tsv', '.mdb')]
    assert not bad, "private data inside ctxshield:\n  " + '\n  '.join(bad)
    print("  ✓ boundary: no .db/.jsonl/.tsv/.mdb files inside ctxshield")


if __name__ == '__main__':
    test_no_claudestorm_imports()
    test_no_private_data_files()
    print("ALL PASS")
    sys.exit(0)
