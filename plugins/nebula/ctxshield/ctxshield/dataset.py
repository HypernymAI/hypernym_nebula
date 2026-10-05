"""
The dataset contract between a private data source (e.g. ClaudeStorm's tools/shield_bridge/) and
ctxshield training. ctxshield never reads a benchmark database — only this format.

    <dataset_dir>/dataset.json        {format, version, created_at, heads: {head: meta}}
    <dataset_dir>/<head>.jsonl        one row per line, sorted by unit_key:
        {"unit_key": str, "group": str, "part": "dev"|"test"|"post", "label": 0|1,
         "fields": {field_name: str, ...}}

head meta: {class_id, section, tier, fields: [names], n_dev, n_pos_dev, n_test, n_pos_test,
            n_post, n_pos_post, rows_sha}
rows_sha = sha256 over sorted lines "unit_key\\tlabel\\tpart\\tgroup\\tfields_sha".

Write-once per (dir, head): rewriting a head with a different rows_sha raises — a changed dataset
is a new version directory, never an overwrite.

NO .get(). No ClaudeStorm imports.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

FORMAT = 'ctxshield.dataset'
FORMAT_VERSION = 1
PARTS = ('dev', 'test', 'post')


def fields_sha(fields: Dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False)
                          .encode('utf-8')).hexdigest()


def rows_sha(rows: List[Dict]) -> str:
    lines = sorted(f"{r['unit_key']}\t{r['label']}\t{r['part']}\t{r['group']}\t"
                   f"{fields_sha(r['fields'])}" for r in rows)
    return hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest()


def validate_row(r: Dict, field_names: List[str]) -> None:
    for k in ('unit_key', 'group', 'part', 'label', 'fields'):
        if k not in r:
            raise ValueError(f"dataset row missing '{k}': {str(r)[:200]}")
    if r['part'] not in PARTS:
        raise ValueError(f"bad part {r['part']!r} in {r['unit_key']}")
    if r['label'] not in (0, 1):
        raise ValueError(f"bad label {r['label']!r} in {r['unit_key']}")
    if sorted(r['fields']) != sorted(field_names):
        raise ValueError(f"fields {sorted(r['fields'])} != declared {sorted(field_names)} "
                         f"in {r['unit_key']}")
    for name, v in r['fields'].items():
        if not isinstance(v, str):
            raise ValueError(f"field {name} of {r['unit_key']} is {type(v).__name__}, not str")


def counts(rows: List[Dict]) -> Dict[str, int]:
    out = {}
    for p in PARTS:
        out[f'n_{p}'] = sum(1 for r in rows if r['part'] == p)
        out[f'n_pos_{p}'] = sum(1 for r in rows if r['part'] == p and r['label'] == 1)
    return out


def write_head(dataset_dir: Path, head: str, meta: Dict, rows: List[Dict]) -> Tuple[str, bool]:
    """Write one head. Returns (rows_sha, written). Same sha already present → (sha, False).
    Different sha already present → RuntimeError (write-once)."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        validate_row(r, meta['fields'])
    groups_by_part = {}
    for r in rows:
        if r['group'] in groups_by_part and groups_by_part[r['group']] != r['part']:
            raise ValueError(f"group {r['group']} appears in parts {groups_by_part[r['group']]} "
                             f"and {r['part']} — split must be by group")
        groups_by_part[r['group']] = r['part']
    rows = sorted(rows, key=lambda r: r['unit_key'])
    sha = rows_sha(rows)
    index_path = dataset_dir / 'dataset.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else \
        {'format': FORMAT, 'version': FORMAT_VERSION, 'created_at': time.time(), 'heads': {}}
    if index['format'] != FORMAT or index['version'] != FORMAT_VERSION:
        raise RuntimeError(f"{index_path}: format {index['format']} v{index['version']} "
                           f"≠ {FORMAT} v{FORMAT_VERSION}")
    if head in index['heads']:
        if index['heads'][head]['rows_sha'] != sha:
            raise RuntimeError(f"REFUSED: {head} already written in {dataset_dir} with rows_sha "
                               f"{index['heads'][head]['rows_sha'][:12]}; new data has {sha[:12]}. "
                               f"A changed dataset is a new version directory.")
        return sha, False
    with open(dataset_dir / f'{head}.jsonl', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps({'unit_key': r['unit_key'], 'group': r['group'], 'part': r['part'],
                                'label': r['label'], 'fields': r['fields']},
                               sort_keys=True, ensure_ascii=False) + '\n')
    index['heads'][head] = dict(meta, **counts(rows), rows_sha=sha)
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True) + '\n')
    return sha, True


def read_head(dataset_dir: Path, head: str) -> Tuple[Dict, List[Dict]]:
    """(meta, rows). Crashes if the file no longer hashes to its recorded rows_sha."""
    dataset_dir = Path(dataset_dir)
    index = json.loads((dataset_dir / 'dataset.json').read_text())
    meta = index['heads'][head]
    rows = []
    with open(dataset_dir / f'{head}.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            validate_row(r, meta['fields'])
            rows.append(r)
    if rows_sha(rows) != meta['rows_sha']:
        raise RuntimeError(f"FATAL: {dataset_dir}/{head}.jsonl fails its rows_sha")
    return meta, rows
