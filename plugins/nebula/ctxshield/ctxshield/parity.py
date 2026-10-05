#!/usr/bin/env python3
"""
T1 parity (build doc §9.6): the native featurizer vs the pure-Python reference, and the native
all-heads gather-dot vs a float64 NumPy dot on the reference features.

    python3 -m ctxshield.parity --features --dataset <dir> [--n 5000] [--out result.json]
    python3 -m ctxshield.parity --scores   --dataset <dir> [--n 5000] [--out result.json]
    python3 -m ctxshield.parity --scores --backend metal --dataset <dir> [--n 5000]

--backend metal|cuda (step 8, T1 GPU part): the GPU scores vs the native CPU scores on the same
units, random weights, hpad 4 and 16; PASS iff |p_gpu − p_cpu| ≤ 1e-5. The backend that ran and
the number of documents that overflowed the on-chip table are reported.

Unit selection is deterministic (sha256 of unit_key order), spread over the dataset's field groups.
Features PASS iff every unit has identical (index, sign) sets and |value| / extras diffs ≤ 1e-6,
for BOTH specs. Scores PASS iff |p_native − p_ref| ≤ 1e-6 (p = sigmoid(logit)), random weights.
A synthetic Unicode stress set is reported separately (it is not part of the pass rule: where
Python and utf8proc normalization disagree, the native result is the spec).

NO .get(). No ClaudeStorm imports.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from ctxshield import dataset as DS
from ctxshield import features as F
from ctxshield import features_ref as R
from ctxshield import spec as S

TOL = 1e-6
TOL_GPU = 1e-5


def groups(dataset_dir: Path) -> Dict[str, List[str]]:
    """field-set key -> heads sharing it."""
    index = json.loads((Path(dataset_dir) / 'dataset.json').read_text())
    g = {}
    for head, meta in sorted(index['heads'].items()):
        key = '|'.join(meta['fields'])
        if key not in g:
            g[key] = []
        g[key].append(head)
    return g


def select(dataset_dir: Path, n: int) -> Dict[str, List[Dict]]:
    """Deterministic units per field group (dedup by unit_key), ~n in total."""
    G = groups(dataset_dir)
    loaded = {key: DS.read_head(Path(dataset_dir), G[key][0])[1] for key in G}
    out, left = {}, n
    # smallest groups first, each taking an equal share of what is left, so the quota a small
    # group cannot fill flows to the larger ones (total = min(n, all rows))
    order = sorted(G, key=lambda k: (len(loaded[k]), k))
    for pos, key in enumerate(order):
        rows = sorted(loaded[key], key=lambda r: hashlib.sha256(r['unit_key'].encode()).hexdigest())
        take = min(len(rows), left // (len(order) - pos))
        out[key] = rows[:take]
        left -= take
    return out


def stress_strings(k: int, seed: int = 7) -> List[str]:
    rng = random.Random(seed)
    pool = ['ﬁ', 'ß', 'İ', 'ı', 'Σ', 'ς', 'K', 'Å', 'é', 'é', '①', 'Ⅻ', 'ｗｉｄｅ', '😀', '中文',
            'ﾊﾟ', '‘', '’', '`', '```', '/usr/bin', 'a_b', 'camelCase', 'os.path', '3.5ms', '42%',
            'deadbeef1', '123e4567-e89b-12d3-a456-426614174000', 'https://x.io/a', 'f.py', "don't",
            ' ', ' ', '\n', ' ', 'x', 'A', '__init__', '(', ')', '.', '?', '́']
    return [''.join(rng.choice(pool) for _ in range(rng.randint(1, 60))) for _ in range(k)]


def compare_features(names: List[str], docs: List[Dict[str, str]], spec: str) -> Dict:
    t0 = time.time()
    indptr, indices, data, extras = F.featurize(names, docs, spec)
    native_ms = (time.time() - t0) * 1000
    bad, worst, ref_ms = [], 0.0, 0.0
    for i, d in enumerate(docs):
        t = time.time()
        ri, rv, re_ = R.featurize(d, S.SPECS[spec]['delex'])
        ref_ms += (time.time() - t) * 1000
        ni = indices[indptr[i]:indptr[i + 1]].tolist()
        nv = data[indptr[i]:indptr[i + 1]]
        if ni != ri:
            bad.append({'doc': i, 'why': 'index set', 'n_native': len(ni), 'n_ref': len(ri),
                        'only_native': len(set(ni) - set(ri)), 'only_ref': len(set(ri) - set(ni))})
            continue
        dv = float(np.max(np.abs(nv.astype(np.float64) - np.array(rv)))) if ri else 0.0
        de = float(np.max(np.abs(extras[i].astype(np.float64) - np.array(re_))))
        worst = max(worst, dv, de)
        if dv > TOL or de > TOL:
            bad.append({'doc': i, 'why': 'values', 'max_val_diff': dv, 'max_extra_diff': de})
    return {'n': len(docs), 'mismatches': len(bad), 'examples': bad[:5], 'max_abs_diff': worst,
            'native_ms': native_ms, 'ref_ms': ref_ms}


def compare_scores(names: List[str], docs: List[Dict[str, str]], spec: str, hpad: int,
                   seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    ne = len(names) + 2
    W = (rng.standard_normal((S.DIM, hpad)) * 0.5).astype(np.float32)
    We = (rng.standard_normal((ne, hpad)) * 0.1).astype(np.float32)
    b = (rng.standard_normal(hpad) * 0.1).astype(np.float32)
    t0 = time.time()
    logits = F.score_logits(names, docs, spec, W, We, b)
    native_ms = (time.time() - t0) * 1000
    worst = 0.0
    for i, d in enumerate(docs):
        ri, rv, re_ = R.featurize(d, S.SPECS[spec]['delex'])
        ref = b.astype(np.float64).copy()
        if ri:
            ref += np.array(rv) @ W[np.array(ri)].astype(np.float64)
        ref += np.array(re_) @ We.astype(np.float64)
        p_ref = 1.0 / (1.0 + np.exp(-ref))
        p_nat = 1.0 / (1.0 + np.exp(-logits[i].astype(np.float64)))
        worst = max(worst, float(np.max(np.abs(p_ref - p_nat))))
    return {'n': len(docs), 'hpad': hpad, 'max_abs_prob_diff': worst, 'native_ms': native_ms,
            'passed': worst <= TOL}


def compare_backend_scores(names: List[str], docs: List[Dict[str, str]], spec: str, hpad: int,
                           seed: int, backend: str) -> Dict:
    rng = np.random.default_rng(seed)
    ne = len(names) + 2
    W = (rng.standard_normal((S.DIM, hpad)) * 0.5).astype(np.float32)
    We = (rng.standard_normal((ne, hpad)) * 0.1).astype(np.float32)
    b = (rng.standard_normal(hpad) * 0.1).astype(np.float32)
    t0 = time.time()
    cpu = F.score_logits(names, docs, spec, W, We, b)
    cpu_ms = (time.time() - t0) * 1000
    scorer = F.make_scorer(backend, names, spec, W, We, b)
    _, rows = F._rows(names, docs)
    scorer.score(rows[:16], 0)                       # warm (pipelines, buffers)
    t0 = time.time()
    gpu = scorer.score(rows, 0)
    gpu_ms = (time.time() - t0) * 1000
    st = scorer.last_stats
    if st['backend'] != backend:
        raise RuntimeError(f"backend mismatch: requested {backend}, ran {st['backend']}")
    p_cpu = 1.0 / (1.0 + np.exp(-cpu.astype(np.float64)))
    p_gpu = 1.0 / (1.0 + np.exp(-gpu.astype(np.float64)))
    worst = float(np.max(np.abs(p_cpu - p_gpu))) if len(docs) else 0.0
    worst_logit = float(np.max(np.abs(cpu.astype(np.float64) - gpu.astype(np.float64)))) if len(docs) else 0.0
    return {'n': len(docs), 'hpad': hpad, 'backend': st['backend'], 'max_abs_prob_diff': worst,
            'max_abs_logit_diff': worst_logit, 'n_overflow_docs': st['n_overflow_docs'],
            'cpu_ms': cpu_ms, 'gpu_ms': gpu_ms, 'passed': worst <= TOL_GPU}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', action='store_true')
    ap.add_argument('--scores', action='store_true')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--backend', type=str, default='cpu', choices=list(F.BACKENDS))
    args = ap.parse_args()
    if not (args.features or args.scores):
        ap.error('need --features or --scores')
    if args.backend != 'cpu':
        if args.features or not args.scores:
            ap.error('--backend metal|cuda is a score comparison: use --scores only')
        F.check_backend(args.backend)
    sel = select(Path(args.dataset), args.n)
    report = {'native': F.native_info(), 'spec_sha': {v: S.spec_sha(v) for v in S.SPECS},
              'groups': {k: len(v) for k, v in sel.items()}, 'features': {}, 'scores': {},
              'stress': {}, 'backend': args.backend}
    ok = True
    for key, rows in sel.items():
        names = key.split('|')
        docs = [r['fields'] for r in rows]
        if args.backend != 'cpu':
            for spec in S.SPECS:
                for hpad in (4, 16):
                    r = compare_backend_scores(names, docs, spec, hpad, 0, args.backend)
                    report['scores'][f'{key}:{spec}:h{hpad}'] = r
                    ok = ok and r['passed']
                    print(f"  scores   {args.backend} vs cpu {spec} {key:<22} hpad {hpad:>2} n {r['n']:>5}  "
                          f"max|Δp| {r['max_abs_prob_diff']:.2e}  max|Δlogit| {r['max_abs_logit_diff']:.2e}  "
                          f"overflow docs {r['n_overflow_docs']:>3}  cpu {r['cpu_ms']:6.1f} ms  "
                          f"{args.backend} {r['gpu_ms']:6.1f} ms")
            continue
        for spec in S.SPECS:
            if args.features:
                r = compare_features(names, docs, spec)
                report['features'][f'{key}:{spec}'] = r
                ok = ok and r['mismatches'] == 0
                print(f"  features {spec} {key:<22} n {r['n']:>5}  mismatches {r['mismatches']:>3}  "
                      f"max|Δ| {r['max_abs_diff']:.2e}  native {r['native_ms']:7.1f} ms  "
                      f"ref {r['ref_ms'] / 1000:6.1f} s")
            if args.scores:
                r = compare_scores(names, docs, spec, 4, 0)
                report['scores'][f'{key}:{spec}'] = r
                ok = ok and r['passed']
                print(f"  scores   {spec} {key:<22} n {r['n']:>5}  max|Δp| {r['max_abs_prob_diff']:.2e}"
                      f"  native {r['native_ms']:7.1f} ms")
    if args.features:
        st = [{'text': s} for s in stress_strings(2000)]
        for spec in S.SPECS:
            r = compare_features(['text'], st, spec)
            report['stress'][spec] = r
            print(f"  stress   {spec} (Unicode/delex, not in pass rule) n {r['n']}  "
                  f"mismatches {r['mismatches']}")
    report['passed'] = ok
    print('PASS' if ok else 'FAIL')
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str) + '\n')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
