#!/usr/bin/env python3
"""
Check a built ctxshield._native against native/golden/golden_v1.bin (build doc §9.4 / T1 / T9):

  cpu   feature indices sha EXACT (tokenizer + hashing identical on this architecture) and
        |p − p_golden| ≤ 1e-5 (the golden logits were made by the NEON CPU kernel on the Mac)
  cuda  |p_cuda − p_golden| ≤ 1e-5 and |p_cuda − p_cpu_here| ≤ 1e-5, overflow docs counted
  --bench  CPU (all cores) vs CUDA wall time by batch size → crossover

    python3 native/cuda/check_golden.py --golden native/golden/golden_v1.bin --backends cpu,cuda \\
            --bench --out native/cuda/results/cuda_l4_<stamp>.json
Run by native/cuda/run_on_l4.sh (which sets SHIELD_CUDA_UNVERIFIED_OK=true for this run only).
Exit 0 iff every requested check passes. NO .get(). No ClaudeStorm imports.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'golden'))
sys.path.insert(0, str(HERE.parents[1]))

import golden_format as G                                                # noqa: E402
from ctxshield import _native                                            # noqa: E402
from ctxshield import features as F                                      # noqa: E402

TOL = 1e-5
SIZES = (1, 16, 256, 1024, 4096, 16384, 65536, 100000)


def probs(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))


def device_info() -> dict:
    out = {'platform': platform.platform(), 'machine': platform.machine()}
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap',
                            '--format=csv,noheader'], capture_output=True, text=True, check=True)
        out['nvidia_smi'] = q.stdout.strip()
    except FileNotFoundError:
        out['nvidia_smi'] = 'nvidia-smi not found'
    return out


def best(fn, reps: int) -> float:
    t = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--golden', required=True)
    ap.add_argument('--backends', default='cpu,cuda')
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    backends = args.backends.split(',')
    for be in backends:
        if be not in ('cpu', 'cuda', 'metal'):
            raise ValueError(f"unknown backend {be}")
        F.check_backend(be)   # CUDA: refuses unless verified or SHIELD_CUDA_UNVERIFIED_OK=true
    meta, recs = G.read(args.golden)
    report = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'golden': str(args.golden),
              'golden_meta': meta, 'native': F.native_info(), 'device': device_info(),
              'backend_available': {be: list(_native.backend_available(be)) for be in backends},
              'records': [], 'bench': {}}
    ok = True
    for rec in recs:
        ne = len(rec['fields']) + 2
        W, We, b = G.make_weights(rec['seed'], rec['hpad'], ne, rec['w_scale'], rec['we_scale'], rec['b_scale'])
        if G.weights_sha(W, We, b) != rec['weights_sha']:
            raise RuntimeError(f"weight recipe mismatch for {rec['group']} (numpy wrap-around arithmetic?)")
        p_gold = probs(rec['logits'])
        row = {'group': rec['group'], 'delex': rec['delex'], 'hpad': rec['hpad'], 'n': len(rec['docs'])}
        p_cpu = None
        for be in backends:
            sc = _native.Scorer(be, rec['fields'], rec['delex'], W, We, b)
            t0 = time.perf_counter()
            z = sc.score(rec['docs'], 0)
            dt = time.perf_counter() - t0
            st = sc.last_stats
            if st['backend'] != be:
                raise RuntimeError(f"backend mismatch: requested {be}, ran {st['backend']}")
            p = probs(z)
            d = float(np.max(np.abs(p - p_gold)))
            r = {'max_abs_prob_diff_vs_golden': d, 'seconds': dt, 'n_overflow_docs': st['n_overflow_docs']}
            passed = d <= TOL
            if be == 'cpu':
                p_cpu = p
                indptr, indices, _, _ = _native.featurize_batch(rec['fields'], rec['docs'], rec['delex'], 0)
                same = G.indices_sha(indptr, indices) == rec['indices_sha']
                r['feature_indices_exact'] = same
                r['nnz_equal'] = bool(np.array_equal(np.diff(indptr).astype(np.uint32), rec['nnz']))
                passed = passed and same
            elif p_cpu is not None:
                r['max_abs_prob_diff_vs_cpu_here'] = float(np.max(np.abs(p - p_cpu)))
                passed = passed and r['max_abs_prob_diff_vs_cpu_here'] <= TOL
            r['passed'] = passed
            ok = ok and passed
            row[be] = r
            print(f"  {rec['group']:<20} {'V2' if rec['delex'] else 'V1'} hpad {rec['hpad']:>2} {be:<5} "
                  f"max|Δp| vs golden {d:.2e}  overflow {st['n_overflow_docs']:>3}  "
                  f"{'PASS' if passed else 'FAIL'}"
                  + (f"  features exact {r['feature_indices_exact']}" if be == 'cpu' else ''))
        report['records'].append(row)
    if args.bench:
        rec = recs[0]   # pilot|reply V1
        ne = len(rec['fields']) + 2
        W, We, b = G.make_weights(rec['seed'], rec['hpad'], ne, rec['w_scale'], rec['we_scale'], rec['b_scale'])
        docs = (rec['docs'] * (max(SIZES) // len(rec['docs']) + 1))[:max(SIZES)]
        scorers = {be: _native.Scorer(be, rec['fields'], rec['delex'], W, We, b) for be in backends}
        for be, sc in scorers.items():
            sc.score(docs[:20000], 0)
        pts = []
        for n in SIZES:
            row = {'n': n}
            for be, sc in scorers.items():
                row[f'{be}_s'] = best(lambda: sc.score(docs[:n], 0), 3 if n <= 16384 else 2)
                row[f'{be}_stats'] = sc.last_stats
            pts.append(row)
            print('  bench n %6d  ' % n + '  '.join(f"{be} {row[be + '_s'] * 1e3:9.2f} ms" for be in scorers))
        report['bench']['points'] = pts
        if 'cuda' in scorers and 'cpu' in scorers:
            cross = None
            for i, p in enumerate(pts):
                if all(q['cuda_s'] < q['cpu_s'] for q in pts[i:]):
                    cross = p['n']
                    break
            report['bench']['crossover_batch'] = cross
            print(f"  CUDA crossover batch: {cross}")
    report['passed'] = ok
    print('PASS' if ok else 'FAIL')
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=str) + '\n')
        print(f"wrote {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
