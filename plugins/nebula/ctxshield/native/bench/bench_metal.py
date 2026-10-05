#!/usr/bin/env python3
"""
Metal vs CPU (all P-cores) throughput by batch size → the crossover batch size (build doc §9.3,
§9.5, T9). Same texts as bench_cpu.py (<dataset>/H13_S6.jsonl, pilot + reply), random weights,
hpad 4. Both backends are warm (resident weights, warmed threads / pipelines). Best of --reps.

    cd ctxshield && python3 native/bench/bench_metal.py [--dataset ../tools/shield_bridge/datasets/v1]
                                                        [--out native/bench/results/metal.json]
crossover = the smallest measured batch size from which Metal is faster at every larger size.
NO .get().
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ctxshield import _native                                            # noqa: E402
from ctxshield import features as F                                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cpu import NAMES, load, weights                               # noqa: E402

SIZES = (1, 4, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 100000)


def best(fn, reps):
    t = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='../tools/shield_bridge/datasets/v1')
    ap.add_argument('--out', default='')
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    F.check_backend('metal')
    docs = load(Path(args.dataset))
    W, We, b = weights()
    report = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'native': F.native_info(),
              'metal_device': _native.backend_available('metal')[1],
              'cpu_threads': args.threads if args.threads > 0 else _native.default_threads(),
              'specs': {}}
    for spec in ('V1', 'V2'):
        big = (docs * (max(SIZES) // len(docs) + 1))[:max(SIZES)]
        names, rows = F._rows(NAMES, big)
        cpu = F.make_scorer('cpu', NAMES, spec, W, We, b)
        gpu = F.make_scorer('metal', NAMES, spec, W, We, b)
        cpu.score(rows[:20000], args.threads)
        gpu.score(rows[:20000], args.threads)
        pts = []
        for n in SIZES:
            sub = rows[:n]
            reps = 5 if n <= 16384 else 3
            tc = best(lambda: cpu.score(sub, args.threads), reps)
            tg = best(lambda: gpu.score(sub, args.threads), reps)
            st = gpu.last_stats
            if st['backend'] != 'metal':
                raise RuntimeError(f"backend mismatch: {st['backend']}")
            pts.append({'n': n, 'cpu_s': tc, 'metal_s': tg, 'metal_tokenize_s': st['t_tokenize_s'],
                        'metal_device_s': st['t_device_s'], 'metal_overflow_docs': st['n_overflow_docs'],
                        'speedup': tc / tg})
            print(f"{spec} n {n:>6}  cpu {tc * 1e3:9.2f} ms  metal {tg * 1e3:9.2f} ms "
                  f"(tokenize {st['t_tokenize_s'] * 1e3:8.2f}, gpu {st['t_device_s'] * 1e3:8.2f})  "
                  f"x{tc / tg:5.2f}  overflow {st['n_overflow_docs']}")
        cross = None
        for i, p in enumerate(pts):
            if all(q['metal_s'] < q['cpu_s'] for q in pts[i:]):
                cross = p['n']
                break
        report['specs'][spec] = {'points': pts, 'crossover_batch': cross}
        print(f"{spec}: crossover batch = {cross}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
