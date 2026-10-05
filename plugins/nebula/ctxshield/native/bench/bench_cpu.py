#!/usr/bin/env python3
"""
CPU benchmark for the ctxshield native kernel (build doc §9.5 / T9).

    cd ctxshield && python3 native/bench/bench_cpu.py [--dataset ../tools/shield_bridge/datasets/v1]
                                                      [--out result.json] [--label before]

Texts: <dataset>/H13_S6.jsonl, fields pilot + reply (the S6 group's fields), every row.
Weights: fixed random float32, hpad = 4 (the S6 group has 4 heads).
Measured for V1 and V2:
  single   one message per call, threads=1, through Python (score_logits); p50 / p99 µs over every
           message (each timed separately, after a warm-up pass)
  kernel   (if the module exposes time_score) the same per message, timed inside C++ — no Python
  core1    all messages in one batch call, threads=1 → µs per message and ns per char
  batch    100,000 messages (rows repeated) in one call, all cores (threads=0) and threads=16, through
           features.score_logits (includes its Python surrogate-cleaning pass), and the native call
           alone on pre-cleaned rows
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
from ctxshield import spec as S                                          # noqa: E402

NAMES = ['pilot', 'reply']


def load(dataset: Path):
    docs = []
    with open(dataset / 'H13_S6.jsonl') as fh:
        for line in fh:
            r = json.loads(line)
            docs.append({'pilot': r['fields']['pilot'], 'reply': r['fields']['reply']})
    return docs


def weights(hpad: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((S.DIM, hpad)) * 0.5).astype(np.float32)
    We = (rng.standard_normal((len(NAMES) + 2, hpad)) * 0.1).astype(np.float32)
    b = (rng.standard_normal(hpad) * 0.1).astype(np.float32)
    return W, We, b


def pct(a, q):
    return float(np.percentile(np.array(a), q))


def bench_spec(docs, spec, W, We, b, n_batch):
    out = {}
    chars = [len(d['pilot']) + len(d['reply']) for d in docs]
    out['n_messages'] = len(docs)
    out['mean_chars'] = float(np.mean(chars))
    # warm-up (thread-local scratch, page faults on W)
    F.score_logits(NAMES, docs[:200], spec, W, We, b, threads=1)
    t = []
    for d in docs:
        t0 = time.perf_counter_ns()
        F.score_logits(NAMES, [d], spec, W, We, b, threads=1)
        t.append((time.perf_counter_ns() - t0) / 1000.0)
    # the §9.5 reference length (S3 measured at a 1,146-char mean): messages of 1,046–1,246 chars
    ref = [i for i, c in enumerate(chars) if 1046 <= c <= 1246]
    out['ref_bucket_n'] = len(ref)
    out['ref_bucket_single_py_p50_us'] = pct([t[i] for i in ref], 50)
    out['ref_bucket_single_py_p99_us'] = pct([t[i] for i in ref], 99)
    out['single_py_p50_us'] = pct(t, 50)
    out['single_py_p99_us'] = pct(t, 99)
    out['single_py_mean_us'] = float(np.mean(t))
    if hasattr(_native, 'time_score'):
        names, rows = F._rows(NAMES, docs)
        ns = _native.time_score(names, rows, F.delex_of(spec), W, We, b, 3)
        us = np.asarray(ns, dtype=np.float64) / 1000.0
        out['single_kernel_p50_us'] = pct(us, 50)
        out['single_kernel_p99_us'] = pct(us, 99)
        out['single_kernel_mean_us'] = float(np.mean(us))
        out['ref_bucket_kernel_p50_us'] = pct(us[ref], 50)
        out['ref_bucket_kernel_p99_us'] = pct(us[ref], 99)
    t0 = time.perf_counter()
    F.score_logits(NAMES, docs, spec, W, We, b, threads=1)
    dt = time.perf_counter() - t0
    out['core1_us_per_msg'] = dt * 1e6 / len(docs)
    out['core1_ns_per_char'] = dt * 1e9 / sum(chars)
    big = (docs * (n_batch // len(docs) + 1))[:n_batch]
    for th, key in ((0, 'batch_all_cores_s'), (16, 'batch_16_threads_s')):
        F.score_logits(NAMES, big[:20000], spec, W, We, b, threads=th)   # warm every thread
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            F.score_logits(NAMES, big, spec, W, We, b, threads=th)
            best = min(best, time.perf_counter() - t0)
        out[key] = best
    # the same batch with the Python-side cleaning (features._rows: surrogate regex) done beforehand:
    # the native call alone (string views + kernel), all cores
    names, rows = F._rows(NAMES, big)
    _native.score_batch(names, rows[:20000], F.delex_of(spec), W, We, b, 0)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        _native.score_batch(names, rows, F.delex_of(spec), W, We, b, 0)
        best = min(best, time.perf_counter() - t0)
    out['batch_native_all_cores_s'] = best
    out['batch_n'] = n_batch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='../tools/shield_bridge/datasets/v1')
    ap.add_argument('--n-batch', type=int, default=100_000)
    ap.add_argument('--out', default='')
    ap.add_argument('--label', default='')
    args = ap.parse_args()
    docs = load(Path(args.dataset))
    W, We, b = weights()
    report = {'label': args.label, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
              'native': F.native_info(), 'specs': {}}
    for spec in ('V1', 'V2'):
        r = bench_spec(docs, spec, W, We, b, args.n_batch)
        report['specs'][spec] = r
        kern = (f"  kernel p50 {r['single_kernel_p50_us']:.1f} p99 {r['single_kernel_p99_us']:.1f} µs"
                f" [1,046–1,246 chars n {r['ref_bucket_n']}: kernel p50 {r['ref_bucket_kernel_p50_us']:.1f}"
                f" p99 {r['ref_bucket_kernel_p99_us']:.1f}, py p50 {r['ref_bucket_single_py_p50_us']:.1f}]"
                if 'single_kernel_p50_us' in r else
                f" [1,046–1,246 chars n {r['ref_bucket_n']}: py p50 {r['ref_bucket_single_py_p50_us']:.1f}"
                f" p99 {r['ref_bucket_single_py_p99_us']:.1f}]")
        print(f"{spec}: n {r['n_messages']} mean {r['mean_chars']:.0f} chars | single(py) p50 "
              f"{r['single_py_p50_us']:.1f} p99 {r['single_py_p99_us']:.1f} µs{kern} | 1-core "
              f"{r['core1_us_per_msg']:.1f} µs/msg {r['core1_ns_per_char']:.1f} ns/char | "
              f"{r['batch_n']} all-cores {r['batch_all_cores_s']:.3f} s, 16 thr "
              f"{r['batch_16_threads_s']:.3f} s (native call alone "
              f"{r['batch_native_all_cores_s']:.3f} s)")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
