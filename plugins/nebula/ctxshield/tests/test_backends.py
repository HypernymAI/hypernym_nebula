#!/usr/bin/env python3
"""
T12 backend honesty + backend score parity on synthetic text (build doc Part X):
  - CPU Scorer (C ABI) == score_batch (same kernel) ≤ 1e-6
  - a backend that is not built / not available RAISES (Scorer, score_batch_metal, features)
  - CUDA refused while backends.json says verified=false (unless SHIELD_CUDA_UNVERIFIED_OK=true)
  - Metal (when available): last_stats['backend'] == 'metal', |p_metal − p_cpu| ≤ 1e-5, including
    documents large enough to overflow the on-chip table (device-memory path)

    cd ctxshield && python3 tests/test_backends.py
NO .get().
"""
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctxshield import _native                                            # noqa: E402
from ctxshield import features as F                                      # noqa: E402
from ctxshield import parity as P                                        # noqa: E402
from ctxshield import spec as S                                          # noqa: E402

NAMES = ['pilot', 'reply']
TOL_GPU = 1e-5


def weights(hpad, seed=3):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((S.DIM, hpad)) * 0.5).astype(np.float32)
    We = (rng.standard_normal((len(NAMES) + 2, hpad)) * 0.1).astype(np.float32)
    b = (rng.standard_normal(hpad) * 0.1).astype(np.float32)
    return W, We, b


def synthetic_docs(n, seed=11):
    rng = random.Random(seed)
    stress = P.stress_strings(400, seed)
    words = ['the', 'Build', 'failed', 'again', "don't", 'core/x.py', '42ms', 'snake_case', 'camelCase',
             'https://x.io/a', '`ls`', '```code```', 'Straße', 'ﬁle', '中文', '😀', 'deadbeef1', '?', '\n']
    docs = []
    for i in range(n):
        k = rng.choice([0, 1, 5, 40, 300, 1500])
        text = ' '.join(rng.choice(words) for _ in range(k)) + rng.choice(stress)
        docs.append({'pilot': rng.choice(stress) + ' ' + ' '.join(rng.choice(words) for _ in range(20)),
                     'reply': text})
    # large documents: on-chip table overflow → device-memory path on the GPU
    big = ' '.join(f'token{j} word{j % 97} x{j * 7919 % 100003}' for j in range(6000))
    docs.append({'pilot': 'WHY', 'reply': big})
    docs.append({'pilot': '', 'reply': ''})
    docs.append({'pilot': 'a' * 50000, 'reply': 'the ' * 20000})
    return docs


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))


def test_cpu_scorer_equals_score_batch():
    docs = synthetic_docs(300)
    for spec in S.SPECS:
        for hpad in (4, 8, 16):
            W, We, b = weights(hpad)
            a = F.score_logits(NAMES, docs, spec, W, We, b)
            sc = F.make_scorer('cpu', NAMES, spec, W, We, b)
            names, rows = F._rows(NAMES, docs)
            c = sc.score(rows, 0)
            assert sc.last_stats['backend'] == 'cpu'
            d = float(np.max(np.abs(sigmoid(a) - sigmoid(c))))
            assert d <= 1e-6, (spec, hpad, d)
    print("  ✓ CPU Scorer (C ABI) == score_batch, V1/V2, hpad 4/8/16")


def raises(fn, what):
    try:
        fn()
    except (RuntimeError, ValueError) as e:
        return str(e)
    raise AssertionError(f"expected an error: {what}")


def test_unavailable_backends_raise():
    W, We, b = weights(4)
    docs = [{'pilot': 'x', 'reply': 'y'}]
    raises(lambda: F.score_logits(NAMES, docs, 'V1', W, We, b, backend='nope'), 'unknown backend')
    raises(lambda: _native.Scorer('nope', NAMES, False, W, We, b), 'unknown backend (native)')
    if not _native.CUDA_BUILT:
        m = raises(lambda: _native.Scorer('cuda', NAMES, False, W, We, b), 'cuda not built')
        assert 'CUDA' in m, m
    elif not ('SHIELD_CUDA_UNVERIFIED_OK' in os.environ and os.environ['SHIELD_CUDA_UNVERIFIED_OK'] == 'true'):
        # built with CUDA but unverified: the native layer refuses too (not only the Python loader)
        m = raises(lambda: _native.Scorer('cuda', NAMES, False, W, We, b), 'cuda unverified (native)')
        assert 'UNTESTED' in m, m
    # the loader refuses CUDA while unverified, whatever the build
    man = F.backends_manifest()
    assert man['backends']['cuda']['verified'] is False
    saved = os.environ['SHIELD_CUDA_UNVERIFIED_OK'] if 'SHIELD_CUDA_UNVERIFIED_OK' in os.environ else None
    if saved is not None:
        del os.environ['SHIELD_CUDA_UNVERIFIED_OK']
    try:
        m = raises(lambda: F.score_logits(NAMES, docs, 'V1', W, We, b, backend='cuda'), 'cuda unverified')
        assert 'verified=false' in m, m
        os.environ['SHIELD_CUDA_UNVERIFIED_OK'] = 'true'
        if not _native.CUDA_BUILT:
            m = raises(lambda: F.score_logits(NAMES, docs, 'V1', W, We, b, backend='cuda'), 'cuda not built')
            assert 'unavailable' in m, m
    finally:
        if saved is None:
            if 'SHIELD_CUDA_UNVERIFIED_OK' in os.environ:
                del os.environ['SHIELD_CUDA_UNVERIFIED_OK']
        else:
            os.environ['SHIELD_CUDA_UNVERIFIED_OK'] = saved
    ok, why = _native.backend_available('metal')
    if not ok:
        raises(lambda: _native.score_batch_metal(NAMES, [['x', 'y']], False, W, We, b), 'metal unavailable')
        raises(lambda: F.score_logits(NAMES, docs, 'V1', W, We, b, backend='metal'), 'metal unavailable')
        print(f"  ✓ Metal unavailable here ({why}) → every Metal entry point raises")
    print("  ✓ unknown / unbuilt / unverified backends raise (CUDA refused while verified=false)")


def test_metal_parity_synthetic():
    ok, why = _native.backend_available('metal')
    if not ok:
        print(f"  - Metal parity not run: {why}")
        return
    docs = synthetic_docs(600)
    for spec in S.SPECS:
        for hpad in (4, 8, 16):
            W, We, b = weights(hpad)
            cpu = F.score_logits(NAMES, docs, spec, W, We, b)
            sc = F.make_scorer('metal', NAMES, spec, W, We, b)
            names, rows = F._rows(NAMES, docs)
            gpu = sc.score(rows, 0)
            st = sc.last_stats
            assert st['backend'] == 'metal', st
            d = float(np.max(np.abs(sigmoid(cpu) - sigmoid(gpu))))
            assert d <= TOL_GPU, (spec, hpad, d)
            assert st['n_overflow_docs'] >= 2, st     # the big docs took the device-memory path
    # > 2 pipelined chunks (16,384 docs each) incl. overflow docs in a later chunk
    many = (synthetic_docs(600) * 70)[:40000]
    W, We, b = weights(4)
    cpu = F.score_logits(NAMES, many, 'V1', W, We, b)
    gpu = F.score_logits(NAMES, many, 'V1', W, We, b, backend='metal')
    d = float(np.max(np.abs(sigmoid(cpu) - sigmoid(gpu))))
    assert d <= TOL_GPU, d
    print(f"  ✓ Metal == CPU ≤ {TOL_GPU} on {len(many)} docs (3 pipelined chunks), max |Δp| {d:.1e}")
    W, We, b = weights(16)
    one = _native.score_batch_metal(NAMES, [['x', 'y']], False, W, We, b)
    assert one.shape == (1, 16)
    print(f"  ✓ Metal == CPU ≤ {TOL_GPU} on {len(docs)} synthetic docs, V1/V2, hpad 4/8/16 "
          f"(overflow docs {st['n_overflow_docs']})")


if __name__ == '__main__':
    test_cpu_scorer_equals_score_batch()
    test_unavailable_backends_raise()
    test_metal_parity_synthetic()
    print("ALL PASS")
