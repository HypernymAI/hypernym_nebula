"""
Native featurizer / scorer front door. The native module is REQUIRED: a missing build is an
ImportError at import time (no Python or numba fallback — build doc §9.6).

Which native core: on Linux the bundled, fully static `ctxshield-scorer` program over a pipe
(_native_static — one binary per CPU runs on every distribution, glibc or musl/Alpine); elsewhere the
pybind11 module `_native`. Same kernel, same API. CTXSHIELD_SCORER=static selects the pipe client on any
platform (development: set CTXSHIELD_SCORER_BIN to a locally built scorer).

    featurize(field_names, docs, spec)                               -> (indptr, indices, data, extras)
    score_logits(field_names, docs, spec, W, We, b, backend='cpu')   -> float32[n, hpad]
    make_scorer(backend, field_names, spec, W, We, b)                -> resident-weights scorer

Backends (build doc §9.2.6, T12): 'cpu' | 'metal' | 'cuda'. A requested backend that is not built,
not available on this machine, or — CUDA — not verified by the L4 run (native/backends.json
`backends.cuda.verified`) RAISES; there is never a silent CPU run. CUDA while unverified is allowed
only with SHIELD_CUDA_UNVERIFIED_OK=true (for the L4 verification run itself).

docs: list of {field_name: text}. field_names: the head group's fields (any order; sorted here).
NO .get(). No ClaudeStorm imports.
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ctxshield import _integrity
from ctxshield import spec as S

USE_STATIC_SCORER = sys.platform.startswith('linux') or (
    'CTXSHIELD_SCORER' in os.environ and os.environ['CTXSHIELD_SCORER'] == 'static')
if USE_STATIC_SCORER:
    from ctxshield import _native_static as _native          # verifies its scorer program (_integrity) on import
else:
    _integrity.verify(Path(importlib.util.find_spec('ctxshield._native').origin))   # BEFORE the native code loads
    from ctxshield import _native

_SURR = re.compile('[\ud800-\udfff]')
BACKENDS = ('cpu', 'metal', 'cuda')
BACKENDS_MANIFEST = Path(__file__).resolve().parents[1] / 'native' / 'backends.json'


def clean(s: str) -> str:
    if s.isascii():                 # no surrogate can be ASCII: skip the regex (~7 µs/message)
        return s
    return _SURR.sub('\ufffd', s)


def _rows(field_names: List[str], docs: List[Dict[str, str]]) -> Tuple[List[str], List[List[str]]]:
    names = sorted(field_names)
    return names, [[clean(d[n]) for n in names] for d in docs]


def delex_of(spec: str) -> bool:
    return S.SPECS[spec]['delex']


def featurize(field_names: List[str], docs: List[Dict[str, str]], spec: str, threads: int = 0):
    names, rows = _rows(field_names, docs)
    return _native.featurize_batch(names, rows, delex_of(spec), threads)


def backends_manifest() -> Dict:
    """native/backends.json — crashes if missing (a GPU backend without its manifest is an error)."""
    return json.loads(BACKENDS_MANIFEST.read_text())


def check_backend(backend: str) -> None:
    """Raise unless `backend` may run here. 'cpu' always may."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (one of {BACKENDS})")
    if backend == 'cpu':
        return
    if backend == 'cuda':
        m = backends_manifest()['backends']['cuda']
        ok = ('SHIELD_CUDA_UNVERIFIED_OK' in os.environ
              and os.environ['SHIELD_CUDA_UNVERIFIED_OK'] == 'true')
        if not m['verified'] and not ok:
            raise RuntimeError(f"ctxshield: CUDA backend refused — {m['status']} "
                               f"(backends.cuda.verified=false in {BACKENDS_MANIFEST}; the L4 run "
                               f"sets SHIELD_CUDA_UNVERIFIED_OK=true explicitly)")
    available, why = _native.backend_available(backend)
    if not available:
        raise RuntimeError(f"ctxshield: backend {backend!r} unavailable: {why}")


def make_scorer(backend: str, field_names: List[str], spec: str, W: np.ndarray, We: np.ndarray,
                b: np.ndarray):
    """A native Scorer with the weights resident in the backend's memory. .score(rows, threads)
    takes rows from _rows(); .last_stats['backend'] is the backend that ran."""
    check_backend(backend)
    return _native.Scorer(backend, sorted(field_names), delex_of(spec), W, We, b)


def score_logits(field_names: List[str], docs: List[Dict[str, str]], spec: str, W: np.ndarray,
                 We: np.ndarray, b: np.ndarray, threads: int = 0, backend: str = 'cpu') -> np.ndarray:
    names, rows = _rows(field_names, docs)
    if backend == 'cpu':
        return _native.score_batch(names, rows, delex_of(spec), W, We, b, threads)
    scorer = make_scorer(backend, names, spec, W, We, b)
    return scorer.score(rows, threads)


def native_info() -> Dict[str, str]:
    return {'utf8proc': _native.utf8proc_version(), 'unicode': _native.unicode_version(),
            'dim_bits': str(_native.DIM_BITS),
            'kernel': 'ctxshield-scorer (static, pipe)' if USE_STATIC_SCORER else 'ctxshield._native (pybind11)'}
