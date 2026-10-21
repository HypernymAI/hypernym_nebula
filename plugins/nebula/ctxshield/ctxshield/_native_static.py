"""
ctxshield._native over a pipe: the same API as the pybind11 module, served by the bundled, fully static
`ctxshield-scorer` program (native/scorer/scorer_main.cpp). Used on Linux — one static binary per CPU runs on
every distribution, glibc or musl (Alpine) — because it needs nothing from the system at run time.

The scorer is started on first use, once per process, and serves every call over stdin/stdout (CXS1 protocol,
documented in scorer_main.cpp). It never opens a socket or a file. If it dies, the next call RAISES with its
exit code — it is not restarted behind your back.

Same honesty rules as the native module: an unknown / unbuilt backend raises (the scorer is CPU only, so
'metal' and 'cuda' report unavailable and refuse to create).

    CTXSHIELD_SCORER_BIN=<path>   use this scorer binary instead of the bundled one (development / tests;
                                  nebula's no-egress guard still only lets the bundled binaries run)
NO .get(). Imports at top.
"""
import atexit
import os
import platform
import struct
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ctxshield import _integrity

DIM_BITS = 18
ABI_VERSION = 1
PROTOCOL = 1
METAL_BUILT = False
CUDA_BUILT = False

_DIM = 1 << DIM_BITS
_BACKENDS = {'cpu': 0, 'metal': 1, 'cuda': 2}
_BACKEND_NAMES = {0: 'cpu', 1: 'metal', 2: 'cuda'}
_ARCH = {'x86_64': 'x86_64', 'amd64': 'x86_64', 'aarch64': 'aarch64', 'arm64': 'aarch64'}
_MAX_CACHED = 16
OP_HELLO, OP_AVAILABLE, OP_FEATURIZE, OP_CREATE, OP_SCORE, OP_DESTROY = 1, 2, 3, 4, 5, 6

STATIC_DIR = Path(__file__).resolve().parent / 'static'


def bundled_binary() -> Path:
    """The bundled scorer for this CPU. ImportError when this CPU has none (same contract as a missing
    native module: no fallback)."""
    m = platform.machine().lower()
    if m not in _ARCH:
        raise ImportError(f"ctxshield: no static scorer for CPU {platform.machine()!r} (x86_64 / aarch64 only)")
    return STATIC_DIR / f'ctxshield-scorer-linux-{_ARCH[m]}'


def _scorer_path() -> Path:
    if 'CTXSHIELD_SCORER_BIN' in os.environ:
        return Path(os.environ['CTXSHIELD_SCORER_BIN']).resolve()
    return bundled_binary()


SCORER = _scorer_path()
if not SCORER.is_file():
    raise ImportError(f"ctxshield: static scorer missing: {SCORER}")
if not os.access(SCORER, os.X_OK):
    raise ImportError(f"ctxshield: static scorer is not executable: {SCORER} (chmod +x)")
if 'CTXSHIELD_SCORER_BIN' not in os.environ:
    _integrity.verify(SCORER)              # the bundled scorer runs only if it is byte-for-byte the pinned build


class _Client:
    """One scorer process per OS process (a forked child starts its own)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc = None
        self._pid = 0
        self.hello = None

    def _start(self) -> None:
        self._proc = subprocess.Popen([str(SCORER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      close_fds=True)
        self._pid = os.getpid()
        h = self._call_locked(OP_HELLO, b'')
        proto, dim_bits, abi, threads = struct.unpack_from('<Iiii', h, 0)
        off = 16
        utf8proc, off = _rd_str(h, off)
        unicode, off = _rd_str(h, off)
        if proto != PROTOCOL or dim_bits != DIM_BITS or abi != ABI_VERSION:
            raise RuntimeError(f"ctxshield: scorer {SCORER} speaks protocol {proto} / dim_bits {dim_bits} / abi {abi};"
                               f" this client needs {PROTOCOL} / {DIM_BITS} / {ABI_VERSION}")
        self.hello = {'threads': threads, 'utf8proc': utf8proc, 'unicode': unicode}

    def _call_locked(self, op: int, payload: bytes) -> bytes:
        p = self._proc
        try:
            p.stdin.write(b'CXS1' + struct.pack('<IQ', op, len(payload)))
            p.stdin.write(payload)
            p.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError(f"ctxshield: scorer {SCORER} exited (code {p.wait()})") from None
        head = p.stdout.read(16)
        if len(head) != 16:
            raise RuntimeError(f"ctxshield: scorer {SCORER} exited (code {p.wait()})")
        if head[:4] != b'CXR1':
            raise RuntimeError(f"ctxshield: scorer {SCORER} sent a bad response header")
        status, n = struct.unpack_from('<IQ', head, 4)
        body = p.stdout.read(n) if n else b''
        if len(body) != n:
            raise RuntimeError(f"ctxshield: scorer {SCORER} exited mid-response (code {p.wait()})")
        if status != 0:
            raise RuntimeError(body.decode('utf-8', 'replace'))
        return body

    def call(self, op: int, payload: bytes) -> bytes:
        with self._lock:
            if self._proc is None or self._pid != os.getpid():
                self._start()
            return self._call_locked(op, payload)

    def info(self) -> Dict:
        with self._lock:
            if self._proc is None or self._pid != os.getpid():
                self._start()
            return self.hello

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._pid == os.getpid():
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
                self._proc.stdout.close()
            self._proc = None

    def alive(self) -> bool:
        return self._proc is not None and self._pid == os.getpid() and self._proc.poll() is None


_CLIENT = _Client()
atexit.register(_CLIENT.close)


def _rd_str(b: bytes, off: int) -> Tuple[str, int]:
    (k,) = struct.unpack_from('<I', b, off)
    off += 4
    return b[off:off + k].decode('utf-8'), off + k


def _str(s: str) -> bytes:
    e = s.encode('utf-8')
    return struct.pack('<I', len(e)) + e


def _names(names: List[str]) -> bytes:
    return struct.pack('<I', len(names)) + b''.join(_str(n) for n in names)


def _docs(docs, nfields: int) -> bytes:
    parts = []
    for d in docs:
        if not isinstance(d, (list, tuple)):
            raise RuntimeError("ctxshield: each doc must be a list of str")
        if len(d) != nfields:
            raise RuntimeError("ctxshield: doc has wrong number of fields")
        for t in d:
            if not isinstance(t, str):
                raise RuntimeError("ctxshield: field text must be str")
            parts.append(t.encode('utf-8'))   # surrogates raise here, as PyUnicode_AsUTF8AndSize does
    lens = np.fromiter((len(x) for x in parts), dtype='<u8', count=len(parts))
    return struct.pack('<QI', len(docs), nfields) + lens.tobytes() + b''.join(parts)


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype='<f4')


def _check_weights(names: List[str], W: np.ndarray, We: np.ndarray, b: np.ndarray) -> int:
    if W.ndim != 2 or W.shape[0] != _DIM:
        raise RuntimeError("ctxshield: W must be [2^18, hpad]")
    hpad = W.shape[1]
    if hpad <= 0 or hpad > 16 or (hpad & 3) != 0:
        raise RuntimeError("ctxshield: hpad must be 4, 8, 12 or 16")
    if We.ndim != 2 or We.shape[0] != len(names) + 2 or We.shape[1] != hpad:
        raise RuntimeError("ctxshield: We must be [n_extras, hpad]")
    if b.ndim != 1 or b.shape[0] != hpad:
        raise RuntimeError("ctxshield: b must be [hpad]")
    return int(hpad)


def _backend_code(name: str) -> int:
    if name not in _BACKENDS:
        raise RuntimeError(f"ctxshield: unknown backend '{name}' (cpu | metal | cuda)")
    return _BACKENDS[name]


def utf8proc_version() -> str:
    return _CLIENT.info()['utf8proc']


def unicode_version() -> str:
    return _CLIENT.info()['unicode']


def default_threads() -> int:
    return _CLIENT.info()['threads']


def backend_available(backend: str) -> Tuple[bool, str]:
    r = _CLIENT.call(OP_AVAILABLE, struct.pack('<i', _backend_code(backend)))
    why, _ = _rd_str(r, 1)
    return r[0] == 1, why


def featurize_batch(field_names: List[str], docs, delex: bool, threads: int = 0):
    """-> (indptr int64[n+1], indices int32[nnz], data float32[nnz], extras float32[n, e])"""
    payload = (_names(list(field_names)) + struct.pack('<Bi', 1 if delex else 0, threads)
               + _docs(docs, len(field_names)))
    r = _CLIENT.call(OP_FEATURIZE, payload)
    n, nnz, ne = struct.unpack_from('<QQI', r, 0)
    off = 20
    indptr = np.frombuffer(r, dtype='<i8', count=n + 1, offset=off).astype(np.int64)
    off += 8 * (n + 1)
    indices = np.frombuffer(r, dtype='<i4', count=nnz, offset=off).astype(np.int32)
    off += 4 * nnz
    data = np.frombuffer(r, dtype='<f4', count=nnz, offset=off).astype(np.float32)
    off += 4 * nnz
    extras = np.frombuffer(r, dtype='<f4', count=n * ne, offset=off).astype(np.float32).reshape(n, ne)
    off += 4 * n * ne
    if off != len(r):
        raise RuntimeError("ctxshield: scorer FEATURIZE response has the wrong size")
    return indptr, indices, data, extras


class Scorer:
    """C ABI handle held by the scorer process (weights resident there)."""

    def __init__(self, backend: str, field_names: List[str], delex: bool, W, We, b):
        self._backend = backend
        self._names = list(field_names)
        code = _backend_code(backend)
        W, We, b = _f32(W), _f32(We), _f32(b)
        self._hpad = _check_weights(self._names, W, We, b)
        payload = (struct.pack('<i', code) + _names(self._names) + struct.pack('<Bi', 1 if delex else 0, self._hpad)
                   + W.tobytes() + We.tobytes() + b.tobytes())
        (self._h,) = struct.unpack('<I', _CLIENT.call(OP_CREATE, payload))
        self._last = {}

    def score(self, docs, threads: int = 0) -> np.ndarray:
        payload = struct.pack('<Ii', self._h, threads) + _docs(docs, len(self._names))
        r = _CLIENT.call(OP_SCORE, payload)
        n = len(docs)
        k = n * self._hpad * 4
        if len(r) != k + 4 + 4 * 8 + 3 * 8:
            raise RuntimeError("ctxshield: scorer SCORE response has the wrong size")
        logits = np.frombuffer(r, dtype='<f4', count=n * self._hpad).astype(np.float32).reshape(n, self._hpad)
        be, n_docs, n_over, n_tok, n_arena, t_tok, t_dev, t_total = struct.unpack_from('<iQQQQddd', r, k)
        if be != _BACKENDS[self._backend]:
            raise RuntimeError(f"ctxshield: backend mismatch: requested {self._backend}, ran {_BACKEND_NAMES[be]}")
        self._last = {'backend': _BACKEND_NAMES[be], 'n_docs': n_docs, 'n_overflow_docs': n_over,
                      'n_tokens': n_tok, 'n_arena_bytes': n_arena, 't_tokenize_s': t_tok,
                      't_device_s': t_dev, 't_total_s': t_total}
        return logits

    def close(self) -> None:
        if self._h and _CLIENT.alive():
            _CLIENT.call(OP_DESTROY, struct.pack('<I', self._h))
        self._h = 0

    def __del__(self):
        if '_h' in self.__dict__:
            self.close()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def last_stats(self) -> Dict:
        return self._last


# score_batch() takes the weights on every call; resending 1–16 MB per call would dominate. Scorers are kept
# per weight set — matched by CONTENT (array_equal against a private copy), so a caller that mutates its
# arrays in place still gets the right answer.
_CACHE: List[Tuple[Tuple, np.ndarray, np.ndarray, np.ndarray, Scorer]] = []


def _cached_scorer(names: List[str], delex: bool, W, We, b) -> Scorer:
    W, We, b = _f32(W), _f32(We), _f32(b)
    key = (tuple(names), bool(delex), W.shape, We.shape, b.shape)
    for i, (k, cW, cWe, cb, sc) in enumerate(_CACHE):
        if k == key and np.array_equal(cb, b) and np.array_equal(cWe, We) and np.array_equal(cW, W):
            _CACHE.append(_CACHE.pop(i))
            return sc
    sc = Scorer('cpu', names, delex, W, We, b)
    _CACHE.append((key, W.copy(), We.copy(), b.copy(), sc))
    if len(_CACHE) > _MAX_CACHED:
        _CACHE.pop(0)[4].close()
    return sc


def score_batch(field_names: List[str], docs, delex: bool, W, We, b, threads: int = 0) -> np.ndarray:
    """CPU -> logits float32[n, hpad]"""
    names = list(field_names)
    _check_weights(names, _f32(W), _f32(We), _f32(b))
    return _cached_scorer(names, delex, W, We, b).score(docs, threads)


def score_batch_metal(field_names: List[str], docs, delex: bool, W, We, b, threads: int = 0) -> np.ndarray:
    """Metal is never built into the static scorer: raises, exactly like the native module without Metal."""
    return Scorer('metal', field_names, delex, W, We, b).score(docs, threads)
