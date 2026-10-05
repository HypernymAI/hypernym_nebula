"""
Pure-Python AUDIT REFERENCE of the feature spec (build doc §5.2c). Slow on purpose: every step is a
plain loop that can be read against the spec. Training and runtime use the native featurizer; this
file exists only so parity (T1) can prove the native code implements the written spec.

    featurize(fields: {name: text}, delex: bool) -> (indices, values, extras)

NO .get(). No ClaudeStorm imports.
"""
import math
import re
import unicodedata
from typing import Dict, List, Tuple

from ctxshield import spec as S

_SURR = re.compile('[\ud800-\udfff]')
_WS = frozenset(chr(c) for c in S.WHITESPACE)
_HEX = frozenset('0123456789abcdefABCDEF')
_EXT = frozenset(S.EXTENSIONS)


def clean(s: str) -> str:
    return _SURR.sub('�', s)


def nfkc(s: str) -> str:
    return unicodedata.normalize('NFKC', s)


def nfkc_casefold(s: str) -> str:
    return unicodedata.normalize('NFKC', unicodedata.normalize('NFKD', s).casefold())


def cat(ch: str) -> str:
    return unicodedata.category(ch)


def is_word(ch: str) -> bool:
    c = cat(ch)
    return c[0] == 'L' or c[0] == 'N' or c == 'Pc' or ch == "'"


def ascii_lower(s: str) -> str:
    return ''.join(chr(ord(c) + 32) if 'A' <= c <= 'Z' else c for c in s)


def tokenize(s: str) -> List[str]:
    toks, cur = [], []
    for ch in s:
        if is_word(ch):
            cur.append(ch)
        elif cur:
            toks.append(''.join(cur))
            cur = []
    if cur:
        toks.append(''.join(cur))
    return toks


# ── delex (V2) ────────────────────────────────────────────────────────────────

def classify(core: str):
    n = len(core)
    low = ascii_lower(core)
    for sch in S.URL_SCHEMES:
        if low.startswith(sch):
            return 'url'
    if n == 36 and all(core[i] == '-' for i in (8, 13, 18, 23)) and \
            all(core[i] in _HEX for i in range(36) if i not in (8, 13, 18, 23)):
        return 'uuid'
    if n >= 7 and all(c in _HEX for c in core) and any(c.isdigit() for c in core) and \
            any(c in 'abcdefABCDEF' for c in core):
        return 'hash'
    if (any(core.startswith(p) for p in S.PATH_PREFIXES) and n > 1) or core.count('/') >= 2:
        return 'path'
    dot = core.rfind('.')
    if dot >= 1 and ascii_lower(core[dot + 1:]) in _EXT:
        return 'path'
    if '_' in core:
        return 'ident'
    for i in range(n - 1):
        if cat(core[i]) == 'Ll' and cat(core[i + 1]) == 'Lu':
            return 'ident'
    for i in range(1, n - 1):
        if core[i] == '.' and cat(core[i - 1])[0] == 'L' and cat(core[i + 1])[0] == 'L':
            return 'ident'
    if n and '0' <= core[0] <= '9':
        j = 1
        while j < n and core[j] in '0123456789.,':
            j += 1
        if j < n and core[j] == '%':
            j += 1
        rest = ascii_lower(core[j:])
        if rest == '' or rest in S.NUM_UNITS:
            return 'num'
    return None


def _chunks(t: str, out: List):
    """Whitespace-split classification of a text run; appends ('t', str) / ('p', name)."""
    i, n = 0, len(t)
    while i < n:
        j = i
        if t[i] in _WS:
            while j < n and t[j] in _WS:
                j += 1
            out.append(('t', t[i:j]))
            i = j
            continue
        while j < n and t[j] not in _WS:
            j += 1
        ch = t[i:j]
        lead = 0
        while lead < len(ch) and ch[lead] in S.LEAD_STRIP:
            lead += 1
        trail = 0
        while trail < len(ch) - lead and ch[len(ch) - 1 - trail] in S.TRAIL_STRIP:
            trail += 1
        core = ch[lead:len(ch) - trail]
        k = classify(core) if core else None
        if k is None:
            out.append(('t', ch))
        else:
            out.append(('t', ch[:lead]))
            out.append(('p', k))
            out.append(('t', ch[len(ch) - trail:]))
        i = j


def delex(s: str) -> List:
    """s = NFKC text. Returns merged segments [('t', str) | ('p', name)]."""
    raw, buf, i, n = [], [], 0, len(s)

    def flush():
        if buf:
            _chunks(''.join(buf), raw)
            buf.clear()

    while i < n:
        if s.startswith('```', i):
            flush()
            j = s.find('```', i + 3)
            raw.append(('p', 'code'))
            i = n if j < 0 else j + 3
            continue
        if s[i] == '`':
            j = s.find('`', i + 1)
            if j >= 0 and j - i <= S.INLINE_MAX and '\n' not in s[i + 1:j]:
                flush()
                raw.append(('p', 'ident'))
                i = j + 1
                continue
        buf.append(s[i])
        i += 1
    flush()
    merged = []
    for kind, v in raw:
        if kind == 't':
            if v == '':
                continue
            if merged and merged[-1][0] == 't':
                merged[-1] = ('t', merged[-1][1] + v)
                continue
        merged.append((kind, v))
    return merged


def tokens_for(text: str, delex_on: bool) -> List[Tuple[str, bool]]:
    if not delex_on:
        return [(t, False) for t in tokenize(nfkc_casefold(text))]
    out = []
    for kind, v in delex(nfkc(text)):
        if kind == 'p':
            out.append((S.PLACEHOLDERS[v], True))
        else:
            out.extend((t, False) for t in tokenize(nfkc_casefold(v)))
    return out


# ── hashing ───────────────────────────────────────────────────────────────────

def fnv(data: bytes) -> int:
    h = S.FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * S.FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def _add(counts: Dict[int, int], field: str, kind: str, payload: str) -> None:
    h = fnv(field.encode('utf-8') + bytes([S.SEP]) + kind.encode('ascii') + bytes([S.SEP])
            + payload.encode('utf-8'))
    idx = h & (S.DIM - 1)
    counts[idx] = (counts[idx] if idx in counts else 0) + (-1 if h >> 63 else 1)


def featurize(fields: Dict[str, str], delex_on: bool) -> Tuple[List[int], List[float], List[float]]:
    counts: Dict[int, int] = {}
    names = sorted(fields)
    for name in names:
        toks = tokens_for(clean(fields[name]), delex_on)
        for k, (t, ph) in enumerate(toks):
            _add(counts, name, 'w1', t)
            if k + 1 < len(toks):
                _add(counts, name, 'w2', t + ' ' + toks[k + 1][0])
            if ph:
                continue
            padded = ' ' + t + ' '
            for n in S.CHAR_NGRAMS:
                for s in range(len(padded) - n + 1):
                    _add(counts, name, 'c', padded[s:s + n])
    items = sorted((i, c) for i, c in counts.items() if c != 0)
    vals = [(1.0 if c > 0 else -1.0) * (1.0 + math.log(abs(c))) for _, c in items]
    norm = math.sqrt(sum(v * v for v in vals))
    vals = [v / norm for v in vals] if norm > 0 else vals
    return [i for i, _ in items], vals, extras(fields)


def extras(fields: Dict[str, str]) -> List[float]:
    names = sorted(fields)
    out = [math.log1p(len(clean(fields[n]))) for n in names]
    q = 'reply' if 'reply' in fields else ('text' if 'text' in fields else None)
    out.append(1.0 if q is not None and '?' in clean(fields[q])[-S.QTAIL_CP:] else 0.0)
    c = 'pilot' if 'pilot' in fields else ('ask' if 'ask' in fields else None)
    if c is None:
        out.append(0.0)
    else:
        t = clean(fields[c])
        letters = sum(1 for ch in t if cat(ch)[0] == 'L')
        upper = sum(1 for ch in t if cat(ch) == 'Lu')
        out.append(upper / letters if letters else 0.0)
    return out
