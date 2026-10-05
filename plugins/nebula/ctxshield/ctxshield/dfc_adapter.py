"""
DFC-facing adapter: score one session's DFC `messages` with the shield heads and produce exactly the
build doc §4.2 output contract (with the revised C5 `shield_unattached`).

    model = TierAModel(model_dir, 'V2', ['H13_S1', 'H06_S2'])     # crashes on any integrity defect
    res = score_session(messages, model)                           # None when the shield is disabled
    top = finalize(res, signals)                                   # mutates signals, returns DF keys

Contract
  score_session(messages, model) -> None (model is None = disabled) or
    {'shield_model_version', 'shield_kernel', 'shield_heads',
     'per_message': {message_idx: {'shield': {head: p}, 'shield_flags': [head]}},
     'shield_counts': {head: n flagged}, 'shield_eligible': {head: n scored},
     'shield_rate': {head: flagged/eligible, or None when nothing was eligible}}
  finalize(res, signals) -> the top-level keys to merge into the DF dict:
    {'shield_model_version', 'shield_kernel', 'shield_counts', 'shield_eligible', 'shield_rate',
     'shield_unattached': [{message_idx, shield, shield_flags}] (scored messages with no signal —
                          by construction the DFC's noise-skipped messages, C5 revised),
     'shield_noise_skipped': len(shield_unattached)}
    and sets on EVERY signal: signal['shield'] = {head: p} ({} when no head applies to it) and
    signal['shield_flags'] = [head] ([] when none). Disabled (res is None): every one of those keys is
    None on the signals and at top level — explicitly dark, never 0.0.

Model interface (Tier A here; a Tier B encoder plugs in the same way):
  .heads   {head: {'section': 'S1'|'S2'|'S6', 'r90': float}}
  .version str (manifest sha[:12])      .kernel str (the backend that actually scores)
  .score(head, docs: List[{field: text}]) -> np.ndarray of calibrated probabilities, len(docs)

NO .get(). Imports at top. Crash loudly, no false guards. No ClaudeStorm imports.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ctxshield import features as F
from ctxshield import segment as SEG
from ctxshield import spec as S

KERNEL_TIER_A = 'ctxshield._native:cpu'
TOP_KEYS = ('shield_model_version', 'shield_kernel', 'shield_counts', 'shield_eligible',
            'shield_rate', 'shield_unattached', 'shield_noise_skipped')


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class TierAModel:
    """Tier A heads (hashed n-gram linear + Platt) loaded from a trained model directory
    (<model_dir>/<spec>/{MANIFEST.json, MANIFEST.sha256, <head>.json, <head>.f32}).

    Every integrity defect is a RuntimeError at construction: MANIFEST.sha256 vs MANIFEST.json,
    every manifest file vs its sha, weights vs weights_sha256, trained spec sha vs the current spec,
    weight length, field list vs the adapter's section fields, a head the manifest lacks.
    """

    def __init__(self, model_dir: Path, spec: str, heads: List[str]):
        if spec not in S.SPECS:
            raise RuntimeError(f"FATAL: unknown feature spec {spec!r}")
        if not heads:
            raise RuntimeError("FATAL: TierAModel needs an explicit, non-empty head list")
        d = Path(model_dir) / spec
        man_path = d / 'MANIFEST.json'
        want = (d / 'MANIFEST.sha256').read_text().split()[0]
        got = _sha256(man_path)
        if got != want:
            raise RuntimeError(f"FATAL: {man_path} sha {got[:12]} != MANIFEST.sha256 {want[:12]}")
        man = json.loads(man_path.read_text())
        for fname, sha in sorted(man['files'].items()):
            fs = _sha256(d / fname)
            if fs != sha:
                raise RuntimeError(f"FATAL: {d / fname} sha {fs[:12]} != manifest {sha[:12]}")
        self.spec = spec
        self.model_dir = d
        self.version = got[:12]
        self.kernel = KERNEL_TIER_A
        self.heads: Dict[str, Dict] = {}
        self._w: Dict[str, Dict] = {}
        for head in heads:
            if head not in man['heads']:
                raise RuntimeError(f"FATAL: head {head} not in {man_path} (has {man['heads']})")
            info = json.loads((d / f'{head}.json').read_text())
            if info['head'] != head or info['spec'] != spec:
                raise RuntimeError(f"FATAL: {head}.json declares head {info['head']} spec {info['spec']}")
            if info['feature_spec_sha'] != S.spec_sha(spec):
                raise RuntimeError(f"FATAL: {head} trained on spec {info['feature_spec_sha'][:12]}, "
                                   f"current {S.spec_sha(spec)[:12]}")
            section = info['section']
            if section not in SEG.SECTION_FIELDS:
                raise RuntimeError(f"FATAL: {head} is section {section}; the DFC adapter builds only "
                                   f"{SEG.SECTIONS}")
            if sorted(info['fields']) != sorted(SEG.SECTION_FIELDS[section]):
                raise RuntimeError(f"FATAL: {head} fields {info['fields']} != adapter "
                                   f"{sorted(SEG.SECTION_FIELDS[section])}")
            w = np.fromfile(d / f'{head}.f32', dtype='<f4')
            if hashlib.sha256(w.tobytes()).hexdigest() != info['weights_sha256']:
                raise RuntimeError(f"FATAL: {head}.f32 fails weights_sha256")
            ne = info['n_extras']
            if ne != len(info['fields']) + 2 or w.shape[0] != S.DIM + ne + 1:
                raise RuntimeError(f"FATAL: {head}.f32 has {w.shape[0]} floats, expected "
                                   f"{S.DIM + len(info['fields']) + 3}")
            W = np.zeros((S.DIM, 4), dtype=np.float32)
            W[:, 0] = w[:S.DIM]
            We = np.zeros((ne, 4), dtype=np.float32)
            We[:, 0] = w[S.DIM:S.DIM + ne]
            b = np.zeros(4, dtype=np.float32)
            b[0] = w[S.DIM + ne]
            self._w[head] = {'W': W, 'We': We, 'b': b, 'fields': list(info['fields']),
                             'a': float(info['platt_a']), 'c': float(info['platt_b'])}
            self.heads[head] = {'section': section, 'r90': float(info['r90']),
                                'class_id': info['class_id']}

    def score(self, head: str, docs: List[Dict[str, str]]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float64)
        h = self._w[head]
        z = F.score_logits(h['fields'], docs, self.spec, h['W'], h['We'], h['b'])[:, 0]
        return 1.0 / (1.0 + np.exp(-(h['a'] * z.astype(np.float64) + h['c'])))


def score_session(messages: List[Dict], model) -> Optional[Dict]:
    """Score every applicable unit of the session. model None = shield disabled -> None."""
    if model is None:
        return None
    units = SEG.build_units(messages)
    per_message: Dict[int, Dict] = {}
    counts, eligible, rate = {}, {}, {}
    for head in sorted(model.heads):
        hinfo = model.heads[head]
        section = hinfo['section']
        if section not in units:
            raise RuntimeError(f"FATAL: model head {head} has section {section}, not built by segment")
        us = units[section]
        p = model.score(head, [SEG.unit_fields(section, u['payload']) for u in us])
        if p.shape[0] != len(us):
            raise RuntimeError(f"FATAL: {head} returned {p.shape[0]} scores for {len(us)} units")
        n_flag = 0
        for u, pi in zip(us, p.tolist()):
            mi = u['message_idx']
            if mi not in per_message:
                per_message[mi] = {'shield': {}, 'shield_flags': []}
            slot = per_message[mi]
            if head in slot['shield']:
                raise RuntimeError(f"FATAL: {head} scored twice on message {mi}")
            slot['shield'][head] = float(pi)
            if pi >= hinfo['r90']:
                slot['shield_flags'].append(head)
                n_flag += 1
        counts[head] = n_flag
        eligible[head] = len(us)
        rate[head] = (n_flag / len(us)) if us else None
    for slot in per_message.values():
        slot['shield_flags'].sort()
    return {'shield_model_version': model.version, 'shield_kernel': model.kernel,
            'shield_heads': sorted(model.heads), 'per_message': per_message,
            'shield_counts': counts, 'shield_eligible': eligible, 'shield_rate': rate}


def finalize(res: Optional[Dict], signals: List[Dict]) -> Dict:
    """Attach per-message scores to the DFC signal dicts (by signal['message_idx']) and return the
    top-level DF keys. Scored messages with no signal go to shield_unattached (C5 revised)."""
    if res is None:
        for s in signals:
            s['shield'] = None
            s['shield_flags'] = None
        return {k: None for k in TOP_KEYS}
    per = res['per_message']
    seen = set()
    for s in signals:
        mi = s['message_idx']
        if mi in seen:
            raise RuntimeError(f"FATAL: two signals carry message_idx {mi}")
        seen.add(mi)
        if mi in per:
            s['shield'] = dict(per[mi]['shield'])
            s['shield_flags'] = list(per[mi]['shield_flags'])
        else:
            s['shield'] = {}
            s['shield_flags'] = []
    unattached = [{'message_idx': mi, 'shield': dict(per[mi]['shield']),
                   'shield_flags': list(per[mi]['shield_flags'])}
                  for mi in sorted(per) if mi not in seen]
    return {'shield_model_version': res['shield_model_version'],
            'shield_kernel': res['shield_kernel'],
            'shield_counts': dict(res['shield_counts']),
            'shield_eligible': dict(res['shield_eligible']),
            'shield_rate': dict(res['shield_rate']),
            'shield_unattached': unattached,
            'shield_noise_skipped': len(unattached)}


class ShieldSession:
    """Holds one loaded model (or None = disabled) for a worker process; load once, score many."""

    def __init__(self, model):
        self.model = model

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def score(self, messages: List[Dict]) -> Optional[Dict]:
        return score_session(messages, self.model)

    def apply(self, messages: List[Dict], signals: List[Dict]) -> Dict:
        return finalize(score_session(messages, self.model), signals)
