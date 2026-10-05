"""
Runtime for the S22 COMBINED heads (build doc S22): native Tier A V2 score + the static regex (strata.fires) as an input,
then Platt. Weights: tools/shield_bridge/models/combined_v1/<head>.f32 = [2^18 hashed][n_extras][regex][bias].

    m = CombinedModel(model_dir, heads)
    p = m.score(head, docs)       # docs = head fields (ctxshield names); the regex input is computed from them
S2 / S6 heads only: their payload is exactly invertible from the fields (S1's earlier_asks list is not).
NO .get(). Integrity defects raise at construction.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ctxshield import features as F
from ctxshield import spec as S
from vendor_strata import fires
from vendor_heads import HEADS


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class CombinedModel:
    def __init__(self, model_dir: Path, heads: List[str]):
        d = Path(model_dir)
        want = (d / 'MANIFEST.sha256').read_text().split()[0]
        if _sha(d / 'MANIFEST.json') != want:
            raise RuntimeError(f"FATAL: {d}/MANIFEST.json fails MANIFEST.sha256")
        man = json.loads((d / 'MANIFEST.json').read_text())
        for fname, sha in man['files'].items():
            if _sha(d / fname) != sha:
                raise RuntimeError(f"FATAL: {d / fname} fails its manifest sha")
        self.version = want[:12]
        self.heads: Dict[str, Dict] = {}
        self._w: Dict[str, Dict] = {}
        for head in heads:
            info = json.loads((d / f'{head}.json').read_text())
            if info['section'] not in ('S2', 'S6'):
                raise RuntimeError(f"FATAL: combined runtime supports S2/S6 heads only, {head} is {info['section']}")
            if info['feature_spec_sha'] != S.spec_sha(info['spec']):
                raise RuntimeError(f"FATAL: {head} trained on another feature spec")
            w = np.fromfile(d / f'{head}.f32', dtype='<f4')
            ne = info['n_extras']
            if w.shape[0] != S.DIM + ne + 2:
                raise RuntimeError(f"FATAL: {head}.f32 has {w.shape[0]} floats, expected {S.DIM + ne + 2}")
            W = np.zeros((S.DIM, 4), dtype=np.float32)
            W[:, 0] = w[:S.DIM]
            We = np.zeros((ne, 4), dtype=np.float32)
            We[:, 0] = w[S.DIM:S.DIM + ne]
            b = np.zeros(4, dtype=np.float32)
            b[0] = w[S.DIM + ne + 1]
            self._w[head] = {'W': W, 'We': We, 'b': b, 'w_regex': float(w[S.DIM + ne]), 'spec': info['spec'],
                             'fields': info['fields'], 'a': info['platt_a'], 'c': info['platt_b']}
            self.heads[head] = {'section': info['section'], 'class_id': info['class_id'], 'r90': info['r90']}

    @staticmethod
    def regex_input(head: str, fields: Dict[str, str]) -> float:
        """strata.fires on the unit payload rebuilt from the head's fields (exact inverse for S2/S6)."""
        h = HEADS[head]
        payload = {key: fields[name] for name, key in h['fields'].items()}
        return 1.0 if fires(h['class_id'], h['section'], payload) else 0.0

    def logits(self, head: str, docs: List[Dict[str, str]]) -> np.ndarray:
        h = self._w[head]
        z = F.score_logits(h['fields'], docs, h['spec'], h['W'], h['We'], h['b'])[:, 0].astype(np.float64)
        return z + h['w_regex'] * np.array([self.regex_input(head, d) for d in docs])

    def score(self, head: str, docs: List[Dict[str, str]]) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float64)
        h = self._w[head]
        return 1.0 / (1.0 + np.exp(-(h['a'] * self.logits(head, docs) + h['c'])))
