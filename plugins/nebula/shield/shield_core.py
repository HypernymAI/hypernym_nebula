"""
Claude Code shield — shared core for the hooks and the MCP server (build doc Part XVIII).

Local only: ctxshield native Tier A heads (and later Tier B), no network. Exchange/field construction
is the benchmark's (`shield_reader.load_session_jsonl` = extract_exchanges' reader + image turns kept, S25) + the `units.py` caps (via ctxshield.shape),
so the text a head sees at runtime is the text it was trained and tested on.

Config: settings.resolve_config() — shipped defaults.json ⊕ ~/.nebula/settings.json (or an explicit --config):
    enabled           false → hooks exit 0 immediately and log 'disabled' (the kill switch)
    model_dir         Tier A models root (<model_dir>/<spec>/...)
    user_prompt       {head: {"spec": .., "threshold": ..}}   scored on the pilot's new turn
    stop              {head: {"spec": .., "threshold": ..}}   scored on this turn's assistant text blocks
    max_rewrites      Stop blocks per pilot message before a visible pass-through
    log_db            SQLite path (every decision, text, score, timing)

NO .get(). Imports at top. Paths passed in. No false guards: errors raise.
"""
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]          # the standalone project root (extracted from ClaudeStorm)
sys.path.insert(0, str(REPO / 'ctxshield'))

from ctxshield import shape                                                   # noqa: E402
from ctxshield.dfc_adapter import TierAModel                                  # noqa: E402
from combined_model import CombinedModel                                      # noqa: E402
from pilot_text import own_words                                              # noqa: E402  (S56)
from shield_reader import FEEDBACK_PREFIX, SHIELD_SKIP_PREFIX, codex_msgs, is_codex_rollout, load_session_jsonl   # noqa: E402  (S25)
from vendor_codebook import CLASSES                                 # noqa: E402

HERE = Path(__file__).resolve().parent
DEFINITION = {c['id']: (c['name'], c['definition']) for c in CLASSES}

MIGRATIONS = [
    [
        """CREATE TABLE IF NOT EXISTS decisions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            event        TEXT NOT NULL,          -- user_prompt / stop / mcp
            session_id   TEXT NOT NULL,
            pilot_sha    TEXT NOT NULL,          -- sha256 of the pilot message this relates to
            attempt      INTEGER NOT NULL,
            text_sha     TEXT NOT NULL,
            text         TEXT NOT NULL,
            scores_json  TEXT,
            fired_json   TEXT,
            decision     TEXT NOT NULL,          -- allow / block / cap_allow / context / disabled
            config_sha   TEXT NOT NULL,
            ms           REAL NOT NULL,
            created_at   REAL NOT NULL
        )""",
        "CREATE INDEX IF NOT EXISTS idx_dec_turn ON decisions(session_id, pilot_sha, event)",
    ],
]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


class DecisionLog:
    def __init__(self, db_path: Path, retention: str):
        if retention not in ('none', 'excerpt', 'full'):
            raise ValueError(f"text_retention must be none/excerpt/full, got {retention!r}")
        self.retention = retention                   # privacy: the decision log keeps text only as the user allows
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(self.db_path), timeout=120.0)
        try:
            c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL, applied_at REAL NOT NULL)")
            row = c.execute("SELECT MAX(version) FROM schema_version").fetchone()
            cur = row[0] if row[0] is not None else 0
            for i in range(cur, len(MIGRATIONS)):
                for stmt in MIGRATIONS[i]:
                    c.execute(stmt)
                c.execute("INSERT INTO schema_version VALUES (?, ?)", (i + 1, time.time()))
            c.commit()
        finally:
            c.close()

    def blocks_so_far(self, session_id: str, pilot_sha: str) -> int:
        c = sqlite3.connect(str(self.db_path), timeout=120.0)
        try:
            return c.execute("SELECT COUNT(*) FROM decisions WHERE session_id=? AND pilot_sha=? AND "
                             "event='stop' AND decision='block'", (session_id, pilot_sha)).fetchone()[0]
        finally:
            c.close()

    def _kept(self, t: str) -> str:
        return t if self.retention == 'full' else (t[:300] if self.retention == 'excerpt' else '')

    def put(self, row: Dict) -> None:
        c = sqlite3.connect(str(self.db_path), timeout=120.0)
        try:
            c.execute("INSERT INTO decisions (event, session_id, pilot_sha, attempt, text_sha, text, scores_json, "
                      "fired_json, decision, config_sha, ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (row['event'], row['session_id'], row['pilot_sha'], row['attempt'], sha(row['text']),
                       self._kept(row['text']), row['scores_json'], row['fired_json'], row['decision'], row['config_sha'],
                       row['ms'], time.time()))
            c.commit()
        finally:
            c.close()


class Shield:
    def __init__(self, config_path: Path):
        raw = Path(config_path).read_text()
        self.cfg = json.loads(raw)
        self.config_sha = sha(raw.strip())
        self.enabled = self.cfg['enabled'] is True
        self.head_models: Dict[str, object] = {}      # head -> TierAModel or CombinedModel
        self.log = None                               # disabled → nothing is opened or created (privacy)
        if not self.enabled:
            return
        self.log = DecisionLog(Path(self.cfg['log_db']), self.cfg['text_retention'])
        groups: Dict[tuple, List[str]] = {}
        for event in ('user_prompt', 'stop'):
            for head, hc in self.cfg[event].items():
                k = (hc['model'], hc['spec'])
                if k not in groups:
                    groups[k] = []
                groups[k].append(head)
        for (kind, spec), heads in groups.items():
            if kind == 'combined':
                m = CombinedModel(Path(self.cfg['combined_dir']), sorted(set(heads)))
            elif kind == 'tierA':
                m = TierAModel(Path(self.cfg['model_dir']), spec, sorted(set(heads)))
            else:
                raise RuntimeError(f"unknown model kind {kind!r}")
            for h in heads:
                self.head_models[h] = m

    # ── field construction (benchmark-identical) ─────────────────────────────
    # Every pilot text goes through own_words (S56): a pasted block is someone else's text, never the pilot's ask.
    # A turn with no pasted block is unchanged byte-for-byte, so the benchmark shapes still hold for it.

    @staticmethod
    def pilot_turn_fields(prompt: str, exchanges: List[Dict]) -> Dict[str, str]:
        """S1 fields for a NEW pilot turn, given the transcript's completed exchanges (units.py S1)."""
        prev = shape.tail(exchanges[-1]['assistant'], shape.CAP_PREV_TAIL) if exchanges else ''
        prior = [shape.head(own_words(e['user']), shape.CAP_EARLIER) for e in exchanges[-2:]]
        return {'pilot': shape.head(own_words(prompt), shape.CAP_PILOT), 'prev': prev,
                'earlier': shape.join_list(prior)}

    @staticmethod
    def text_block_fields(pilot: str, blocks: List[str]) -> List[Dict[str, str]]:
        """S2 fields, one per assistant text block of this turn (units.py S2)."""
        ask = shape.head(own_words(pilot), shape.CAP_ASK)
        return [{'ask': ask, 'text': shape.head(b, shape.CAP_BLOCK)} for b in blocks]

    @staticmethod
    def reply_fields(pilot: str, blocks: List[str]) -> Dict[str, str]:
        """S6 fields: the pilot's turn + the WHOLE reply (units.py S6: text parts joined by newline)."""
        return {'pilot': shape.head(own_words(pilot), shape.CAP_PILOT),
                'reply': shape.headtail('\n'.join(blocks), shape.CAP_REPLY_HEAD, shape.CAP_REPLY_TAIL)}

    @classmethod
    def stop_docs(cls, pilot: str, blocks: List[str]) -> Dict[str, List[Dict[str, str]]]:
        return {'S2': cls.text_block_fields(pilot, blocks), 'S6': [cls.reply_fields(pilot, blocks)]}

    @staticmethod
    def is_real_pilot(text: str) -> bool:
        t = text.strip()
        return bool(t) and not t.startswith(SHIELD_SKIP_PREFIX) and not t.startswith(FEEDBACK_PREFIX)

    # ── scoring ──────────────────────────────────────────────────────────────

    def score_heads(self, event: str, docs_by_section: Dict[str, List[Dict[str, str]]]
                    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Each head scores the docs of ITS section (S1 / S2 / S6); max per head; heads at/over threshold."""
        scores, fired = {}, {}
        for head, hc in self.cfg[event].items():
            m = self.head_models[head]
            s = m.score(head, docs_by_section[m.heads[head]['section']])
            scores[head] = float(s.max()) if len(s) else 0.0
            if len(s) and scores[head] >= hc['threshold']:
                fired[head] = scores[head]
        return scores, fired

    def certified(self, event: str, fired: Dict[str, float]) -> Dict[str, float]:
        """Only certified heads may enforce; working-not-certified heads are recorded."""
        return {h: s for h, s in fired.items() if self.cfg[event][h]['certified'] is True}

    def reframe_text(self, fired: Dict[str, float], event: str) -> str:
        lines = []
        for head in sorted(fired):
            cid = self.head_models[head].heads[head]['class_id']
            name, definition = DEFINITION[cid]
            lines.append(f"- {name} ({head} {fired[head]:.2f} ≥ {self.cfg[event][head]['threshold']:.2f}): {definition}")
        return '\n'.join(lines)


def turn_of(msgs: List[Dict]) -> Tuple[str, List[str]]:
    """(last real pilot message, its assistant text blocks so far) from message records (shield_reader.read_msgs). The
    Stop hook reads it from its window of the transcript (2026-10-21), which always opens at or before the last real
    pilot message, so it is what transcript_turn reads from the whole file."""
    pilot, blocks = '', []
    for r in msgs:
        if r['type'] == 'user':
            if r['results'] > 0:
                continue                          # a tool result, not the pilot
            if not Shield.is_real_pilot(r['text']):
                continue
            pilot, blocks = r['text'], []
        else:
            for kind, text in r['blocks']:
                if kind == 'text':
                    if text.strip():
                        blocks.append(text)
    return pilot, blocks


def transcript_turn(path: str) -> Tuple[str, List[str], List[Dict]]:
    """(last real pilot message, this turn's assistant text blocks, completed exchanges) — the
    benchmark's exchange reader over the live transcript. A Codex rollout is read through codex_msgs (2026-10-19)."""
    ex = load_session_jsonl(Path(path))
    pilot, blocks = '', []
    if is_codex_rollout(Path(path)):
        pilot, blocks = turn_of(codex_msgs(Path(path)))
        return pilot, blocks, ex
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if 'type' not in d or 'message' not in d or not isinstance(d['message'], dict):
                continue
            m = d['message']
            content = m['content'] if 'content' in m else ''
            if d['type'] == 'user':
                if isinstance(content, list) and any(isinstance(b, dict) and 'type' in b and
                                                     b['type'] == 'tool_result' for b in content):
                    continue
                t = content if isinstance(content, str) else '\n'.join(
                    b['text'] for b in content if isinstance(b, dict) and 'type' in b and b['type'] == 'text'
                    and 'text' in b) if isinstance(content, list) else ''
                if not Shield.is_real_pilot(t):
                    continue
                pilot, blocks = t, []
            elif d['type'] == 'assistant' and isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and 'type' in b and b['type'] == 'text' and 'text' in b and b['text'].strip():
                        blocks.append(b['text'])
    return pilot, blocks, ex
