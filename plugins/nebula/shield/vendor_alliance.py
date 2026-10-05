"""
VENDORED from ClaudeStorm core/alliance_analyzer.py (sha256 0ec7a5672d6dfd1570eadcc25c1fa226bcafe35e06dbe35bc2a2148bcc62bc03): STOP, WORD, content, jaccard — sliced byte-exact
with ast by tools/extract_from_claudestorm.py. Do not edit here; re-extract.
"""
import re


STOP = frozenset("""
a an the and or but if then than that this these those is are was were be been being am
do does did doing done have has had having i you he she it we they me him her us them my
your his its our their mine yours what which who whom whose when where why how all any
both each few more most other some such no nor not only own same so too very can will
just should now to of in on at by for with about against between into through during
before after above below from up down out off over under again further once here there
""".split())


WORD = re.compile(r"[a-z0-9_]+")


def content(text: str) -> frozenset:
    """Content words, lowercased, function words stripped."""
    return frozenset(w for w in WORD.findall(text.lower())
                     if w not in STOP and len(w) > 2)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
