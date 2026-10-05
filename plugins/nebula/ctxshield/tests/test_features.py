#!/usr/bin/env python3
"""
Feature spec tests (build doc §5.2c, T1 part 2): delex golden cases, tokenizer cases, and that the
native featurizer equals the reference on every case.

    cd ctxshield && python3 tests/test_features.py

NO .get().
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctxshield import features as F                                      # noqa: E402
from ctxshield import features_ref as R                                  # noqa: E402

# (input text, expected V2 token sequence; placeholders written ⟨name⟩)
DELEX = [
    ('see ```py\nx = 1\n``` done', ['see', '⟨code⟩', 'done']),
    ('open ```never closed', ['open', '⟨code⟩']),
    ('run `ls -la` now', ['run', '⟨ident⟩', 'now']),
    ('a ` b', ['a', 'b']),
    ('tick `a\nb` x', ['tick', 'a', 'b', 'x']),
    ('go to https://x.io/a?b=1 please', ['go', 'to', '⟨url⟩', 'please']),
    ('HTTP://CAPS.io', ['⟨url⟩']),
    ('id 123e4567-e89b-12d3-a456-426614174000.', ['id', '⟨uuid⟩']),
    ('commit deadbee1 and abcdefa', ['commit', '⟨hash⟩', 'and', 'abcdefa']),
    ('path /usr/bin ok', ['path', '⟨path⟩', 'ok']),
    ('a/b/c and/or ~/x ./y ../z', ['⟨path⟩', 'and', 'or', '⟨path⟩', '⟨path⟩', '⟨path⟩']),
    ('edit core/dimensional_field_computer.py now', ['edit', '⟨path⟩', 'now']),
    ('file README.md, done', ['file', '⟨path⟩', 'done']),
    ('call compute_dimensional_field()', ['call', '⟨ident⟩']),
    ('the getValue method', ['the', '⟨ident⟩', 'method']),
    ('use os.path here', ['use', '⟨ident⟩', 'here']),
    ('took 3.5ms and 42% of 1,024 at 2x', ['took', '⟨num⟩', 'and', '⟨num⟩', 'of', '⟨num⟩', 'at',
                                          '⟨num⟩']),
    ('version 3.14 ok', ['version', '⟨num⟩', 'ok']),
    ('("quoted_name").', ['⟨ident⟩']),
    ("it's DFC's job", ["it's", "dfc's", 'job']),
    ('ＦＵＬＬ width', ['full', 'width']),
    ('Straße ﬁle', ['strasse', 'file']),
    ('e.g. this', ['⟨ident⟩', 'this']),
]


def test_delex_golden():
    for text, want in DELEX:
        got = [t for t, _ in R.tokens_for(text, True)]
        assert got == want, f"{text!r}: {got} != {want}"
    print(f"  ✓ delex golden: {len(DELEX)} cases (reference)")


def test_native_equals_reference_on_golden():
    docs = [{'text': t} for t, _ in DELEX]
    for spec, dlx in (('V1', False), ('V2', True)):
        indptr, indices, data, extras = F.featurize(['text'], docs, spec)
        for i, d in enumerate(docs):
            ri, rv, _ = R.featurize(d, dlx)
            assert indices[indptr[i]:indptr[i + 1]].tolist() == ri, (spec, d)
    print("  ✓ native == reference on every golden case, V1 and V2")


def test_v2_changes_code_text_v1_does_not():
    d = [{'text': 'edit core/dimensional_field_computer.py and call compute_dimensional_field'}]
    a = F.featurize(['text'], d, 'V1')[1].tolist()
    b = F.featurize(['text'], d, 'V2')[1].tolist()
    assert a != b and len(b) < len(a), (len(a), len(b))
    print(f"  ✓ V2 delexicalizes code vocabulary ({len(a)} → {len(b)} active features)")


def test_empty_and_surrogates():
    docs = [{'text': ''}, {'text': 'bad \ud800 surrogate'}]
    indptr, indices, data, extras = F.featurize(['text'], docs, 'V1')
    assert indptr[1] == 0 and extras[0][0] == 0.0
    assert indptr[2] > 0
    print("  ✓ empty field → no features; lone surrogate → U+FFFD, no crash")


# Step 8: the native normalizer splits a field before every ASCII character (ASCII runs lower-cased
# byte-wise, only [ASCII starter + non-ASCII run] segments through utf8proc). These cases sit on
# those boundaries: combining marks after ASCII, compositions, Hangul jamo, expansions, casefold
# expansions, tokens that start ASCII and continue non-ASCII, and the same inside delex segments.
BOUNDARY = [
    'e\u0301 cafe\u0301s CAFE\u0301', 'A\u030a\u0327x', 'x\u0301\u0327\u0301y', 'a\u0308\u0304b',
    '\u1100\u1161\u11a8 \u1100\u1161 k\u1161', 'ǅemal ǈ', 'ﬃ ﬁx aﬁb', 'Straße STRASSE', 'İstanbul',
    'ΣΑΣ σας', 'K\u212a Å\u212b', 'x\u00b2 \u2460\u2461', 'ｗｉｄｅ Ｗide', 'abc\u0301def ghi\u0327',
    'it\u2019s don\u2019t', 'n\u0303o\u0303 N\u0303', 'q\u0307\u0323 q\u0323\u0307',
    'mixed`e\u0301`tick and `A\u030a` path/e\u0301/x.py https://e\u0301.io 3\u0301ms',
    'a_\u0301b camel\u0301Case e\u0301.g\u0301 1,000\u0301', '\u0301leading mark', 'end mark\u0301',
    '\u00e9\u0301 \u1e9b\u0323', 'a\u200bb c\u00a0d e\u3000f', "'\u0301' ''",
]


def test_segmented_normalization_boundaries():
    docs = [{'text': t} for t in BOUNDARY] + [{'text': ' '.join(BOUNDARY)}]
    for spec, dlx in (('V1', False), ('V2', True)):
        indptr, indices, data, extras = F.featurize(['text'], docs, spec)
        for i, d in enumerate(docs):
            ri, rv, re_ = R.featurize(d, dlx)
            assert indices[indptr[i]:indptr[i + 1]].tolist() == ri, (spec, d)
            assert max([0.0] + [abs(float(a) - b) for a, b in zip(data[indptr[i]:indptr[i + 1]], rv)]) <= 1e-6
            assert max(abs(float(a) - b) for a, b in zip(extras[i], re_)) <= 1e-6
    print(f"  ✓ native == reference on {len(docs)} normalization-boundary cases, V1 and V2")


if __name__ == '__main__':
    test_delex_golden()
    test_native_equals_reference_on_golden()
    test_v2_changes_code_text_v1_does_not()
    test_empty_and_surrogates()
    test_segmented_normalization_boundaries()
    print("ALL PASS")
