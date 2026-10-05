"""
FEATURE SPEC constants (build doc §5.2c). Their sha256 is the spec sha recorded in every manifest.
The native featurizer (native/src/featurize.cpp) and the audit reference (features_ref.py) both
implement exactly this; changing any value here is a new spec version.

NO .get(). No ClaudeStorm imports.
"""
import hashlib
import json

DIM_BITS = 18
DIM = 1 << DIM_BITS
SEP = 0x1F
FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
KINDS = ('w1', 'w2', 'c')          # r2: one KIND for all char n-grams (build doc 5.2d)
CHAR_NGRAMS = (3, 4, 5)
UNICODE_VERSION = '15.1.0'
UTF8PROC_VERSION = '2.9.0'

# §5.2c whitespace = Python str.isspace() set
WHITESPACE = (tuple(range(0x09, 0x0E)) + tuple(range(0x1C, 0x21)) + (0x85, 0xA0, 0x1680)
              + tuple(range(0x2000, 0x200B)) + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000))
LEAD_STRIP = '([{<"\''
TRAIL_STRIP = ')]}>"\',.;:!?'
URL_SCHEMES = ('http://', 'https://', 'file://')
PATH_PREFIXES = ('/', '~/', './', '../')
EXTENSIONS = ('py', 'js', 'ts', 'tsx', 'rs', 'go', 'c', 'cc', 'cpp', 'h', 'hpp', 'm', 'mm',
              'metal', 'cu', 'java', 'kt', 'swift', 'rb', 'php', 'sh', 'md', 'json', 'yaml',
              'yml', 'toml', 'sql', 'db', 'txt', 'html', 'css')
NUM_UNITS = ('ms', 'kb', 'mb', 'gb', 's', 'x')
INLINE_MAX = 200
PLACEHOLDERS = {'code': '⟨code⟩', 'ident': '⟨ident⟩', 'url': '⟨url⟩', 'uuid': '⟨uuid⟩',
                'hash': '⟨hash⟩', 'path': '⟨path⟩', 'num': '⟨num⟩'}
QTAIL_CP = 200
SPECS = {'V1': {'delex': False}, 'V2': {'delex': True}}


def spec_sha(version: str) -> str:
    if version not in SPECS:
        raise KeyError(f"unknown spec {version}")
    blob = json.dumps({
        'version': version, 'delex': SPECS[version]['delex'], 'dim_bits': DIM_BITS, 'sep': SEP,
        'fnv': [FNV_OFFSET, FNV_PRIME], 'kinds': KINDS, 'char_ngrams': CHAR_NGRAMS,
        'unicode': UNICODE_VERSION, 'utf8proc': UTF8PROC_VERSION, 'whitespace': WHITESPACE,
        'lead': LEAD_STRIP, 'trail': TRAIL_STRIP, 'url': URL_SCHEMES, 'path': PATH_PREFIXES,
        'ext': EXTENSIONS, 'units': NUM_UNITS, 'inline_max': INLINE_MAX,
        'placeholders': PLACEHOLDERS, 'qtail': QTAIL_CP,
        'rules': 'build doc 5.2c 2026-09-26 + 5.2d r2'}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()
