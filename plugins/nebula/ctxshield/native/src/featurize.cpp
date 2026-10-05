// FEATURE SPEC implementation (build doc §5.2c + r2 §5.2d). Mirrors ctxshield/features_ref.py step
// for step; parity (T1) proves the two agree. Streaming FNV-1a: no feature string is ever built.
//
// Step 8 speed pass (2026-09-27) — same features, same values:
//  * Tokens are spans into ONE reused per-thread arena laid out ' ' tok ' ' tok ' ', so w1, w2 and
//    the padded ' '+tok+' ' char n-gram windows are all contiguous byte ranges of the arena: no
//    Token objects, no per-token std::string, no padded string, no code-point vectors for ASCII.
//  * Normalization is SEGMENTED: every position just before an ASCII character is a normalization
//    boundary for NFKC and NFKC+casefold (an ASCII char has combining class 0, never composes as the
//    second element of a pair, and nothing reorders across it). So ASCII runs are lower-cased
//    byte-wise and only [ASCII starter] + [non-ASCII run] segments go through utf8proc
//    (utf8proc_decompose + utf8proc_normalize_utf32 into a reused buffer = utf8proc_map without the
//    malloc/re-encode/re-decode). Proven by parity on the frozen units + the Unicode stress set.
//  * Accumulator = open-addressing table sized from the exact insertion count (≥ 4×, power of two),
//    tagged with a generation so it is never cleared; replaces the 2^18 dense counts + stamps.
#include "shield.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#ifdef SHIELD_PROFILE
#include <chrono>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "utf8proc.h"

namespace ctxshield {

#ifdef SHIELD_PROFILE
double g_prof[8];
#define PROF(k) do { auto _t = std::chrono::steady_clock::now(); g_prof[k] += std::chrono::duration<double>(_t - _p).count(); _p = _t; } while (0)
#else
#define PROF(k) do {} while (0)
#endif

namespace {

constexpr uint8_t kSep = 0x1F;
constexpr int kInlineMax = 200;
constexpr int kQtail = 200;
constexpr int kGenBits = 32 - kDimBits;           // 14
constexpr uint32_t kGenMax = (1u << kGenBits) - 1;
constexpr size_t kMinCap = 1024;
constexpr size_t kMaxCap = size_t(1) << (kDimBits + 1);   // load ≤ 0.5 even with all 2^18 indices

const char* const kKinds[3] = {"w1", "w2", "c"};   // r2 (build doc 5.2d)
const char* const kExt[] = {"py", "js", "ts", "tsx", "rs", "go", "c", "cc", "cpp", "h", "hpp",
                            "m", "mm", "metal", "cu", "java", "kt", "swift", "rb", "php", "sh",
                            "md", "json", "yaml", "yml", "toml", "sql", "db", "txt", "html", "css"};
const char* const kUnits[] = {"ms", "kb", "mb", "gb", "s", "x"};
const char* const kSchemes[] = {"http://", "https://", "file://"};
const char* const kPathPrefix[] = {"/", "~/", "./", "../"};

// Placeholder token bytes (UTF-8 of U+27E8 name U+27E9), indexed by Ph.
enum Ph { PH_NONE = -1, PH_CODE, PH_IDENT, PH_URL, PH_UUID, PH_HASH, PH_PATH, PH_NUM };
const char* const kPhText[] = {"⟨code⟩", "⟨ident⟩", "⟨url⟩",
                               "⟨uuid⟩", "⟨hash⟩", "⟨path⟩",
                               "⟨num⟩"};

constexpr int kNFKC = UTF8PROC_COMPOSE | UTF8PROC_COMPAT;
constexpr int kNFKCCF = UTF8PROC_COMPOSE | UTF8PROC_COMPAT | UTF8PROC_CASEFOLD;

// ASCII word table: 0 = separator, else the case-folded byte. Word = L*, N*, Pc ('_') or U+0027.
struct AsciiTable {
    uint8_t v[128];
    AsciiTable() {
        for (int c = 0; c < 128; ++c) v[c] = 0;
        for (int c = 'a'; c <= 'z'; ++c) v[c] = static_cast<uint8_t>(c);
        for (int c = 'A'; c <= 'Z'; ++c) v[c] = static_cast<uint8_t>(c + 32);
        for (int c = '0'; c <= '9'; ++c) v[c] = static_cast<uint8_t>(c);
        v[static_cast<int>('_')] = '_';
        v[static_cast<int>('\'')] = '\'';
    }
};
const AsciiTable kAscii;

inline uint64_t fnv_bytes(uint64_t h, const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h;
}
inline uint64_t fnv_byte(uint64_t h, uint8_t b) {
    h ^= b;
    return h * kFnvPrime;
}

inline bool is_ws_slow(int32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}
// ASCII class bits: 1 = whitespace (§5.2c str.isspace set), 2 = delex lead-strip, 4 = trail-strip
struct DelexTable {
    uint8_t v[128];
    DelexTable() {
        for (int c = 0; c < 128; ++c) v[c] = is_ws_slow(c) ? 1 : 0;
        for (const char* p = "([{<\"'"; *p; ++p) v[static_cast<int>(*p)] |= 2;
        for (const char* p = ")]}>\"',.;:!?"; *p; ++p) v[static_cast<int>(*p)] |= 4;
    }
};
const DelexTable kDelex;
inline bool is_ws(int32_t c) { return c < 0x80 ? (kDelex.v[c] & 1) != 0 : is_ws_slow(c); }
inline int cat(int32_t c) { return static_cast<int>(utf8proc_category(c)); }
inline bool is_letter(int32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    int k = cat(c);
    return k >= 1 && k <= 5;
}
inline bool is_ll(int32_t c) { return c < 0x80 ? (c >= 'a' && c <= 'z') : cat(c) == UTF8PROC_CATEGORY_LL; }
inline bool is_lu(int32_t c) { return c < 0x80 ? (c >= 'A' && c <= 'Z') : cat(c) == UTF8PROC_CATEGORY_LU; }
inline bool is_word_cp(int32_t c) {
    if (c < 0x80) return kAscii.v[c] != 0;
    int k = cat(c);
    return (k >= 1 && k <= 5) || (k >= 9 && k <= 12);
}
inline bool is_hex(int32_t c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
inline int32_t alower(int32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

inline size_t count_cp(const uint8_t* p, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) k += (p[i] & 0xC0) != 0x80;
    return k;
}

// ── tokenizer: appends tokens of one field to sc.arena / sc.toks ─────────────────────────────

struct Tok {
    Scratch& sc;
    uint8_t field;
    bool open = false;
    uint32_t a = 0, ncp = 0;
    bool ascii = true;
    size_t first;   // index of the first token of this field in sc.toks

    Tok(Scratch& s, uint8_t f) : sc(s), field(f), first(s.toks.size()) {
        ensure(1);
        sc.arena[sc.n++] = ' ';
    }
    inline void ensure(size_t k) {
        if (sc.n + k > sc.arena.size()) sc.arena.resize(std::max(sc.arena.size() * 2, sc.n + k + 256));
    }
    inline void close() {
        if (!open) return;
        ensure(1);
        sc.toks.push_back(TokSpan{a, static_cast<uint32_t>(sc.n - a), ncp, field, 0,
                                  static_cast<uint8_t>(ascii ? 1 : 0), 0});
        sc.arena[sc.n++] = ' ';
        open = false;
    }
    // ASCII run; `T` = uint8_t (bytes) or int32_t (code points < 0x80). Run-structured: skip a
    // separator run, copy a word run (case-folded), so the only unpredictable branches are the
    // run ends.
    template <typename T>
    void ascii_run(const T* s, size_t len) {
        ensure(2 * len + 2);
        const uint8_t* tbl = kAscii.v;
        uint8_t* w = sc.arena.data();
        size_t n = sc.n, i = 0;
        if (open) {   // a token that started in the previous segment continues
            while (i < len) {
                const uint8_t c = tbl[static_cast<uint8_t>(s[i])];
                if (!c) break;
                w[n++] = c;
                ++i;
                ++ncp;
            }
            if (i == len) { sc.n = n; return; }
            sc.toks.push_back(TokSpan{a, static_cast<uint32_t>(n - a), ncp, field, 0,
                                      static_cast<uint8_t>(ascii ? 1 : 0), 0});
            w[n++] = ' ';
            open = false;
        }
        while (i < len) {
            while (i < len && !tbl[static_cast<uint8_t>(s[i])]) ++i;
            if (i == len) break;
            const size_t start = i, ta = n;
            do {
                w[n++] = tbl[static_cast<uint8_t>(s[i])];
                ++i;
            } while (i < len && tbl[static_cast<uint8_t>(s[i])]);
            if (i == len) {   // may continue in the next segment
                open = true;
                a = static_cast<uint32_t>(ta);
                ncp = static_cast<uint32_t>(i - start);
                ascii = true;
                break;
            }
            sc.toks.push_back(TokSpan{static_cast<uint32_t>(ta), static_cast<uint32_t>(n - ta),
                                      static_cast<uint32_t>(i - start), field, 0, 1, 0});
            w[n++] = ' ';
        }
        sc.n = n;
    }
    inline void cp(int32_t c) {
        if (c < 0x80) {
            uint8_t b = kAscii.v[c];
            if (b) {
                ensure(2);
                if (!open) { open = true; a = static_cast<uint32_t>(sc.n); ncp = 0; ascii = true; }
                sc.arena[sc.n++] = b;
                ++ncp;
            } else {
                close();
            }
            return;
        }
        if (is_word_cp(c)) {
            ensure(5);
            if (!open) { open = true; a = static_cast<uint32_t>(sc.n); ncp = 0; ascii = true; }
            utf8proc_ssize_t m = utf8proc_encode_char(c, sc.arena.data() + sc.n);
            sc.n += static_cast<size_t>(m);
            ++ncp;
            ascii = false;
        } else {
            close();
        }
    }
    void placeholder(Ph k) {
        close();
        size_t L = std::strlen(kPhText[k]);
        ensure(L + 1);
        uint32_t pa = static_cast<uint32_t>(sc.n);
        std::memcpy(sc.arena.data() + sc.n, kPhText[k], L);
        sc.n += L;
        sc.toks.push_back(TokSpan{pa, static_cast<uint32_t>(L), 0, field, 1, 0, 0});
        sc.arena[sc.n++] = ' ';
    }
    void finish() {
        ensure(1);
        close();
        for (size_t t = first; t + 1 < sc.toks.size(); ++t) sc.toks[t].next = 1;
    }
};

// utf8proc_map(s, opts) minus malloc/re-encode: decompose into sc.ubuf, compose in place.
// Returns the number of code points in sc.ubuf.
size_t umap_cps(Scratch& sc, const uint8_t* s, size_t n, int opts) {
    auto o = static_cast<utf8proc_option_t>(opts);
    for (;;) {
        utf8proc_ssize_t r = utf8proc_decompose(s, static_cast<utf8proc_ssize_t>(n), sc.ubuf.data(),
                                                static_cast<utf8proc_ssize_t>(sc.ubuf.size()), o);
        if (r < 0) throw std::runtime_error(std::string("ctxshield: utf8proc: ") + utf8proc_errmsg(r));
        if (static_cast<size_t>(r) > sc.ubuf.size()) {
            sc.ubuf.resize(static_cast<size_t>(r) * 2 + 64);
            continue;
        }
        utf8proc_ssize_t m = utf8proc_normalize_utf32(sc.ubuf.data(), r, o);
        if (m < 0) throw std::runtime_error(std::string("ctxshield: utf8proc: ") + utf8proc_errmsg(m));
        return static_cast<size_t>(m);
    }
}

// NFKC+casefold + tokenize a UTF-8 byte range (V1 field, or a V2 text segment after encoding).
void tok_v1_bytes(Tok& T, Scratch& sc, const uint8_t* s, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t r = i;
        while (r < n && s[r] < 0x80) ++r;
        if (r == n) { T.ascii_run(s + i, n - i); return; }
        size_t seg = r > i ? r - 1 : r;          // the ASCII char before the run is its starter
        if (seg > i) T.ascii_run(s + i, seg - i);
        size_t q = r;
        while (q < n && s[q] >= 0x80) ++q;
        size_t m = umap_cps(sc, s + seg, q - seg, kNFKCCF);
        for (size_t k = 0; k < m; ++k) T.cp(sc.ubuf[k]);
        i = q;
    }
}

// Same over a code-point range (V2 text segments): ASCII runs direct, non-ASCII segments encoded.
void tok_v1_cps(Tok& T, Scratch& sc, const int32_t* c, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t r = i;
        while (r < n && c[r] < 0x80) ++r;
        if (r == n) { T.ascii_run(c + i, n - i); return; }
        size_t seg = r > i ? r - 1 : r;
        if (seg > i) T.ascii_run(c + i, seg - i);
        size_t q = r;
        while (q < n && c[q] >= 0x80) ++q;
        sc.bbuf.resize((q - seg) * 4 + 4);
        size_t nb = 0;
        for (size_t k = seg; k < q; ++k)
            nb += static_cast<size_t>(utf8proc_encode_char(c[k], sc.bbuf.data() + nb));
        // umap_cps writes sc.ubuf; bbuf is a separate buffer
        size_t m = umap_cps(sc, sc.bbuf.data(), nb, kNFKCCF);
        for (size_t k = 0; k < m; ++k) T.cp(sc.ubuf[k]);
        i = q;
    }
}

// ── delex (V2) ────────────────────────────────────────────────────────────────

bool starts_ascii_ci(const int32_t* c, size_t a, size_t b, const char* lit) {
    size_t L = std::strlen(lit);
    if (b - a < L) return false;
    for (size_t k = 0; k < L; ++k) if (alower(c[a + k]) != static_cast<int32_t>(lit[k])) return false;
    return true;
}
bool starts_exact(const int32_t* c, size_t a, size_t b, const char* lit) {
    size_t L = std::strlen(lit);
    if (b - a < L) return false;
    for (size_t k = 0; k < L; ++k) if (c[a + k] != static_cast<int32_t>(lit[k])) return false;
    return true;
}
bool equals_ascii_ci(const int32_t* c, size_t a, size_t b, const char* lit) {
    return b - a == std::strlen(lit) && starts_ascii_ci(c, a, b, lit);
}

// Rule order exactly as §5.2c (url, uuid, hash, path, ident, num; first match wins). One pre-scan
// collects the facts each rule needs, so a rule that cannot fire is skipped without its own scan.
Ph classify(const int32_t* c, size_t a, size_t b) {
    const size_t n = b - a;
    size_t slashes = 0;
    bool dot = false, us = false, all_hex = true, dig = false, hexlet = false, lower = false;
    for (size_t i = a; i < b; ++i) {
        const int32_t x = c[i];
        const bool d = x >= '0' && x <= '9';
        const bool hl = (x >= 'a' && x <= 'f') || (x >= 'A' && x <= 'F');
        dig |= d;
        hexlet |= hl;
        all_hex &= d | hl;
        slashes += x == '/';
        dot |= x == '.';
        us |= x == '_';
        lower |= !(x >= 'A' && x <= 'Z') && !d;   // anything that could be category Ll
    }
    if (n >= 7) {
        const int32_t f = alower(c[a]);
        if (f == 'h' || f == 'f')
            for (const char* sch : kSchemes) if (starts_ascii_ci(c, a, b, sch)) return PH_URL;
    }
    if (n == 36) {
        bool ok = true;
        for (size_t i = 0; i < 36 && ok; ++i) {
            bool dash = (i == 8 || i == 13 || i == 18 || i == 23);
            ok = dash ? c[a + i] == '-' : is_hex(c[a + i]);
        }
        if (ok) return PH_UUID;
    }
    if (n >= 7 && all_hex && dig && hexlet) return PH_HASH;
    const int32_t f0 = c[a];
    if (f0 == '/' || f0 == '~' || f0 == '.') {
        bool pre = false;
        for (const char* p : kPathPrefix) if (starts_exact(c, a, b, p)) pre = true;
        if (pre && n > 1) return PH_PATH;
    }
    if (slashes >= 2) return PH_PATH;
    if (dot) {
        for (size_t i = b; i > a; --i) {
            if (c[i - 1] == '.') {
                size_t d = i - 1;
                if (d >= a + 1)
                    for (const char* e : kExt) if (equals_ascii_ci(c, d + 1, b, e)) return PH_PATH;
                break;   // rfind: only the LAST dot counts
            }
        }
    }
    if (us) return PH_IDENT;
    if (lower)
        for (size_t i = a; i + 1 < b; ++i)
            if (is_ll(c[i]) && is_lu(c[i + 1])) return PH_IDENT;
    if (dot)
        for (size_t i = a + 1; i + 1 < b; ++i)
            if (c[i] == '.' && is_letter(c[i - 1]) && is_letter(c[i + 1])) return PH_IDENT;
    if (f0 >= '0' && f0 <= '9') {
        size_t j = a + 1;
        while (j < b && ((c[j] >= '0' && c[j] <= '9') || c[j] == '.' || c[j] == ',')) ++j;
        if (j < b && c[j] == '%') ++j;
        if (j == b) return PH_NUM;
        for (const char* u : kUnits) if (equals_ascii_ci(c, j, b, u)) return PH_NUM;
    }
    return PH_NONE;
}

// Emits text ranges and placeholders in order. Text between placeholders is always a contiguous
// range of the NFKC code points (merged segments of features_ref.delex are contiguous: a flush only
// happens at a construct, which always emits a placeholder).
struct DelexEmit {
    Tok& T;
    Scratch& sc;
    const int32_t* c;
    size_t text_start = 0;
    void ph(size_t pa, size_t pb, Ph k) {
        if (pa > text_start) tok_v1_cps(T, sc, c + text_start, pa - text_start);
        T.close();
        T.placeholder(k);
        text_start = pb;
    }
    void end(size_t n) {
        if (n > text_start) tok_v1_cps(T, sc, c + text_start, n - text_start);
    }
};

void chunks(const int32_t* t, size_t i, size_t n, DelexEmit& E) {
    while (i < n) {
        size_t j = i;
        if (is_ws(t[i])) {
            while (j < n && is_ws(t[j])) ++j;
            i = j;
            continue;
        }
        while (j < n && !is_ws(t[j])) ++j;
        size_t len = j - i, lead = 0, trail = 0;
        while (lead < len && t[i + lead] < 0x80 && (kDelex.v[t[i + lead]] & 2)) ++lead;
        while (trail < len - lead && t[j - 1 - trail] < 0x80 && (kDelex.v[t[j - 1 - trail]] & 4)) ++trail;
        size_t ca = i + lead, cb = j - trail;
        Ph k = (cb > ca) ? classify(t, ca, cb) : PH_NONE;
        if (k != PH_NONE) E.ph(ca, cb, k);
        i = j;
    }
}

void delex_field(Tok& T, Scratch& sc, const int32_t* s, size_t n) {
    DelexEmit E{T, sc, s};
    size_t buf = 0;   // start of the pending plain-text run
    size_t i = 0;
    while (i < n) {
        if (i + 2 < n && s[i] == '`' && s[i + 1] == '`' && s[i + 2] == '`') {
            chunks(s, buf, i, E);
            size_t j = i + 3, end = n;
            for (; j + 2 < n; ++j)
                if (s[j] == '`' && s[j + 1] == '`' && s[j + 2] == '`') { end = j + 3; break; }
            E.ph(i, end, PH_CODE);
            i = buf = end;
            continue;
        }
        if (s[i] == '`') {
            size_t lim = std::min(n, i + static_cast<size_t>(kInlineMax) + 1);
            size_t j = i + 1;
            while (j < lim && s[j] != '`') ++j;
            if (j < lim) {   // closing backtick with j - i <= kInlineMax
                bool nl = false;
                for (size_t k = i + 1; k < j; ++k) if (s[k] == '\n') { nl = true; break; }
                if (!nl) {
                    chunks(s, buf, i, E);
                    E.ph(i, j + 1, PH_IDENT);
                    i = buf = j + 1;
                    continue;
                }
            }
        }
        ++i;
    }
    chunks(s, buf, n, E);
    E.end(n);
}

// NFKC code points of a field (segmented like the V1 path; no case fold).
void nfkc_cps(Scratch& sc, const uint8_t* s, size_t n) {
    size_t w = 0;
    if (sc.cps.size() < n + 16) sc.cps.resize(n * 2 + 16);
    size_t i = 0;
    while (i < n) {
        size_t r = i;
        while (r < n && s[r] < 0x80) ++r;
        const size_t seg = (r == n) ? n : (r > i ? r - 1 : r);
        if (sc.cps.size() < w + (seg - i)) sc.cps.resize((w + (seg - i)) * 2 + 16);
        int32_t* o = sc.cps.data() + w;
        for (size_t k = i; k < seg; ++k) *o++ = s[k];
        w += seg - i;
        if (r == n) break;
        size_t q = r;
        while (q < n && s[q] >= 0x80) ++q;
        const size_t m = umap_cps(sc, s + seg, q - seg, kNFKC);
        if (sc.cps.size() < w + m) sc.cps.resize((w + m) * 2 + 16);
        std::memcpy(sc.cps.data() + w, sc.ubuf.data(), m * sizeof(int32_t));
        w += m;
        i = q;
    }
    sc.ncps = w;
}

// ── accumulator ───────────────────────────────────────────────────────────────
// Phase 1 (hash_tokens) computes every insertion hash of the document into sc.hbuf — pure
// arithmetic, no memory dependencies, so the out-of-order core overlaps the FNV chains of many
// windows. Phase 2 (insert_all) folds them into the table; the home-slot case (hit or empty) is
// branch-free, only a genuine collision takes the probe loop.

void begin_doc(Scratch& sc, size_t adds) {
    size_t cap = kMinCap;
    while (cap < 4 * adds && cap < kMaxCap) cap <<= 1;
    if (sc.table.size() < cap) {
        sc.table.assign(std::max(cap, sc.table.size()), 0);
        sc.gen = 0;
    }
    if (++sc.gen > kGenMax) {   // generation wrap: clear once every 16k documents
        std::fill(sc.table.begin(), sc.table.end(), 0);
        sc.gen = 1;
    }
    sc.mask = static_cast<uint32_t>(cap - 1);
    if (sc.touched.size() < adds) sc.touched.resize(adds * 2);
    if (sc.hbuf.size() < adds) sc.hbuf.resize(adds * 2);
    sc.ntouched = 0;
}

// returns the number of hashes written to sc.hbuf (== the adds count of tokenize_doc)
size_t hash_tokens(const FieldSet& fs, Scratch& sc) {
    uint64_t* H = sc.hbuf.data();
    size_t nh = 0;
    const uint8_t* A = sc.arena.data();
    const TokSpan* tk = sc.toks.data();
    const size_t nt = sc.toks.size();
    for (size_t t = 0; t < nt; ++t) {
        const TokSpan& k = tk[t];
        const auto& pre = fs.pre[k.field];
        const uint8_t* p = A + k.a;
        H[nh++] = fnv_bytes(pre[0], p, k.len);
        if (k.next) {
            const TokSpan& nx = tk[t + 1];
            H[nh++] = fnv_bytes(pre[1], p, nx.a + nx.len - k.a);
        }
        if (k.ph) continue;
        const uint8_t* P = p - 1;   // ' ' + tok + ' '
        const uint64_t pc = pre[2];
        if (k.ascii) {
            const size_t np = static_cast<size_t>(k.len) + 2;   // >= 3
            size_t s = 0;
            for (; s + 5 <= np; ++s) {   // interior windows: c3, c4, c5 all exist
                uint64_t h = fnv_byte(fnv_byte(fnv_byte(pc, P[s]), P[s + 1]), P[s + 2]);
                H[nh] = h;
                h = fnv_byte(h, P[s + 3]);
                H[nh + 1] = h;
                H[nh + 2] = fnv_byte(h, P[s + 4]);
                nh += 3;
            }
            for (; s + 3 <= np; ++s) {
                uint64_t h = fnv_byte(fnv_byte(fnv_byte(pc, P[s]), P[s + 1]), P[s + 2]);
                H[nh++] = h;
                if (s + 4 <= np) H[nh++] = fnv_byte(h, P[s + 3]);
            }
        } else {
            // code-point byte offsets of the padded token (lead bytes), plus the end
            const size_t nb = static_cast<size_t>(k.len) + 2;
            sc.offs.clear();
            for (size_t i = 0; i < nb; ++i) if ((P[i] & 0xC0) != 0x80) sc.offs.push_back(static_cast<uint32_t>(i));
            sc.offs.push_back(static_cast<uint32_t>(nb));
            const uint32_t* o = sc.offs.data();
            const size_t np = sc.offs.size() - 1;   // padded code points
            for (size_t s = 0; s + 3 <= np; ++s) {
                uint64_t h = fnv_bytes(pc, P + o[s], o[s + 3] - o[s]);
                H[nh++] = h;
                if (s + 4 > np) continue;
                h = fnv_bytes(h, P + o[s + 3], o[s + 4] - o[s + 3]);
                H[nh++] = h;
                if (s + 5 > np) continue;
                H[nh++] = fnv_bytes(h, P + o[s + 4], o[s + 5] - o[s + 4]);
            }
        }
    }
    return nh;
}

void insert_all(Scratch& sc, size_t nh) {
    const uint64_t* H = sc.hbuf.data();
    uint64_t* T = sc.table.data();
    uint32_t* tch = sc.touched.data();
    size_t nt = 0;
    const uint32_t mask = sc.mask;
    const uint32_t gtag = sc.gen << kDimBits;
    for (size_t k = 0; k < nh; ++k) {
        const uint64_t h = H[k];
        const uint32_t idx = static_cast<uint32_t>(h) & kDimMask;
        const uint32_t tag = gtag | idx;
        const int32_t d = 1 - 2 * static_cast<int32_t>(h >> 63);
        uint32_t s = idx & mask;
        uint64_t e = T[s];
        uint32_t et = static_cast<uint32_t>(e);
        bool hit = et == tag;
        bool empty = (et & ~kDimMask) != gtag;   // stale generation = empty
        while (!(hit | empty)) {                 // collision: linear probe (rare at load <= 0.5)
            s = (s + 1) & mask;
            e = T[s];
            et = static_cast<uint32_t>(e);
            hit = et == tag;
            empty = (et & ~kDimMask) != gtag;
        }
        const int32_t c = (hit ? static_cast<int32_t>(e >> 32) : 0) + d;
        T[s] = static_cast<uint64_t>(tag) | (static_cast<uint64_t>(static_cast<uint32_t>(c)) << 32);
        tch[nt] = s;
        nt += empty ? 1 : 0;
    }
    sc.ntouched = nt;
}

// 1 + ln(k) for small k, computed with the same std::log once: identical values, no per-feature log.
struct LogTable {
    double v[65];
    LogTable() { for (int k = 1; k <= 64; ++k) v[k] = 1.0 + std::log(static_cast<double>(k)); v[0] = 0.0; }
};
const LogTable kLog;

}  // namespace

FieldSet::FieldSet(const std::vector<std::string>& sorted_names) : names(sorted_names) {
    for (size_t i = 1; i < names.size(); ++i)
        if (!(names[i - 1] < names[i])) throw std::runtime_error("ctxshield: field names must be sorted, unique");
    if (names.size() > 255) throw std::runtime_error("ctxshield: too many fields");
    int reply = -1, text = -1, pilot = -1, ask = -1;
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == "reply") reply = static_cast<int>(i);
        if (names[i] == "text") text = static_cast<int>(i);
        if (names[i] == "pilot") pilot = static_cast<int>(i);
        if (names[i] == "ask") ask = static_cast<int>(i);
        std::array<uint64_t, 3> p{};
        for (int k = 0; k < 3; ++k) {
            uint64_t h = fnv_bytes(kFnvOffset, reinterpret_cast<const uint8_t*>(names[i].data()), names[i].size());
            h = fnv_byte(h, kSep);
            h = fnv_bytes(h, reinterpret_cast<const uint8_t*>(kKinds[k]), std::strlen(kKinds[k]));
            p[static_cast<size_t>(k)] = fnv_byte(h, kSep);
        }
        pre.push_back(p);
    }
    reply_or_text = reply >= 0 ? reply : text;
    pilot_or_ask = pilot >= 0 ? pilot : ask;
}

Scratch::Scratch() {
    arena.resize(1 << 16);
    toks.reserve(1 << 12);
    ubuf.resize(1 << 12);
    touched.reserve(1 << 14);
    offs.reserve(256);
}

size_t tokenize_doc(const FieldSet& fs, const std::string_view* texts, bool dlx, Scratch& sc) {
    sc.n = 0;
    sc.toks.clear();
    for (size_t f = 0; f < fs.names.size(); ++f) {
        const uint8_t* s = reinterpret_cast<const uint8_t*>(texts[f].data());
        const size_t n = texts[f].size();
        Tok T(sc, static_cast<uint8_t>(f));
        if (!dlx) {
            tok_v1_bytes(T, sc, s, n);
        } else {
#ifdef SHIELD_PROFILE
            auto _p = std::chrono::steady_clock::now();
#endif
            nfkc_cps(sc, s, n);
            PROF(6);
            delex_field(T, sc, sc.cps.data(), sc.ncps);
            PROF(7);
        }
        T.finish();
    }
    size_t adds = 0;
    for (const TokSpan& t : sc.toks) adds += token_adds(t);
    return adds;
}

void compute_extras(const FieldSet& fs, const std::string_view* texts, float* out) {
    const size_t nf = fs.names.size();
    for (size_t f = 0; f < nf; ++f) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(texts[f].data());
        out[f] = static_cast<float>(std::log1p(static_cast<double>(count_cp(p, texts[f].size()))));
    }
    out[nf] = 0.0f;
    out[nf + 1] = 0.0f;
    if (fs.reply_or_text >= 0) {
        const std::string_view r = texts[fs.reply_or_text];
        const uint8_t* p = reinterpret_cast<const uint8_t*>(r.data());
        // start of the last kQtail code points
        size_t i = r.size(), k = 0;
        while (i > 0 && k < static_cast<size_t>(kQtail)) {
            --i;
            if ((p[i] & 0xC0) != 0x80) ++k;
        }
        for (size_t j = i; j < r.size(); ++j) if (p[j] == '?') { out[nf] = 1.0f; break; }
    }
    if (fs.pilot_or_ask >= 0) {
        const std::string_view t = texts[fs.pilot_or_ask];
        const uint8_t* p = reinterpret_cast<const uint8_t*>(t.data());
        const size_t n = t.size();
        size_t letters = 0, upper = 0, i = 0;
        while (i < n) {
            if (p[i] < 0x80) {
                uint8_t c = p[i];
                if (c >= 'A' && c <= 'Z') { ++letters; ++upper; }
                else if (c >= 'a' && c <= 'z') ++letters;
                ++i;
                continue;
            }
            int32_t c;
            utf8proc_ssize_t m = utf8proc_iterate(p + i, static_cast<utf8proc_ssize_t>(n - i), &c);
            if (m <= 0) throw std::runtime_error("ctxshield: invalid UTF-8 in input");
            int k = cat(c);
            if (k >= 1 && k <= 5) ++letters;
            if (k == UTF8PROC_CATEGORY_LU) ++upper;
            i += static_cast<size_t>(m);
        }
        out[nf + 1] = letters ? static_cast<float>(static_cast<double>(upper) / static_cast<double>(letters)) : 0.0f;
    }
}


namespace {

// Tokenize + hash + accumulate one document into sc (table slots listed in sc.touched).
inline void accumulate_doc(const FieldSet& fs, const std::string_view* texts, bool dlx, Scratch& sc) {
#ifdef SHIELD_PROFILE
    auto _p = std::chrono::steady_clock::now();
#endif
    const size_t adds = tokenize_doc(fs, texts, dlx, sc);
    PROF(0);
    begin_doc(sc, adds);
    const size_t nh = hash_tokens(fs, sc);
    if (nh != adds) throw std::logic_error("ctxshield: hash count != token_adds");
    PROF(1);
    insert_all(sc, nh);
    PROF(2);
}

// Non-zero slots → sc.xi (index) / sc.x (sign·(1+ln|c|)); returns 1/L2-norm (1.0 when empty).
// Four partial sums break the add dependency chain (float64; order differs from the reference's
// sorted sum only at the 1e-16 level).
inline double finalize_doc(Scratch& sc) {
    const size_t m = sc.ntouched;
    if (sc.x.size() < m) { sc.x.resize(m * 2); sc.xi.resize(m * 2); }
    double* X = sc.x.data();
    uint32_t* XI = sc.xi.data();
    const uint64_t* T = sc.table.data();
    const uint32_t* tch = sc.touched.data();
    size_t k = 0;
    for (size_t q = 0; q < m; ++q) {
        const uint64_t e = T[tch[q]];
        const int32_t c = static_cast<int32_t>(e >> 32);
        const int32_t a = c > 0 ? c : -c;
        const double mag = a <= 64 ? kLog.v[a] : 1.0 + std::log(static_cast<double>(a));
        XI[k] = static_cast<uint32_t>(e) & kDimMask;
        X[k] = c > 0 ? mag : -mag;
        k += c != 0;   // a cancelled count (c == 0) is overwritten by the next slot
    }
    double n0 = 0, n1 = 0, n2 = 0, n3 = 0;
    size_t j = 0;
    for (; j + 4 <= k; j += 4) {
        n0 += X[j] * X[j];
        n1 += X[j + 1] * X[j + 1];
        n2 += X[j + 2] * X[j + 2];
        n3 += X[j + 3] * X[j + 3];
    }
    for (; j < k; ++j) n0 += X[j] * X[j];
    sc.nx = k;
    const double norm = std::sqrt((n0 + n1) + (n2 + n3));
    return norm > 0 ? 1.0 / norm : 1.0;
}

}  // namespace

void featurize(const FieldSet& fs, const std::string_view* texts, bool dlx, Scratch& sc,
               Features& out, bool sorted) {
    accumulate_doc(fs, texts, dlx, sc);
    if (sorted) {
        // sort the claimed slots by feature index
        std::sort(sc.touched.begin(), sc.touched.begin() + static_cast<long>(sc.ntouched), [&](uint32_t x, uint32_t y) {
            return (static_cast<uint32_t>(sc.table[x]) & kDimMask) < (static_cast<uint32_t>(sc.table[y]) & kDimMask);
        });
    }
    const double inv = finalize_doc(sc);
    out.idx.assign(sc.xi.begin(), sc.xi.begin() + static_cast<long>(sc.nx));
    out.val.resize(sc.nx);
    for (size_t k = 0; k < sc.nx; ++k) out.val[k] = static_cast<float>(sc.x[k] * inv);
    out.extras.resize(static_cast<size_t>(fs.n_extras()));
    compute_extras(fs, texts, out.extras.data());
}

void featurize_score(const FieldSet& fs, const std::string_view* texts, bool dlx, Scratch& sc,
                     const float* W, const float* We, const float* b, int hpad, float* logits) {
    if (hpad <= 0 || hpad > kMaxHeadsPad || (hpad & 3) != 0)
        throw std::runtime_error("ctxshield: hpad must be 4, 8, 12 or 16");
    accumulate_doc(fs, texts, dlx, sc);
#ifdef SHIELD_PROFILE
    auto _p = std::chrono::steady_clock::now();
#endif
    const double inv = finalize_doc(sc);
    PROF(3);
    float ex[256 + 2];
    compute_extras(fs, texts, ex);
    PROF(4);
    gather_dot(sc.xi.data(), sc.x.data(), inv, sc.nx, ex, static_cast<size_t>(fs.n_extras()), W, We, b,
               hpad, logits);
    PROF(5);
}

int default_threads() {
#if defined(__APPLE__)
    int v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &v, &len, nullptr, 0) == 0 && v > 0) return v;
#endif
    unsigned h = std::thread::hardware_concurrency();
    if (h == 0) throw std::runtime_error("ctxshield: cannot determine CPU count");
    return static_cast<int>(h);
}

const char* utf8proc_version_string() { return utf8proc_version(); }
const char* unicode_version_string() { return utf8proc_unicode_version(); }

}  // namespace ctxshield
