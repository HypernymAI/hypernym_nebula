// ctxshield native core — the interface every backend implements (build doc Part IX, §5.2c/§5.2d).
//
// Step 3 (2026-09-26): CPU backend, exposed to Python through pybind11 (python/bind.cpp).
// Step 8 (2026-09-27): CPU speed pass (byte-level tokenizer into one reused arena, open-addressing
// accumulator), the plain-C ABI at the bottom of this file, and the Metal / CUDA backends behind it.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ctxshield {

constexpr int kDimBits = 18;
constexpr uint32_t kDim = 1u << kDimBits;
constexpr uint32_t kDimMask = kDim - 1;
constexpr int kMaxHeadsPad = 16;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

// One document = the head group's fields: `texts` points at names.size() views aligned with the
// sorted field names. Texts are valid UTF-8 (the Python wrapper replaces lone surrogates with
// U+FFFD first); invalid UTF-8 raises.
struct FieldSet {
    std::vector<std::string> names;             // sorted
    std::vector<std::array<uint64_t, 3>> pre;   // FNV state after FIELD 0x1F KIND 0x1F, KIND = w1, w2, c
    int reply_or_text = -1;                     // index used for the qtail extra (-1 = none)
    int pilot_or_ask = -1;                      // index used for the caps extra (-1 = none)
    explicit FieldSet(const std::vector<std::string>& sorted_names);
    int n_extras() const { return static_cast<int>(names.size()) + 2; }
};

struct Features {
    std::vector<uint32_t> idx;        // sorted ascending, unique (unsorted when sorted=false)
    std::vector<float> val;           // L2-normalised over the hashed block
    std::vector<float> extras;        // n_extras
};

// A token = a span of the arena. Arena layout per field: ' ' tok ' ' tok ' ' ... — so
//   w1 payload  = arena[a, a+len)
//   w2 payload  = arena[a, next.a + next.len)          (the single 0x20 between them is in the arena)
//   padded tok  = arena[a-1, a+len+1)                   (' ' + tok + ' ', for the char n-grams)
struct TokSpan {
    uint32_t a;       // byte offset of the token in the doc arena
    uint32_t len;     // bytes
    uint32_t ncp;     // code points
    uint8_t field;    // index into FieldSet::names
    uint8_t ph;       // 1 = delex placeholder (w1/w2 only)
    uint8_t ascii;    // 1 = every byte is a code point
    uint8_t next;     // 1 = the next token is in the same field (w2 exists)
};

// Per-thread scratch, reused across documents (no allocation on the hot path once warm).
struct Scratch {
    // tokenizer output for the current document
    std::vector<uint8_t> arena;
    size_t n = 0;                     // arena bytes used
    std::vector<TokSpan> toks;
    // utf8proc / delex work buffers
    std::vector<int32_t> ubuf;        // decompose output (code points)
    std::vector<int32_t> cps;         // V2: NFKC code points of the field: [0, ncps)
    size_t ncps = 0;
    std::vector<uint8_t> bbuf;        // V2: UTF-8 of a code-point segment
    std::vector<uint32_t> offs;       // code-point byte offsets of one padded non-ASCII token
    // accumulator: open addressing, slot = {tag = gen<<18 | idx, signed count}
    std::vector<uint64_t> table;      // low 32 bits tag, high 32 bits count (int32)
    std::vector<uint32_t> touched;    // claimed slots of the current document: [0, ntouched)
    size_t ntouched = 0;
    std::vector<uint64_t> hbuf;       // every insertion hash of the current document
    uint32_t gen = 0;
    uint32_t mask = 0;
    std::vector<double> x;            // sublinear values of the non-zero slots: [0, nx)
    std::vector<uint32_t> xi;         // their feature indices
    size_t nx = 0;
    Scratch();
};

// Tokenize every field of one document into sc.arena / sc.toks (spans carry the field index).
// Returns the exact number of hash insertions the document will make (w1 + w2 + char n-grams).
size_t tokenize_doc(const FieldSet& fs, const std::string_view* texts, bool delex,
                    Scratch& sc);

// Number of hash insertions made by one token (w1 + w2 + char n-grams).
inline size_t token_adds(const TokSpan& t) {
    size_t k = 1 + t.next;
    if (!t.ph) {
        size_t p = t.ncp + 2;
        k += (p >= 3 ? p - 2 : 0) + (p >= 4 ? p - 3 : 0) + (p >= 5 ? p - 4 : 0);
    }
    return k;
}

// Dense extras from the raw field texts (§5.2c), written to out[0 .. n_extras).
void compute_extras(const FieldSet& fs, const std::string_view* texts, float* out);

// sorted=false skips the index sort (scoring does not need order; the set and values are the same).
void featurize(const FieldSet& fs, const std::string_view* texts, bool delex,
               Scratch& scratch, Features& out, bool sorted = true);

// logits[k] = b[k] + sum_f val_f * W[idx_f * hpad + k] + sum_e extras_e * We[e * hpad + k]
void score_features(const Features& f, const float* W, const float* We, const float* b,
                    int hpad, float* logits);

// Same dot with val_f = float(x_f * inv) (the value featurize() emits), extras ex[0 .. ne).
void gather_dot(const uint32_t* idx, const double* x, double inv, size_t n, const float* ex,
                size_t ne, const float* W, const float* We, const float* b, int hpad,
                float* logits);

// featurize + score one document without materializing Features (the CPU scoring hot path).
void featurize_score(const FieldSet& fs, const std::string_view* texts, bool delex, Scratch& sc,
                     const float* W, const float* We, const float* b, int hpad, float* logits);

const char* utf8proc_version_string();
const char* unicode_version_string();

// P-core count (hw.perflevel0.physicalcpu on Apple), else hardware_concurrency.
int default_threads();

}  // namespace ctxshield

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Plain-C ABI (build doc §9.2.6): one scorer handle per (backend, head group, weights).
// Every backend implements the same call. A backend that is not built, not available on this
// machine, or (CUDA) not verified is an ERROR at create time — never a quiet CPU run.
// All calls return 0 on success, non-zero on error with a message in err[0 .. errlen).
// ─────────────────────────────────────────────────────────────────────────────────────────────
extern "C" {

enum shield_backend { SHIELD_BACKEND_CPU = 0, SHIELD_BACKEND_METAL = 1, SHIELD_BACKEND_CUDA = 2 };

typedef struct shield_scorer shield_scorer;

typedef struct shield_stats {
    int backend;                 // the backend that ran
    size_t n_docs;
    size_t n_overflow_docs;      // GPU: docs whose on-chip table overflowed → device-memory table
    size_t n_tokens;
    size_t n_arena_bytes;
    double t_tokenize_s;         // CPU tokenization (GPU backends) / 0 (CPU backend)
    double t_device_s;           // GPU command buffer wall time (GPU backends)
    double t_total_s;
} shield_stats;

int shield_abi_version(void);
const char* shield_backend_name(int backend);
// 1 = built into this module AND usable on this machine; 0 otherwise. why[0..whylen) says why not.
int shield_backend_available(int backend, char* why, size_t whylen);

// Weights: W [2^18][hpad], We [n_extras][hpad], b [hpad], float32, head-interleaved.
// hpad ∈ {4, 8, 12, 16}. Weights are copied into the backend's memory.
int shield_scorer_create(int backend, const char* const* field_names, int n_fields, int delex,
                         const float* W, const float* We, const float* b, int hpad,
                         shield_scorer** out, char* err, size_t errlen);
// texts: n_docs × n_fields pointers (row-major, fields in the sorted order given at create),
// lens the byte lengths. logits: n_docs × hpad.
int shield_scorer_score(shield_scorer* s, const char* const* texts, const size_t* lens,
                        size_t n_docs, int threads, float* logits, shield_stats* stats,
                        char* err, size_t errlen);
void shield_scorer_destroy(shield_scorer* s);

}  // extern "C"
