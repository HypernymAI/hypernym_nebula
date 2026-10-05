// Internal backend interface behind the C ABI (include/shield.h). One instance = one head group
// (sorted field names + spec) + one weight set, resident in the backend's memory.
#pragma once
#include <memory>
#include <string>
#include <string_view>

#include "shield.h"

namespace ctxshield {

struct GroupWeights {
    std::vector<std::string> names;   // sorted
    bool delex = false;
    int hpad = 0;
    const float* W = nullptr;         // [2^18][hpad]
    const float* We = nullptr;        // [n_extras][hpad]
    const float* b = nullptr;         // [hpad]
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual int kind() const = 0;
    // texts: n_docs × n_fields views (row-major); logits: n_docs × hpad
    virtual void score(const std::string_view* texts, size_t n_docs, int threads, float* logits,
                       shield_stats* st) = 0;
};

std::unique_ptr<Backend> make_cpu_backend(const GroupWeights& g);

// Metal (src/metal/metal_backend.mm) — present only when built with SHIELD_METAL.
bool metal_built();
bool metal_available(std::string& why);
std::unique_ptr<Backend> make_metal_backend(const GroupWeights& g);

// CUDA (src/cuda/cuda_backend.cpp) — present only when built with SHIELD_CUDA. UNTESTED.
bool cuda_built();
bool cuda_available(std::string& why);
std::unique_ptr<Backend> make_cuda_backend(const GroupWeights& g);

// GPU host helper shared by Metal and CUDA: CPU tokenization of a batch into flat arrays.
struct GpuTok {            // 16 bytes, mirrored by the kernels
    uint32_t a;            // arena offset (absolute in the batch arena)
    uint32_t len;          // bytes
    uint32_t w2_end;       // arena end of the w2 payload (0 = no w2)
    uint32_t meta;         // field | ph << 8 | ascii << 9
};
struct GpuPart {           // one worker's contiguous range of documents
    size_t d0 = 0, d1 = 0;
    std::vector<uint8_t> arena;
    std::vector<GpuTok> toks;      // offsets local to the part
    std::vector<uint32_t> ntok;    // tokens per doc
};
struct GpuBatch {          // reused across calls (capacity is kept)
    std::vector<GpuPart> parts;
    std::vector<uint32_t> doc_adds;    // exact hash insertions per doc
    std::vector<float> extras;         // n_docs × n_extras
    size_t n_docs = 0, arena_bytes = 0, n_toks = 0;
};
// CPU tokenization of docs [0, n_docs) (all threads). Then gpu_pack writes the flat arrays straight
// into the (GPU-visible) destinations, in parallel: arena bytes, tokens with absolute offsets, and
// doc_tok[n_docs + 1] token offsets.
void gpu_tokenize(const FieldSet& fs, bool delex, const std::string_view* texts, size_t n_docs,
                  int threads, GpuBatch& out);
void gpu_pack(const GpuBatch& b, uint8_t* arena, GpuTok* toks, uint32_t* doc_tok, int threads);

}  // namespace ctxshield
