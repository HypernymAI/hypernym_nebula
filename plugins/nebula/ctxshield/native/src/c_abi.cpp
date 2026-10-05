// Plain-C ABI over the backends (include/shield.h, build doc §9.2.6) + the CPU backend + the CPU
// tokenization shared by the GPU hosts. A requested backend that is not built / not available /
// (CUDA) not verified is an error — never a quiet CPU run.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "parallel.h"
#include "shield.h"

namespace ctxshield {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class CpuBackend final : public Backend {
public:
    explicit CpuBackend(const GroupWeights& g)
        : fs_(g.names), delex_(g.delex), hpad_(g.hpad),
          W_(g.W, g.W + static_cast<size_t>(kDim) * static_cast<size_t>(g.hpad)),
          We_(g.We, g.We + static_cast<size_t>(fs_.n_extras()) * static_cast<size_t>(g.hpad)),
          b_(g.b, g.b + g.hpad) {}
    int kind() const override { return SHIELD_BACKEND_CPU; }
    void score(const std::string_view* texts, size_t n, int threads, float* logits,
               shield_stats* st) override {
        const double t0 = now_s();
        const size_t nf = fs_.names.size();
        parallel_chunks(n, threads, [&](size_t a, size_t e) {
            Scratch& sc = thread_scratch();
            for (size_t i = a; i < e; ++i)
                featurize_score(fs_, texts + i * nf, delex_, sc, W_.data(), We_.data(), b_.data(), hpad_,
                                logits + i * static_cast<size_t>(hpad_));
        });
        st->backend = SHIELD_BACKEND_CPU;
        st->n_docs = n;
        st->t_total_s = now_s() - t0;
    }

private:
    FieldSet fs_;
    bool delex_;
    int hpad_;
    std::vector<float> W_, We_, b_;
};

void set_err(char* err, size_t errlen, const std::string& m) {
    if (err && errlen) std::snprintf(err, errlen, "%s", m.c_str());
}

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(const GroupWeights& g) { return std::make_unique<CpuBackend>(g); }

void gpu_tokenize(const FieldSet& fs, bool delex, const std::string_view* texts, size_t n_docs,
                  int threads, GpuBatch& out) {
    const size_t nf = fs.names.size();
    const size_t ne = static_cast<size_t>(fs.n_extras());
    int T = threads > 0 ? threads : default_threads();
    // a few parts per thread so uneven documents balance; each part is one contiguous doc range
    size_t nparts = std::max<size_t>(1, std::min<size_t>(n_docs, static_cast<size_t>(T) * 4));
    if (out.parts.size() < nparts) out.parts.resize(nparts);
    const size_t chunk = (n_docs + nparts - 1) / nparts;
    out.n_docs = n_docs;
    out.doc_adds.resize(n_docs);
    out.extras.resize(n_docs * ne);
    parallel_chunks(nparts, T, [&](size_t pa, size_t pb) {
        Scratch& sc = thread_scratch();
        for (size_t p = pa; p < pb; ++p) {
            GpuPart& P = out.parts[p];
            P.d0 = std::min(n_docs, p * chunk);
            P.d1 = std::min(n_docs, P.d0 + chunk);
            P.arena.clear();
            P.toks.clear();
            P.ntok.clear();
            for (size_t d = P.d0; d < P.d1; ++d) {
                const std::string_view* tx = texts + d * nf;
                out.doc_adds[d] = static_cast<uint32_t>(tokenize_doc(fs, tx, delex, sc));
                compute_extras(fs, tx, out.extras.data() + d * ne);
                const size_t base = P.arena.size();
                if (base + sc.n > 0xFFFFFFF0ULL) throw std::runtime_error("ctxshield: GPU batch arena > 4 GB; split the batch");
                P.arena.insert(P.arena.end(), sc.arena.begin(), sc.arena.begin() + static_cast<long>(sc.n));
                const size_t nt = sc.toks.size();
                const size_t tb = P.toks.size();
                P.toks.resize(tb + nt);
                GpuTok* g = P.toks.data() + tb;
                const uint32_t b32 = static_cast<uint32_t>(base);
                for (size_t t = 0; t < nt; ++t) {
                    const TokSpan& k = sc.toks[t];
                    g[t].a = b32 + k.a;
                    g[t].len = k.len;
                    g[t].w2_end = k.next ? b32 + sc.toks[t + 1].a + sc.toks[t + 1].len : 0u;
                    g[t].meta = static_cast<uint32_t>(k.field) | (static_cast<uint32_t>(k.ph) << 8) |
                                (static_cast<uint32_t>(k.ascii) << 9);
                }
                P.ntok.push_back(static_cast<uint32_t>(nt));
            }
        }
    });
    size_t ab = 0, tb = 0;
    for (size_t p = 0; p < nparts; ++p) {
        ab += out.parts[p].arena.size();
        tb += out.parts[p].toks.size();
    }
    for (size_t p = nparts; p < out.parts.size(); ++p) {   // unused parts of an earlier, larger call
        out.parts[p].d0 = out.parts[p].d1 = 0;
        out.parts[p].arena.clear();
        out.parts[p].toks.clear();
        out.parts[p].ntok.clear();
    }
    if (ab > 0xFFFFFFF0ULL || tb > 0xFFFFFFF0ULL)
        throw std::runtime_error("ctxshield: GPU batch arena/tokens > 4 G; split the batch");
    out.arena_bytes = ab;
    out.n_toks = tb;
}

void gpu_pack(const GpuBatch& b, uint8_t* arena, GpuTok* toks, uint32_t* doc_tok, int threads) {
    const size_t np = b.parts.size();
    std::vector<size_t> abase(np), tbase(np);
    size_t ab = 0, tb = 0;
    for (size_t p = 0; p < np; ++p) {
        abase[p] = ab;
        tbase[p] = tb;
        ab += b.parts[p].arena.size();
        tb += b.parts[p].toks.size();
    }
    doc_tok[0] = 0;
    parallel_chunks(np, threads, [&](size_t pa, size_t pe) {
        for (size_t p = pa; p < pe; ++p) {
            const GpuPart& P = b.parts[p];
            if (!P.arena.empty()) std::memcpy(arena + abase[p], P.arena.data(), P.arena.size());
            const uint32_t a0 = static_cast<uint32_t>(abase[p]);
            GpuTok* dst = toks + tbase[p];
            for (size_t t = 0; t < P.toks.size(); ++t) {
                GpuTok g = P.toks[t];
                g.a += a0;
                if (g.w2_end) g.w2_end += a0;
                dst[t] = g;
            }
            uint32_t acc = static_cast<uint32_t>(tbase[p]);
            for (size_t i = 0; i < P.ntok.size(); ++i) {
                acc += P.ntok[i];
                doc_tok[P.d0 + i + 1] = acc;
            }
        }
    });
}

#ifndef SHIELD_HAVE_METAL
bool metal_built() { return false; }
bool metal_available(std::string& why) { why = "Metal backend not built into this module (build.sh metal)"; return false; }
std::unique_ptr<Backend> make_metal_backend(const GroupWeights&) {
    throw std::runtime_error("ctxshield: Metal backend not built into this module (build.sh metal)");
}
#endif
#ifndef SHIELD_HAVE_CUDA
bool cuda_built() { return false; }
bool cuda_available(std::string& why) { why = "CUDA backend not built into this module (-DSHIELD_CUDA=ON, UNTESTED)"; return false; }
std::unique_ptr<Backend> make_cuda_backend(const GroupWeights&) {
    throw std::runtime_error("ctxshield: CUDA backend not built into this module (-DSHIELD_CUDA=ON)");
}
#endif

}  // namespace ctxshield

using namespace ctxshield;

struct shield_scorer {
    std::unique_ptr<Backend> be;
    size_t n_fields;
    int hpad;
};

extern "C" {

int shield_abi_version(void) { return 1; }

const char* shield_backend_name(int backend) {
    switch (backend) {
        case SHIELD_BACKEND_CPU: return "cpu";
        case SHIELD_BACKEND_METAL: return "metal";
        case SHIELD_BACKEND_CUDA: return "cuda";
        default: return "unknown";
    }
}

int shield_backend_available(int backend, char* why, size_t whylen) {
    std::string w;
    bool ok = false;
    switch (backend) {
        case SHIELD_BACKEND_CPU: ok = true; break;
        case SHIELD_BACKEND_METAL: ok = metal_available(w); break;
        case SHIELD_BACKEND_CUDA: ok = cuda_available(w); break;
        default: w = "unknown backend";
    }
    set_err(why, whylen, w);
    return ok ? 1 : 0;
}

int shield_scorer_create(int backend, const char* const* field_names, int n_fields, int delex,
                         const float* W, const float* We, const float* b, int hpad,
                         shield_scorer** out, char* err, size_t errlen) {
    try {
        *out = nullptr;
        if (hpad <= 0 || hpad > kMaxHeadsPad || (hpad & 3) != 0)
            throw std::runtime_error("ctxshield: hpad must be 4, 8, 12 or 16");
        GroupWeights g;
        for (int i = 0; i < n_fields; ++i) g.names.emplace_back(field_names[i]);
        g.delex = delex != 0;
        g.hpad = hpad;
        g.W = W;
        g.We = We;
        g.b = b;
        std::string why;
        std::unique_ptr<Backend> be;
        switch (backend) {
            case SHIELD_BACKEND_CPU: be = make_cpu_backend(g); break;
            case SHIELD_BACKEND_METAL:
                if (!metal_available(why)) throw std::runtime_error("ctxshield: Metal unavailable: " + why);
                be = make_metal_backend(g);
                break;
            case SHIELD_BACKEND_CUDA:
                if (!cuda_available(why)) throw std::runtime_error("ctxshield: CUDA unavailable: " + why);
                be = make_cuda_backend(g);
                break;
            default: throw std::runtime_error("ctxshield: unknown backend");
        }
        *out = new shield_scorer{std::move(be), g.names.size(), hpad};
        return 0;
    } catch (const std::exception& e) {
        set_err(err, errlen, e.what());
        return 1;
    }
}

int shield_scorer_score(shield_scorer* s, const char* const* texts, const size_t* lens,
                        size_t n_docs, int threads, float* logits, shield_stats* stats,
                        char* err, size_t errlen) {
    try {
        std::vector<std::string_view> v(n_docs * s->n_fields);
        for (size_t i = 0; i < v.size(); ++i) v[i] = std::string_view(texts[i], lens[i]);
        shield_stats st{};
        s->be->score(v.data(), n_docs, threads, logits, &st);
        if (stats) *stats = st;
        return 0;
    } catch (const std::exception& e) {
        set_err(err, errlen, e.what());
        return 1;
    }
}

void shield_scorer_destroy(shield_scorer* s) { delete s; }

}  // extern "C"
