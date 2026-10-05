// CUDA host for ctxshield (build doc §9.4) — UNTESTED: no NVIDIA device on the build machine.
// Mirrors src/metal/metal_backend.mm: CPU tokenization (all cores) into PINNED host buffers →
// async H2D on the chunk's stream → score_tg (one block per doc) → D2H of logits + overflow flags;
// two streams double-buffer so the CPU tokenizes chunk c+1 while the GPU runs chunk c; overflowed
// documents are re-run by score_dev with global-memory tables. Every CUDA error raises.
//
// Refused while unverified: cuda_available() is false unless the module was built with
// SHIELD_CUDA_VERIFIED=1 (only after run_on_l4.sh passes) or SHIELD_CUDA_UNVERIFIED_OK=true is set
// in the environment (the L4 verification run itself). The Python loader enforces the same via
// native/backends.json (features.check_backend).
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "shield.h"
#include "shield_cuda.h"

#ifndef SHIELD_CUDA_VERIFIED
#define SHIELD_CUDA_VERIFIED 0
#endif

namespace ctxshield {

namespace {

constexpr size_t kChunk = 16384;
constexpr size_t kDevSlotsPerPass = size_t(1) << 25;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ctxshield: CUDA ") + what + ": " + cudaGetErrorString(e));
}

// device buffer, grow-only
struct DBuf {
    void* p = nullptr;
    size_t cap = 0;
    void need(size_t bytes) {
        bytes = std::max<size_t>(bytes, 16);
        if (bytes <= cap) return;
        if (p) ck(cudaFree(p), "cudaFree");
        p = nullptr;
        cap = bytes + bytes / 4;
        ck(cudaMalloc(&p, cap), "cudaMalloc");
    }
    ~DBuf() { if (p) cudaFree(p); }
};
// pinned host buffer, grow-only
struct HBuf {
    void* p = nullptr;
    size_t cap = 0;
    void need(size_t bytes) {
        bytes = std::max<size_t>(bytes, 16);
        if (bytes <= cap) return;
        if (p) ck(cudaFreeHost(p), "cudaFreeHost");
        p = nullptr;
        cap = bytes + bytes / 4;
        ck(cudaHostAlloc(&p, cap, cudaHostAllocDefault), "cudaHostAlloc");
    }
    ~HBuf() { if (p) cudaFreeHost(p); }
};

class CudaBackend final : public Backend {
public:
    explicit CudaBackend(const GroupWeights& g)
        : fs_(g.names), delex_(g.delex), hpad_(g.hpad), ne_(fs_.n_extras()) {
        ck(shield_cuda_prepare(), "cudaFuncSetAttribute(score_tg)");
        const size_t wb = static_cast<size_t>(kDim) * static_cast<size_t>(hpad_) * sizeof(float);
        W_.need(wb);
        ck(cudaMemcpy(W_.p, g.W, wb, cudaMemcpyHostToDevice), "upload W");
        We_.need(static_cast<size_t>(ne_) * hpad_ * sizeof(float));
        ck(cudaMemcpy(We_.p, g.We, static_cast<size_t>(ne_) * hpad_ * sizeof(float), cudaMemcpyHostToDevice), "upload We");
        b_.need(static_cast<size_t>(hpad_) * sizeof(float));
        ck(cudaMemcpy(b_.p, g.b, static_cast<size_t>(hpad_) * sizeof(float), cudaMemcpyHostToDevice), "upload b");
        std::vector<uint64_t> pre;
        for (const auto& p : fs_.pre) pre.insert(pre.end(), p.begin(), p.end());
        pre_.need(pre.size() * sizeof(uint64_t));
        ck(cudaMemcpy(pre_.p, pre.data(), pre.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "upload pre");
        for (auto& S : sets_) {
            ck(cudaStreamCreate(&S.stream), "cudaStreamCreate");
            ck(cudaEventCreate(&S.ev0), "cudaEventCreate");
            ck(cudaEventCreate(&S.ev1), "cudaEventCreate");
        }
    }
    ~CudaBackend() override {
        for (auto& S : sets_) {
            if (S.stream) cudaStreamDestroy(S.stream);
            if (S.ev0) cudaEventDestroy(S.ev0);
            if (S.ev1) cudaEventDestroy(S.ev1);
        }
    }

    int kind() const override { return SHIELD_BACKEND_CUDA; }

    void score(const std::string_view* texts, size_t n, int threads, float* logits,
               shield_stats* st) override {
        const double t0 = now_s();
        st->backend = SHIELD_BACKEND_CUDA;
        st->n_docs = n;
        if (n == 0) { st->t_total_s = now_s() - t0; return; }
        if (n > 0xFFFFFFF0ULL) throw std::runtime_error("ctxshield: CUDA batch too large");
        const size_t nf = fs_.names.size();
        acc_ = Acc{};
        size_t c = 0;
        for (size_t base = 0; base < n; base += kChunk, ++c) {
            Set& S = sets_[c & 1];
            if (S.busy) finish(S, logits);
            const size_t cnt = std::min(kChunk, n - base);
            const double ta = now_s();
            gpu_tokenize(fs_, delex_, texts + base * nf, cnt, threads, S.batch);
            S.h_arena.need(S.batch.arena_bytes);
            S.h_toks.need(S.batch.n_toks * sizeof(GpuTok));
            S.h_doctok.need((cnt + 1) * sizeof(uint32_t));
            gpu_pack(S.batch, static_cast<uint8_t*>(S.h_arena.p), static_cast<GpuTok*>(S.h_toks.p),
                     static_cast<uint32_t*>(S.h_doctok.p), threads);
            S.h_extras.need(S.batch.extras.size() * sizeof(float));
            std::memcpy(S.h_extras.p, S.batch.extras.data(), S.batch.extras.size() * sizeof(float));
            acc_.tok += now_s() - ta;
            acc_.tokens += S.batch.n_toks;
            acc_.bytes += S.batch.arena_bytes;
            const size_t outb = cnt * static_cast<size_t>(hpad_) * sizeof(float);
            S.d_arena.need(S.batch.arena_bytes);
            S.d_toks.need(S.batch.n_toks * sizeof(GpuTok));
            S.d_doctok.need((cnt + 1) * sizeof(uint32_t));
            S.d_extras.need(S.batch.extras.size() * sizeof(float));
            S.d_out.need(outb);
            S.d_flags.need(cnt * sizeof(uint32_t));
            S.h_out.need(outb);
            S.h_flags.need(cnt * sizeof(uint32_t));
            ck(cudaEventRecord(S.ev0, S.stream), "cudaEventRecord");
            ck(cudaMemcpyAsync(S.d_arena.p, S.h_arena.p, S.batch.arena_bytes, cudaMemcpyHostToDevice, S.stream), "H2D arena");
            ck(cudaMemcpyAsync(S.d_toks.p, S.h_toks.p, S.batch.n_toks * sizeof(GpuTok), cudaMemcpyHostToDevice, S.stream), "H2D toks");
            ck(cudaMemcpyAsync(S.d_doctok.p, S.h_doctok.p, (cnt + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, S.stream), "H2D doc_tok");
            ck(cudaMemcpyAsync(S.d_extras.p, S.h_extras.p, S.batch.extras.size() * sizeof(float), cudaMemcpyHostToDevice, S.stream), "H2D extras");
            ck(shield_cuda_score_tg(static_cast<const uint8_t*>(S.d_arena.p), S.d_toks.p,
                                    static_cast<const uint32_t*>(S.d_doctok.p), static_cast<const float*>(S.d_extras.p),
                                    static_cast<const float*>(W_.p), static_cast<const float*>(We_.p),
                                    static_cast<const float*>(b_.p), static_cast<const uint64_t*>(pre_.p),
                                    static_cast<uint32_t>(cnt), static_cast<uint32_t>(hpad_), static_cast<uint32_t>(ne_),
                                    static_cast<float*>(S.d_out.p), static_cast<uint32_t*>(S.d_flags.p), S.stream),
               "score_tg launch");
            ck(cudaMemcpyAsync(S.h_flags.p, S.d_flags.p, cnt * sizeof(uint32_t), cudaMemcpyDeviceToHost, S.stream), "D2H flags");
            ck(cudaEventRecord(S.ev1, S.stream), "cudaEventRecord");
            S.base = base;
            S.cnt = cnt;
            S.busy = true;
        }
        for (size_t k = 0; k < 2; ++k) {
            Set& S = sets_[(c + k) & 1];
            if (S.busy) finish(S, logits);
        }
        st->t_tokenize_s = acc_.tok;
        st->t_device_s = acc_.gpu;
        st->n_overflow_docs = acc_.ovf;
        st->n_tokens = acc_.tokens;
        st->n_arena_bytes = acc_.bytes;
        st->t_total_s = now_s() - t0;
    }

private:
    struct Set {
        cudaStream_t stream = nullptr;
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        HBuf h_arena, h_toks, h_doctok, h_extras, h_out, h_flags;
        DBuf d_arena, d_toks, d_doctok, d_extras, d_out, d_flags;
        GpuBatch batch;
        size_t base = 0, cnt = 0;
        bool busy = false;
    };

    void finish(Set& S, float* logits) {
        ck(cudaStreamSynchronize(S.stream), "score_tg");
        float ms = 0.f;
        ck(cudaEventElapsedTime(&ms, S.ev0, S.ev1), "cudaEventElapsedTime");
        acc_.gpu += ms / 1000.0;
        S.busy = false;
        const uint32_t* fl = static_cast<const uint32_t*>(S.h_flags.p);
        std::vector<uint32_t> ovf;
        for (size_t i = 0; i < S.cnt; ++i) if (fl[i]) ovf.push_back(static_cast<uint32_t>(i));
        acc_.ovf += ovf.size();
        size_t k = 0;
        while (k < ovf.size()) {
            std::vector<uint32_t> docs, sizes;
            std::vector<uint64_t> offs;
            size_t total = 0;
            while (k < ovf.size()) {
                size_t cap = 1024;
                const size_t adds = S.batch.doc_adds[ovf[k]];
                while (cap < 2 * adds) cap <<= 1;
                if (!docs.empty() && total + cap > kDevSlotsPerPass) break;
                docs.push_back(ovf[k]);
                offs.push_back(total);
                sizes.push_back(static_cast<uint32_t>(cap));
                total += cap;
                ++k;
            }
            dk_.need(total * sizeof(uint32_t));
            dc_.need(total * sizeof(int32_t));
            odocs_.need(docs.size() * sizeof(uint32_t));
            ooffs_.need(offs.size() * sizeof(uint64_t));
            osizes_.need(sizes.size() * sizeof(uint32_t));
            ck(cudaMemsetAsync(dk_.p, 0, total * sizeof(uint32_t), S.stream), "memset K");
            ck(cudaMemsetAsync(dc_.p, 0, total * sizeof(int32_t), S.stream), "memset C");
            ck(cudaMemcpyAsync(odocs_.p, docs.data(), docs.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, S.stream), "H2D ovf docs");
            ck(cudaMemcpyAsync(ooffs_.p, offs.data(), offs.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, S.stream), "H2D ovf offs");
            ck(cudaMemcpyAsync(osizes_.p, sizes.data(), sizes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, S.stream), "H2D ovf sizes");
            ck(shield_cuda_score_dev(static_cast<const uint8_t*>(S.d_arena.p), S.d_toks.p,
                                     static_cast<const uint32_t*>(S.d_doctok.p), static_cast<const float*>(S.d_extras.p),
                                     static_cast<const float*>(W_.p), static_cast<const float*>(We_.p),
                                     static_cast<const float*>(b_.p), static_cast<const uint64_t*>(pre_.p),
                                     static_cast<uint32_t>(hpad_), static_cast<uint32_t>(ne_),
                                     static_cast<float*>(S.d_out.p), static_cast<uint32_t*>(dk_.p),
                                     static_cast<int32_t*>(dc_.p), static_cast<const uint32_t*>(odocs_.p),
                                     static_cast<const uint64_t*>(ooffs_.p), static_cast<const uint32_t*>(osizes_.p),
                                     static_cast<uint32_t>(docs.size()), S.stream),
               "score_dev launch");
            ck(cudaStreamSynchronize(S.stream), "score_dev");
        }
        const size_t outb = S.cnt * static_cast<size_t>(hpad_) * sizeof(float);
        ck(cudaMemcpyAsync(S.h_out.p, S.d_out.p, outb, cudaMemcpyDeviceToHost, S.stream), "D2H logits");
        ck(cudaStreamSynchronize(S.stream), "D2H logits");
        std::memcpy(logits + S.base * static_cast<size_t>(hpad_), S.h_out.p, outb);
    }

    FieldSet fs_;
    bool delex_;
    int hpad_;
    int ne_;
    DBuf W_, We_, b_, pre_, dk_, dc_, odocs_, ooffs_, osizes_;
    Set sets_[2];
    struct Acc { double tok = 0, gpu = 0; size_t ovf = 0, tokens = 0, bytes = 0; } acc_;
};

}  // namespace

bool cuda_built() { return true; }

bool cuda_available(std::string& why) {
#if !SHIELD_CUDA_VERIFIED
    const char* ok = std::getenv("SHIELD_CUDA_UNVERIFIED_OK");
    if (ok == nullptr || std::string(ok) != "true") {
        why = "UNTESTED — no NVIDIA device verification yet (backends.cuda.verified=false); set "
              "SHIELD_CUDA_UNVERIFIED_OK=true only for the L4 verification run (native/cuda/run_on_l4.sh)";
        return false;
    }
#endif
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) { why = std::string("cudaGetDeviceCount: ") + cudaGetErrorString(e); return false; }
    if (n == 0) { why = "no CUDA device"; return false; }
    cudaDeviceProp p;
    const cudaError_t e2 = cudaGetDeviceProperties(&p, 0);
    if (e2 != cudaSuccess) { why = std::string("cudaGetDeviceProperties: ") + cudaGetErrorString(e2); return false; }
    why = std::string("CUDA device: ") + p.name + " sm_" + std::to_string(p.major) + std::to_string(p.minor);
    return true;
}

std::unique_ptr<Backend> make_cuda_backend(const GroupWeights& g) { return std::make_unique<CudaBackend>(g); }

}  // namespace ctxshield
