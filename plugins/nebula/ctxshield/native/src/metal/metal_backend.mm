// Metal host for ctxshield (build doc §9.3). Unified memory: every buffer is MTLStorageModeShared
// (the CPU writes token spans straight into GPU-visible memory; no blit). Weights are uploaded once
// per scorer, head-interleaved exactly as on the CPU (W[idx][hpad]). The metallib is embedded in
// the extension (metallib_data.h, generated at build time from shield.metal).
//
// Per call: CPU tokenization (all P-cores) → score_tg (one threadgroup per document, on-chip table)
// → the documents whose on-chip table overflowed are re-run by score_dev with device-memory tables
// → logits. The backend that ran is reported in shield_stats; any Metal error raises.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "metallib_data.h"
#include "shield.h"

namespace ctxshield {

namespace {

struct MetalParams {   // == Params in shield.metal
    uint32_t n_docs;
    uint32_t hpad;
    uint32_t ne;
    uint32_t limit;
    uint32_t base;
    uint32_t pad0, pad1, pad2;
};

constexpr uint32_t kSlots = 7600;                 // score_tg_7600: 30,400 B of threadgroup memory
constexpr uint32_t kLimit = kSlots * 3 / 4;       // claim limit before a document overflows
constexpr uint32_t kTgThreads = 512;              // ≤ 32 simdgroups (kMaxSimdTg in the kernel)
constexpr uint32_t kMaxGridChunk = 65535;         // documents per dispatch
constexpr size_t kDevSlotsPerPass = size_t(1) << 25;   // device tables per overflow pass (256 MB)
constexpr size_t kChunk = 16384;                  // documents per pipelined GPU chunk

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string ns_err(NSError* e) {
    if (e == nil) return "unknown Metal error";
    return std::string([[e localizedDescription] UTF8String]);
}

class MetalBackend final : public Backend {
public:
    explicit MetalBackend(const GroupWeights& g)
        : fs_(g.names), delex_(g.delex), hpad_(g.hpad), ne_(fs_.n_extras()) {
        dev_ = MTLCreateSystemDefaultDevice();
        if (dev_ == nil) throw std::runtime_error("ctxshield: no Metal device");
        queue_ = [dev_ newCommandQueue];
        if (queue_ == nil) throw std::runtime_error("ctxshield: Metal newCommandQueue failed");
        dispatch_data_t data = dispatch_data_create(kShieldMetallib, kShieldMetallibSize, nullptr,
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError* err = nil;
        id<MTLLibrary> lib = [dev_ newLibraryWithData:data error:&err];
        if (lib == nil) throw std::runtime_error("ctxshield: Metal library load failed: " + ns_err(err));
        ptg_ = pipeline(lib, "score_tg_7600");
        pdev_ = pipeline(lib, "score_dev");
        if (ptg_.maxTotalThreadsPerThreadgroup < kTgThreads)
            throw std::runtime_error("ctxshield: score_tg needs 256 threads per threadgroup, device allows " +
                                     std::to_string(ptg_.maxTotalThreadsPerThreadgroup));
        dev_threads_ = std::min<NSUInteger>(1024, pdev_.maxTotalThreadsPerThreadgroup) / 32 * 32;
        if (dev_threads_ == 0) throw std::runtime_error("ctxshield: score_dev pipeline has no threads");
        const size_t wbytes = static_cast<size_t>(kDim) * static_cast<size_t>(hpad_) * sizeof(float);
        W_ = [dev_ newBufferWithBytes:g.W length:wbytes options:MTLResourceStorageModeShared];
        We_ = [dev_ newBufferWithBytes:g.We length:static_cast<size_t>(ne_) * hpad_ * sizeof(float)
                               options:MTLResourceStorageModeShared];
        b_ = [dev_ newBufferWithBytes:g.b length:static_cast<size_t>(hpad_) * sizeof(float)
                              options:MTLResourceStorageModeShared];
        std::vector<uint64_t> pre;
        for (const auto& p : fs_.pre) pre.insert(pre.end(), p.begin(), p.end());
        pre_ = [dev_ newBufferWithBytes:pre.data() length:pre.size() * sizeof(uint64_t)
                                options:MTLResourceStorageModeShared];
        if (W_ == nil || We_ == nil || b_ == nil || pre_ == nil)
            throw std::runtime_error("ctxshield: Metal weight buffer allocation failed");
    }

    int kind() const override { return SHIELD_BACKEND_METAL; }

    void score(const std::string_view* texts, size_t n, int threads, float* logits,
               shield_stats* st) override {
        @autoreleasepool {
            score_impl(texts, n, threads, logits, st);
        }
    }

private:
    // One in-flight chunk per buffer set: while the GPU runs chunk c, the CPU tokenizes chunk c+1
    // into the other set (double buffering). A set is finished (wait → overflow pass → logits)
    // before it is reused.
    struct BufSet {
        id<MTLBuffer> arena = nil, toks = nil, doctok = nil, extras = nil, out = nil, flags = nil;
        id<MTLCommandBuffer> cb = nil;
        GpuBatch batch;
        size_t base = 0, cnt = 0;
        bool busy = false;
    };

    void score_impl(const std::string_view* texts, size_t n, int threads, float* logits,
                    shield_stats* st) {
        const double t0 = now_s();
        st->backend = SHIELD_BACKEND_METAL;
        st->n_docs = n;
        if (n == 0) { st->t_total_s = now_s() - t0; return; }
        if (n > 0xFFFFFFF0ULL) throw std::runtime_error("ctxshield: Metal batch too large");
        const size_t nf = fs_.names.size();
        acc_ = Acc{};
        size_t c = 0;
        for (size_t base = 0; base < n; base += kChunk, ++c) {
            BufSet& S = sets_[c & 1];
            if (S.busy) finish(S, logits);
            const size_t cnt = std::min(kChunk, n - base);
            const double ta = now_s();
            gpu_tokenize(fs_, delex_, texts + base * nf, cnt, threads, S.batch);
            grow(S.arena, S.batch.arena_bytes);
            grow(S.toks, S.batch.n_toks * sizeof(GpuTok));
            grow(S.doctok, (cnt + 1) * sizeof(uint32_t));
            gpu_pack(S.batch, static_cast<uint8_t*>([S.arena contents]),
                     static_cast<GpuTok*>([S.toks contents]), static_cast<uint32_t*>([S.doctok contents]),
                     threads);
            put(S.extras, S.batch.extras.data(), S.batch.extras.size() * sizeof(float), 16);
            grow(S.out, cnt * static_cast<size_t>(hpad_) * sizeof(float));
            grow(S.flags, cnt * sizeof(uint32_t));
            acc_.tok += now_s() - ta;
            acc_.tokens += S.batch.n_toks;
            acc_.bytes += S.batch.arena_bytes;
            S.cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [S.cb computeCommandEncoder];
            [enc setComputePipelineState:ptg_];
            bind_set(enc, S);
            [enc setBuffer:S.flags offset:0 atIndex:10];
            for (size_t b0 = 0; b0 < cnt; b0 += kMaxGridChunk) {
                const size_t m = std::min<size_t>(kMaxGridChunk, cnt - b0);
                MetalParams P{static_cast<uint32_t>(cnt), static_cast<uint32_t>(hpad_),
                              static_cast<uint32_t>(ne_), kLimit, static_cast<uint32_t>(b0), 0, 0, 0};
                [enc setBytes:&P length:sizeof(P) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(m, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kTgThreads, 1, 1)];
            }
            [enc endEncoding];
            [S.cb commit];
            S.base = base;
            S.cnt = cnt;
            S.busy = true;
        }
        for (size_t k = 0; k < 2; ++k) {   // finish in submission order
            BufSet& S = sets_[(c + k) & 1];
            if (S.busy) finish(S, logits);
        }
        st->t_tokenize_s = acc_.tok;
        st->t_device_s = acc_.gpu;
        st->n_overflow_docs = acc_.ovf;
        st->n_tokens = acc_.tokens;
        st->n_arena_bytes = acc_.bytes;
        st->t_total_s = now_s() - t0;
    }

    void finish(BufSet& S, float* logits) {
        [S.cb waitUntilCompleted];
        check(S.cb, "score_tg");
        acc_.gpu += S.cb.GPUEndTime - S.cb.GPUStartTime;
        S.cb = nil;
        S.busy = false;
        // documents whose on-chip table overflowed → device-memory tables
        const uint32_t* fl = static_cast<const uint32_t*>([S.flags contents]);
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
            grow(dk_, total * sizeof(uint32_t));
            grow(dc_, total * sizeof(int32_t));
            std::memset([dk_ contents], 0, total * sizeof(uint32_t));
            std::memset([dc_ contents], 0, total * sizeof(int32_t));
            put(odocs_, docs.data(), docs.size() * sizeof(uint32_t), 16);
            put(ooffs_, offs.data(), offs.size() * sizeof(uint64_t), 16);
            put(osizes_, sizes.data(), sizes.size() * sizeof(uint32_t), 16);
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pdev_];
            bind_set(enc, S);
            MetalParams P{static_cast<uint32_t>(S.cnt), static_cast<uint32_t>(hpad_),
                          static_cast<uint32_t>(ne_), 0, 0, 0, 0, 0};
            [enc setBytes:&P length:sizeof(P) atIndex:8];
            [enc setBuffer:dk_ offset:0 atIndex:10];
            [enc setBuffer:dc_ offset:0 atIndex:11];
            [enc setBuffer:odocs_ offset:0 atIndex:12];
            [enc setBuffer:ooffs_ offset:0 atIndex:13];
            [enc setBuffer:osizes_ offset:0 atIndex:14];
            [enc dispatchThreadgroups:MTLSizeMake(docs.size(), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(dev_threads_, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            check(cb, "score_dev");
            acc_.gpu += cb.GPUEndTime - cb.GPUStartTime;
        }
        std::memcpy(logits + S.base * static_cast<size_t>(hpad_), [S.out contents],
                    S.cnt * static_cast<size_t>(hpad_) * sizeof(float));
    }

    void bind_set(id<MTLComputeCommandEncoder> enc, const BufSet& S) {
        [enc setBuffer:S.arena offset:0 atIndex:0];
        [enc setBuffer:S.toks offset:0 atIndex:1];
        [enc setBuffer:S.doctok offset:0 atIndex:2];
        [enc setBuffer:S.extras offset:0 atIndex:3];
        [enc setBuffer:W_ offset:0 atIndex:4];
        [enc setBuffer:We_ offset:0 atIndex:5];
        [enc setBuffer:b_ offset:0 atIndex:6];
        [enc setBuffer:pre_ offset:0 atIndex:7];
        [enc setBuffer:S.out offset:0 atIndex:9];
    }

    id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, const char* name) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (fn == nil) throw std::runtime_error(std::string("ctxshield: Metal function missing: ") + name);
        NSError* err = nil;
        id<MTLComputePipelineState> p = [dev_ newComputePipelineStateWithFunction:fn error:&err];
        if (p == nil) throw std::runtime_error(std::string("ctxshield: Metal pipeline ") + name + ": " + ns_err(err));
        return p;
    }
    void grow(id<MTLBuffer> __strong& buf, size_t bytes) {
        bytes = std::max<size_t>(bytes, 16);
        if (buf != nil && [buf length] >= bytes) return;
        buf = [dev_ newBufferWithLength:bytes + bytes / 4 options:MTLResourceStorageModeShared];
        if (buf == nil) throw std::runtime_error("ctxshield: Metal buffer allocation failed (" + std::to_string(bytes) + " B)");
    }
    void put(id<MTLBuffer> __strong& buf, const void* src, size_t bytes, size_t min_bytes) {
        grow(buf, std::max(bytes, min_bytes));
        if (bytes) std::memcpy([buf contents], src, bytes);
    }
    static void check(id<MTLCommandBuffer> cb, const char* what) {
        if (cb.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error(std::string("ctxshield: Metal ") + what + " failed: " + ns_err(cb.error));
    }

    FieldSet fs_;
    bool delex_;
    int hpad_;
    int ne_;
    id<MTLDevice> dev_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> ptg_ = nil, pdev_ = nil;
    NSUInteger dev_threads_ = 0;
    id<MTLBuffer> W_ = nil, We_ = nil, b_ = nil, pre_ = nil;
    id<MTLBuffer> dk_ = nil, dc_ = nil, odocs_ = nil, ooffs_ = nil, osizes_ = nil;
    BufSet sets_[2];
    struct Acc { double tok = 0, gpu = 0; size_t ovf = 0, tokens = 0, bytes = 0; } acc_;
};

}  // namespace

bool metal_built() { return true; }

bool metal_available(std::string& why) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (d == nil) { why = "no Metal device on this machine"; return false; }
        if (![d supportsFamily:MTLGPUFamilyMetal3]) { why = "Metal device lacks the Metal 3 family"; return false; }
        why = std::string("Metal device: ") + [[d name] UTF8String];
        return true;
    }
}

std::unique_ptr<Backend> make_metal_backend(const GroupWeights& g) {
    @autoreleasepool {
        return std::make_unique<MetalBackend>(g);
    }
}

}  // namespace ctxshield
