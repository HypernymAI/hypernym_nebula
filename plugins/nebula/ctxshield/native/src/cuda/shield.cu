// ctxshield CUDA kernels (build doc §9.4) — UNTESTED: no NVIDIA device on the build machine.
// Line-for-line port of src/metal/shield.metal, whose parity vs the CPU IS tested (≤ 1e-5):
//   * one block per document; w1/w2 one token per thread, char n-gram windows parallel over the
//     document's arena positions (contiguous range per thread, one binary search, walk forward);
//   * score_tg: shared-memory open-addressing table of kCudaSlots packed u32 slots (index+1 in bits
//     0..18, signed count in bits 19..31); > 3/4 claimed or a count leaving ±4095 → flags[doc] = 1
//     and the host re-runs that document with score_dev (global-memory table, u32 key + i32 count);
//   * __ldg float4 weight gathers (head-interleaved W[idx][hpad]); warp-shuffle reductions.
// Target sm_89 (NVIDIA L4). Verified only by native/cuda/run_on_l4.sh (golden_v1.bin ≤ 1e-5).
#include <cstdint>

#include "shield_cuda.h"

namespace {

struct GpuTok {   // == ctxshield::GpuTok (src/backend.h)
    uint32_t a;
    uint32_t len;
    uint32_t w2_end;
    uint32_t meta;   // field | ph << 8 | ascii << 9
};
static_assert(sizeof(GpuTok) == 16, "GpuTok layout");

constexpr uint64_t kFnvPrime = 0x100000001b3ULL;
constexpr uint32_t kMask18 = (1u << 18) - 1u;
constexpr int kMaxWarps = 32;

__device__ __forceinline__ uint64_t fnv_byte(uint64_t h, uint8_t b) { return (h ^ b) * kFnvPrime; }
__device__ __forceinline__ uint64_t ldg_u64(const uint64_t* p) {
    return static_cast<uint64_t>(__ldg(reinterpret_cast<const unsigned long long*>(p)));
}

struct SharedIns {
    uint32_t* T;
    uint32_t* claimed;
    uint32_t* ovf;
    uint32_t limit;
    __device__ bool insert(uint64_t h) {
        const uint32_t idx = static_cast<uint32_t>(h) & kMask18;
        const uint32_t key = idx + 1u;
        const bool neg = (h >> 63) != 0ULL;
        const uint32_t delta = neg ? 0xFFF80000u : 0x00080000u;
        uint32_t s = idx % kCudaSlots;
        for (uint32_t probe = 0; probe < kCudaSlots; ++probe) {
            uint32_t v = *reinterpret_cast<volatile uint32_t*>(&T[s]);
            if (v == 0u) {
                if (*reinterpret_cast<volatile uint32_t*>(claimed) >= limit) {
                    atomicExch(ovf, 1u);
                    return false;
                }
                const uint32_t old = atomicCAS(&T[s], 0u, key | delta);
                if (old == 0u) {
                    atomicAdd(claimed, 1u);
                    return true;
                }
                v = old;
            }
            if ((v & 0x7FFFFu) == key) {
                const uint32_t old = atomicAdd(&T[s], delta);
                const int oc = static_cast<int>(old) >> 19;
                if ((!neg && oc == 4095) || (neg && oc == -4096)) {
                    atomicExch(ovf, 1u);
                    return false;
                }
                return true;
            }
            s = (s + 1u == kCudaSlots) ? 0u : s + 1u;
        }
        atomicExch(ovf, 1u);
        return false;
    }
};

struct GlobalIns {
    uint32_t* K;
    int32_t* C;
    uint32_t mask;
    __device__ bool insert(uint64_t h) {
        const uint32_t idx = static_cast<uint32_t>(h) & kMask18;
        const uint32_t key = idx + 1u;
        const int d = (h >> 63) != 0ULL ? -1 : 1;
        uint32_t s = idx & mask;
        for (;;) {
            uint32_t v = *reinterpret_cast<volatile uint32_t*>(&K[s]);
            if (v == 0u) {
                const uint32_t old = atomicCAS(&K[s], 0u, key);
                if (old == 0u) {
                    atomicAdd(&C[s], d);
                    return true;
                }
                v = old;
            }
            if (v == key) {
                atomicAdd(&C[s], d);
                return true;
            }
            s = (s + 1u) & mask;
        }
    }
};

template <typename Ins>
__device__ bool hash_words(const uint8_t* __restrict__ A, const GpuTok k, const uint64_t* __restrict__ pre,
                           Ins& ins) {
    const uint32_t f = k.meta & 0xFFu;
    uint64_t h = ldg_u64(pre + f * 3u);
    for (uint32_t i = k.a; i < k.a + k.len; ++i) h = fnv_byte(h, __ldg(A + i));
    if (!ins.insert(h)) return false;
    if (k.w2_end != 0u) {
        h = ldg_u64(pre + f * 3u + 1u);
        for (uint32_t i = k.a; i < k.w2_end; ++i) h = fnv_byte(h, __ldg(A + i));
        if (!ins.insert(h)) return false;
    }
    return true;
}

template <typename Ins>
__device__ bool hash_window(const uint8_t* __restrict__ A, const GpuTok k, uint32_t s, uint64_t pc,
                            Ins& ins) {
    const uint32_t end = k.a + k.len + 1u;
    if ((k.meta >> 9) & 1u) {
        uint64_t g = fnv_byte(fnv_byte(fnv_byte(pc, __ldg(A + s)), __ldg(A + s + 1u)), __ldg(A + s + 2u));
        if (!ins.insert(g)) return false;
        if (s + 4u <= end) {
            g = fnv_byte(g, __ldg(A + s + 3u));
            if (!ins.insert(g)) return false;
            if (s + 5u <= end) {
                g = fnv_byte(g, __ldg(A + s + 4u));
                if (!ins.insert(g)) return false;
            }
        }
        return true;
    }
    if ((__ldg(A + s) & 0xC0u) == 0x80u) return true;
    uint64_t g = pc;
    uint32_t pos = s;
    for (uint32_t n = 1u; n <= 5u; ++n) {
        do {
            g = fnv_byte(g, __ldg(A + pos));
            ++pos;
        } while (pos < end && (__ldg(A + pos) & 0xC0u) == 0x80u);
        if (n >= 3u && !ins.insert(g)) return false;
        if (pos >= end) break;
    }
    return true;
}

__device__ __forceinline__ GpuTok load_tok(const GpuTok* toks, uint32_t t) {
    const uint4 v = __ldg(reinterpret_cast<const uint4*>(toks) + t);
    return GpuTok{v.x, v.y, v.z, v.w};
}

template <typename Ins>
__device__ void hash_doc(const uint8_t* __restrict__ A, const GpuTok* __restrict__ toks, uint32_t t0,
                         uint32_t t1, const uint64_t* __restrict__ pre, Ins& ins) {
    if (t1 == t0) return;
    const uint32_t tid = threadIdx.x, tgs = blockDim.x;
    for (uint32_t t = t0 + tid; t < t1; t += tgs)
        if (!hash_words(A, load_tok(toks, t), pre, ins)) return;
    const uint32_t lo = load_tok(toks, t0).a - 1u;
    const GpuTok last = load_tok(toks, t1 - 1u);
    const uint32_t hi = last.a + last.len - 1u;
    const uint32_t chunk = (hi - lo + tgs - 1u) / tgs;
    const uint32_t p0 = lo + tid * chunk;
    if (p0 >= hi) return;
    const uint32_t p1 = min(hi, p0 + chunk);
    uint32_t L = t0, R = t1;
    while (R - L > 1u) {
        const uint32_t m = (L + R) >> 1;
        if (load_tok(toks, m).a - 1u <= p0) L = m; else R = m;
    }
    uint32_t t = L;
    GpuTok k = load_tok(toks, t);
    uint64_t pc = ldg_u64(pre + (k.meta & 0xFFu) * 3u + 2u);
    for (uint32_t p = p0; p < p1; ++p) {
        while (t + 1u < t1 && load_tok(toks, t + 1u).a - 1u <= p) {
            ++t;
            k = load_tok(toks, t);
            pc = ldg_u64(pre + (k.meta & 0xFFu) * 3u + 2u);
        }
        if (p + 1u >= k.a + k.len) continue;
        if ((k.meta >> 8) & 1u) continue;
        if (!hash_window(A, k, p, pc, ins)) return;
    }
}

__device__ __forceinline__ void contribute(int c, uint32_t idx, const float4* __restrict__ W, uint32_t L,
                                           float& n2, float4* acc) {
    const float x = (c > 0 ? 1.0f : -1.0f) * (1.0f + logf(static_cast<float>(c > 0 ? c : -c)));
    n2 += x * x;
    const float4* w = W + static_cast<size_t>(idx) * L;
    for (uint32_t l = 0; l < L; ++l) {
        const float4 q = __ldg(w + l);
        acc[l].x += x * q.x;
        acc[l].y += x * q.y;
        acc[l].z += x * q.z;
        acc[l].w += x * q.w;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// warp + block reduction; logits = b + acc/‖x‖ + extras·We (threads 0 .. hpad-1)
__device__ void reduce_write(float n2, float4* acc, uint32_t L, float* part, uint32_t doc,
                             const float* __restrict__ extras, const float* __restrict__ We,
                             const float* __restrict__ b, uint32_t hpad, uint32_t ne,
                             float* __restrict__ out) {
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5, nw = (blockDim.x + 31u) >> 5;
    n2 = warp_sum(n2);
    float a[16];
    for (uint32_t l = 0; l < 4; ++l) {
        a[4 * l + 0] = l < L ? warp_sum(acc[l].x) : 0.f;
        a[4 * l + 1] = l < L ? warp_sum(acc[l].y) : 0.f;
        a[4 * l + 2] = l < L ? warp_sum(acc[l].z) : 0.f;
        a[4 * l + 3] = l < L ? warp_sum(acc[l].w) : 0.f;
    }
    if (lane == 0u) {
        part[warp * 17u] = n2;
        for (uint32_t j = 0; j < 16u; ++j) part[warp * 17u + 1u + j] = a[j];
    }
    __syncthreads();
    if (tid < hpad) {
        float nn = 0.f, s = 0.f;
        for (uint32_t w = 0; w < nw; ++w) {
            nn += part[w * 17u];
            s += part[w * 17u + 1u + tid];
        }
        const float inv = nn > 0.f ? 1.0f / sqrtf(nn) : 1.0f;
        float o = __ldg(b + tid) + s * inv;
        for (uint32_t e = 0; e < ne; ++e) o += __ldg(extras + static_cast<size_t>(doc) * ne + e) * __ldg(We + e * hpad + tid);
        out[static_cast<size_t>(doc) * hpad + tid] = o;
    }
}

__global__ void __launch_bounds__(kCudaBlock)
score_tg(const uint8_t* __restrict__ A, const GpuTok* __restrict__ toks,
         const uint32_t* __restrict__ doc_tok, const float* __restrict__ extras,
         const float4* __restrict__ W, const float* __restrict__ We, const float* __restrict__ b,
         const uint64_t* __restrict__ pre, uint32_t hpad, uint32_t ne, float* __restrict__ out,
         uint32_t* __restrict__ flags) {
    extern __shared__ uint32_t smem[];
    uint32_t* T = smem;
    uint32_t* claimed = smem + kCudaSlots;
    uint32_t* ovf = smem + kCudaSlots + 1;
    float* part = reinterpret_cast<float*>(smem + kCudaSlots + 2);
    const uint32_t doc = blockIdx.x, tid = threadIdx.x;
    for (uint32_t s = tid; s < kCudaSlots; s += blockDim.x) T[s] = 0u;
    if (tid == 0u) { *claimed = 0u; *ovf = 0u; }
    __syncthreads();
    SharedIns ins{T, claimed, ovf, kCudaLimit};
    hash_doc(A, toks, __ldg(doc_tok + doc), __ldg(doc_tok + doc + 1u), pre, ins);
    __syncthreads();
    const bool overflowed = *reinterpret_cast<volatile uint32_t*>(ovf) != 0u;
    if (tid == 0u) flags[doc] = overflowed ? 1u : 0u;
    if (overflowed) return;   // uniform across the block
    const uint32_t L = hpad / 4u;
    float n2 = 0.f;
    float4 acc[4] = {make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f),
                     make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)};
    for (uint32_t s = tid; s < kCudaSlots; s += blockDim.x) {
        const uint32_t v = T[s];
        if (v == 0u) continue;
        const int c = static_cast<int>(v) >> 19;
        if (c == 0) continue;
        contribute(c, (v & 0x7FFFFu) - 1u, W, L, n2, acc);
    }
    reduce_write(n2, acc, L, part, doc, extras, We, b, hpad, ne, out);
}

__global__ void __launch_bounds__(kCudaDevBlock)
score_dev(const uint8_t* __restrict__ A, const GpuTok* __restrict__ toks,
          const uint32_t* __restrict__ doc_tok, const float* __restrict__ extras,
          const float4* __restrict__ W, const float* __restrict__ We, const float* __restrict__ b,
          const uint64_t* __restrict__ pre, uint32_t hpad, uint32_t ne, float* __restrict__ out,
          uint32_t* K, int32_t* C, const uint32_t* __restrict__ ovf_docs,
          const uint64_t* __restrict__ ovf_off, const uint32_t* __restrict__ ovf_size) {
    __shared__ float part[kMaxWarps * 17];
    const uint32_t g = blockIdx.x, tid = threadIdx.x;
    const uint32_t doc = ovf_docs[g];
    const uint64_t off = ovf_off[g];
    const uint32_t size = ovf_size[g];
    uint32_t* Kd = K + off;
    int32_t* Cd = C + off;
    GlobalIns ins{Kd, Cd, size - 1u};
    hash_doc(A, toks, doc_tok[doc], doc_tok[doc + 1u], pre, ins);
    __threadfence_block();
    __syncthreads();
    const uint32_t L = hpad / 4u;
    float n2 = 0.f;
    float4 acc[4] = {make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f),
                     make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)};
    for (uint32_t s = tid; s < size; s += blockDim.x) {
        const uint32_t key = *reinterpret_cast<volatile uint32_t*>(&Kd[s]);
        if (key == 0u) continue;
        const int c = *reinterpret_cast<volatile int32_t*>(&Cd[s]);
        if (c == 0) continue;
        contribute(c, key - 1u, W, L, n2, acc);
    }
    reduce_write(n2, acc, L, part, doc, extras, We, b, hpad, ne, out);
}

}  // namespace

extern "C" {

size_t shield_cuda_tg_smem_bytes(void) {
    return (static_cast<size_t>(kCudaSlots) + 2u) * sizeof(uint32_t) + static_cast<size_t>(kMaxWarps) * 17u * sizeof(float);
}

cudaError_t shield_cuda_prepare(void) {
    // L4 / Ada: up to 99 KB dynamic shared memory per block on opt-in (build doc §9.4)
    return cudaFuncSetAttribute(score_tg, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(shield_cuda_tg_smem_bytes()));
}

cudaError_t shield_cuda_score_tg(const uint8_t* A, const void* toks, const uint32_t* doc_tok,
                                 const float* extras, const float* W, const float* We,
                                 const float* b, const uint64_t* pre, uint32_t n_docs,
                                 uint32_t hpad, uint32_t ne, float* out, uint32_t* flags,
                                 cudaStream_t stream) {
    if (n_docs == 0u) return cudaSuccess;
    score_tg<<<n_docs, kCudaBlock, shield_cuda_tg_smem_bytes(), stream>>>(
        A, static_cast<const GpuTok*>(toks), doc_tok, extras, reinterpret_cast<const float4*>(W), We, b,
        pre, hpad, ne, out, flags);
    return cudaGetLastError();
}

cudaError_t shield_cuda_score_dev(const uint8_t* A, const void* toks, const uint32_t* doc_tok,
                                  const float* extras, const float* W, const float* We,
                                  const float* b, const uint64_t* pre, uint32_t hpad, uint32_t ne,
                                  float* out, uint32_t* K, int32_t* C, const uint32_t* ovf_docs,
                                  const uint64_t* ovf_off, const uint32_t* ovf_size,
                                  uint32_t n_ovf, cudaStream_t stream) {
    if (n_ovf == 0u) return cudaSuccess;
    score_dev<<<n_ovf, kCudaDevBlock, 0, stream>>>(
        A, static_cast<const GpuTok*>(toks), doc_tok, extras, reinterpret_cast<const float4*>(W), We, b,
        pre, hpad, ne, out, K, C, ovf_docs, ovf_off, ovf_size);
    return cudaGetLastError();
}

}  // extern "C"
