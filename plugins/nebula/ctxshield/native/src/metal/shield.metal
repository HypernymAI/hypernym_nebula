// ctxshield Metal kernels (build doc §9.3). The CPU emits token spans (tokenization with Unicode
// categories stays on the CPU: branchy, cheap); the GPU hashes every w1 / w2 / char n-gram of the
// tokens and accumulates them, then finalizes (sublinear + L2) and runs the head-interleaved
// all-heads gather-dot — ONE threadgroup per document.
//
// §9.3 lists Kernel A (hash per token) and Kernel B (accumulate + score) separately; here they are
// fused into one kernel per path (the hashes never leave the threadgroup — no intermediate buffer).
//
//   score_tg<NS>  threadgroup-memory open-addressing table of NS packed slots:
//                 bits 0..18 = feature index + 1 (0 = empty), bits 19..31 = signed count (±4095).
//                 A document that claims more than 3/4 of the slots, or whose count would leave
//                 ±4095, sets its overflow flag and writes nothing; the host re-runs exactly those
//                 documents with score_dev (counted, reported as n_overflow_docs).
//   score_dev     device-memory table per overflowed document (u32 key + i32 count, ≥ 2× the
//                 document's exact insertion count, power of two).
//
// Mirrors src/featurize.cpp (hashing) + src/score.cpp (dot). Parity vs the CPU: ≤ 1e-5 on
// probabilities (float32 log / sqrt / re-associated sums), measured by ctxshield.parity --backend metal.
#include <metal_stdlib>
using namespace metal;

struct GpuTok {        // == ctxshield::GpuTok (src/backend.h)
    uint a;            // arena offset of the token
    uint len;          // bytes
    uint w2_end;       // arena end of the w2 payload, 0 = none
    uint meta;         // field | ph << 8 | ascii << 9
};

struct Params {        // == MetalParams (metal_backend.mm)
    uint n_docs;
    uint hpad;
    uint ne;
    uint limit;        // score_tg: max claimed slots
    uint base;         // score_tg: document index of threadgroup 0 of this dispatch
    uint pad0, pad1, pad2;
};

constant ulong kFnvPrime = 0x100000001b3UL;
constant uint kMask18 = (1u << 18) - 1u;
constant uint kMaxSimdTg = 32;     // score_tg threadgroups ≤ 1024 threads
constant uint kMaxSimdDev = 32;    // score_dev threadgroups ≤ 1024 threads

inline ulong fnv_byte(ulong h, uchar b) { return (h ^ ulong(b)) * kFnvPrime; }

// The insertion hashes of one document, split for load balance (a thread per token would wait on
// the longest token of the document): (1) w1 + w2, one token per thread; (2) the char n-gram
// windows, parallel over the doc's arena POSITIONS — each thread takes a contiguous range of
// window starts, binary-searches its first token once, then walks forward. Same multiset of hashes
// as the CPU (order is irrelevant to the counts).
template <typename Ins>
inline bool hash_words(device const uchar* A, GpuTok k, constant ulong* pre, thread Ins& ins) {
    const uint f = k.meta & 0xFFu;
    ulong h = pre[f * 3u];
    for (uint i = k.a; i < k.a + k.len; ++i) h = fnv_byte(h, A[i]);
    if (!ins.insert(h)) return false;
    if (k.w2_end != 0u) {
        h = pre[f * 3u + 1u];
        for (uint i = k.a; i < k.w2_end; ++i) h = fnv_byte(h, A[i]);
        if (!ins.insert(h)) return false;
    }
    return true;
}

// c3 → c4 → c5 of the window starting at arena position s of token k (' ' + tok + ' ' = [a-1, a+len+1))
template <typename Ins>
inline bool hash_window(device const uchar* A, GpuTok k, uint s, ulong pc, thread Ins& ins) {
    const uint end = k.a + k.len + 1u;
    if ((k.meta >> 9) & 1u) {   // ASCII: bytes are code points
        ulong g = fnv_byte(fnv_byte(fnv_byte(pc, A[s]), A[s + 1u]), A[s + 2u]);
        if (!ins.insert(g)) return false;
        if (s + 4u <= end) {
            g = fnv_byte(g, A[s + 3u]);
            if (!ins.insert(g)) return false;
            if (s + 5u <= end) {
                g = fnv_byte(g, A[s + 4u]);
                if (!ins.insert(g)) return false;
            }
        }
        return true;
    }
    if ((A[s] & 0xC0u) == 0x80u) return true;   // not a code-point start
    ulong g = pc;
    uint pos = s;
    for (uint n = 1u; n <= 5u; ++n) {
        do {
            g = fnv_byte(g, A[pos]);
            ++pos;
        } while (pos < end && (A[pos] & 0xC0u) == 0x80u);
        if (n >= 3u && !ins.insert(g)) return false;
        if (pos >= end) break;
    }
    return true;
}

template <typename Ins>
inline void hash_doc(device const uchar* A, device const GpuTok* toks, uint t0, uint t1,
                     constant ulong* pre, uint tid, uint tgs, thread Ins& ins) {
    if (t1 == t0) return;
    for (uint t = t0 + tid; t < t1; t += tgs)
        if (!hash_words(A, toks[t], pre, ins)) return;
    // window starts of token t are [a-1, a+len-1) (at least 3 bytes to the padded end)
    const uint lo = toks[t0].a - 1u;
    const uint hi = toks[t1 - 1u].a + toks[t1 - 1u].len - 1u;
    const uint chunk = (hi - lo + tgs - 1u) / tgs;
    const uint p0 = lo + tid * chunk;
    if (p0 >= hi) return;
    const uint p1 = min(hi, p0 + chunk);
    uint L = t0, R = t1;   // largest t with toks[t].a - 1 <= p0
    while (R - L > 1u) {
        const uint m = (L + R) >> 1;
        if (toks[m].a - 1u <= p0) L = m; else R = m;
    }
    uint t = L;
    GpuTok k = toks[t];
    ulong pc = pre[(k.meta & 0xFFu) * 3u + 2u];
    for (uint p = p0; p < p1; ++p) {
        while (t + 1u < t1 && toks[t + 1u].a - 1u <= p) {
            ++t;
            k = toks[t];
            pc = pre[(k.meta & 0xFFu) * 3u + 2u];
        }
        if (p + 1u >= k.a + k.len) continue;    // past this token's last window start
        if ((k.meta >> 8) & 1u) continue;       // placeholder: no char n-grams
        if (!hash_window(A, k, p, pc, ins)) return;
    }
}

template <uint NS>
struct TgIns {
    threadgroup atomic_uint* T;
    threadgroup atomic_uint* claimed;
    threadgroup atomic_uint* ovf;
    uint limit;
    bool insert(ulong h) {
        const uint idx = uint(h) & kMask18;
        const uint key = idx + 1u;
        const bool neg = (h >> 63) != 0UL;
        const uint delta = neg ? 0xFFF80000u : 0x00080000u;   // ∓1 in the count field
        uint s = idx % NS;
        for (uint probe = 0u; probe < NS;) {
            uint v = atomic_load_explicit(&T[s], memory_order_relaxed);
            if (v == 0u) {
                if (atomic_load_explicit(claimed, memory_order_relaxed) >= limit) {
                    atomic_store_explicit(ovf, 1u, memory_order_relaxed);
                    return false;
                }
                uint expected = 0u;
                if (atomic_compare_exchange_weak_explicit(&T[s], &expected, key | delta,
                                                          memory_order_relaxed, memory_order_relaxed)) {
                    atomic_fetch_add_explicit(claimed, 1u, memory_order_relaxed);
                    return true;
                }
                if (expected == 0u) continue;   // spurious CAS failure: retry this slot
                v = expected;
            }
            if ((v & 0x7FFFFu) == key) {
                const uint old = atomic_fetch_add_explicit(&T[s], delta, memory_order_relaxed);
                const int oc = int(old) >> 19;
                if ((!neg && oc == 4095) || (neg && oc == -4096)) {   // count left ±4095
                    atomic_store_explicit(ovf, 1u, memory_order_relaxed);
                    return false;
                }
                return true;
            }
            s = (s + 1u == NS) ? 0u : s + 1u;
            ++probe;
        }
        atomic_store_explicit(ovf, 1u, memory_order_relaxed);
        return false;
    }
};

struct DevIns {
    device atomic_uint* K;
    device atomic_int* C;
    uint mask;
    bool insert(ulong h) {
        const uint idx = uint(h) & kMask18;
        const uint key = idx + 1u;
        const int d = (h >> 63) != 0UL ? -1 : 1;
        uint s = idx & mask;
        for (;;) {
            uint v = atomic_load_explicit(&K[s], memory_order_relaxed);
            if (v == 0u) {
                uint expected = 0u;
                if (atomic_compare_exchange_weak_explicit(&K[s], &expected, key, memory_order_relaxed,
                                                          memory_order_relaxed)) {
                    atomic_fetch_add_explicit(&C[s], d, memory_order_relaxed);
                    return true;
                }
                if (expected == 0u) continue;
                v = expected;
            }
            if (v == key) {
                atomic_fetch_add_explicit(&C[s], d, memory_order_relaxed);
                return true;
            }
            s = (s + 1u) & mask;
        }
    }
};

// one feature's contribution: x = sign·(1+ln|c|), n2 += x², acc += x·W[idx]
inline void contribute(int c, uint idx, device const float4* W, uint L, thread float& n2,
                       thread float4* acc) {
    const float x = (c > 0 ? 1.0f : -1.0f) * (1.0f + precise::log(float(abs(c))));
    n2 += x * x;
    device const float4* w = W + idx * L;
    for (uint l = 0u; l < L; ++l) acc[l] += x * w[l];
}

// simd + threadgroup reduction, then logits = b + acc/‖x‖ + extras·We (threads 0 .. hpad-1)
template <uint MAXSG>
inline void reduce_write(float n2, thread float4* acc, uint L, threadgroup float (*part)[17],
                         uint sg, uint lane, uint nsg, uint tid, uint doc,
                         device const float* extras, device const float* We, device const float* b,
                         constant Params& P, device float* out) {
    n2 = simd_sum(n2);
    for (uint l = 0u; l < L; ++l) acc[l] = simd_sum(acc[l]);
    if (lane == 0u) {
        part[sg][0] = n2;
        for (uint l = 0u; l < L; ++l)
            for (uint j = 0u; j < 4u; ++j) part[sg][1u + 4u * l + j] = acc[l][j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < P.hpad) {
        float nn = 0.0f, a = 0.0f;
        for (uint g = 0u; g < nsg; ++g) {
            nn += part[g][0];
            a += part[g][1u + tid];
        }
        const float inv = nn > 0.0f ? 1.0f / precise::sqrt(nn) : 1.0f;
        float o = b[tid] + a * inv;
        for (uint e = 0u; e < P.ne; ++e) o += extras[doc * P.ne + e] * We[e * P.hpad + tid];
        out[doc * P.hpad + tid] = o;
    }
}

template <uint NS>
kernel void score_tg(device const uchar* A [[buffer(0)]],
                     device const GpuTok* toks [[buffer(1)]],
                     device const uint* doc_tok [[buffer(2)]],
                     device const float* extras [[buffer(3)]],
                     device const float4* W [[buffer(4)]],
                     device const float* We [[buffer(5)]],
                     device const float* b [[buffer(6)]],
                     constant ulong* pre [[buffer(7)]],
                     constant Params& P [[buffer(8)]],
                     device float* out [[buffer(9)]],
                     device uint* flags [[buffer(10)]],
                     uint gid [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint tgs [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint nsg [[simdgroups_per_threadgroup]]) {
    const uint doc = P.base + gid;
    threadgroup atomic_uint T[NS];
    threadgroup atomic_uint claimed;
    threadgroup atomic_uint ovf;
    threadgroup float part[kMaxSimdTg][17];
    for (uint s = tid; s < NS; s += tgs) atomic_store_explicit(&T[s], 0u, memory_order_relaxed);
    if (tid == 0u) {
        atomic_store_explicit(&claimed, 0u, memory_order_relaxed);
        atomic_store_explicit(&ovf, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TgIns<NS> ins{T, &claimed, &ovf, P.limit};
    const uint t0 = doc_tok[doc], t1 = doc_tok[doc + 1u];
    hash_doc(A, toks, t0, t1, pre, tid, tgs, ins);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool overflowed = atomic_load_explicit(&ovf, memory_order_relaxed) != 0u;
    if (tid == 0u) flags[doc] = overflowed ? 1u : 0u;
    if (overflowed) return;   // uniform across the threadgroup
    const uint L = P.hpad / 4u;
    float n2 = 0.0f;
    float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint s = tid; s < NS; s += tgs) {
        const uint v = atomic_load_explicit(&T[s], memory_order_relaxed);
        if (v == 0u) continue;
        const int c = int(v) >> 19;
        if (c == 0) continue;
        contribute(c, (v & 0x7FFFFu) - 1u, W, L, n2, acc);
    }
    reduce_write<kMaxSimdTg>(n2, acc, L, part, sg, lane, nsg, tid, doc, extras, We, b, P, out);
}

template [[host_name("score_tg_7600")]] kernel void score_tg<7600>(
    device const uchar*, device const GpuTok*, device const uint*, device const float*,
    device const float4*, device const float*, device const float*, constant ulong*,
    constant Params&, device float*, device uint*, uint, uint, uint, uint, uint, uint);
template [[host_name("score_tg_4000")]] kernel void score_tg<4000>(
    device const uchar*, device const GpuTok*, device const uint*, device const float*,
    device const float4*, device const float*, device const float*, constant ulong*,
    constant Params&, device float*, device uint*, uint, uint, uint, uint, uint, uint);

kernel void score_dev(device const uchar* A [[buffer(0)]],
                      device const GpuTok* toks [[buffer(1)]],
                      device const uint* doc_tok [[buffer(2)]],
                      device const float* extras [[buffer(3)]],
                      device const float4* W [[buffer(4)]],
                      device const float* We [[buffer(5)]],
                      device const float* b [[buffer(6)]],
                      constant ulong* pre [[buffer(7)]],
                      constant Params& P [[buffer(8)]],
                      device float* out [[buffer(9)]],
                      device atomic_uint* K [[buffer(10)]],
                      device atomic_int* C [[buffer(11)]],
                      device const uint* ovf_docs [[buffer(12)]],
                      device const ulong* ovf_off [[buffer(13)]],
                      device const uint* ovf_size [[buffer(14)]],
                      uint g [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tgs [[threads_per_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float part[kMaxSimdDev][17];
    const uint doc = ovf_docs[g];
    const ulong off = ovf_off[g];
    const uint size = ovf_size[g];
    device atomic_uint* Kd = K + off;
    device atomic_int* Cd = C + off;
    DevIns ins{Kd, Cd, size - 1u};
    const uint t0 = doc_tok[doc], t1 = doc_tok[doc + 1u];
    hash_doc(A, toks, t0, t1, pre, tid, tgs, ins);
    threadgroup_barrier(mem_flags::mem_device);
    const uint L = P.hpad / 4u;
    float n2 = 0.0f;
    float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint s = tid; s < size; s += tgs) {
        const uint key = atomic_load_explicit(&Kd[s], memory_order_relaxed);
        if (key == 0u) continue;
        const int c = atomic_load_explicit(&Cd[s], memory_order_relaxed);
        if (c == 0) continue;
        contribute(c, key - 1u, W, L, n2, acc);
    }
    reduce_write<kMaxSimdDev>(n2, acc, L, part, sg, lane, nsg, tid, doc, extras, We, b, P, out);
}
