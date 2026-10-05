// All heads of a group at once (build doc §9.2.3): weights are head-interleaved W[idx][hpad], so
// ONE gather per active feature fetches every head's weight; NEON FMA over hpad/4 lanes.
// Step 8: four independent accumulator sets (the FMA latency chain was the bound, not the
// gathers), software prefetch of the W rows 16 features ahead, and a fused entry point that takes
// the un-normalized values + 1/norm straight from the accumulator (no Features materialized).
#include "shield.h"

#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CTXSHIELD_NEON 1
#endif

namespace ctxshield {

namespace {

constexpr size_t kPrefetch = 16;

// VAL(k) yields the float feature value of feature k.
template <int L, typename ValFn>
inline void dot_lanes(const uint32_t* idx, size_t n, ValFn val, const float* ex, size_t ne,
                      const float* W, const float* We, const float* b, float* logits) {
    constexpr size_t H = 4 * L;
#if CTXSHIELD_NEON
    float32x4_t a0[L], a1[L], a2[L], a3[L];
    for (int l = 0; l < L; ++l) {
        a0[l] = vld1q_f32(b + 4 * l);
        a1[l] = vdupq_n_f32(0.f);
        a2[l] = vdupq_n_f32(0.f);
        a3[l] = vdupq_n_f32(0.f);
    }
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        if (k + kPrefetch + 4 <= n) {
            __builtin_prefetch(W + static_cast<size_t>(idx[k + kPrefetch]) * H);
            __builtin_prefetch(W + static_cast<size_t>(idx[k + kPrefetch + 1]) * H);
            __builtin_prefetch(W + static_cast<size_t>(idx[k + kPrefetch + 2]) * H);
            __builtin_prefetch(W + static_cast<size_t>(idx[k + kPrefetch + 3]) * H);
        }
        const float* w0 = W + static_cast<size_t>(idx[k]) * H;
        const float* w1 = W + static_cast<size_t>(idx[k + 1]) * H;
        const float* w2 = W + static_cast<size_t>(idx[k + 2]) * H;
        const float* w3 = W + static_cast<size_t>(idx[k + 3]) * H;
        const float v0 = val(k), v1 = val(k + 1), v2 = val(k + 2), v3 = val(k + 3);
        for (int l = 0; l < L; ++l) {
            a0[l] = vfmaq_n_f32(a0[l], vld1q_f32(w0 + 4 * l), v0);
            a1[l] = vfmaq_n_f32(a1[l], vld1q_f32(w1 + 4 * l), v1);
            a2[l] = vfmaq_n_f32(a2[l], vld1q_f32(w2 + 4 * l), v2);
            a3[l] = vfmaq_n_f32(a3[l], vld1q_f32(w3 + 4 * l), v3);
        }
    }
    for (; k < n; ++k) {
        const float* w = W + static_cast<size_t>(idx[k]) * H;
        const float v = val(k);
        for (int l = 0; l < L; ++l) a0[l] = vfmaq_n_f32(a0[l], vld1q_f32(w + 4 * l), v);
    }
    for (size_t e = 0; e < ne; ++e) {
        const float* w = We + e * H;
        for (int l = 0; l < L; ++l) a1[l] = vfmaq_n_f32(a1[l], vld1q_f32(w + 4 * l), ex[e]);
    }
    for (int l = 0; l < L; ++l)
        vst1q_f32(logits + 4 * l, vaddq_f32(vaddq_f32(a0[l], a1[l]), vaddq_f32(a2[l], a3[l])));
#else
    float acc[4][H];
    for (size_t h = 0; h < H; ++h) { acc[0][h] = b[h]; acc[1][h] = acc[2][h] = acc[3][h] = 0.f; }
    for (size_t k = 0; k < n; ++k) {
        if (k + kPrefetch < n) __builtin_prefetch(W + static_cast<size_t>(idx[k + kPrefetch]) * H);
        const float* w = W + static_cast<size_t>(idx[k]) * H;
        const float v = val(k);
        float* a = acc[k & 3];
        for (size_t h = 0; h < H; ++h) a[h] += v * w[h];
    }
    for (size_t e = 0; e < ne; ++e) {
        const float* w = We + e * H;
        for (size_t h = 0; h < H; ++h) acc[1][h] += ex[e] * w[h];
    }
    for (size_t h = 0; h < H; ++h) logits[h] = (acc[0][h] + acc[1][h]) + (acc[2][h] + acc[3][h]);
#endif
}

template <typename ValFn>
void dispatch(const uint32_t* idx, size_t n, ValFn val, const float* ex, size_t ne, const float* W,
              const float* We, const float* b, int hpad, float* logits) {
    switch (hpad) {
        case 4: dot_lanes<1>(idx, n, val, ex, ne, W, We, b, logits); break;
        case 8: dot_lanes<2>(idx, n, val, ex, ne, W, We, b, logits); break;
        case 12: dot_lanes<3>(idx, n, val, ex, ne, W, We, b, logits); break;
        case 16: dot_lanes<4>(idx, n, val, ex, ne, W, We, b, logits); break;
        default: throw std::runtime_error("ctxshield: hpad must be 4, 8, 12 or 16");
    }
}

}  // namespace

void score_features(const Features& f, const float* W, const float* We, const float* b,
                    int hpad, float* logits) {
    const float* v = f.val.data();
    dispatch(f.idx.data(), f.idx.size(), [v](size_t k) { return v[k]; }, f.extras.data(),
             f.extras.size(), W, We, b, hpad, logits);
}

void gather_dot(const uint32_t* idx, const double* x, double inv, size_t n, const float* ex,
                size_t ne, const float* W, const float* We, const float* b, int hpad,
                float* logits) {
    // the same float value featurize() emits: float(x * inv)
    dispatch(idx, n, [x, inv](size_t k) { return static_cast<float>(x[k] * inv); }, ex, ne, W, We,
             b, hpad, logits);
}

}  // namespace ctxshield
