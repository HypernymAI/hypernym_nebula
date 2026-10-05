// C-linkage launchers exported by shield.cu (compiled by nvcc) and called by cuda_backend.cpp
// (compiled by the host C++ compiler). Build doc §9.4 — UNTESTED until run_on_l4.sh passes.
#pragma once
#include <cstddef>
#include <cstdint>

#ifndef SHIELD_CUDA_SYNTAX_SHIM
#include <cuda_runtime.h>
#endif

extern "C" {

// Same launch geometry as Metal: one block per document. Shared-memory table of kCudaSlots packed
// slots (bits 0..18 = index + 1, bits 19..31 = signed count); overflow → flags[doc] = 1.
constexpr uint32_t kCudaSlots = 8192;
constexpr uint32_t kCudaLimit = kCudaSlots * 3 / 4;
constexpr int kCudaBlock = 256;
constexpr int kCudaDevBlock = 1024;

cudaError_t shield_cuda_prepare(void);   // opt-in dynamic shared memory for score_tg
size_t shield_cuda_tg_smem_bytes(void);

cudaError_t shield_cuda_score_tg(const uint8_t* A, const void* toks, const uint32_t* doc_tok,
                                 const float* extras, const float* W, const float* We,
                                 const float* b, const uint64_t* pre, uint32_t n_docs,
                                 uint32_t hpad, uint32_t ne, float* out, uint32_t* flags,
                                 cudaStream_t stream);

cudaError_t shield_cuda_score_dev(const uint8_t* A, const void* toks, const uint32_t* doc_tok,
                                  const float* extras, const float* W, const float* We,
                                  const float* b, const uint64_t* pre, uint32_t hpad, uint32_t ne,
                                  float* out, uint32_t* K, int32_t* C, const uint32_t* ovf_docs,
                                  const uint64_t* ovf_off, const uint32_t* ovf_size,
                                  uint32_t n_ovf, cudaStream_t stream);

}  // extern "C"
