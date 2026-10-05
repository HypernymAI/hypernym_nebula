// SYNTAX-CHECK SHIM ONLY — never used to build. Declares the subset of the CUDA runtime / device API
// that src/cuda/{shield.cu, cuda_backend.cpp} use, so the sources can be kept compiler-clean on a
// machine with no CUDA toolkit (build doc §9.4: "kept compiler-clean against the host-side C++"):
//   clang -x cuda --cuda-host-only / --cuda-device-only --cuda-gpu-arch=sm_89 -nocudainc -fsyntax-only
// Signatures follow the CUDA 12 runtime headers; the real build (run_on_l4.sh) uses <cuda_runtime.h>.
#pragma once
#include <cstddef>

#ifdef __CUDA__
#include "__clang_cuda_builtin_vars.h"
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#else
#define __global__
#define __device__
#define __host__
#endif

enum cudaError { cudaSuccess = 0, cudaErrorInvalidValue = 1 };
typedef enum cudaError cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
#define cudaHostAllocDefault 0x00
struct dim3 {
    unsigned int x, y, z;
    __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};
struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
};
struct float4 { float x, y, z, w; };
struct uint4 { unsigned int x, y, z, w; };

extern "C" {
__host__ const char* cudaGetErrorString(cudaError_t);
__host__ cudaError_t cudaGetLastError(void);
__host__ cudaError_t cudaMalloc(void** p, size_t n);
__host__ cudaError_t cudaFree(void* p);
__host__ cudaError_t cudaHostAlloc(void** p, size_t n, unsigned int flags);
__host__ cudaError_t cudaFreeHost(void* p);
__host__ cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind k);
__host__ cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind k, cudaStream_t s);
__host__ cudaError_t cudaMemsetAsync(void* dst, int v, size_t n, cudaStream_t s);
__host__ cudaError_t cudaStreamCreate(cudaStream_t* s);
__host__ cudaError_t cudaStreamDestroy(cudaStream_t s);
__host__ cudaError_t cudaStreamSynchronize(cudaStream_t s);
__host__ cudaError_t cudaEventCreate(cudaEvent_t* e);
__host__ cudaError_t cudaEventDestroy(cudaEvent_t e);
__host__ cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s);
__host__ cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b);
__host__ cudaError_t cudaGetDeviceCount(int* n);
__host__ cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int dev);
__host__ int cudaConfigureCall(dim3 grid, dim3 block, size_t smem = 0, cudaStream_t s = 0);
__host__ cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value);
}
template <class T>
static __inline__ __host__ cudaError_t cudaFuncSetAttribute(T* entry, cudaFuncAttribute attr, int value) {
    return ::cudaFuncSetAttribute(reinterpret_cast<const void*>(entry), attr, value);
}

#ifdef __CUDA__
__device__ unsigned int atomicCAS(unsigned int* a, unsigned int cmp, unsigned int v);
__device__ unsigned int atomicAdd(unsigned int* a, unsigned int v);
__device__ int atomicAdd(int* a, int v);
__device__ unsigned int atomicExch(unsigned int* a, unsigned int v);
__device__ float __shfl_down_sync(unsigned int mask, float v, unsigned int delta, int width = 32);
__device__ unsigned char __ldg(const unsigned char* p);
__device__ unsigned int __ldg(const unsigned int* p);
__device__ unsigned long long __ldg(const unsigned long long* p);
__device__ float __ldg(const float* p);
__device__ float4 __ldg(const float4* p);
__device__ uint4 __ldg(const uint4* p);
__device__ void __syncthreads(void);
__device__ void __threadfence_block(void);
__device__ float logf(float);
__device__ float sqrtf(float);
__device__ unsigned int min(unsigned int, unsigned int);
__device__ float4 make_float4(float x, float y, float z, float w);
#endif
