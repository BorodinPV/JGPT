#include "jgpt_cudnn_convert.h"
#include "jgpt_cuda_stream.cuh"
#include "jgpt_cuda_error_macros.cuh"
#include <climits>

__global__ void jgpt_f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] = __float2half_rn(src[i]);
    }
}

__global__ void jgpt_f16_to_f32_kernel(const __half* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

extern "C" void jgpt_extra_f32_to_f16(const float* src, __half* dst, size_t n) {
    if (src == nullptr || dst == nullptr || n == 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    while (n > 0) {
        const int chunk = n > (size_t) INT_MAX ? INT_MAX : (int) n;
        const int blocks = (chunk + 255) / 256;
        jgpt_f32_to_f16_kernel<<<blocks, 256, 0, kTensorCudaStream>>>(src, dst, chunk);
        src += chunk;
        dst += chunk;
        n -= (size_t) chunk;
    }
    CUDA_KERNEL_CHECK();
}

extern "C" void jgpt_extra_f16_to_f32(const __half* src, float* dst, size_t n) {
    if (src == nullptr || dst == nullptr || n == 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    while (n > 0) {
        const int chunk = n > (size_t) INT_MAX ? INT_MAX : (int) n;
        const int blocks = (chunk + 255) / 256;
        jgpt_f16_to_f32_kernel<<<blocks, 256, 0, kTensorCudaStream>>>(src, dst, chunk);
        src += chunk;
        dst += chunk;
        n -= (size_t) chunk;
    }
    CUDA_KERNEL_CHECK();
}
