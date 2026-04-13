#pragma once

#include <cublas_v2.h>
#include <cstdio>

#include "jgpt_cuda_stream.cuh"

namespace jgpt_cuda_detail {

/**
 * Создаёт cuBLAS-дескриптор с TF32 и привязкой к единому stream TensorOpsGPU.
 * При ошибке пишет в stderr с префиксом {@code log_ctx} (например {@code "TensorOpsGPU"} или {@code "TensorOpsGPU extra"}).
 * @return handle или nullptr
 */
inline cublasHandle_t create_cublas_for_jgpt_stream(const char* log_ctx) {
    jgpt_cuda_ensure_stream();
    cublasHandle_t h = nullptr;
    cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cublasCreate failed: %d\n", log_ctx, static_cast<int>(st));
        return nullptr;
    }
    st = cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cublasSetMathMode failed: %d\n", log_ctx, static_cast<int>(st));
        cublasDestroy(h);
        return nullptr;
    }
    cudaStream_t stream = jgpt_cuda_stream_handle();
    if (stream == nullptr) {
        fprintf(stderr, "%s: CUDA stream unavailable after jgpt_cuda_ensure_stream (cublas)\n", log_ctx);
        cublasDestroy(h);
        return nullptr;
    }
    st = cublasSetStream(h, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cublasSetStream failed: %d\n", log_ctx, static_cast<int>(st));
        cublasDestroy(h);
        return nullptr;
    }
    return h;
}

}  // namespace jgpt_cuda_detail
