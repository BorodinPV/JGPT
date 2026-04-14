#pragma once

#include <cublas_v2.h>
#include <cstdio>

#include "jgpt_cuda_stream.cuh"

namespace jgpt_cuda_detail {

/**
 * Создаёт cuBLAS-дескриптор с TF32 и привязкой к единому stream TensorOpsGPU.
 * При ошибке пишет в stderr с префиксом {@code log_ctx}.
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

/** Thread-local cuBLAS handle (единый для всех translation unit). */
inline cublasHandle_t& cublas_thread_local_handle() {
    thread_local cublasHandle_t h = nullptr;
    return h;
}

/** Получить или создать thread-local cuBLAS handle. */
inline cublasHandle_t get_cublas_handle() {
    cublasHandle_t& h = cublas_thread_local_handle();
    if (h != nullptr) {
        return h;
    }
    h = create_cublas_for_jgpt_stream("TensorOpsGPU");
    return h;
}

/** Освободить thread-local cuBLAS handle (вызывать при очистке ресурсов потока). */
inline void destroy_cublas_thread_local_handle() {
    cublasHandle_t& h = cublas_thread_local_handle();
    if (h != nullptr) {
        cublasDestroy(h);
        h = nullptr;
    }
}

}  // namespace jgpt_cuda_detail
