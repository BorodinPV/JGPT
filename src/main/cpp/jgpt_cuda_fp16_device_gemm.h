#pragma once

#include <cublas_v2.h>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/** Согласовано с Java {@code JGPT_FP16_MATMUL} / {@code TensorOpsGPU.useFp16Matmul()}. */
void jgpt_device_fp16_gemm_set(int enabled);
int jgpt_device_fp16_gemm_enabled(void);
void jgpt_device_fp16_gemm_cleanup(void);
void jgpt_device_fp16_gemm_prewarm(size_t nelemA, size_t nelemB);

/**
 * Row-major {@code C[M×N] = op(A)*op(B)} — та же конвенция, что {@code matmulGPUDeviceEx}
 * ({@code Sgemm(opB, opA, N, M, K, B, A, C)}).
 * При FP16: A/B → half, compute {@code CUBLAS_COMPUTE_32F_FAST_16F}, C остаётся FP32.
 */
cublasStatus_t jgpt_cublas_device_gemm_rowmajor(
        cublasHandle_t handle,
        int transposeA,
        int transposeB,
        int M,
        int K,
        int N,
        const float* A,
        const float* B,
        float* C,
        float alpha,
        float beta);

/**
 * Обёртка над {@code cublasSgemmStridedBatched} / {@code GemmStridedBatchedEx}
 * (аргументы в column-major порядке cuBLAS).
 */
cublasStatus_t jgpt_cublas_device_gemm_strided_colmajor(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const float* A,
        int lda,
        long long strideA,
        const float* B,
        int ldb,
        long long strideB,
        float* C,
        int ldc,
        long long strideC,
        int batchCount,
        float alpha,
        float beta);

#ifdef __cplusplus
}
#endif
