/* Device GEMM: TF32 Sgemm или FP16 GemmEx (staging float→half).
 * Included from jgpt_cuda.cu after kernels (needs launch_float_to_half_n).
 */

#include "jgpt_cuda_fp16_device_gemm.h"
#include "jgpt_cuda_tls_blob.cuh"
#include "jgpt_cuda_size_check.cuh"
#include <cuda_fp16.h>
#include <library_types.h>
#include <atomic>

static std::atomic<int> g_device_fp16_gemm{0};
static thread_local jgpt_cuda_tls::TlsDeviceBlob tl_dev_fp16_Ah;
static thread_local jgpt_cuda_tls::TlsDeviceBlob tl_dev_fp16_Bh;

void jgpt_device_fp16_gemm_set(int enabled) {
    g_device_fp16_gemm.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

int jgpt_device_fp16_gemm_enabled(void) {
    return g_device_fp16_gemm.load(std::memory_order_relaxed);
}

void jgpt_device_fp16_gemm_cleanup(void) {
    tl_dev_fp16_Ah.free_cached();
    tl_dev_fp16_Bh.free_cached();
}

static bool jgpt_dev_fp16_ensure_ab(size_t nelemA, size_t nelemB, __half** outA, __half** outB) {
    if (jgpt_alloc_n_elem_overflows(nelemA, sizeof(__half)) || jgpt_alloc_n_elem_overflows(nelemB, sizeof(__half))) {
        fprintf(stderr, "jgpt_dev_fp16_ensure_ab: size overflow a=%zu b=%zu\n", nelemA, nelemB);
        return false;
    }
    if (!tl_dev_fp16_Ah.grow_to_fit(nelemA * sizeof(__half))) {
        fprintf(stderr, "jgpt_dev_fp16_ensure_ab: Ah alloc failed (%zu elems)\n", nelemA);
        return false;
    }
    if (!tl_dev_fp16_Bh.grow_to_fit(nelemB * sizeof(__half))) {
        fprintf(stderr, "jgpt_dev_fp16_ensure_ab: Bh alloc failed (%zu elems)\n", nelemB);
        return false;
    }
    *outA = static_cast<__half*>(tl_dev_fp16_Ah.ptr);
    *outB = static_cast<__half*>(tl_dev_fp16_Bh.ptr);
    return *outA != nullptr && *outB != nullptr;
}

void jgpt_device_fp16_gemm_prewarm(size_t nelemA, size_t nelemB) {
    if (!g_device_fp16_gemm.load(std::memory_order_relaxed) || nelemA == 0U || nelemB == 0U) {
        return;
    }
    __half *a = nullptr, *b = nullptr;
    (void) jgpt_dev_fp16_ensure_ab(nelemA, nelemB, &a, &b);
}

static size_t jgpt_colmajor_operand_elems(
        cublasOperation_t trans, int dim0, int dim1, int ld, long long stride, int batchCount) {
    /* trans=N: matrix is dim0×dim1 (m×k for A), elems = ld * dim1; trans=T: dim1×dim0, elems = ld * dim0. */
    size_t one = trans == CUBLAS_OP_N ? (size_t) ld * (size_t) dim1 : (size_t) ld * (size_t) dim0;
    if (stride == 0LL) {
        return one;
    }
    if (stride < 0LL || batchCount <= 0) {
        return one;
    }
    return (size_t) batchCount * (size_t) stride;
}

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
        float beta) {
    if (handle == nullptr || A == nullptr || B == nullptr || C == nullptr || M <= 0 || K <= 0 || N <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transposeA ? M : K;
    int ldb = transposeB ? K : N;
    if (!g_device_fp16_gemm.load(std::memory_order_relaxed)) {
        return cublasSgemm(handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
    }
    const size_t nelemA = (size_t) M * (size_t) K;
    const size_t nelemB = (size_t) K * (size_t) N;
    __half *Ah = nullptr, *Bh = nullptr;
    if (!jgpt_dev_fp16_ensure_ab(nelemA, nelemB, &Ah, &Bh)) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    jgpt_cuda_ensure_stream();
    launch_float_to_half_n(A, Ah, nelemA);
    launch_float_to_half_n(B, Bh, nelemB);
    return cublasGemmEx(
            handle,
            opB,
            opA,
            N,
            M,
            K,
            &alpha,
            Bh,
            CUDA_R_16F,
            ldb,
            Ah,
            CUDA_R_16F,
            lda,
            &beta,
            C,
            CUDA_R_32F,
            N,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

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
        float beta) {
    if (handle == nullptr || A == nullptr || B == nullptr || C == nullptr || m <= 0 || n <= 0 || k <= 0
            || batchCount <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!g_device_fp16_gemm.load(std::memory_order_relaxed)) {
        return cublasSgemmStridedBatched(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                strideA,
                B,
                ldb,
                strideB,
                &beta,
                C,
                ldc,
                strideC,
                batchCount);
    }
    /* A: op(A) is m×k; B: op(B) is k×n */
    const size_t nelemA = jgpt_colmajor_operand_elems(transa, m, k, lda, strideA, batchCount);
    const size_t nelemB = jgpt_colmajor_operand_elems(transb, k, n, ldb, strideB, batchCount);
    __half *Ah = nullptr, *Bh = nullptr;
    if (!jgpt_dev_fp16_ensure_ab(nelemA, nelemB, &Ah, &Bh)) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    jgpt_cuda_ensure_stream();
    launch_float_to_half_n(A, Ah, nelemA);
    launch_float_to_half_n(B, Bh, nelemB);
    return cublasGemmStridedBatchedEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            &alpha,
            Ah,
            CUDA_R_16F,
            lda,
            strideA,
            Bh,
            CUDA_R_16F,
            ldb,
            strideB,
            &beta,
            C,
            CUDA_R_32F,
            ldc,
            strideC,
            batchCount,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
