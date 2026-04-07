#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <jni.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <climits>
#include <cstring>
#include <mutex>

#include "jgpt_cuda_stream.cuh"
#include "jgpt_cuda_ffn_link.h"
#include "jgpt_cuda_graph_prewarm.h"
#include "jgpt_cuda_size_check.cuh"

/* ========== Thread-safe cuBLAS handle per thread (extra ops / strided batched GEMM) ========== */
static thread_local cublasHandle_t tl_extra_cublas_handle = nullptr;

static cublasHandle_t get_extra_cublas_handle() {
    if (tl_extra_cublas_handle != nullptr) {
        return tl_extra_cublas_handle;
    }
    /* Дескриптор thread_local — на поток один create; g_stream защищён mutex в jgpt_cuda.cu. */
    jgpt_cuda_ensure_stream();
    cublasHandle_t h = nullptr;
    cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "TensorOpsGPU extra: cublasCreate failed: %d\n", static_cast<int>(st));
        return nullptr;
    }
    st = cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "TensorOpsGPU extra: cublasSetMathMode failed: %d\n", static_cast<int>(st));
        cublasDestroy(h);
        return nullptr;
    }
    cudaStream_t stream = jgpt_cuda_stream_handle();
    if (stream == nullptr) {
        fprintf(stderr, "TensorOpsGPU extra: CUDA stream unavailable after jgpt_cuda_ensure_stream (cublas)\n");
        cublasDestroy(h);
        return nullptr;
    }
    st = cublasSetStream(h, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "TensorOpsGPU extra: cublasSetStream failed: %d\n", static_cast<int>(st));
        cublasDestroy(h);
        return nullptr;
    }
    tl_extra_cublas_handle = h;
    return tl_extra_cublas_handle;
}

extern "C" void jgpt_cuda_extra_warmup_cublas(void) {
    (void) get_extra_cublas_handle();
}

static void destroy_extra_cublas_handle() {
    if (tl_extra_cublas_handle) {
        cublasDestroy(tl_extra_cublas_handle);
        tl_extra_cublas_handle = nullptr;
    }
}

/* ========== Device buffers cached for CE loss (freed in jgpt_cuda_extra_cleanup) ========== */
static thread_local float* tl_ce_d_logits = nullptr;
static thread_local float* tl_ce_d_targets = nullptr;
static thread_local float* tl_ce_d_grad = nullptr;
static thread_local float* tl_ce_d_loss_sum = nullptr;
static thread_local unsigned int* tl_ce_d_valid = nullptr;
static thread_local size_t tl_ce_logits_bytes = 0;
static thread_local size_t tl_ce_targets_bytes = 0;

/** Pinned host для async D2H CE loss/valid (без sync в JNI launch). */
static thread_local float* tl_ce_h_async_loss = nullptr;
static thread_local unsigned int* tl_ce_h_async_valid = nullptr;

static void ce_async_host_free() {
    if (tl_ce_h_async_loss != nullptr) {
        cudaFreeHost(tl_ce_h_async_loss);
        tl_ce_h_async_loss = nullptr;
    }
    if (tl_ce_h_async_valid != nullptr) {
        cudaFreeHost(tl_ce_h_async_valid);
        tl_ce_h_async_valid = nullptr;
    }
}

static void ce_async_host_ensure() {
    if (tl_ce_h_async_loss == nullptr) {
        cudaError_t err =
                cudaHostAlloc(reinterpret_cast<void**>(&tl_ce_h_async_loss), sizeof(float), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "ce_async_host_ensure(loss): %s\n", cudaGetErrorString(err));
            return;
        }
    }
    if (tl_ce_h_async_valid == nullptr) {
        cudaError_t err = cudaHostAlloc(
                reinterpret_cast<void**>(&tl_ce_h_async_valid), sizeof(unsigned int), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "ce_async_host_ensure(valid): %s\n", cudaGetErrorString(err));
            return;
        }
    }
}

/** Пinned host: копия float targets перед async H2D (не держим jarray во время work на stream). */
static thread_local float* tl_ce_h_pinned_flt_targets = nullptr;
static thread_local size_t tl_ce_h_pinned_flt_n = 0;

static void ce_pinned_flt_targets_free() {
    if (tl_ce_h_pinned_flt_targets != nullptr) {
        cudaFreeHost(tl_ce_h_pinned_flt_targets);
        tl_ce_h_pinned_flt_targets = nullptr;
        tl_ce_h_pinned_flt_n = 0;
    }
}

static void ce_pinned_flt_targets_ensure(size_t nfloats) {
    if (nfloats == 0) {
        return;
    }
    if (nfloats > tl_ce_h_pinned_flt_n) {
        ce_pinned_flt_targets_free();
        cudaError_t err = cudaHostAlloc(
                reinterpret_cast<void**>(&tl_ce_h_pinned_flt_targets), nfloats * sizeof(float), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "ce_pinned_flt_targets_ensure: %s\n", cudaGetErrorString(err));
            return;
        }
        tl_ce_h_pinned_flt_n = nfloats;
    }
}

static void ce_free_cached() {
    cudaFree(tl_ce_d_logits);
    cudaFree(tl_ce_d_targets);
    cudaFree(tl_ce_d_grad);
    cudaFree(tl_ce_d_loss_sum);
    cudaFree(tl_ce_d_valid);
    tl_ce_d_logits = tl_ce_d_targets = tl_ce_d_grad = nullptr;
    tl_ce_d_loss_sum = nullptr;
    tl_ce_d_valid = nullptr;
    tl_ce_logits_bytes = tl_ce_targets_bytes = 0;
    ce_async_host_free();
    ce_pinned_flt_targets_free();
}

/**
 * Увеличивает пару device-буферов CE (logits + grad) до {@code bytes_logits}.
 * При ошибке второго {@code cudaMalloc} освобождает первый и сбрасывает кэш — без рассинхрона {@code tl_ce_logits_bytes}.
 */
static bool ce_ensure_logits_grad_buffers(size_t bytes_logits) {
    if (bytes_logits <= tl_ce_logits_bytes) {
        return true;
    }
    cudaFree(tl_ce_d_logits);
    cudaFree(tl_ce_d_grad);
    tl_ce_d_logits = nullptr;
    tl_ce_d_grad = nullptr;
    tl_ce_logits_bytes = 0;

    cudaError_t e_logits = cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_logits), bytes_logits);
    if (e_logits != cudaSuccess) {
        fprintf(
                stderr,
                "ce_ensure_logits_grad_buffers: cudaMalloc(logits) %zu bytes: %s\n",
                bytes_logits,
                cudaGetErrorString(e_logits));
        return false;
    }
    cudaError_t e_grad = cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_grad), bytes_logits);
    if (e_grad != cudaSuccess) {
        fprintf(
                stderr,
                "ce_ensure_logits_grad_buffers: cudaMalloc(grad) %zu bytes: %s\n",
                bytes_logits,
                cudaGetErrorString(e_grad));
        cudaFree(tl_ce_d_logits);
        tl_ce_d_logits = nullptr;
        tl_ce_d_grad = nullptr;
        tl_ce_logits_bytes = 0;
        return false;
    }
    tl_ce_logits_bytes = bytes_logits;
    return true;
}

/* ========== Thread-local JNI scratch (grow-only; освобождается в jgpt_cuda_extra_cleanup) ========== */

static thread_local void* tl_softmax_pair = nullptr;
static thread_local size_t tl_softmax_bytes_per = 0;

static bool softmax_pair_ensure(size_t bytes_per, float** d_src, float** d_dst) {
    jgpt_cuda_ensure_stream();
    if (bytes_per == 0U) {
        return false;
    }
    if (bytes_per > tl_softmax_bytes_per) {
        cudaFree(tl_softmax_pair);
        tl_softmax_pair = nullptr;
        tl_softmax_bytes_per = 0;
        if (cudaMalloc(&tl_softmax_pair, bytes_per * 2U) != cudaSuccess) {
            return false;
        }
        tl_softmax_bytes_per = bytes_per;
    }
    *d_src = static_cast<float*>(tl_softmax_pair);
    *d_dst = reinterpret_cast<float*>(static_cast<unsigned char*>(tl_softmax_pair) + bytes_per);
    return true;
}

static void softmax_pair_free_cached() {
    cudaFree(tl_softmax_pair);
    tl_softmax_pair = nullptr;
    tl_softmax_bytes_per = 0;
}

/* ========== Thread-local pool for AdamW JNI (grow-only; freed in jgpt_cuda_extra_cleanup) ========== */
static thread_local float* tl_adamw_d_param = nullptr;
static thread_local float* tl_adamw_d_grad = nullptr;
static thread_local float* tl_adamw_d_m = nullptr;
static thread_local float* tl_adamw_d_v = nullptr;
static thread_local size_t tl_adamw_bytes = 0;

static bool adamw_pool_ensure(size_t bytes) {
    jgpt_cuda_ensure_stream();
    if (bytes == 0U) {
        return false;
    }
    if (tl_adamw_bytes >= bytes) {
        return true;
    }
    cudaFree(tl_adamw_d_param);
    cudaFree(tl_adamw_d_grad);
    cudaFree(tl_adamw_d_m);
    cudaFree(tl_adamw_d_v);
    tl_adamw_d_param = nullptr;
    tl_adamw_d_grad = nullptr;
    tl_adamw_d_m = nullptr;
    tl_adamw_d_v = nullptr;
    tl_adamw_bytes = 0;
    if (cudaMalloc(reinterpret_cast<void**>(&tl_adamw_d_param), bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&tl_adamw_d_grad), bytes) != cudaSuccess) {
        cudaFree(tl_adamw_d_param);
        tl_adamw_d_param = nullptr;
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&tl_adamw_d_m), bytes) != cudaSuccess) {
        cudaFree(tl_adamw_d_param);
        cudaFree(tl_adamw_d_grad);
        tl_adamw_d_param = nullptr;
        tl_adamw_d_grad = nullptr;
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&tl_adamw_d_v), bytes) != cudaSuccess) {
        cudaFree(tl_adamw_d_param);
        cudaFree(tl_adamw_d_grad);
        cudaFree(tl_adamw_d_m);
        tl_adamw_d_param = nullptr;
        tl_adamw_d_grad = nullptr;
        tl_adamw_d_m = nullptr;
        return false;
    }
    tl_adamw_bytes = bytes;
    return true;
}

static void adamw_pool_free_cached() {
    cudaFree(tl_adamw_d_param);
    cudaFree(tl_adamw_d_grad);
    cudaFree(tl_adamw_d_m);
    cudaFree(tl_adamw_d_v);
    tl_adamw_d_param = nullptr;
    tl_adamw_d_grad = nullptr;
    tl_adamw_d_m = nullptr;
    tl_adamw_d_v = nullptr;
    tl_adamw_bytes = 0;
}

static thread_local void* tl_attn_bwd_host = nullptr;
static thread_local size_t tl_attn_bwd_host_total = 0;

static bool attn_bwd_host_ensure(
        size_t bytesProb, size_t bytesQK, size_t bytesV, float** d_go, float** d_p, float** d_q, float** d_k,
        float** d_v, float** d_dp, float** d_ds,
        float** d_gq, float** d_gk, float** d_gv) {
    size_t total = 3U * bytesProb + 4U * bytesQK + 3U * bytesV;
    if (total > tl_attn_bwd_host_total) {
        cudaFree(tl_attn_bwd_host);
        tl_attn_bwd_host = nullptr;
        tl_attn_bwd_host_total = 0;
        if (cudaMalloc(&tl_attn_bwd_host, total) != cudaSuccess) {
            return false;
        }
        tl_attn_bwd_host_total = total;
    }
    unsigned char* base = static_cast<unsigned char*>(tl_attn_bwd_host);
    size_t off = 0;
    *d_go = reinterpret_cast<float*>(base + off);
    off += bytesV;
    *d_p = reinterpret_cast<float*>(base + off);
    off += bytesProb;
    *d_q = reinterpret_cast<float*>(base + off);
    off += bytesQK;
    *d_k = reinterpret_cast<float*>(base + off);
    off += bytesQK;
    *d_v = reinterpret_cast<float*>(base + off);
    off += bytesV;
    *d_dp = reinterpret_cast<float*>(base + off);
    off += bytesProb;
    *d_ds = reinterpret_cast<float*>(base + off);
    off += bytesProb;
    *d_gq = reinterpret_cast<float*>(base + off);
    off += bytesQK;
    *d_gk = reinterpret_cast<float*>(base + off);
    off += bytesQK;
    *d_gv = reinterpret_cast<float*>(base + off);
    off += bytesV;
    (void) off;
    return true;
}

static void attn_bwd_host_free_cached() {
    cudaFree(tl_attn_bwd_host);
    tl_attn_bwd_host = nullptr;
    tl_attn_bwd_host_total = 0;
}

static thread_local void* tl_attn_bwd_aux = nullptr;
static thread_local size_t tl_attn_bwd_aux_total = 0;

static bool attn_bwd_aux_ensure(
        size_t bytesProb, size_t bytesV, float** d_dp, float** d_ds) {
    size_t total = 2U * bytesProb;
    if (total > tl_attn_bwd_aux_total) {
        cudaFree(tl_attn_bwd_aux);
        tl_attn_bwd_aux = nullptr;
        tl_attn_bwd_aux_total = 0;
        if (cudaMalloc(&tl_attn_bwd_aux, total) != cudaSuccess) {
            return false;
        }
        tl_attn_bwd_aux_total = total;
    }
    unsigned char* base = static_cast<unsigned char*>(tl_attn_bwd_aux);
    size_t off = 0;
    *d_dp = reinterpret_cast<float*>(base + off);
    off += bytesProb;
    *d_ds = reinterpret_cast<float*>(base + off);
    (void) bytesV;
    return true;
}

static void attn_bwd_aux_free_cached() {
    cudaFree(tl_attn_bwd_aux);
    tl_attn_bwd_aux = nullptr;
    tl_attn_bwd_aux_total = 0;
}

// ========== Ядра (float32) ==========

__global__ void softmax_last_dim_kernel(const float* src, float* dst, int nrows, int inner) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * inner;
    float maxv = -INFINITY;
    for (int j = 0; j < inner; j++) {
        maxv = fmaxf(maxv, src[base + j]);
    }
    float sum = 0.f;
    for (int j = 0; j < inner; j++) {
        float e = expf(src[base + j] - maxv);
        dst[base + j] = e;
        sum += e;
    }
    float inv = 1.f / sum;
    for (int j = 0; j < inner; j++) {
        dst[base + j] *= inv;
    }
}

/**
 * Softmax по последней оси (ветка «fp16» в API): раньше exp шёл через FP16-округление диффа;
 * это давало sum=0 и Inf/NaN в вероятностях. Считаем exp в FP32, как в softmax_last_dim_kernel.
 */
__global__ void softmax_last_dim_kernel_fp16(const float* src, float* dst, int nrows, int inner) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * inner;
    float maxv = -INFINITY;
    for (int j = 0; j < inner; j++) {
        maxv = fmaxf(maxv, src[base + j]);
    }
    float sum = 0.f;
    for (int j = 0; j < inner; j++) {
        float diff = src[base + j] - maxv;
        float e = expf(diff);
        dst[base + j] = e;
        sum += e;
    }
    float inv = 1.f / fmaxf(sum, 1e-12f);
    for (int j = 0; j < inner; j++) {
        dst[base + j] *= inv;
    }
}

// ========== Block-per-row softmax (оптимизированная версия для inner >= 64) ==========
//
// Проблема «одна нить на строку»: все нити варпа обращаются к разным строкам →
// нет коалесценции. При inner=512 каждая нить делает 3 последовательных цикла по 512 эл.
//
// Решение: один блок на строку (gridDim.x = nrows, blockDim.x = kSoftmaxBlockDim=256).
// Нити блока читают ОДНУ строку с шагом blockDim.x — warp читает 32 смежных float → отличная
// коалесценция. Редукция max/sum — через warp-shuffle + shared memory между warpами.

static constexpr int kSoftmaxBlockDim       = 256;         // нитей на блок
static constexpr int kSoftmaxNumWarps       = kSoftmaxBlockDim / 32;  // 8
static constexpr int kSoftmaxBlockThreshold = 64;          // при inner >= этого → block-per-row

__device__ __forceinline__ float warpReduceMax32(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    return v;
}

__device__ __forceinline__ float warpReduceSum32(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

// Вызывается всеми kSoftmaxBlockDim нитями; возвращает broadcast-max блока.
__device__ float blockReduceMax256(float v, float* smem) {
    v = warpReduceMax32(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
    __syncthreads();
    float bv = (threadIdx.x < kSoftmaxNumWarps) ? smem[threadIdx.x] : -INFINITY;
    if (threadIdx.x < 32) bv = warpReduceMax32(bv);
    if (threadIdx.x == 0) smem[0] = bv;
    __syncthreads();
    return smem[0];
}

// Вызывается всеми kSoftmaxBlockDim нитями; возвращает broadcast-sum блока.
__device__ float blockReduceSum256(float v, float* smem) {
    v = warpReduceSum32(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
    __syncthreads();
    float bv = (threadIdx.x < kSoftmaxNumWarps) ? smem[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) bv = warpReduceSum32(bv);
    if (threadIdx.x == 0) smem[0] = bv;
    __syncthreads();
    return smem[0];
}

/**
 * Softmax по последней оси, block-per-row.
 * Запуск: gridDim.x=nrows, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void softmax_last_dim_block_kernel(
        const float* __restrict__ src, float* __restrict__ dst, int nrows, int inner) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int tid = threadIdx.x;
    const float* rowSrc = src + (ptrdiff_t)row * inner;
    float*       rowDst = dst + (ptrdiff_t)row * inner;

    float maxv = -INFINITY;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        maxv = fmaxf(maxv, rowSrc[j]);
    maxv = blockReduceMax256(maxv, smem);

    float sumv = 0.f;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        sumv += expf(rowSrc[j] - maxv);
    sumv = blockReduceSum256(sumv, smem);

    const float inv = 1.f / fmaxf(sumv, 1e-12f);
    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        rowDst[j] = expf(rowSrc[j] - maxv) * inv;
}

/**
 * Слитый (scale + causal-mask + softmax), block-per-row для attention.
 * Устраняет отдельные вызовы scale_inplace и add_mask_inplace — читает d_scores только один раз.
 * Для causal mask: row % inner — query-позиция; mask[qPos * inner + j] добавляется к scaled logit.
 * Запуск: gridDim.x=nrows, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void softmax_scaled_masked_block_kernel(
        const float* __restrict__ src, float* __restrict__ dst,
        const float* __restrict__ mask, float scale,
        int nrows, int inner) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int tid = threadIdx.x;
    const float* rowSrc = src + (ptrdiff_t)row * inner;
    float*       rowDst = dst + (ptrdiff_t)row * inner;
    const int qPos = row % inner;  // query-позиция в последовательности для causal mask

    float maxv = -INFINITY;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        float v = rowSrc[j] * scale + (mask ? mask[(ptrdiff_t)qPos * inner + j] : 0.f);
        maxv = fmaxf(maxv, v);
    }
    maxv = blockReduceMax256(maxv, smem);

    float sumv = 0.f;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        float v = rowSrc[j] * scale + (mask ? mask[(ptrdiff_t)qPos * inner + j] : 0.f);
        sumv += expf(v - maxv);
    }
    sumv = blockReduceSum256(sumv, smem);

    const float inv = 1.f / fmaxf(sumv, 1e-12f);
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        float v = rowSrc[j] * scale + (mask ? mask[(ptrdiff_t)qPos * inner + j] : 0.f);
        rowDst[j] = expf(v - maxv) * inv;
    }
}

/** In-place: один указатель без src/dst restrict-aliasing (для graph-aux размером bytesProb вместо 2×). */
__global__ void softmax_scaled_masked_inplace_block_kernel(
        float* __restrict__ buf, const float* __restrict__ mask, float scale, int nrows, int inner) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int tid = threadIdx.x;
    float* rowBuf = buf + (ptrdiff_t)row * inner;
    const int qPos = row % inner;

    float maxv = -INFINITY;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        float v = rowBuf[j] * scale + (mask ? mask[(ptrdiff_t)qPos * inner + j] : 0.f);
        maxv = fmaxf(maxv, v);
    }
    maxv = blockReduceMax256(maxv, smem);

    float sumv = 0.f;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        float v = rowBuf[j] * scale + (mask ? mask[(ptrdiff_t)qPos * inner + j] : 0.f);
        sumv += expf(v - maxv);
    }
    sumv = blockReduceSum256(sumv, smem);

    const float inv = 1.f / fmaxf(sumv, 1e-12f);
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        float v = rowBuf[j] * scale + (mask ? mask[(ptrdiff_t)qPos * inner + j] : 0.f);
        rowBuf[j] = expf(v - maxv) * inv;
    }
}

/**
 * Одна строка = один позиционный токен [batch*seq]: стабильный softmax, CE loss (atomic),
 * градиент (p - one_hot) * scale. Невалидный target: строка градиента в ноль, в loss не входит.
 */
__global__ void cross_entropy_softmax_grad_loss_kernel(const float* logits, const float* targets_f,
                                                       float* grad, float scale, float* loss_sum,
                                                       unsigned int* valid_count, int nrows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * vocab;
    int t = (int)targets_f[row];
    if (t < 0 || t >= vocab) {
        for (int v = 0; v < vocab; ++v) {
            grad[base + v] = 0.f;
        }
        return;
    }
    float maxv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        maxv = fmaxf(maxv, logits[base + v]);
    }
    float sumExp = 0.f;
    for (int v = 0; v < vocab; ++v) {
        sumExp += expf(logits[base + v] - maxv);
    }
    float logProb = logits[base + t] - maxv - logf(sumExp);
    atomicAdd(loss_sum, -logProb);
    atomicAdd(valid_count, 1U);
    float invSum = 1.f / sumExp;
    for (int v = 0; v < vocab; ++v) {
        float p = expf(logits[base + v] - maxv) * invSum;
        float g = p;
        if (v == t) {
            g -= 1.f;
        }
        grad[base + v] = g * scale;
    }
}

/**
 * Fused CE + grad (ветка use_fp16_softmax в JNI): exp в FP32, без half-округления диффа (sumExp=0 / Inf).
 */
__global__ void cross_entropy_softmax_grad_loss_kernel_fp16(const float* logits, const float* targets_f,
                                                              float* grad, float scale, float* loss_sum,
                                                              unsigned int* valid_count, int nrows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * vocab;
    int t = (int)targets_f[row];
    if (t < 0 || t >= vocab) {
        for (int v = 0; v < vocab; ++v) {
            grad[base + v] = 0.f;
        }
        return;
    }
    float maxv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        maxv = fmaxf(maxv, logits[base + v]);
    }
    float sumExp = 0.f;
    for (int v = 0; v < vocab; ++v) {
        float diff = logits[base + v] - maxv;
        sumExp += expf(diff);
    }
    sumExp = fmaxf(sumExp, 1e-12f);
    float logProb = logits[base + t] - maxv - logf(sumExp);
    atomicAdd(loss_sum, -logProb);
    atomicAdd(valid_count, 1U);
    float invSum = 1.f / sumExp;
    for (int v = 0; v < vocab; ++v) {
        float diff = logits[base + v] - maxv;
        float p = expf(diff) * invSum;
        float g = p;
        if (v == t) {
            g -= 1.f;
        }
        grad[base + v] = g * scale;
    }
}

/** CE + softmax + grad; targets как int32 на строку (как (int)targets_f[row] в float-версии). */
__global__ void cross_entropy_softmax_grad_loss_kernel_i32(const float* logits, const int* targets_i,
                                                           float* grad, float scale, float* loss_sum,
                                                           unsigned int* valid_count, int nrows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * vocab;
    int t = targets_i[row];
    if (t < 0 || t >= vocab) {
        for (int v = 0; v < vocab; ++v) {
            grad[base + v] = 0.f;
        }
        return;
    }
    float maxv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        maxv = fmaxf(maxv, logits[base + v]);
    }
    float sumExp = 0.f;
    for (int v = 0; v < vocab; ++v) {
        sumExp += expf(logits[base + v] - maxv);
    }
    float logProb = logits[base + t] - maxv - logf(sumExp);
    atomicAdd(loss_sum, -logProb);
    atomicAdd(valid_count, 1U);
    float invSum = 1.f / sumExp;
    for (int v = 0; v < vocab; ++v) {
        float p = expf(logits[base + v] - maxv) * invSum;
        float g = p;
        if (v == t) {
            g -= 1.f;
        }
        grad[base + v] = g * scale;
    }
}

__global__ void cross_entropy_softmax_grad_loss_kernel_fp16_i32(const float* logits, const int* targets_i,
                                                                float* grad, float scale, float* loss_sum,
                                                                unsigned int* valid_count, int nrows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * vocab;
    int t = targets_i[row];
    if (t < 0 || t >= vocab) {
        for (int v = 0; v < vocab; ++v) {
            grad[base + v] = 0.f;
        }
        return;
    }
    float maxv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        maxv = fmaxf(maxv, logits[base + v]);
    }
    float sumExp = 0.f;
    for (int v = 0; v < vocab; ++v) {
        float diff = logits[base + v] - maxv;
        sumExp += expf(diff);
    }
    sumExp = fmaxf(sumExp, 1e-12f);
    float logProb = logits[base + t] - maxv - logf(sumExp);
    atomicAdd(loss_sum, -logProb);
    atomicAdd(valid_count, 1U);
    float invSum = 1.f / sumExp;
    for (int v = 0; v < vocab; ++v) {
        float diff = logits[base + v] - maxv;
        float p = expf(diff) * invSum;
        float g = p;
        if (v == t) {
            g -= 1.f;
        }
        grad[base + v] = g * scale;
    }
}

__global__ void layer_norm_fwd_kernel(const float* src, const float* gamma, const float* beta,
                                     float* dst, int outer, int lastDim, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer) {
        return;
    }
    int base = i * lastDim;
    float mean = 0.f;
    for (int j = 0; j < lastDim; j++) {
        mean += src[base + j];
    }
    mean /= (float) lastDim;
    float var = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float d = src[base + j] - mean;
        var += d * d;
    }
    var /= (float) lastDim;
    float invStd = rsqrtf(var + eps);
    for (int j = 0; j < lastDim; j++) {
        float normalized = (src[base + j] - mean) * invStd;
        dst[base + j] = normalized * gamma[j] + beta[j];
    }
}

__global__ void rms_norm_fwd_kernel(const float* src, const float* gamma, float* dst,
                                    int outer, int lastDim, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer) {
        return;
    }
    int base = i * lastDim;
    float sumSq = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float v = src[base + j];
        sumSq += v * v;
    }
    float rms = sqrtf(sumSq / (float) lastDim + eps);
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;
    for (int j = 0; j < lastDim; j++) {
        dst[base + j] = src[base + j] * invRms * gamma[j];
    }
}

/**
 * Ветка «fp16» в API: раньше x/γ округлялись до half → при больших |x| half давал Inf, далее Q/K/V и softmax ломались.
 * Считаем как rms_norm_fwd_kernel (FP32), выход по-прежнему float.
 */
__global__ void rms_norm_fwd_kernel_fp16(const float* src, const float* gamma, float* dst,
                                         int outer, int lastDim, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer) {
        return;
    }
    int base = i * lastDim;
    float sumSq = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float v = src[base + j];
        sumSq += v * v;
    }
    float rms = sqrtf(sumSq / (float) lastDim + eps);
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;
    for (int j = 0; j < lastDim; j++) {
        dst[base + j] = src[base + j] * invRms * gamma[j];
    }
}

/**
 * RMSNorm forward, block-per-row.
 * Один блок kSoftmaxBlockDim нитей обрабатывает одну строку.
 * При lastDim=256: ровно 1 нить на элемент, серийные циклы исчезают.
 * Запуск: gridDim.x=outer, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void rms_norm_fwd_block_kernel(
        const float* __restrict__ src, const float* __restrict__ gamma,
        float* __restrict__ dst, int outer, int lastDim, float eps) {
    extern __shared__ float smem[];
    const int o = blockIdx.x;
    if (o >= outer) return;
    const int tid = threadIdx.x;

    float sumSq = 0.f;
    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim)
        sumSq += src[o * lastDim + j] * src[o * lastDim + j];
    sumSq = blockReduceSum256(sumSq, smem);

    float rms = sqrtf(sumSq / (float) lastDim + eps);
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;

    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim)
        dst[o * lastDim + j] = src[o * lastDim + j] * invRms * gamma[j];
}

static constexpr int kRmsNormFwdBlockThreshold = 64;

#define CUDA_CHECK_X(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

#define CUDA_KERNEL_CHECK_RV(rv) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return (rv); \
        } \
    } while (0)

static void launch_rms_norm_fwd(const float* src, const float* gamma, float* dst,
        int outer, int lastDim, float eps) {
    if (lastDim >= kRmsNormFwdBlockThreshold) {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        rms_norm_fwd_block_kernel<<<outer, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(
                src, gamma, dst, outer, lastDim, eps);
    } else {
        int threads = jgpt_cuda_get_optimal_block_size();
        int blocks = (outer + threads - 1) / threads;
        rms_norm_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
                src, gamma, dst, outer, lastDim, eps);
    }
    CUDA_KERNEL_CHECK();
}

__device__ float gelu_tanh_dev(float x) {
    const float SQRT_2_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    float x3 = x * x * x;
    float tanhArg = SQRT_2_PI * (x + COEF * x3);
    return 0.5f * x * (1.0f + tanhf(tanhArg));
}

__global__ void gelu_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = gelu_tanh_dev(src[i]);
    }
}

__global__ void sigmoid_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = src[i];
        float s;
        if (x >= 20.f) {
            s = 1.f;
        } else if (x <= -20.f) {
            s = 0.f;
        } else {
            s = 1.f / (1.f + expf(-x));
        }
        dst[i] = s;
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void mul_scalar_kernel(const float* a, float* b, int n, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i] * scalar;
    }
}

__global__ void scale_inplace_kernel(float* a, int n, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= scalar;
    }
}

__global__ void sum_squares_kernel(const float* src, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = src[i];
        atomicAdd(out, v * v);
    }
}

/** Для устойчивой суммы квадратов: далее cublasDdot(dst,dst) в double. */
__global__ void float_to_double_kernel(const float* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (double) src[i];
    }
}

__global__ void adamw_step_kernel(
        float* param, const float* grad, float* m, float* v,
        float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, float invBias1, float invBias2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float gi = grad[i];
    float mi = beta1 * m[i] + (1.f - beta1) * gi;
    float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float mHat = mi * invBias1;
    float vHat = vi * invBias2;
    param[i] -= learningRate * (mHat / (sqrtf(vHat) + epsilon) + weightDecay * param[i]);
}

/** Один launch: по блоку на сегмент (разные device-указатели), потоки обходят длину с шагом blockDim.x. */
__global__ void adamw_step_kernel_segments(
        const uintptr_t* d_param_ptrs,
        const uintptr_t* d_grad_ptrs,
        const uintptr_t* d_m_ptrs,
        const uintptr_t* d_v_ptrs,
        const int* d_lengths,
        int num_segments,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        float invBias1,
        float invBias2) {
    int seg = blockIdx.x;
    if (seg >= num_segments) {
        return;
    }
    int n = d_lengths[seg];
    if (n <= 0) {
        return;
    }
    float* param = reinterpret_cast<float*>(d_param_ptrs[seg]);
    const float* grad = reinterpret_cast<const float*>(d_grad_ptrs[seg]);
    float* m = reinterpret_cast<float*>(d_m_ptrs[seg]);
    float* v = reinterpret_cast<float*>(d_v_ptrs[seg]);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float gi = grad[i];
        float mi = beta1 * m[i] + (1.f - beta1) * gi;
        float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float mHat = mi * invBias1;
        float vHat = vi * invBias2;
        param[i] -= learningRate * (mHat / (sqrtf(vHat) + epsilon) + weightDecay * param[i]);
    }
}

__global__ void embedding_token_fwd_kernel(
        const float* tokens, const float* weights, float* out, int batch, int seqLen, int dModel, int vocabSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    int b = t0 / seqLen;
    int token = (int) tokens[b * seqLen + s];
    if (token < 0 || token >= vocabSize) {
        out[idx] = 0.f;
        return;
    }
    out[idx] = weights[(size_t) token * (size_t) dModel + (size_t) j];
}

/**
 * Ветка «fp16» в API: раньше вес строки эмбеддинга прогоняли через half → крупные веса давали Inf в out.
 * Считаем как embedding_token_fwd_kernel (без квантования веса).
 */
__global__ void embedding_token_fwd_kernel_fp16(
        const float* tokens, const float* weights, float* out, int batch, int seqLen, int dModel, int vocabSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    int b = t0 / seqLen;
    int token = (int) tokens[b * seqLen + s];
    if (token < 0 || token >= vocabSize) {
        out[idx] = 0.f;
        return;
    }
    out[idx] = weights[(size_t) token * (size_t) dModel + (size_t) j];
}

__global__ void embedding_token_bwd_kernel(
        const float* tokens, const float* gradOut, float* gradWeights,
        int batch, int seqLen, int dModel, int vocabSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    int b = t0 / seqLen;
    int token = (int) tokens[b * seqLen + s];
    if (token < 0 || token >= vocabSize) {
        return;
    }
    atomicAdd(&gradWeights[token * dModel + j], gradOut[idx]);
}

__global__ void embedding_position_bwd_kernel(
        const float* gradCombined, float* gradWeights,
        int batch, int seqLen, int dModel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    atomicAdd(&gradWeights[s * dModel + j], gradCombined[idx]);
}

/** x[b,s,j] += posWeights[posRowStart + s, j] (таблица [>=posRowStart+seqLen, dModel] row-major). */
__global__ void add_position_embedding_broadcast_kernel(
        float* x, const float* posWeights, int posRowStart, int batch, int seqLen, int dModel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    x[idx] += posWeights[(size_t)(posRowStart + s) * (size_t) dModel + (size_t) j];
}

__global__ void apply_causal_mask_3d_kernel(const float* scores, const float* mask, float* out,
                                            int batch, int seqLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * seqLen;
    if (idx >= total) {
        return;
    }
    int j = idx % seqLen;
    int i = idx / seqLen;
    int queryRow = i % seqLen;
    out[idx] = scores[idx] + mask[queryRow * seqLen + j];
}

__global__ void transpose_2d_last_kernel(const float* src, float* dst, int d0, int d1, int d2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2;
    if (idx >= total) {
        return;
    }
    int c = idx % d2;
    int t = idx / d2;
    int b = t % d1;
    int a = t / d1;
    int dstIdx = a * (d2 * d1) + c * d1 + b;
    dst[dstIdx] = src[idx];
}

__global__ void split_heads_kernel(const float* src, float* dst, int batch, int seqLen, int dModel, int numHeads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int dHead = dModel / numHeads;
    int j = idx % dModel;
    int t = idx / dModel;
    int i = t % seqLen;
    int b = t / seqLen;
    int h = j / dHead;
    int jd = j % dHead;
    int r0 = numHeads * seqLen * dHead;
    int r1 = seqLen * dHead;
    int r2 = dHead;
    int dstIdx = b * r0 + h * r1 + i * r2 + jd;
    dst[dstIdx] = src[idx];
}

/** Головы K или V в row-major [batch, H, seq, dHead]: копия среза батча в кэш {@code head * maxSeqLen * dHead + pos * dHead}. */
__global__ void kv_heads4d_to_cache_kernel(
        const float* src, float* dst, int numHeads, int seqLen, int maxSeqLen, int dHead) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numHeads * seqLen * dHead;
    if (idx >= total) {
        return;
    }
    int j = idx % dHead;
    int s = (idx / dHead) % seqLen;
    int h = idx / (seqLen * dHead);
    int srcO = h * (seqLen * dHead) + s * dHead + j;
    int dstO = h * (maxSeqLen * dHead) + s * dHead + j;
    dst[dstO] = src[srcO];
}

__global__ void concat_heads_kernel(const float* src, float* dst, int batch, int numHeads, int seqLen, int dHead) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dModel = numHeads * dHead;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t = idx / dModel;
    int i = t % seqLen;
    int b = t / seqLen;
    int h = j / dHead;
    int jd = j % dHead;
    int s0 = numHeads * seqLen * dHead;
    int s1 = seqLen * dHead;
    int s2 = dHead;
    int s3 = 1;
    int srcIdx = b * s0 + h * s1 + i * s2 + jd * s3;
    dst[idx] = src[srcIdx];
}

__global__ void rope_4d_kernel(const float* src, float* dst, int batch, int numHeads, int seqLen, int dHead,
                               const int* positions, int posLen, int posBaseOffset) {
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfPairs = dHead / 2;
    int total = batch * numHeads * seqLen * halfPairs;
    if (pairIdx >= total) {
        return;
    }
    int j = pairIdx % halfPairs;
    int tmp = pairIdx / halfPairs;
    int i = tmp % seqLen;
    tmp /= seqLen;
    int h = tmp % numHeads;
    int b = tmp / numHeads;

    int s0 = numHeads * seqLen * dHead;
    int s1 = seqLen * dHead;
    int s2 = dHead;
    int s3 = 1;
    int base = b * s0 + h * s1 + i * s2;
    int idx1 = base + (2 * j) * s3;
    int idx2 = base + (2 * j + 1) * s3;
    int p = (positions != nullptr && i < posLen) ? positions[i] : (i + posBaseOffset);
    float theta = (float) p / powf(10000.f, (2.f * (float) j) / (float) dHead);
    float c = cosf(theta);
    float s = sinf(theta);
    float x1 = src[idx1];
    float x2 = src[idx2];
    dst[idx1] = x1 * c - x2 * s;
    dst[idx2] = x1 * s + x2 * c;
}

__global__ void rope_4d_bwd_kernel(const float* gradY, float* gradX, int batch, int numHeads, int seqLen, int dHead,
                                   const int* positions, int posLen, int posBaseOffset) {
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfPairs = dHead / 2;
    int total = batch * numHeads * seqLen * halfPairs;
    if (pairIdx >= total) {
        return;
    }
    int j = pairIdx % halfPairs;
    int tmp = pairIdx / halfPairs;
    int i = tmp % seqLen;
    tmp /= seqLen;
    int h = tmp % numHeads;
    int b = tmp / numHeads;

    int s0 = numHeads * seqLen * dHead;
    int s1 = seqLen * dHead;
    int s2 = dHead;
    int base = b * s0 + h * s1 + i * s2;
    int idx1 = base + 2 * j;
    int idx2 = base + 2 * j + 1;
    int p = (positions != nullptr && i < posLen) ? positions[i] : (i + posBaseOffset);
    float theta = (float) p / powf(10000.f, (2.f * (float) j) / (float) dHead);
    float c = cosf(theta);
    float s = sinf(theta);
    float gy1 = gradY[idx1];
    float gy2 = gradY[idx2];
    gradX[idx1] += gy1 * c + gy2 * s;
    gradX[idx2] += -gy1 * s + gy2 * c;
}

__global__ void softmax_last_dim_bwd_kernel(const float* gOut, const float* p, float* gIn, int nrows, int inner) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * inner;
    float dot = 0.f;
    for (int j = 0; j < inner; j++) {
        dot += p[base + j] * gOut[base + j];
    }
    for (int j = 0; j < inner; j++) {
        gIn[base + j] += p[base + j] * (gOut[base + j] - dot);
    }
}

/**
 * Backward softmax по последней оси, block-per-row. VJP: gIn[j] += p[j]*(gOut[j] - dot(p,gOut)).
 * Запуск: gridDim.x=nrows, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void softmax_last_dim_bwd_block_kernel(
        const float* __restrict__ gOut, const float* __restrict__ p,
        float* __restrict__ gIn, int nrows, int inner) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int tid = threadIdx.x;
    const ptrdiff_t base = (ptrdiff_t)row * inner;

    float dot = 0.f;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        dot += p[base + j] * gOut[base + j];
    dot = blockReduceSum256(dot, smem);

    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        gIn[base + j] += p[base + j] * (gOut[base + j] - dot);
}

/* ========== Missing helper kernels (были пропущены) ========== */

/** Транспонирование последних двух измерений: [batch, d1, d2] → [batch, d2, d1] */
static __global__ void transpose_last2_3d_kernel(const float* __restrict__ src, float* __restrict__ dst,
        int batch, int d1, int d2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d1 * d2;
    if (idx >= total) {
        return;
    }
    int k = idx % d2;
    int tmp = idx / d2;
    int j = tmp % d1;
    int b = tmp / d1;
    // dst[b][k][j] = src[b][j][k]
    int dstIdx = (b * d2 + k) * d1 + j;
    dst[dstIdx] = src[idx];
}

/** Scale inplace (extra версия для attention) */
static __global__ void scale_inplace_kernel_extra(float* __restrict__ a, int n, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= scalar;
    }
}

/** Добавление causal mask inplace: scores += mask */
static __global__ void add_mask_inplace_kernel(float* __restrict__ scores, const float* __restrict__ mask,
        int batch, int seqLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * seqLen;
    if (idx >= total) {
        return;
    }
    int j = idx % seqLen;
    int row = idx / seqLen;
    int i = row % seqLen;
    scores[idx] += mask[i * seqLen + j];
}

__global__ void multiply_bwd_kernel(const float* gOut, const float* a, const float* b, float* gA, float* gB, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gA[i] += gOut[i] * b[i];
        gB[i] += gOut[i] * a[i];
    }
}

__global__ void accumulate_add_kernel(float* acc, const float* delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        acc[i] += delta[i];
    }
}

/** acc[i] += delta[i] * scale (scale = -1 для вычитания delta из acc). */
__global__ void accumulate_scaled_add_kernel(float* acc, const float* delta, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        acc[i] += delta[i] * scale;
    }
}

__global__ void gelu_bwd_kernel(const float* gOut, const float* inp, float* gIn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float x = inp[i];
    const float SQRT_2_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    float x3 = x * x * x;
    float tanhArg = SQRT_2_PI * (x + COEF * x3);
    float t = tanhf(tanhArg);
    float sech2 = 1.f - t * t;
    float dtanh = SQRT_2_PI * (1.f + 3.f * COEF * x * x) * sech2;
    float ddx = 0.5f * x * dtanh + 0.5f * (1.f + t);
    gIn[i] += gOut[i] * ddx;
}

__global__ void sigmoid_bwd_kernel(const float* gOut, const float* inp, float* gIn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float x = inp[i];
    float s;
    if (x >= 20.f) {
        s = 1.f;
    } else if (x <= -20.f) {
        s = 0.f;
    } else {
        s = 1.f / (1.f + expf(-x));
    }
    gIn[i] += gOut[i] * s * (1.f - s);
}

__global__ void relu_bwd_kernel(const float* gOut, const float* inp, float* gIn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && inp[i] > 0.f) {
        gIn[i] += gOut[i];
    }
}

__global__ void layer_norm_bwd_kernel(const float* gOut, const float* x, const float* gamma, float eps,
                                      float* gX, float* gGamma, float* gBeta, int outer, int lastDim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer) {
        return;
    }
    int base = i * lastDim;
    float mean = 0.f;
    for (int j = 0; j < lastDim; j++) {
        mean += x[base + j];
    }
    mean /= (float) lastDim;
    float var = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float d = x[base + j] - mean;
        var += d * d;
    }
    var /= (float) lastDim;
    float invStd = rsqrtf(var + eps);

    float sumDh = 0.f;
    float sumDhXh = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float xh = (x[base + j] - mean) * invStd;
        float dh = gOut[base + j] * gamma[j];
        sumDh += dh;
        sumDhXh += dh * xh;
    }
    float meanDh = sumDh / (float) lastDim;
    float meanDhXh = sumDhXh / (float) lastDim;

    for (int j = 0; j < lastDim; j++) {
        float xh = (x[base + j] - mean) * invStd;
        atomicAdd(&gGamma[j], gOut[base + j] * xh);
        atomicAdd(&gBeta[j], gOut[base + j]);
        float dh = gOut[base + j] * gamma[j];
        float dx = invStd * (dh - meanDh - xh * meanDhXh);
        gX[base + j] += dx;
    }
}

__global__ void rms_norm_bwd_kernel(const float* gOut, const float* x, const float* gamma, float eps,
                                    float* gX, float* gGamma, int outer, int lastDim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outer) {
        return;
    }
    int base = o * lastDim;
    float sumSq = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float v = x[base + j];
        sumSq += v * v;
    }
    float ms = sumSq / (float) lastDim;
    float rms = sqrtf(ms + eps);
    // Согласовано с rms_norm_fwd_kernel*: при не-конечном rms не даём NaN расползаться по ∂x/∂γ.
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;

    float sumXGrad = 0.f;
    for (int j = 0; j < lastDim; j++) {
        sumXGrad += x[base + j] * gamma[j] * gOut[base + j];
    }
    float meanXGrad = sumXGrad / (float) lastDim;

    for (int j = 0; j < lastDim; j++) {
        float xj = x[base + j];
        int idx = base + j;
        atomicAdd(&gGamma[j], gOut[idx] * xj * invRms);
        gX[idx] += invRms * gamma[j] * gOut[idx] - invRms * invRms * invRms * xj * meanXGrad;
    }
}

/**
 * RMSNorm backward, block-per-row. Одна строка = один блок kSoftmaxBlockDim нитей.
 * При lastDim = d_model = 256: ровно 1 нить на элемент, серийные циклы исчезают.
 * Запуск: gridDim.x=outer, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void rms_norm_bwd_block_kernel(
        const float* __restrict__ gOut, const float* __restrict__ x,
        const float* __restrict__ gamma, float eps,
        float* __restrict__ gX, float* __restrict__ gGamma, int outer, int lastDim) {
    extern __shared__ float smem[];
    const int o = blockIdx.x;
    if (o >= outer) return;
    const int tid = threadIdx.x;

    // Вычислить sumSq = sum(x[j]^2) по строке
    float sumSq = 0.f;
    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim)
        sumSq += x[o * lastDim + j] * x[o * lastDim + j];
    sumSq = blockReduceSum256(sumSq, smem);

    float rms = sqrtf(sumSq / (float) lastDim + eps);
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;

    // Вычислить sumXGrad = sum(x[j] * gamma[j] * gOut[j])
    float localXG = 0.f;
    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim)
        localXG += x[o * lastDim + j] * gamma[j] * gOut[o * lastDim + j];
    float sumXGrad = blockReduceSum256(localXG, smem);
    float meanXGrad = sumXGrad / (float) lastDim;

    // Записать gX; gGamma накапливается атомарно (across outer строк)
    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim) {
        int idx = o * lastDim + j;
        atomicAdd(&gGamma[j], gOut[idx] * x[idx] * invRms);
        gX[idx] += invRms * gamma[j] * gOut[idx] - invRms * invRms * invRms * x[idx] * meanXGrad;
    }
}

static constexpr int kRmsNormBwdBlockThreshold = 64; // block-per-row при lastDim >= этого

static void launch_rms_norm_bwd(const float* gOut, const float* x, const float* gamma, float eps,
        float* gX, float* gGamma, int outer, int lastDim) {
    if (lastDim >= kRmsNormBwdBlockThreshold) {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        rms_norm_bwd_block_kernel<<<outer, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(
                gOut, x, gamma, eps, gX, gGamma, outer, lastDim);
    } else {
        int threads = jgpt_cuda_get_optimal_block_size();
        int blocks = (outer + threads - 1) / threads;
        rms_norm_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
                gOut, x, gamma, eps, gX, gGamma, outer, lastDim);
    }
    CUDA_KERNEL_CHECK();
}

static void launch_softmax_last_dim(const float* src, float* dst, int nrows, int inner, bool use_fp16_softmax) {
    if (inner >= kSoftmaxBlockThreshold) {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        softmax_last_dim_block_kernel<<<nrows, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(src, dst, nrows, inner);
    } else {
        int threads = jgpt_cuda_get_optimal_block_size();
        int blocks = (nrows + threads - 1) / threads;
        if (use_fp16_softmax) {
            softmax_last_dim_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(src, dst, nrows, inner);
        } else {
            softmax_last_dim_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(src, dst, nrows, inner);
        }
    }
    CUDA_KERNEL_CHECK();
}

static void launch_softmax_last_dim_bwd(const float* gOut, const float* p, float* gIn, int nrows, int inner) {
    if (inner >= kSoftmaxBlockThreshold) {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        softmax_last_dim_bwd_block_kernel<<<nrows, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(gOut, p, gIn, nrows, inner);
    } else {
        int threads = jgpt_cuda_get_optimal_block_size();
        int blocks = (nrows + threads - 1) / threads;
        softmax_last_dim_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(gOut, p, gIn, nrows, inner);
    }
    CUDA_KERNEL_CHECK();
}

static void launch_cross_entropy(const float* logits, const float* targets, float* grad, float scale,
        float* loss_sum, unsigned int* valid, int nrows, int vocab, bool use_fp16_softmax) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (nrows + threads - 1) / threads;
    if (use_fp16_softmax) {
        cross_entropy_softmax_grad_loss_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(
                logits, targets, grad, scale, loss_sum, valid, nrows, vocab);
    } else {
        cross_entropy_softmax_grad_loss_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
                logits, targets, grad, scale, loss_sum, valid, nrows, vocab);
    }
    CUDA_KERNEL_CHECK();
}

static void launch_cross_entropy_i32(const float* logits, const int* targets_i, float* grad, float scale,
        float* loss_sum, unsigned int* valid, int nrows, int vocab, bool use_fp16_softmax) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (nrows + threads - 1) / threads;
    if (use_fp16_softmax) {
        cross_entropy_softmax_grad_loss_kernel_fp16_i32<<<blocks, threads, 0, kTensorCudaStream>>>(
                logits, targets_i, grad, scale, loss_sum, valid, nrows, vocab);
    } else {
        cross_entropy_softmax_grad_loss_kernel_i32<<<blocks, threads, 0, kTensorCudaStream>>>(
                logits, targets_i, grad, scale, loss_sum, valid, nrows, vocab);
    }
    CUDA_KERNEL_CHECK();
}

__global__ void float_token_ids_to_int32_kernel(const float* src, int* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = static_cast<int>(src[i]);
    }
}

static void launch_float_to_int32_targets(const float* d_float_src, int* d_int_dst, int n) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    float_token_ids_to_int32_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_float_src, d_int_dst, n);
    CUDA_KERNEL_CHECK();
}

static inline void* jni_direct_ptr(JNIEnv* env, jobject buf, jlong byte_off, jlong need_bytes, const char* ctx) {
    if (buf == nullptr) {
        fprintf(stderr, "%s: null buffer\n", ctx);
        return nullptr;
    }
    void* addr = env->GetDirectBufferAddress(buf);
    if (addr == nullptr) {
        fprintf(stderr, "%s: not a direct buffer\n", ctx);
        return nullptr;
    }
    jlong cap = env->GetDirectBufferCapacity(buf);
    if (byte_off < 0 || need_bytes < 0 || byte_off > cap || need_bytes > cap - byte_off) {
        fprintf(stderr, "%s: range out of capacity (off=%lld need=%lld cap=%lld)\n", ctx, (long long) byte_off,
                (long long) need_bytes, (long long) cap);
        return nullptr;
    }
    return static_cast<void*>(static_cast<char*>(addr) + byte_off);
}

static bool batched_sgemm_row_major_extra(
        const float* d_A, const float* d_B, float* d_C,
        int batchCount, int M, int K, int N, float alpha, float beta) {
    cublasHandle_t handle = get_extra_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm: cuBLAS handle unavailable\n");
        return false;
    }
    long long strideA = (long long) M * (long long) K;
    long long strideB = (long long) K * (long long) N;
    long long strideC = (long long) M * (long long) N;
    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_B,
            N,
            strideB,
            d_A,
            K,
            strideA,
            &beta,
            d_C,
            N,
            strideC,
            batchCount);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm failed: status %d\n", (int) st);
        return false;
    }
    return true;
}

// Row-major C(M×N) = A(M×K) × B^T,  где B хранится как N×K row-major.
// Устраняет необходимость явно транспонировать B перед GEMM.
static bool batched_sgemm_row_major_transB(
        const float* d_A, const float* d_B, float* d_C,
        int batchCount, int M, int K, int N, float alpha, float beta) {
    cublasHandle_t handle = get_extra_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm (transB): cuBLAS handle unavailable\n");
        return false;
    }
    long long strideA = (long long) M * K;
    long long strideB = (long long) N * K;   // B is N×K row-major
    long long strideC = (long long) M * N;
    // cuBLAS col-major: C_col(N×M) = B_col^T(N×K) × A_col(K×M)
    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,   // transpose B (K×N col-major → N×K)
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, K, strideB,    // B: N×K row-major = K×N col-major, ldb=K
            d_A, K, strideA,    // A: M×K row-major = K×M col-major, lda=K
            &beta,
            d_C, N, strideC,
            batchCount);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm (transB) failed: %d\n", (int) st);
        return false;
    }
    return true;
}

// Row-major C(M×N) = A^T × B,  где A хранится как K×M row-major, B — K×N row-major.
// Устраняет необходимость явно транспонировать A перед GEMM.
static bool batched_sgemm_row_major_transA(
        const float* d_A, const float* d_B, float* d_C,
        int batchCount, int M, int K, int N, float alpha, float beta) {
    cublasHandle_t handle = get_extra_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm (transA): cuBLAS handle unavailable\n");
        return false;
    }
    long long strideA = (long long) K * M;   // A is K×M row-major
    long long strideB = (long long) K * N;   // B is K×N row-major
    long long strideC = (long long) M * N;
    // cuBLAS col-major: C_col(N×M) = B_col(N×K) × A_col^T(K×M)
    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,   // transpose A (M×K col-major → K×M)
            N, M, K,
            &alpha,
            d_B, N, strideB,    // B: K×N row-major = N×K col-major, ldb=N
            d_A, M, strideA,    // A: K×M row-major = M×K col-major, lda=M
            &beta,
            d_C, N, strideC,
            batchCount);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm (transA) failed: %d\n", (int) st);
        return false;
    }
    return true;
}

/* Non-graph attention forward scratch (mask slot included). May be reallocated by non-graph ops. */
static thread_local void* tl_attn_fwd_aux = nullptr;
static thread_local size_t tl_attn_fwd_aux_total = 0;

static bool attn_fwd_aux_ensure(
        size_t bytesQK,
        size_t bytesProb,
        size_t bytesMask,
        bool with_mask,
        float** d_scores,
        float** d_probs,
        float** d_mask) {
    size_t total = bytesProb * 2U;
    if (with_mask) {
        total += bytesMask;
    }
    if (total > tl_attn_fwd_aux_total) {
        cudaFree(tl_attn_fwd_aux);
        tl_attn_fwd_aux = nullptr;
        tl_attn_fwd_aux_total = 0;
        if (cudaMalloc(&tl_attn_fwd_aux, total) != cudaSuccess) {
            return false;
        }
        tl_attn_fwd_aux_total = total;
    }
    unsigned char* base = static_cast<unsigned char*>(tl_attn_fwd_aux);
    size_t off = 0;
    *d_scores = reinterpret_cast<float*>(base + off);
    off += bytesProb;
    *d_probs = reinterpret_cast<float*>(base + off);
    off += bytesProb;
    if (with_mask) {
        *d_mask = reinterpret_cast<float*>(base + off);
    } else {
        *d_mask = nullptr;
    }
    (void) bytesQK;
    return true;
}

/*
 * Отдельный буфер только для CUDA-graph пути (scaledDotProductAttentionForwardGPUDeviceResident
 * и jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas).
 *
 * КРИТИЧЕСКИ ВАЖНО: этот буфер НЕ должен совпадать с tl_attn_fwd_aux.
 * При захвате CUDA graph GPU-адреса d_scores/d_probs фиксируются в графе.
 * Если tl_attn_fwd_aux перевыделяется другими операциями (например, генерацией текста через
 * scaledDotProductAttentionForwardGPUDevice с маской), GPU-адреса в графе устаревают →
 * cudaGraphLaunch падает с "illegal memory access".
 * Использование отдельного буфера (не tl_attn_fwd_aux) гарантирует, что
 * нe-graph операции не могут изменить указатели, захваченные в графе.
 *
 * Глобальный (процессный) буфер с mutex: один и тот же device-адрес для всех слоёв декодера на одном GPU —
 * иначе thread_local копии не делят память между вызовами и при глубоком стеке растёт пик VRAM.
 *
 * Размер = bytesProb (одна матрица B×S×S): QK^T пишет logits, затем in-place softmax → probs, затем GEMM(P,V).
 * Раньше было 2×bytesProb (отдельные scores/probs); уполовинивание снимает ~100 MiB+ VRAM на типичных S и снижает OOM на cudaGraphLaunch.
 */
static std::mutex g_attn_fwd_graph_aux_mu;
static void* g_attn_fwd_graph_aux = nullptr;
static size_t g_attn_fwd_graph_aux_total = 0;

/** Workspace без слота под mask: mask передаётся отдельным device-указателем (для CUDA graph / без H2D probs). */
static bool attn_fwd_aux_ensure_qk_probs_only(
        size_t bytesQK,
        size_t bytesProb,
        float** d_scores,
        float** d_probs) {
    size_t total = bytesProb;
    std::lock_guard<std::mutex> lock(g_attn_fwd_graph_aux_mu);
    if (total > g_attn_fwd_graph_aux_total
            || (g_attn_fwd_graph_aux != nullptr && total < g_attn_fwd_graph_aux_total)) {
        cudaFree(g_attn_fwd_graph_aux);
        g_attn_fwd_graph_aux = nullptr;
        g_attn_fwd_graph_aux_total = 0;
        if (cudaMalloc(&g_attn_fwd_graph_aux, total) != cudaSuccess) {
            return false;
        }
        g_attn_fwd_graph_aux_total = total;
    }
    unsigned char* base = static_cast<unsigned char*>(g_attn_fwd_graph_aux);
    float* p = reinterpret_cast<float*>(base);
    *d_scores = p;
    *d_probs = p;
    (void) bytesQK;
    return true;
}

void jgpt_cuda_decoder_graph_debug_aux_snapshot(
        uintptr_t* fwd_ptr, uintptr_t* graph_ptr, size_t* fwd_sz, size_t* graph_sz) {
    if (fwd_ptr != nullptr) {
        *fwd_ptr = reinterpret_cast<uintptr_t>(tl_attn_fwd_aux);
    }
    if (fwd_sz != nullptr) {
        *fwd_sz = tl_attn_fwd_aux_total;
    }
    std::lock_guard<std::mutex> lock(g_attn_fwd_graph_aux_mu);
    if (graph_ptr != nullptr) {
        *graph_ptr = reinterpret_cast<uintptr_t>(g_attn_fwd_graph_aux);
    }
    if (graph_sz != nullptr) {
        *graph_sz = g_attn_fwd_graph_aux_total;
    }
}

static bool attn_fwd_run_core(
        float* d_q,
        float* d_k,
        float* d_v,
        float* d_scores,
        float* d_probs,
        float* d_out,
        float* d_mask,
        int batch,
        int seqLen,
        int dK,
        int dV,
        float scale,
        bool use_fp16_softmax) {
    (void) use_fp16_softmax;
    // Q × K^T → scores: transB GEMM (нет явного транспонирования K)
    if (!batched_sgemm_row_major_transB(d_q, d_k, d_scores, batch, seqLen, dK, seqLen, 1.0f, 0.0f)) {
        return false;
    }
    // Слитый scale + causal-mask + softmax: читаем d_scores один раз вместо трёх отдельных проходов
    int nrows = batch * seqLen;
    {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        if (d_probs == d_scores) {
            softmax_scaled_masked_inplace_block_kernel<<<nrows, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(
                    d_scores, d_mask, scale, nrows, seqLen);
        } else {
            softmax_scaled_masked_block_kernel<<<nrows, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(
                    d_scores, d_probs, d_mask, scale, nrows, seqLen);
        }
    }
    CUDA_KERNEL_CHECK_RV(false);
    if (!batched_sgemm_row_major_extra(d_probs, d_v, d_out, batch, seqLen, seqLen, dV, 1.0f, 0.0f)) {
        return false;
    }
    return true;
}

// ============================================================
//  FlashAttention-2 (causal, forward + backward)
//  Layout: Q/K/V/O  = [BH, S, Dh]  (BH = batch * numHeads)
//          LSE      = [BH, S]       log-sum-exp per query row
//          D        = [BH, S]       dot(dO, O) per query row (bwd scratch)
//  Tile sizes: Br = Bc = kFaBr.  Head dim: kFaDh (compile-time).
// ============================================================

static constexpr int kFaDh = 16;    // d_head
static constexpr int kFaBr = 64;    // query tile rows  (= block size for fwd / dQ)
static constexpr int kFaBc = 64;    // KV tile rows     (= block size for dKdV)

static_assert(kFaDh > 0 && (kFaDh % 2) == 0, "FlashAttention: d_head must be positive even");

/** Поднимает лимит dynamic shared memory, если плитка (Br/Bc/Dh) требует больше 48 KiB на блок. */
static cudaError_t flash_kernel_ensure_dyn_smem(const void* kernel, size_t dynamic_shmem) {
    constexpr size_t kDefaultDynSmem = 48u * 1024u;
    if (dynamic_shmem <= kDefaultDynSmem) {
        return cudaSuccess;
    }
    if (dynamic_shmem > static_cast<size_t>(INT_MAX)) {
        return cudaErrorInvalidValue;
    }
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shmem));
}

/* Thread-local scratch for D vector (small: BH*S floats); LSE пишется в буфер вызывающего. */
static thread_local float* tl_fa_D    = nullptr;
static thread_local size_t tl_fa_D_bytes   = 0;

static bool fa_ensure_D(size_t bytes) {
    if (bytes <= tl_fa_D_bytes) return true;
    void* raw = nullptr;
    if (cudaMalloc(&raw, bytes) != cudaSuccess) {
        fprintf(stderr, "fa_ensure_D: cudaMalloc(%zu) failed\n", bytes);
        return false;
    }
    if (tl_fa_D != nullptr) {
        (void) cudaFree(tl_fa_D);
    }
    tl_fa_D = static_cast<float*>(raw);
    tl_fa_D_bytes = bytes;
    return true;
}

// ----------------------------------------------------------------
//  Forward kernel
//  One block per (bh, q_tile).  Block: kFaBr threads.
//  Shared memory layout (floats):
//    q_smem [kFaBr][kFaDh]   — query tile, loaded once
//    k_smem [kFaBc][kFaDh]   — key tile, per KV step
//    v_smem [kFaBc][kFaDh]   — value tile, per KV step
//    s_smem [kFaBc][kFaBr]   — S_ij transposed: s_smem[j][i] = S[i][j]
//                               (transposed layout avoids bank conflicts when
//                                all threads read their own s column)
// ----------------------------------------------------------------
__global__ void __launch_bounds__(kFaBr)
flash_attn_fwd_kernel(
    const float* __restrict__ Q,    // [BH, S, Dh]
    const float* __restrict__ K,    // [BH, S, Dh]
    const float* __restrict__ V,    // [BH, S, Dh]
    float* __restrict__ O,          // [BH, S, Dh]
    float* __restrict__ LSE,        // [BH, S]
    int BH, int S, float scale)
{
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const int bh     = blockIdx.x / num_q_tiles;
    const int q_tile = blockIdx.x % num_q_tiles;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int qi  = q_tile * kFaBr + tid;

    const ptrdiff_t bh_off = (ptrdiff_t)bh * S * kFaDh;
    const float* Qp   = Q   + bh_off;
    const float* Kp   = K   + bh_off;
    const float* Vp   = V   + bh_off;
    float*       Op   = O   + bh_off;
    float*       LSEp = LSE + (ptrdiff_t)bh * S;

    // Smem
    extern __shared__ float smem[];
    float* q_smem = smem;                                         // [kFaBr][kFaDh]
    float* k_smem = q_smem + kFaBr * kFaDh;                      // [kFaBc][kFaDh]
    float* v_smem = k_smem + kFaBc * kFaDh;                      // [kFaBc][kFaDh]
    float* s_smem = v_smem + kFaBc * kFaDh;                      // [kFaBc][kFaBr] transposed

    // Load Q tile once (one row per thread)
    if (qi < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++)
            q_smem[tid * kFaDh + d] = Qp[qi * kFaDh + d];
    } else {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++)
            q_smem[tid * kFaDh + d] = 0.f;
    }
    __syncthreads();

    // Per-thread accumulators
    float o_reg[kFaDh];
    #pragma unroll
    for (int d = 0; d < kFaDh; d++) o_reg[d] = 0.f;
    float mi = -INFINITY;
    float li = 0.f;

    const int num_kv_tiles = (S + kFaBc - 1) / kFaBc;

    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        const int kv_start = kv_t * kFaBc;
        // Causal: skip tiles where ALL keys are beyond this query tile's last row
        if (kv_start > q_tile * kFaBr + kFaBr - 1) break;

        // Collaborative load of K and V tiles (kFaBr threads load kFaBc rows)
        for (int row = tid; row < kFaBc; row += kFaBr) {
            const int kj = kv_start + row;
            if (kj < S) {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[row * kFaDh + d] = Kp[kj * kFaDh + d];
                    v_smem[row * kFaDh + d] = Vp[kj * kFaDh + d];
                }
            } else {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[row * kFaDh + d] = 0.f;
                    v_smem[row * kFaDh + d] = 0.f;
                }
            }
        }
        __syncthreads();

        /* Все потоки блока обязаны выполнять одинаковое число __syncthreads() на итерацию.
         * Нельзя вызывать __syncthreads() только для qi >= S (ветка continue) — иначе UB CUDA. */
        if (qi < S) {
            // Compute S_ij for this thread's query row, store transposed: s_smem[j][tid]
            #pragma unroll
            for (int j = 0; j < kFaBc; j++) {
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dot = __fmaf_rn(q_smem[tid * kFaDh + d], k_smem[j * kFaDh + d], dot);
                const int kj = kv_start + j;
                float s = dot * scale;
                if (qi < kj || kj >= S) s = -INFINITY;  // causal + boundary
                s_smem[j * kFaBr + tid] = s;             // transposed: [j][tid]
            }

            // Online softmax
            float mij = -INFINITY;
            #pragma unroll
            for (int j = 0; j < kFaBc; j++)
                mij = fmaxf(mij, s_smem[j * kFaBr + tid]);

            float lij = 0.f;
            #pragma unroll
            for (int j = 0; j < kFaBc; j++) {
                float p = __expf(s_smem[j * kFaBr + tid] - mij);
                s_smem[j * kFaBr + tid] = p;
                lij += p;
            }

            const float mi_new = fmaxf(mi, mij);
            const float alpha  = __expf(mi  - mi_new);
            const float beta   = __expf(mij - mi_new);
            const float li_new = alpha * li + beta * lij;

            // O update: O_new = alpha*O_old + beta * sum_j(p_j * V_j)
            #pragma unroll
            for (int d = 0; d < kFaDh; d++) {
                float vacc = 0.f;
                #pragma unroll
                for (int j = 0; j < kFaBc; j++)
                    vacc = __fmaf_rn(s_smem[j * kFaBr + tid], v_smem[j * kFaDh + d], vacc);
                o_reg[d] = alpha * o_reg[d] + beta * vacc;
            }

            mi = mi_new;
            li = li_new;
        }
        __syncthreads();
    }

    if (qi < S) {
        const float inv_l = (li > 0.f) ? __frcp_rn(li) : 0.f;
        #pragma unroll
        for (int d = 0; d < kFaDh; d++)
            Op[qi * kFaDh + d] = o_reg[d] * inv_l;
        LSEp[qi] = mi + __logf(fmaxf(li, 1e-12f));
    }
}

// ----------------------------------------------------------------
//  D precompute kernel: D[bh][qi] = dot(dO[bh,qi], O[bh,qi])
//  Grid: ceil(BH*S / 64),  Block: 64.
// ----------------------------------------------------------------
__global__ void flash_attn_compute_D_kernel(
    const float* __restrict__ dO,  // [BH, S, kFaDh]
    const float* __restrict__ O,   // [BH, S, kFaDh]
    float* __restrict__ D,         // [BH, S]
    int BH, int S)
{
    const int idx = blockIdx.x * 64 + threadIdx.x;
    const int bh  = idx / S;
    const int qi  = idx % S;
    if (bh >= BH || qi >= S) return;

    const ptrdiff_t base = (ptrdiff_t)bh * S * kFaDh + (ptrdiff_t)qi * kFaDh;
    float acc = 0.f;
    #pragma unroll
    for (int d = 0; d < kFaDh; d++)
        acc += dO[base + d] * O[base + d];
    D[(ptrdiff_t)bh * S + qi] = acc;
}

// ----------------------------------------------------------------
//  Backward: dK and dV
//  One block per (bh, kv_tile).  Block: kFaBc threads.
//  Each block iterates over all Q tiles, accumulates dK/dV in registers.
//  Shared memory:
//    k_smem [kFaBc][kFaDh]   — key tile for this block (fixed)
//    v_smem [kFaBc][kFaDh]   — value tile (fixed)
//    q_smem [kFaBr][kFaDh]   — query tile (per Q step)
//    do_smem[kFaBr][kFaDh]   — dO tile (per Q step)
// ----------------------------------------------------------------
__global__ void __launch_bounds__(kFaBc)
flash_attn_bwd_dkdv_kernel(
    const float* __restrict__ Q,    // [BH, S, Dh]
    const float* __restrict__ K,    // [BH, S, Dh]
    const float* __restrict__ V,    // [BH, S, Dh]
    const float* __restrict__ dO,   // [BH, S, Dh]
    const float* __restrict__ LSE,  // [BH, S]
    const float* __restrict__ D,    // [BH, S]
    float* __restrict__ dK,         // [BH, S, Dh]
    float* __restrict__ dV,         // [BH, S, Dh]
    int BH, int S, float scale)
{
    const int num_kv_tiles = (S + kFaBc - 1) / kFaBc;
    const int bh      = blockIdx.x / num_kv_tiles;
    const int kv_tile = blockIdx.x % num_kv_tiles;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int kj  = kv_tile * kFaBc + tid;

    const ptrdiff_t bh_off = (ptrdiff_t)bh * S * kFaDh;
    const float* Qp   = Q   + bh_off;
    const float* Kp   = K   + bh_off;
    const float* Vp   = V   + bh_off;
    const float* dOp  = dO  + bh_off;
    const float* LSEp = LSE + (ptrdiff_t)bh * S;
    const float* Dp   = D   + (ptrdiff_t)bh * S;
    float*       dKp  = dK  + bh_off;
    float*       dVp  = dV  + bh_off;

    extern __shared__ float smem[];
    float* k_smem  = smem;                                          // [kFaBc][kFaDh]
    float* v_smem  = k_smem  + kFaBc * kFaDh;                     // [kFaBc][kFaDh]
    float* q_smem  = v_smem  + kFaBc * kFaDh;                     // [kFaBr][kFaDh]
    float* do_smem = q_smem  + kFaBr * kFaDh;                     // [kFaBr][kFaDh]

    // Load K/V tile once for this block
    if (kj < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            k_smem[tid * kFaDh + d] = Kp[kj * kFaDh + d];
            v_smem[tid * kFaDh + d] = Vp[kj * kFaDh + d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            k_smem[tid * kFaDh + d] = 0.f;
            v_smem[tid * kFaDh + d] = 0.f;
        }
    }
    __syncthreads();

    // dK/dV accumulators in registers
    float dk_reg[kFaDh], dv_reg[kFaDh];
    #pragma unroll
    for (int d = 0; d < kFaDh; d++) { dk_reg[d] = 0.f; dv_reg[d] = 0.f; }

    // For causal: only Q tiles where qi_start <= kj (= kv_tile*kFaBc + kFaBc-1 at most)
    // So q_tile_start = kv_tile * kFaBc / kFaBr (integer division — first q_tile with any valid qi)
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const int q_tile_start = (kv_tile * kFaBc) / kFaBr;

    for (int q_tile = q_tile_start; q_tile < num_q_tiles; q_tile++) {
        const int qi_base = q_tile * kFaBr;

        // Load Q and dO tiles (kFaBc threads load kFaBr rows, strided)
        for (int row = tid; row < kFaBr; row += kFaBc) {
            const int qi = qi_base + row;
            if (qi < S) {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    q_smem [row * kFaDh + d] = Qp [qi * kFaDh + d];
                    do_smem[row * kFaDh + d] = dOp[qi * kFaDh + d];
                }
            } else {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    q_smem [row * kFaDh + d] = 0.f;
                    do_smem[row * kFaDh + d] = 0.f;
                }
            }
        }
        __syncthreads();

        // For kFaBc=kFaBr=64: each thread computes its KV column (tid) across all Br query rows.
        // Compute P_ij and dS_ij, accumulate dK[tid] and dV[tid].
        if (kj < S) {
            #pragma unroll
            for (int i = 0; i < kFaBr; i++) {
                const int qi = qi_base + i;
                if (qi >= S) break;

                // S_ij = dot(Q[qi], K[kj]) * scale
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dot = __fmaf_rn(q_smem[i * kFaDh + d], k_smem[tid * kFaDh + d], dot);
                float s = dot * scale;
                if (qi < kj) s = -INFINITY;  // causal

                // P_ij = exp(S_ij - LSE[qi])
                float lse_qi = LSEp[qi];
                float p = (s == -INFINITY) ? 0.f : __expf(s - lse_qi);

                // dV[kj] += P_ij * dO[qi]  → accumulate in dv_reg
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dv_reg[d] = __fmaf_rn(p, do_smem[i * kFaDh + d], dv_reg[d]);

                // dP_ij = dot(dO[qi], V[kj])
                float dp = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dp = __fmaf_rn(do_smem[i * kFaDh + d], v_smem[tid * kFaDh + d], dp);

                // dS_ij = P_ij * (dP_ij - D[qi]) * scale
                float ds = p * (dp - Dp[qi]) * scale;

                // dK[kj] += dS_ij * Q[qi]
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dk_reg[d] = __fmaf_rn(ds, q_smem[i * kFaDh + d], dk_reg[d]);
            }
        }
        __syncthreads();
    }

    // Write dK, dV
    if (kj < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            dKp[kj * kFaDh + d] = dk_reg[d];
            dVp[kj * kFaDh + d] = dv_reg[d];
        }
    }
}

// ----------------------------------------------------------------
//  Backward: dQ
//  One block per (bh, q_tile).  Block: kFaBr threads.
//  Each block iterates over KV tiles (causal: only kv_t with kv_start <= qi).
//  Shared memory:
//    q_smem [kFaBr][kFaDh]   — query tile (fixed)
//    do_smem[kFaBr][kFaDh]   — dO tile (fixed)
//    k_smem [kFaBc][kFaDh]   — key tile (per KV step)
//    v_smem [kFaBc][kFaDh]   — value tile (per KV step)
// ----------------------------------------------------------------
__global__ void __launch_bounds__(kFaBr)
flash_attn_bwd_dq_kernel(
    const float* __restrict__ Q,    // [BH, S, Dh]
    const float* __restrict__ K,    // [BH, S, Dh]
    const float* __restrict__ V,    // [BH, S, Dh]
    const float* __restrict__ dO,   // [BH, S, Dh]
    const float* __restrict__ LSE,  // [BH, S]
    const float* __restrict__ D,    // [BH, S]
    float* __restrict__ dQ,         // [BH, S, Dh]
    int BH, int S, float scale)
{
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const int bh     = blockIdx.x / num_q_tiles;
    const int q_tile = blockIdx.x % num_q_tiles;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int qi  = q_tile * kFaBr + tid;

    const ptrdiff_t bh_off = (ptrdiff_t)bh * S * kFaDh;
    const float* Qp   = Q   + bh_off;
    const float* Kp   = K   + bh_off;
    const float* Vp   = V   + bh_off;
    const float* dOp  = dO  + bh_off;
    const float* LSEp = LSE + (ptrdiff_t)bh * S;
    const float* Dp   = D   + (ptrdiff_t)bh * S;
    float*       dQp  = dQ  + bh_off;

    extern __shared__ float smem[];
    float* q_smem  = smem;                                          // [kFaBr][kFaDh]
    float* do_smem = q_smem  + kFaBr * kFaDh;                     // [kFaBr][kFaDh]
    float* k_smem  = do_smem + kFaBr * kFaDh;                     // [kFaBc][kFaDh]
    float* v_smem  = k_smem  + kFaBc * kFaDh;                     // [kFaBc][kFaDh]

    // Load Q and dO tiles once
    if (qi < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            q_smem [tid * kFaDh + d] = Qp [qi * kFaDh + d];
            do_smem[tid * kFaDh + d] = dOp[qi * kFaDh + d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            q_smem [tid * kFaDh + d] = 0.f;
            do_smem[tid * kFaDh + d] = 0.f;
        }
    }
    __syncthreads();

    float dq_reg[kFaDh];
    #pragma unroll
    for (int d = 0; d < kFaDh; d++) dq_reg[d] = 0.f;

    float mi  = (qi < S) ? LSEp[qi] : 0.f;
    float di  = (qi < S) ? Dp  [qi] : 0.f;

    const int num_kv_tiles = (S + kFaBc - 1) / kFaBc;

    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        const int kv_start = kv_t * kFaBc;
        if (kv_start > q_tile * kFaBr + kFaBr - 1) break;  // causal: no more valid KV tiles

        // Load K and V tiles
        for (int row = tid; row < kFaBc; row += kFaBr) {
            const int kj = kv_start + row;
            if (kj < S) {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[row * kFaDh + d] = Kp[kj * kFaDh + d];
                    v_smem[row * kFaDh + d] = Vp[kj * kFaDh + d];
                }
            } else {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[row * kFaDh + d] = 0.f;
                    v_smem[row * kFaDh + d] = 0.f;
                }
            }
        }
        __syncthreads();

        if (qi < S) {
            // Each thread computes its query row across all Bc KV positions
            #pragma unroll
            for (int j = 0; j < kFaBc; j++) {
                const int kj = kv_start + j;
                if (kj >= S) break;

                // S_ij = dot(Q[qi], K[kj]) * scale
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dot = __fmaf_rn(q_smem[tid * kFaDh + d], k_smem[j * kFaDh + d], dot);
                float s = dot * scale;
                if (qi < kj) s = -INFINITY;

                float p = (s == -INFINITY) ? 0.f : __expf(s - mi);

                // dP_ij = dot(dO[qi], V[kj])
                float dp = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dp = __fmaf_rn(do_smem[tid * kFaDh + d], v_smem[j * kFaDh + d], dp);

                float ds = p * (dp - di) * scale;

                // dQ[qi] += dS_ij * K[kj]
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dq_reg[d] = __fmaf_rn(ds, k_smem[j * kFaDh + d], dq_reg[d]);
            }
        }
        __syncthreads();
    }

    if (qi < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++)
            dQp[qi * kFaDh + d] = dq_reg[d];
    }
}

// ----------------------------------------------------------------
//  Host-side launchers
// ----------------------------------------------------------------
static bool flash_attn_sync_stream_ok(const char* ctx) {
    return jgpt_cuda_sync_stream_unless_capturing(ctx) != 0;
}

static bool flash_attn_fwd_run(
        const float* d_q, const float* d_k, const float* d_v,
        float* d_o, float* d_lse,
        int BH, int S, float scale)
{
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const long long grid_ll = (long long) BH * (long long) num_q_tiles;
    if (grid_ll <= 0LL || grid_ll > static_cast<long long>(INT_MAX)) {
        fprintf(
                stderr,
                "flash_attn_fwd: grid overflow BH=%d S=%d num_q_tiles=%d (blocks=%lld)\n",
                BH,
                S,
                num_q_tiles,
                static_cast<long long>(grid_ll));
        return false;
    }
    const int grid = static_cast<int>(grid_ll);
    constexpr size_t smem =
        (kFaBr + 2 * kFaBc) * kFaDh * sizeof(float)   // q + k + v tiles
        + kFaBc * kFaBr * sizeof(float);               // s_smem [kFaBc][kFaBr]
    cudaError_t sa = flash_kernel_ensure_dyn_smem((const void*)flash_attn_fwd_kernel, smem);
    if (sa != cudaSuccess) {
        fprintf(stderr, "flash_attn_fwd: cudaFuncSetAttribute smem=%zu: %s\n", smem, cudaGetErrorString(sa));
        return false;
    }
    flash_attn_fwd_kernel<<<grid, kFaBr, smem, kTensorCudaStream>>>(
            d_q, d_k, d_v, d_o, d_lse, BH, S, scale);
    CUDA_KERNEL_CHECK_RV(false);
    return flash_attn_sync_stream_ok("flash_attn_fwd");
}

static bool flash_attn_bwd_run(
        const float* d_q, const float* d_k, const float* d_v,
        const float* d_o, const float* d_do,
        const float* d_lse,
        float* d_dq, float* d_dk, float* d_dv,
        int BH, int S, float scale)
{
    const size_t qkv_bytes = (size_t)BH * (size_t)S * (size_t)kFaDh * sizeof(float);
    if (cudaMemsetAsync(d_dq, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }
    if (cudaMemsetAsync(d_dk, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }
    if (cudaMemsetAsync(d_dv, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }

    // 1. Compute D[bh][qi] = dot(dO[qi], O[qi])
    const long long total_rows_ll = (long long) BH * (long long) S;
    if (total_rows_ll <= 0LL || total_rows_ll > static_cast<long long>(INT_MAX)) {
        fprintf(
                stderr,
                "flash_attn_bwd: total_rows overflow BH=%d S=%d (rows=%lld)\n",
                BH,
                S,
                static_cast<long long>(total_rows_ll));
        return false;
    }
    const int total_rows = static_cast<int>(total_rows_ll);
    const int d_grid = (total_rows + 63) / 64;
    flash_attn_compute_D_kernel<<<d_grid, 64, 0, kTensorCudaStream>>>(
            d_do, d_o, tl_fa_D, BH, S);
    CUDA_KERNEL_CHECK_RV(false);

    // 2. dK, dV kernel (one block per kv_tile)
    constexpr size_t smem_dkdv =
        2 * kFaBc * kFaDh * sizeof(float)                                // k + v fixed
        + 2 * kFaBr * kFaDh * sizeof(float);                             // q + do per q_tile
    const int num_kv_tiles = (S + kFaBc - 1) / kFaBc;
    const long long grid_dkdv_ll = (long long) BH * (long long) num_kv_tiles;
    if (grid_dkdv_ll <= 0LL || grid_dkdv_ll > static_cast<long long>(INT_MAX)) {
        fprintf(
                stderr,
                "flash_attn_bwd dkdv: grid overflow BH=%d S=%d num_kv_tiles=%d (blocks=%lld)\n",
                BH,
                S,
                num_kv_tiles,
                static_cast<long long>(grid_dkdv_ll));
        return false;
    }
    const int grid_dkdv = static_cast<int>(grid_dkdv_ll);
    cudaError_t sb = flash_kernel_ensure_dyn_smem((const void*)flash_attn_bwd_dkdv_kernel, smem_dkdv);
    if (sb != cudaSuccess) {
        fprintf(stderr, "flash_attn_bwd dkdv: cudaFuncSetAttribute smem=%zu: %s\n", smem_dkdv, cudaGetErrorString(sb));
        return false;
    }
    flash_attn_bwd_dkdv_kernel<<<grid_dkdv, kFaBc, smem_dkdv, kTensorCudaStream>>>(
            d_q, d_k, d_v, d_do, d_lse, tl_fa_D,
            d_dk, d_dv, BH, S, scale);
    CUDA_KERNEL_CHECK_RV(false);

    // 3. dQ kernel (one block per q_tile)
    constexpr size_t smem_dq =
        2 * kFaBr * kFaDh * sizeof(float)                                // q + do fixed
        + 2 * kFaBc * kFaDh * sizeof(float);                             // k + v per kv_tile
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const long long grid_dq_ll = (long long) BH * (long long) num_q_tiles;
    if (grid_dq_ll <= 0LL || grid_dq_ll > static_cast<long long>(INT_MAX)) {
        fprintf(
                stderr,
                "flash_attn_bwd dq: grid overflow BH=%d S=%d num_q_tiles=%d (blocks=%lld)\n",
                BH,
                S,
                num_q_tiles,
                static_cast<long long>(grid_dq_ll));
        return false;
    }
    const int grid_dq = static_cast<int>(grid_dq_ll);
    sb = flash_kernel_ensure_dyn_smem((const void*)flash_attn_bwd_dq_kernel, smem_dq);
    if (sb != cudaSuccess) {
        fprintf(stderr, "flash_attn_bwd dq: cudaFuncSetAttribute smem=%zu: %s\n", smem_dq, cudaGetErrorString(sb));
        return false;
    }
    flash_attn_bwd_dq_kernel<<<grid_dq, kFaBr, smem_dq, kTensorCudaStream>>>(
            d_q, d_k, d_v, d_do, d_lse, tl_fa_D,
            d_dq, BH, S, scale);
    CUDA_KERNEL_CHECK_RV(false);
    return flash_attn_sync_stream_ok("flash_attn_bwd");
}

// ========== JNI ==========
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_softmaxLastDimGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jfloatArray h_dst, jint batch, jint mid, jint inner,
    jboolean useFp16Softmax) {
    (void) clazz;
    int nrows = batch * mid;
    if (nrows <= 0 || inner <= 0) {
        return;
    }
    if (check_size_overflow((size_t) nrows, (size_t) inner, sizeof(float))) {
        fprintf(stderr, "softmaxLastDimGPU: size overflow\n");
        return;
    }
    size_t bytes = (size_t) nrows * (size_t) inner * sizeof(float);
    float *d_src = nullptr, *d_dst = nullptr;
    if (!softmax_pair_ensure(bytes, &d_src, &d_dst)) {
        return;
    }

    jfloat* p_src = env->GetFloatArrayElements(h_src, nullptr);
    if (!p_src) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_src, p_src, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, p_src, JNI_ABORT);

    launch_softmax_last_dim(d_src, d_dst, nrows, inner, useFp16Softmax == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    jfloat* p_dst = env->GetFloatArrayElements(h_dst, nullptr);
    if (!p_dst) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(p_dst, d_dst, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_dst, p_dst, 0);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_crossEntropySoftmaxGradLossGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_logits, jfloatArray h_targets, jfloatArray h_grad, jint batch,
    jint seqLen, jint vocab, jfloat gradScaleOverTotalTokens, jfloatArray h_lossOut, jboolean useFp16Softmax) {
    (void) clazz;
    int nrows = batch * seqLen;
    if (nrows <= 0 || vocab <= 0) {
        jfloat* plo = env->GetFloatArrayElements(h_lossOut, nullptr);
        if (plo) {
            plo[0] = 0.f;
            env->ReleaseFloatArrayElements(h_lossOut, plo, 0);
        }
        return;
    }
    if (check_size_overflow((size_t) nrows, (size_t) vocab, sizeof(float))) {
        fprintf(stderr, "crossEntropySoftmaxGradLossGPU: size overflow\n");
        return;
    }
    jgpt_cuda_ensure_stream();
    size_t bytes_logits = (size_t) nrows * (size_t) vocab * sizeof(float);
    size_t bytes_tgt = (size_t) nrows * sizeof(float);
    if (!ce_ensure_logits_grad_buffers(bytes_logits)) {
        jfloat* plo = env->GetFloatArrayElements(h_lossOut, nullptr);
        if (plo) {
            plo[0] = 0.f;
            env->ReleaseFloatArrayElements(h_lossOut, plo, 0);
        }
        return;
    }
    if (bytes_tgt > tl_ce_targets_bytes) {
        cudaFree(tl_ce_d_targets);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_targets), bytes_tgt));
        tl_ce_targets_bytes = bytes_tgt;
    }
    if (tl_ce_d_loss_sum == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_loss_sum), sizeof(float)));
    }
    if (tl_ce_d_valid == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_valid), sizeof(unsigned int)));
    }

    jfloat* plog = env->GetFloatArrayElements(h_logits, nullptr);
    jfloat* ptgt = env->GetFloatArrayElements(h_targets, nullptr);
    if (!plog || !ptgt) {
        jfloat* plo = env->GetFloatArrayElements(h_lossOut, nullptr);
        if (plo) {
            plo[0] = 0.f;
            env->ReleaseFloatArrayElements(h_lossOut, plo, 0);
        }
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(tl_ce_d_logits, plog, bytes_logits, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(tl_ce_d_targets, ptgt, bytes_tgt, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_logits, plog, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_targets, ptgt, JNI_ABORT);

    CUDA_CHECK_X(cudaMemsetAsync(tl_ce_d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(tl_ce_d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy(tl_ce_d_logits, tl_ce_d_targets, tl_ce_d_grad, gradScaleOverTotalTokens, tl_ce_d_loss_sum,
            tl_ce_d_valid, nrows, vocab, useFp16Softmax == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    jfloat* pgrad = env->GetFloatArrayElements(h_grad, nullptr);
    if (!pgrad) {
        jfloat* plo = env->GetFloatArrayElements(h_lossOut, nullptr);
        if (plo) {
            plo[0] = 0.f;
            env->ReleaseFloatArrayElements(h_lossOut, plo, 0);
        }
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgrad, tl_ce_d_grad, bytes_logits, cudaMemcpyDeviceToHost, kTensorCudaStream));

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_X(cudaMemcpyAsync(&h_loss_sum, tl_ce_d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(&h_valid, tl_ce_d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    env->ReleaseFloatArrayElements(h_grad, pgrad, 0);

    jfloat* plo = env->GetFloatArrayElements(h_lossOut, nullptr);
    if (plo) {
        plo[0] = (h_valid == 0U) ? 0.f : (h_loss_sum / (float)h_valid);
        env->ReleaseFloatArrayElements(h_lossOut, plo, 0);
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_crossEntropySoftmaxGradLossGPUDirect(
    JNIEnv* env, jclass clazz, jobject logits_buf, jlong logits_byte_off, jobject targets_buf, jlong targets_byte_off,
    jfloatArray h_grad, jint batch, jint seq_len, jint vocab, jfloat grad_scale_over_total_tokens,
    jfloatArray h_loss_out, jboolean use_fp16_softmax) {
    (void) clazz;
    int nrows = batch * seq_len;
    if (nrows <= 0 || vocab <= 0) {
        jfloat* plo = env->GetFloatArrayElements(h_loss_out, nullptr);
        if (plo) {
            plo[0] = 0.f;
            env->ReleaseFloatArrayElements(h_loss_out, plo, 0);
        }
        return;
    }
    if (check_size_overflow((size_t) nrows, (size_t) vocab, sizeof(float))) {
        fprintf(stderr, "crossEntropySoftmaxGradLossGPUDirect: size overflow\n");
        return;
    }
    jgpt_cuda_ensure_stream();
    size_t bytes_logits = (size_t) nrows * (size_t) vocab * sizeof(float);
    size_t bytes_tgt = (size_t) nrows * sizeof(float);

    void* plog = jni_direct_ptr(env, logits_buf, logits_byte_off, (jlong) bytes_logits, "CE direct logits");
    void* ptgt = jni_direct_ptr(env, targets_buf, targets_byte_off, (jlong) bytes_tgt, "CE direct targets");
    if (!plog || !ptgt) {
        jfloat* plo = env->GetFloatArrayElements(h_loss_out, nullptr);
        if (plo) {
            plo[0] = 0.f;
            env->ReleaseFloatArrayElements(h_loss_out, plo, 0);
        }
        return;
    }

    if (!ce_ensure_logits_grad_buffers(bytes_logits)) {
        jfloat* plo = env->GetFloatArrayElements(h_loss_out, nullptr);
        if (plo) {
            plo[0] = 0.f;
            env->ReleaseFloatArrayElements(h_loss_out, plo, 0);
        }
        return;
    }
    if (bytes_tgt > tl_ce_targets_bytes) {
        cudaFree(tl_ce_d_targets);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_targets), bytes_tgt));
        tl_ce_targets_bytes = bytes_tgt;
    }
    if (tl_ce_d_loss_sum == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_loss_sum), sizeof(float)));
    }
    if (tl_ce_d_valid == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_valid), sizeof(unsigned int)));
    }

    CUDA_CHECK_X(cudaMemcpyAsync(tl_ce_d_logits, plog, bytes_logits, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(tl_ce_d_targets, ptgt, bytes_tgt, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    CUDA_CHECK_X(cudaMemsetAsync(tl_ce_d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(tl_ce_d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy(tl_ce_d_logits, tl_ce_d_targets, tl_ce_d_grad, grad_scale_over_total_tokens, tl_ce_d_loss_sum,
            tl_ce_d_valid, nrows, vocab, use_fp16_softmax == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    jfloat* pgrad = env->GetFloatArrayElements(h_grad, nullptr);
    if (!pgrad) {
        jfloat* plo = env->GetFloatArrayElements(h_loss_out, nullptr);
        if (plo) {
            plo[0] = 0.f;
            env->ReleaseFloatArrayElements(h_loss_out, plo, 0);
        }
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgrad, tl_ce_d_grad, bytes_logits, cudaMemcpyDeviceToHost, kTensorCudaStream));

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_X(cudaMemcpyAsync(&h_loss_sum, tl_ce_d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(&h_valid, tl_ce_d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    env->ReleaseFloatArrayElements(h_grad, pgrad, 0);

    jfloat* plo = env->GetFloatArrayElements(h_loss_out, nullptr);
    if (plo) {
        plo[0] = (h_valid == 0U) ? 0.f : (h_loss_sum / (float) h_valid);
        env->ReleaseFloatArrayElements(h_loss_out, plo, 0);
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_copyHostFloatBufferToGpuIntTokenIds(
    JNIEnv* env, jclass clazz, jobject float_buf, jlong float_byte_off, jint n_tokens, jlong d_dst_int) {
    (void) clazz;
    if (n_tokens <= 0 || d_dst_int == 0) {
        return;
    }
    size_t bytes_f = (size_t) n_tokens * sizeof(float);
    void* pf = jni_direct_ptr(env, float_buf, float_byte_off, (jlong) bytes_f, "copyFloatToInt targets");
    if (!pf) {
        return;
    }
    if (bytes_f > tl_ce_targets_bytes) {
        cudaFree(tl_ce_d_targets);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_targets), bytes_f));
        tl_ce_targets_bytes = bytes_f;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(tl_ce_d_targets, pf, bytes_f, cudaMemcpyHostToDevice, kTensorCudaStream));
    int* ddst = reinterpret_cast<int*>(static_cast<uintptr_t>(d_dst_int));
    launch_float_to_int32_targets(tl_ce_d_targets, ddst, n_tokens);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_layerNormGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_x, jfloatArray h_gamma, jfloatArray h_beta, jfloatArray h_out,
    jint outer, jint lastDim, jfloat eps) {
    (void) clazz;
    if (outer <= 0 || lastDim <= 0) {
        return;
    }
    size_t bytes_x = (size_t) outer * (size_t) lastDim * sizeof(float);
    size_t bytes_g = (size_t) lastDim * sizeof(float);
    float *d_x = nullptr, *d_g = nullptr, *d_b = nullptr, *d_o = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_x));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_g), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_o), bytes_x));

    jfloat* px = env->GetFloatArrayElements(h_x, nullptr);
    jfloat* pg = env->GetFloatArrayElements(h_gamma, nullptr);
    jfloat* pb = env->GetFloatArrayElements(h_beta, nullptr);
    if (!px || !pg || !pb) {
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_b);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px, bytes_x, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, pg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_b, pb, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gamma, pg, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_beta, pb, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (outer + threads - 1) / threads;
    layer_norm_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_x, d_g, d_b, d_o, outer, lastDim, eps);
    CUDA_KERNEL_CHECK();
    jfloat* po = env->GetFloatArrayElements(h_out, nullptr);
    if (!po) {
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_b);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po, d_o, bytes_x, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_out, po, 0);

    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_o);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_rmsNormGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_x, jfloatArray h_gamma, jfloatArray h_out, jint outer,
    jint lastDim, jfloat eps, jboolean useFp16) {
    (void) clazz;
    if (outer <= 0 || lastDim <= 0) {
        return;
    }
    size_t bytes_x = (size_t) outer * (size_t) lastDim * sizeof(float);
    size_t bytes_g = (size_t) lastDim * sizeof(float);
    float *d_x = nullptr, *d_g = nullptr, *d_o = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_x));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_g), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_o), bytes_x));

    jfloat* px = env->GetFloatArrayElements(h_x, nullptr);
    jfloat* pg = env->GetFloatArrayElements(h_gamma, nullptr);
    if (!px || !pg) {
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px, bytes_x, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, pg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gamma, pg, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_fwd(d_x, d_g, d_o, outer, lastDim, eps);
    jfloat* po = env->GetFloatArrayElements(h_out, nullptr);
    if (!po) {
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po, d_o, bytes_x, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_out, po, 0);

    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_o);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_rmsNormGPUDevice(
    JNIEnv* env, jclass clazz, jlong dX, jlong dGamma, jfloat eps, jlong dOut, jint outer, jint lastDim,
    jboolean useFp16) {
    (void) env;
    (void) clazz;
    if (dX == 0 || dGamma == 0 || dOut == 0 || outer <= 0 || lastDim <= 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    const float* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(dX));
    const float* gamma = reinterpret_cast<const float*>(static_cast<uintptr_t>(dGamma));
    float* out = reinterpret_cast<float*>(static_cast<uintptr_t>(dOut));
    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_fwd(x, gamma, out, outer, lastDim, eps);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_rmsNormMatmulLmHeadGPUDevice(
    JNIEnv* env, jclass clazz,
    jlong dX,
    jlong dGamma,
    jfloat eps,
    jlong dNormOut,
    jlong dW,
    jlong dLogits,
    jint rows,
    jint dModel,
    jint vocab,
    jboolean useFp16Rms) {
    (void) env;
    (void) clazz;
    if (dX == 0 || dGamma == 0 || dNormOut == 0 || dW == 0 || dLogits == 0) {
        return;
    }
    if (rows <= 0 || dModel <= 0 || vocab <= 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    const float* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(dX));
    const float* gamma = reinterpret_cast<const float*>(static_cast<uintptr_t>(dGamma));
    float* normOut = reinterpret_cast<float*>(static_cast<uintptr_t>(dNormOut));
    const float* w = reinterpret_cast<const float*>(static_cast<uintptr_t>(dW));
    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_fwd(x, gamma, normOut, rows, dModel, eps);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = get_extra_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "rmsNormMatmulLmHeadGPUDevice: cuBLAS handle unavailable\n");
        return;
    }
    cublasStatus_t st =
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, vocab, rows, dModel, &alpha, w, vocab, normOut, dModel, &beta, logits, vocab);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "rmsNormMatmulLmHeadGPUDevice: cublasSgemm failed (status %d)\n", (int) st);
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_rmsNormMatmulFfnW1W3GPUDevice(
    JNIEnv* env,
    jclass clazz,
    jlong dX,
    jlong dGamma,
    jfloat eps,
    jlong dNormOut,
    jlong dW1,
    jlong dW3,
    jlong dH1,
    jlong dGate,
    jint rows,
    jint dModel,
    jint dInt,
    jboolean useFp16Rms) {
    (void) env;
    (void) clazz;
    if (dX == 0 || dGamma == 0 || dNormOut == 0 || dW1 == 0 || dW3 == 0 || dH1 == 0 || dGate == 0) {
        return;
    }
    if (rows <= 0 || dModel <= 0 || dInt <= 0) {
        return;
    }
    const float* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(dX));
    const float* gamma = reinterpret_cast<const float*>(static_cast<uintptr_t>(dGamma));
    float* normOut = reinterpret_cast<float*>(static_cast<uintptr_t>(dNormOut));
    const float* w1 = reinterpret_cast<const float*>(static_cast<uintptr_t>(dW1));
    const float* w3 = reinterpret_cast<const float*>(static_cast<uintptr_t>(dW3));
    float* h1 = reinterpret_cast<float*>(static_cast<uintptr_t>(dH1));
    float* gate = reinterpret_cast<float*>(static_cast<uintptr_t>(dGate));

    jgpt_cuda_ensure_stream();
    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_fwd(x, gamma, normOut, rows, dModel, eps);
    if (jgpt_cuda_ffn_w1w3_strided_batched_device(normOut, w1, w3, h1, gate, rows, dModel, dInt) != 1) {
        fprintf(stderr, "rmsNormMatmulFfnW1W3GPUDevice: W1+W3 strided batched failed\n");
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_geluGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jfloatArray h_dst, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    jfloat* pa = env->GetFloatArrayElements(h_src, nullptr);
    if (!pa) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, pa, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, n);
    jfloat* pb = env->GetFloatArrayElements(h_dst, nullptr);
    if (!pb) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pb, d_b, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_dst, pb, 0);
    cudaFree(d_a);
    cudaFree(d_b);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sigmoidGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jfloatArray h_dst, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    jfloat* pa = env->GetFloatArrayElements(h_src, nullptr);
    if (!pa) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, pa, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, n);
    jfloat* pb = env->GetFloatArrayElements(h_dst, nullptr);
    if (!pb) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pb, d_b, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_dst, pb, 0);
    cudaFree(d_a);
    cudaFree(d_b);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sigmoidGPUDevice(
    JNIEnv* env, jclass clazz, jlong dSrc, jlong dDst, jint n) {
    (void) env;
    (void) clazz;
    if (dSrc == 0 || dDst == 0 || n <= 0) {
        return;
    }
    const float* src = reinterpret_cast<const float*>(static_cast<uintptr_t>(dSrc));
    float* dst = reinterpret_cast<float*>(static_cast<uintptr_t>(dDst));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(src, dst, n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_multiplyGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_a, jfloatArray h_b, jfloatArray h_c, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));
    jfloat* pa = env->GetFloatArrayElements(h_a, nullptr);
    jfloat* pb = env->GetFloatArrayElements(h_b, nullptr);
    if (!pa || !pb) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_b, pb, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_a, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_b, pb, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, d_c, n);
    jfloat* pc = env->GetFloatArrayElements(h_c, nullptr);
    if (!pc) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pc, d_c, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_c, pc, 0);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_multiplyGPUDevice(
    JNIEnv* env, jclass clazz, jlong dA, jlong dB, jlong dC, jint n) {
    (void) env;
    (void) clazz;
    if (dA == 0 || dB == 0 || dC == 0 || n <= 0) {
        return;
    }
    const float* a = reinterpret_cast<const float*>(static_cast<uintptr_t>(dA));
    const float* b = reinterpret_cast<const float*>(static_cast<uintptr_t>(dB));
    float* c = reinterpret_cast<float*>(static_cast<uintptr_t>(dC));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(a, b, c, n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_multiplyScalarGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_a, jfloatArray h_b, jint n, jfloat scalar) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    jfloat* pa = env->GetFloatArrayElements(h_a, nullptr);
    if (!pa) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_a, pa, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    mul_scalar_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, n, scalar);
    jfloat* pb = env->GetFloatArrayElements(h_b, nullptr);
    if (!pb) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pb, d_b, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_b, pb, 0);
    cudaFree(d_a);
    cudaFree(d_b);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_applyCausalMask3DGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_scores, jfloatArray h_mask, jfloatArray h_out, jint batch,
    jint seqLen) {
    (void) clazz;
    int total = batch * seqLen * seqLen;
    if (total <= 0) {
        return;
    }
    size_t bytes_s = (size_t) total * sizeof(float);
    size_t bytes_m = (size_t) seqLen * seqLen * sizeof(float);
    float *d_s = nullptr, *d_m = nullptr, *d_o = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_s), bytes_s));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_m), bytes_m));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_o), bytes_s));
    jfloat* ps = env->GetFloatArrayElements(h_scores, nullptr);
    jfloat* pm = env->GetFloatArrayElements(h_mask, nullptr);
    if (!ps || !pm) {
        cudaFree(d_s);
        cudaFree(d_m);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_s, ps, bytes_s, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_m, pm, bytes_m, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_scores, ps, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_mask, pm, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    apply_causal_mask_3d_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_s, d_m, d_o, batch, seqLen);
    jfloat* po = env->GetFloatArrayElements(h_out, nullptr);
    if (!po) {
        cudaFree(d_s);
        cudaFree(d_m);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po, d_o, bytes_s, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_out, po, 0);
    cudaFree(d_s);
    cudaFree(d_m);
    cudaFree(d_o);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_transpose2DLastGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jfloatArray h_dst, jint d0, jint d1, jint d2) {
    (void) clazz;
    int total = d0 * d1 * d2;
    if (total <= 0) {
        return;
    }
    size_t bytes = (size_t) total * sizeof(float);
    float *d_s = nullptr, *d_d = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_s), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_d), bytes));
    jfloat* ps = env->GetFloatArrayElements(h_src, nullptr);
    if (!ps) {
        cudaFree(d_s);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_s, ps, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, ps, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    transpose_2d_last_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_s, d_d, d0, d1, d2);
    jfloat* pd = env->GetFloatArrayElements(h_dst, nullptr);
    if (!pd) {
        cudaFree(d_s);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pd, d_d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_dst, pd, 0);
    cudaFree(d_s);
    cudaFree(d_d);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_splitHeadsGPUDevice(
    JNIEnv* env, jclass clazz, jlong dSrc, jlong dDst, jint batch, jint seqLen, jint dModel, jint numHeads) {
    (void) env;
    (void) clazz;
    int total = batch * seqLen * dModel;
    if (dSrc == 0 || dDst == 0 || total <= 0) {
        return;
    }
    const float* src = reinterpret_cast<const float*>(static_cast<uintptr_t>(dSrc));
    float* dst = reinterpret_cast<float*>(static_cast<uintptr_t>(dDst));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    split_heads_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(src, dst, batch, seqLen, dModel, numHeads);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_concatHeadsGPUDevice(
    JNIEnv* env, jclass clazz, jlong dSrc, jlong dDst, jint batch, jint numHeads, jint seqLen, jint dHead) {
    (void) env;
    (void) clazz;
    int dModel = numHeads * dHead;
    int total = batch * seqLen * dModel;
    if (dSrc == 0 || dDst == 0 || total <= 0) {
        return;
    }
    const float* src = reinterpret_cast<const float*>(static_cast<uintptr_t>(dSrc));
    float* dst = reinterpret_cast<float*>(static_cast<uintptr_t>(dDst));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    concat_heads_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(src, dst, batch, numHeads, seqLen, dHead);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_copyKvHeads4dToCacheGPUDevice(
    JNIEnv* env,
    jclass clazz,
    jlong dSrc,
    jlong dDst,
    jint numHeads,
    jint seqLen,
    jint maxSeqLen,
    jint dHead,
    jint batchIdx,
    jint batch) {
    (void) env;
    (void) clazz;
    if (dSrc == 0 || dDst == 0 || numHeads <= 0 || seqLen <= 0 || maxSeqLen <= 0 || dHead <= 0) {
        return;
    }
    if (batch <= 0 || batchIdx < 0 || batchIdx >= batch) {
        return;
    }
    size_t sliceFloats = (size_t) numHeads * seqLen * dHead;
    const float* srcBase = reinterpret_cast<const float*>(static_cast<uintptr_t>(dSrc));
    srcBase += (size_t) batchIdx * sliceFloats;
    float* dst = reinterpret_cast<float*>(static_cast<uintptr_t>(dDst));
    int total = numHeads * seqLen * dHead;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    kv_heads4d_to_cache_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            srcBase, dst, numHeads, seqLen, maxSeqLen, dHead);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_applyRoPE4DGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jfloatArray h_dst, jint batch, jint numHeads, jint seqLen,
    jint dHead, jintArray h_positions, jint posBaseOffset) {
    (void) clazz;
    if (dHead % 2 != 0) {
        return;
    }
    int halfPairs = dHead / 2;
    int total = batch * numHeads * seqLen * halfPairs;
    if (total <= 0) {
        return;
    }
    size_t bytes = (size_t) batch * numHeads * seqLen * dHead * sizeof(float);
    float *d_s = nullptr, *d_d = nullptr;
    int *d_pos = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_s), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_d), bytes));
    jfloat* ps = env->GetFloatArrayElements(h_src, nullptr);
    if (!ps) {
        cudaFree(d_s);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_s, ps, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, ps, JNI_ABORT);

    int posLen = seqLen;
    jint* posHost = nullptr;
    if (h_positions != nullptr) {
        posLen = env->GetArrayLength(h_positions);
        posHost = env->GetIntArrayElements(h_positions, nullptr);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_pos), (size_t) posLen * sizeof(int)));
        CUDA_CHECK_X(cudaMemcpyAsync(d_pos, posHost, (size_t) posLen * sizeof(int), cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
        env->ReleaseIntArrayElements(h_positions, posHost, JNI_ABORT);
    }

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    rope_4d_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_s, d_d, batch, numHeads, seqLen, dHead, d_pos, posLen, posBaseOffset);
    jfloat* pd = env->GetFloatArrayElements(h_dst, nullptr);
    if (!pd) {
        cudaFree(d_s);
        cudaFree(d_d);
        if (d_pos) {
            cudaFree(d_pos);
        }
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pd, d_d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_dst, pd, 0);
    cudaFree(d_s);
    cudaFree(d_d);
    if (d_pos) {
        cudaFree(d_pos);
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_applyRoPE4DGPUDevice(
    JNIEnv* env, jclass clazz, jlong dSrc, jlong dDst, jint batch, jint numHeads, jint seqLen, jint dHead,
    jintArray h_positions, jint posBaseOffset) {
    (void) clazz;
    if (dSrc == 0 || dDst == 0 || dHead % 2 != 0) {
        return;
    }
    int halfPairs = dHead / 2;
    int total = batch * numHeads * seqLen * halfPairs;
    if (total <= 0) {
        return;
    }
    float* src = reinterpret_cast<float*>(static_cast<uintptr_t>(dSrc));
    float* dst = reinterpret_cast<float*>(static_cast<uintptr_t>(dDst));
    int* d_pos = nullptr;
    int posLen = seqLen;
    if (h_positions != nullptr) {
        posLen = env->GetArrayLength(h_positions);
        jint* posHost = env->GetIntArrayElements(h_positions, nullptr);
        if (!posHost) {
            return;
        }
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_pos), (size_t) posLen * sizeof(int)));
        CUDA_CHECK_X(cudaMemcpyAsync(d_pos, posHost, (size_t) posLen * sizeof(int), cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
        env->ReleaseIntArrayElements(h_positions, posHost, JNI_ABORT);
    }

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    rope_4d_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            src, dst, batch, numHeads, seqLen, dHead, d_pos, posLen, posBaseOffset);
    if (d_pos) {
        cudaFree(d_pos);
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_applyRoPEBackward4DGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gradY, jfloatArray h_gradX, jint batch, jint numHeads, jint seqLen,
    jint dHead, jintArray h_positions, jint posBaseOffset) {
    (void) clazz;
    if (dHead % 2 != 0) {
        return;
    }
    int halfPairs = dHead / 2;
    int total = batch * numHeads * seqLen * halfPairs;
    if (total <= 0) {
        return;
    }
    size_t bytes = (size_t) batch * numHeads * seqLen * dHead * sizeof(float);
    float *d_gy = nullptr, *d_gx = nullptr;
    int *d_pos = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gy), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gx), bytes));
    jfloat* pgy = env->GetFloatArrayElements(h_gradY, nullptr);
    jfloat* pgx = env->GetFloatArrayElements(h_gradX, nullptr);
    if (!pgy || !pgx) {
        cudaFree(d_gy);
        cudaFree(d_gx);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_gy, pgy, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gx, pgx, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gradY, pgy, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gradX, pgx, JNI_ABORT);

    int posLen = seqLen;
    jint* posHost = nullptr;
    if (h_positions != nullptr) {
        posLen = env->GetArrayLength(h_positions);
        posHost = env->GetIntArrayElements(h_positions, nullptr);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_pos), (size_t) posLen * sizeof(int)));
        CUDA_CHECK_X(cudaMemcpyAsync(d_pos, posHost, (size_t) posLen * sizeof(int), cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
        env->ReleaseIntArrayElements(h_positions, posHost, JNI_ABORT);
    }

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    rope_4d_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_gy, d_gx, batch, numHeads, seqLen, dHead, d_pos, posLen, posBaseOffset);
    pgx = env->GetFloatArrayElements(h_gradX, nullptr);
    if (!pgx) {
        cudaFree(d_gy);
        cudaFree(d_gx);
        if (d_pos) {
            cudaFree(d_pos);
        }
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgx, d_gx, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gradX, pgx, 0);
    cudaFree(d_gy);
    cudaFree(d_gx);
    if (d_pos) {
        cudaFree(d_pos);
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_applyRoPEBackward4DGPUDevice(
    JNIEnv* env, jclass clazz, jlong dGradY, jlong dGradX, jint batch, jint numHeads, jint seqLen, jint dHead,
    jint posBaseOffset) {
    (void) env;
    (void) clazz;
    if (dGradY == 0 || dGradX == 0 || dHead % 2 != 0) {
        return;
    }
    int halfPairs = dHead / 2;
    int total = batch * numHeads * seqLen * halfPairs;
    if (total <= 0) {
        return;
    }
    const float* gradY = reinterpret_cast<const float*>(static_cast<uintptr_t>(dGradY));
    float* gradX = reinterpret_cast<float*>(static_cast<uintptr_t>(dGradX));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    /*
     * rope_4d_bwd_kernel накапливает в gradX через +=. На пути JNI буфер предзаполняется копией с хоста;
     * здесь вызывающий GPU-код обязан обнулить gradX до вызова (см. TransformerBackward: QHeads.clear()).
     */
    rope_4d_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            gradY, gradX, batch, numHeads, seqLen, dHead, nullptr, seqLen, posBaseOffset);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenForwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_tokens, jfloatArray h_weights, jfloatArray h_out, jint batch,
    jint seqLen, jint dModel, jint vocabSize, jboolean useFp16) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0) {
        return;
    }
    size_t tokBytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    size_t wBytes = (size_t) vocabSize * (size_t) dModel * sizeof(float);
    size_t outBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    float *d_tok = nullptr, *d_w = nullptr, *d_out = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tokBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_w), wBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_out), outBytes));
    jfloat* ptok = env->GetFloatArrayElements(h_tokens, nullptr);
    jfloat* pw = env->GetFloatArrayElements(h_weights, nullptr);
    if (!ptok || !pw) {
        cudaFree(d_tok);
        cudaFree(d_w);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_w, pw, wBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_tokens, ptok, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_weights, pw, JNI_ABORT);
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    if (useFp16 == JNI_TRUE) {
        embedding_token_fwd_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    } else {
        embedding_token_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    }
    jfloat* po = env->GetFloatArrayElements(h_out, nullptr);
    if (!po) {
        cudaFree(d_tok);
        cudaFree(d_w);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po, d_out, outBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_out, po, 0);
    cudaFree(d_tok);
    cudaFree(d_w);
    cudaFree(d_out);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenForwardGPUDirect(
    JNIEnv* env, jclass clazz, jobject token_byte_buffer, jfloatArray h_weights, jfloatArray h_out, jint batch,
    jint seqLen, jint dModel, jint vocabSize, jboolean useFp16) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0) {
        return;
    }
    void* tok_host = env->GetDirectBufferAddress(token_byte_buffer);
    if (tok_host == nullptr) {
        return;
    }
    jlong cap_bytes = env->GetDirectBufferCapacity(token_byte_buffer);
    size_t tok_bytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    if (cap_bytes < 0 || (jlong) tok_bytes > cap_bytes) {
        return;
    }
    size_t wBytes = (size_t) vocabSize * (size_t) dModel * sizeof(float);
    size_t outBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    float *d_tok = nullptr, *d_w = nullptr, *d_out = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tok_bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_w), wBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_out), outBytes));
    jfloat* pw = env->GetFloatArrayElements(h_weights, nullptr);
    if (!pw) {
        cudaFree(d_tok);
        cudaFree(d_w);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, tok_host, tok_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_w, pw, wBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_weights, pw, JNI_ABORT);
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    if (useFp16 == JNI_TRUE) {
        embedding_token_fwd_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    } else {
        embedding_token_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    }
    jfloat* po = env->GetFloatArrayElements(h_out, nullptr);
    if (!po) {
        cudaFree(d_tok);
        cudaFree(d_w);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po, d_out, outBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_out, po, 0);
    cudaFree(d_tok);
    cudaFree(d_w);
    cudaFree(d_out);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenForwardGPUDeviceWeights(
    JNIEnv* env, jclass clazz, jfloatArray h_tokens, jlong d_weights_ptr, jfloatArray h_out, jint batch,
    jint seqLen, jint dModel, jint vocabSize, jboolean useFp16) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0) {
        return;
    }
    float* d_w = reinterpret_cast<float*>(static_cast<uintptr_t>(d_weights_ptr));
    if (d_w == nullptr) {
        return;
    }
    size_t tokBytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    size_t outBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    float *d_tok = nullptr, *d_out = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tokBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_out), outBytes));
    jfloat* ptok = env->GetFloatArrayElements(h_tokens, nullptr);
    if (!ptok) {
        cudaFree(d_tok);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_tokens, ptok, JNI_ABORT);
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    if (useFp16 == JNI_TRUE) {
        embedding_token_fwd_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    } else {
        embedding_token_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    }
    jfloat* po = env->GetFloatArrayElements(h_out, nullptr);
    if (!po) {
        cudaFree(d_tok);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po, d_out, outBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_out, po, 0);
    cudaFree(d_tok);
    cudaFree(d_out);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenForwardGPUDeviceWeightsToDevice(
    JNIEnv* env, jclass clazz, jfloatArray h_tokens, jlong d_weights_ptr, jlong d_out_ptr, jint batch, jint seqLen,
    jint dModel, jint vocabSize, jboolean useFp16) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0) {
        return;
    }
    float* d_w = reinterpret_cast<float*>(static_cast<uintptr_t>(d_weights_ptr));
    float* d_out = reinterpret_cast<float*>(static_cast<uintptr_t>(d_out_ptr));
    if (d_w == nullptr || d_out == nullptr) {
        return;
    }
    size_t tokBytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    float* d_tok = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tokBytes));
    jfloat* ptok = env->GetFloatArrayElements(h_tokens, nullptr);
    if (!ptok) {
        cudaFree(d_tok);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_tokens, ptok, JNI_ABORT);
    jgpt_cuda_ensure_stream();
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    if (useFp16 == JNI_TRUE) {
        embedding_token_fwd_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    } else {
        embedding_token_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "embeddingTokenForwardGPUDeviceWeightsToDevice: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_tok);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenForwardGPUDirectDeviceWeights(
    JNIEnv* env, jclass clazz, jobject token_byte_buffer, jlong d_weights_ptr, jfloatArray h_out, jint batch,
    jint seqLen, jint dModel, jint vocabSize, jboolean useFp16) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0) {
        return;
    }
    float* d_w = reinterpret_cast<float*>(static_cast<uintptr_t>(d_weights_ptr));
    if (d_w == nullptr) {
        return;
    }
    void* tok_host = env->GetDirectBufferAddress(token_byte_buffer);
    if (tok_host == nullptr) {
        return;
    }
    jlong cap_bytes = env->GetDirectBufferCapacity(token_byte_buffer);
    size_t tok_bytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    if (cap_bytes < 0 || (jlong) tok_bytes > cap_bytes) {
        return;
    }
    size_t outBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    float *d_tok = nullptr, *d_out = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tok_bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_out), outBytes));
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, tok_host, tok_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    if (useFp16 == JNI_TRUE) {
        embedding_token_fwd_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    } else {
        embedding_token_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    }
    jfloat* po = env->GetFloatArrayElements(h_out, nullptr);
    if (!po) {
        cudaFree(d_tok);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po, d_out, outBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_out, po, 0);
    cudaFree(d_tok);
    cudaFree(d_out);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenForwardGPUDirectDeviceWeightsToDevice(
    JNIEnv* env, jclass clazz, jobject token_byte_buffer, jlong d_weights_ptr, jlong d_out_ptr, jint batch,
    jint seqLen, jint dModel, jint vocabSize, jboolean useFp16) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0) {
        return;
    }
    float* d_w = reinterpret_cast<float*>(static_cast<uintptr_t>(d_weights_ptr));
    float* d_out = reinterpret_cast<float*>(static_cast<uintptr_t>(d_out_ptr));
    if (d_w == nullptr || d_out == nullptr) {
        return;
    }
    void* tok_host = env->GetDirectBufferAddress(token_byte_buffer);
    if (tok_host == nullptr) {
        return;
    }
    jlong cap_bytes = env->GetDirectBufferCapacity(token_byte_buffer);
    size_t tok_bytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    if (cap_bytes < 0 || (jlong) tok_bytes > cap_bytes) {
        return;
    }
    float* d_tok = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tok_bytes));
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, tok_host, tok_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    jgpt_cuda_ensure_stream();
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    if (useFp16 == JNI_TRUE) {
        embedding_token_fwd_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    } else {
        embedding_token_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_tok, d_w, d_out, batch, seqLen, dModel, vocabSize);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "embeddingTokenForwardGPUDirectDeviceWeightsToDevice: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_tok);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_tokens, jfloatArray h_gradOut, jfloatArray h_gradWeights,
    jint batch, jint seqLen, jint dModel, jint vocabSize) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0) {
        return;
    }
    size_t tokBytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    size_t goBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    size_t gwBytes = (size_t) vocabSize * (size_t) dModel * sizeof(float);
    float *d_tok = nullptr, *d_go = nullptr, *d_gw = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tokBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), goBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gw), gwBytes));
    jfloat* ptok = env->GetFloatArrayElements(h_tokens, nullptr);
    jfloat* pgo = env->GetFloatArrayElements(h_gradOut, nullptr);
    jfloat* pgw = env->GetFloatArrayElements(h_gradWeights, nullptr);
    if (!ptok || !pgo || !pgw) {
        cudaFree(d_tok); cudaFree(d_go); cudaFree(d_gw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, goBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gw, pgw, gwBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_tokens, ptok, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gradOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gradWeights, pgw, JNI_ABORT);
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    embedding_token_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_go, d_gw, batch, seqLen, dModel, vocabSize);
    pgw = env->GetFloatArrayElements(h_gradWeights, nullptr);
    if (!pgw) {
        cudaFree(d_tok); cudaFree(d_go); cudaFree(d_gw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgw, d_gw, gwBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gradWeights, pgw, 0);
    cudaFree(d_tok);
    cudaFree(d_go);
    cudaFree(d_gw);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenBackwardGPUDeviceGradWeights(
    JNIEnv* env, jclass clazz, jfloatArray h_tokens, jfloatArray h_gradOut, jint batch, jint seqLen, jint dModel,
    jint vocabSize, jlong gradWeightsDevicePtr) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0) {
        return;
    }
    float* d_gw = reinterpret_cast<float*>(static_cast<uintptr_t>(gradWeightsDevicePtr));
    if (d_gw == nullptr) {
        return;
    }
    size_t tokBytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    size_t goBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    float *d_tok = nullptr, *d_go = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tokBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), goBytes));
    jfloat* ptok = env->GetFloatArrayElements(h_tokens, nullptr);
    jfloat* pgo = env->GetFloatArrayElements(h_gradOut, nullptr);
    if (!ptok || !pgo) {
        cudaFree(d_tok);
        cudaFree(d_go);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, goBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_tokens, ptok, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gradOut, pgo, JNI_ABORT);
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    embedding_token_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_go, d_gw, batch, seqLen, dModel, vocabSize);
    cudaFree(d_tok);
    cudaFree(d_go);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingPositionBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gradCombined, jfloatArray h_gradWeights,
    jint batch, jint seqLen, jint dModel) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0) {
        return;
    }
    size_t gcBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    size_t gwBytes = (size_t) seqLen * (size_t) dModel * sizeof(float);
    float *d_gc = nullptr, *d_gw = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gc), gcBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gw), gwBytes));
    jfloat* pgc = env->GetFloatArrayElements(h_gradCombined, nullptr);
    jfloat* pgw = env->GetFloatArrayElements(h_gradWeights, nullptr);
    if (!pgc || !pgw) {
        cudaFree(d_gc); cudaFree(d_gw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_gc, pgc, gcBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gw, pgw, gwBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gradCombined, pgc, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gradWeights, pgw, JNI_ABORT);
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    embedding_position_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_gc, d_gw, batch, seqLen, dModel);
    pgw = env->GetFloatArrayElements(h_gradWeights, nullptr);
    if (!pgw) {
        cudaFree(d_gc); cudaFree(d_gw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgw, d_gw, gwBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gradWeights, pgw, 0);
    cudaFree(d_gc);
    cudaFree(d_gw);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingPositionBackwardGPUDeviceGradWeights(
    JNIEnv* env, jclass clazz, jfloatArray h_gradCombined, jint batch, jint seqLen, jint dModel,
    jlong gradWeightsDevicePtr) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0) {
        return;
    }
    float* d_gw = reinterpret_cast<float*>(static_cast<uintptr_t>(gradWeightsDevicePtr));
    if (d_gw == nullptr) {
        return;
    }
    size_t gcBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    float* d_gc = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gc), gcBytes));
    jfloat* pgc = env->GetFloatArrayElements(h_gradCombined, nullptr);
    if (!pgc) {
        cudaFree(d_gc);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_gc, pgc, gcBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gradCombined, pgc, JNI_ABORT);
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    embedding_position_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_gc, d_gw, batch, seqLen, dModel);
    cudaFree(d_gc);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingTokenBackwardGPUDeviceGradWeightsDeviceGrad(
    JNIEnv* env, jclass clazz, jfloatArray h_tokens, jlong dGradOut, jint batch, jint seqLen, jint dModel,
    jint vocabSize, jlong gradWeightsDevicePtr) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || vocabSize <= 0 || dGradOut == 0) {
        return;
    }
    float* d_go = reinterpret_cast<float*>(static_cast<uintptr_t>(dGradOut));
    float* d_gw = reinterpret_cast<float*>(static_cast<uintptr_t>(gradWeightsDevicePtr));
    if (d_gw == nullptr) {
        return;
    }
    size_t tokBytes = (size_t) batch * (size_t) seqLen * sizeof(float);
    float* d_tok = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tokBytes));
    jfloat* ptok = env->GetFloatArrayElements(h_tokens, nullptr);
    if (!ptok) {
        cudaFree(d_tok);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_tokens, ptok, JNI_ABORT);
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    embedding_token_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_tok, d_go, d_gw, batch, seqLen, dModel, vocabSize);
    cudaFree(d_tok);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_embeddingPositionBackwardGPUDeviceGradWeightsDeviceGrad(
    JNIEnv* env, jclass clazz, jlong dGradCombined, jint batch, jint seqLen, jint dModel,
    jlong gradWeightsDevicePtr) {
    (void) env;
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || dGradCombined == 0) {
        return;
    }
    float* d_gc = reinterpret_cast<float*>(static_cast<uintptr_t>(dGradCombined));
    float* d_gw = reinterpret_cast<float*>(static_cast<uintptr_t>(gradWeightsDevicePtr));
    if (d_gw == nullptr) {
        return;
    }
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    embedding_position_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_gc, d_gw, batch, seqLen, dModel);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_addPositionEmbeddingGPUDeviceWeightsWithOffset(
    JNIEnv* env, jclass clazz, jfloatArray h_x, jlong posWeightsDevicePtr, jint batch, jint seqLen, jint dModel,
    jint posRowStart) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0) {
        return;
    }
    if (posRowStart < 0) {
        return;
    }
    const float* d_pw = reinterpret_cast<const float*>(static_cast<uintptr_t>(posWeightsDevicePtr));
    if (d_pw == nullptr) {
        return;
    }
    size_t xBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    float* d_x = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), xBytes));
    jfloat* px = env->GetFloatArrayElements(h_x, nullptr);
    if (!px) {
        cudaFree(d_x);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px, xBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
    jgpt_cuda_ensure_stream();
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    add_position_embedding_broadcast_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
        d_x, d_pw, posRowStart, batch, seqLen, dModel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "addPositionEmbeddingGPUDeviceWeights: %s\n", cudaGetErrorString(err));
    }
    px = env->GetFloatArrayElements(h_x, nullptr);
    if (!px) {
        cudaFree(d_x);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(px, d_x, xBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_x, px, 0);
    cudaFree(d_x);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_addPositionEmbeddingGPUDeviceBuffersWithOffset(
    JNIEnv* env, jclass clazz, jlong d_x_ptr, jlong d_pos_weights_ptr, jint batch, jint seqLen, jint dModel,
    jint posRowStart) {
    (void) env;
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0) {
        return;
    }
    if (posRowStart < 0) {
        return;
    }
    float* d_x = reinterpret_cast<float*>(static_cast<uintptr_t>(d_x_ptr));
    const float* d_pw = reinterpret_cast<const float*>(static_cast<uintptr_t>(d_pos_weights_ptr));
    if (d_x == nullptr || d_pw == nullptr) {
        return;
    }
    jgpt_cuda_ensure_stream();
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    add_position_embedding_broadcast_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
        d_x, d_pw, posRowStart, batch, seqLen, dModel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "addPositionEmbeddingGPUDeviceBuffers: %s\n", cudaGetErrorString(err));
    }
}

/** Host x и contiguous строки позиций [seqLen*dModel]; тот же kernel, что и addPositionEmbeddingGPUDeviceWeightsWithOffset с posRowStart=0. */
JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_addPositionEmbeddingGPUHostPosSlice(
    JNIEnv* env, jclass clazz, jfloatArray h_x, jfloatArray h_pos_rows, jint batch, jint seqLen, jint dModel) {
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0) {
        return;
    }
    size_t xBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    size_t posBytes = (size_t) seqLen * (size_t) dModel * sizeof(float);
    float* d_x = nullptr;
    float* d_pw = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), xBytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_pw), posBytes));
    jfloat* px = env->GetFloatArrayElements(h_x, nullptr);
    jfloat* ppos = env->GetFloatArrayElements(h_pos_rows, nullptr);
    if (!px || !ppos) {
        if (px) {
            env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
        }
        if (ppos) {
            env->ReleaseFloatArrayElements(h_pos_rows, ppos, JNI_ABORT);
        }
        cudaFree(d_x);
        cudaFree(d_pw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px, xBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_pw, ppos, posBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_pos_rows, ppos, JNI_ABORT);
    jgpt_cuda_ensure_stream();
    int total = batch * seqLen * dModel;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    add_position_embedding_broadcast_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
        d_x, d_pw, 0, batch, seqLen, dModel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "addPositionEmbeddingGPUHostPosSlice: %s\n", cudaGetErrorString(err));
    }
    px = env->GetFloatArrayElements(h_x, nullptr);
    if (!px) {
        cudaFree(d_x);
        cudaFree(d_pw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(px, d_x, xBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_x, px, 0);
    cudaFree(d_x);
    cudaFree(d_pw);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_scaledDotProductAttentionForwardGPUDevice(
    JNIEnv* env,
    jclass clazz,
    jlong dQPtr,
    jlong dKPtr,
    jlong dVPtr,
    jfloatArray h_mask,
    jlong dOutPtr,
    jfloatArray h_probs,
    jint batch,
    jint seqLen,
    jint dKDim,
    jint dVDim,
    jfloat scale,
    jboolean useFp16Softmax) {
    (void) clazz;
    if (dQPtr == 0 || dKPtr == 0 || dVPtr == 0 || dOutPtr == 0 || batch <= 0 || seqLen <= 0 || dKDim <= 0
            || dVDim <= 0) {
        return;
    }
    float* pq = reinterpret_cast<float*>(static_cast<uintptr_t>(dQPtr));
    float* pk = reinterpret_cast<float*>(static_cast<uintptr_t>(dKPtr));
    float* pv = reinterpret_cast<float*>(static_cast<uintptr_t>(dVPtr));
    float* pout = reinterpret_cast<float*>(static_cast<uintptr_t>(dOutPtr));

    size_t bytesQK = (size_t) batch * (size_t) seqLen * (size_t) dKDim * sizeof(float);
    size_t bytesProb = (size_t) batch * (size_t) seqLen * (size_t) seqLen * sizeof(float);
    size_t bytesMask = (size_t) seqLen * (size_t) seqLen * sizeof(float);
    const bool with_mask = h_mask != nullptr;
    float *d_scores = nullptr, *d_probs = nullptr, *d_mask_dev = nullptr;
    if (!attn_fwd_aux_ensure(bytesQK, bytesProb, bytesMask, with_mask, &d_scores, &d_probs, &d_mask_dev)) {
        return;
    }

    if (with_mask) {
        jfloat* pm = env->GetFloatArrayElements(h_mask, nullptr);
        if (!pm) {
            return;
        }
        CUDA_CHECK_X(cudaMemcpyAsync(d_mask_dev, pm, bytesMask, cudaMemcpyHostToDevice, kTensorCudaStream));
        env->ReleaseFloatArrayElements(h_mask, pm, JNI_ABORT);
    }

    jgpt_cuda_ensure_stream();
    if (!attn_fwd_run_core(
                pq,
                pk,
                pv,
                d_scores,
                d_probs,
                pout,
                d_mask_dev,
                batch,
                seqLen,
                dKDim,
                dVDim,
                scale,
                useFp16Softmax == JNI_TRUE)) {
        return;
    }

    if (h_probs != nullptr) {
        jfloat* pprobs = env->GetFloatArrayElements(h_probs, nullptr);
        if (!pprobs) {
            return;
        }
        CUDA_CHECK_X(cudaMemcpyAsync(pprobs, d_probs, bytesProb, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
        env->ReleaseFloatArrayElements(h_probs, pprobs, 0);
    }
    /* Без D2H в JVM: порядок с следующим *GPUDevice на kTensorCudaStream сохраняется; граница шага — synchronizeStream. */
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_scaledDotProductAttentionForwardGPUDeviceResident(
    JNIEnv* env,
    jclass clazz,
    jlong dQPtr,
    jlong dKPtr,
    jlong dVPtr,
    jlong dOutPtr,
    jlong dMaskPtr,
    jlong dProbsOutPtr,
    jint batch,
    jint seqLen,
    jint dKDim,
    jint dVDim,
    jfloat scale,
    jboolean useFp16Softmax) {
    (void) env;
    (void) clazz;
    if (dQPtr == 0 || dKPtr == 0 || dVPtr == 0 || dOutPtr == 0 || batch <= 0 || seqLen <= 0 || dKDim <= 0
            || dVDim <= 0) {
        return;
    }
    float* pq = reinterpret_cast<float*>(static_cast<uintptr_t>(dQPtr));
    float* pk = reinterpret_cast<float*>(static_cast<uintptr_t>(dKPtr));
    float* pv = reinterpret_cast<float*>(static_cast<uintptr_t>(dVPtr));
    float* pout = reinterpret_cast<float*>(static_cast<uintptr_t>(dOutPtr));

    size_t bytesQK = (size_t) batch * (size_t) seqLen * (size_t) dKDim * sizeof(float);
    size_t bytesProb = (size_t) batch * (size_t) seqLen * (size_t) seqLen * sizeof(float);
    float *d_scores = nullptr, *d_probs = nullptr;
    if (!attn_fwd_aux_ensure_qk_probs_only(bytesQK, bytesProb, &d_scores, &d_probs)) {
        return;
    }

    float* d_mask_ptr = nullptr;
    if (dMaskPtr != 0) {
        d_mask_ptr = reinterpret_cast<float*>(static_cast<uintptr_t>(dMaskPtr));
    }

    jgpt_cuda_ensure_stream();
    if (!attn_fwd_run_core(
                pq,
                pk,
                pv,
                d_scores,
                d_probs,
                pout,
                d_mask_ptr,
                batch,
                seqLen,
                dKDim,
                dVDim,
                static_cast<float>(scale),
                useFp16Softmax == JNI_TRUE)) {
        return;
    }

    if (dProbsOutPtr != 0) {
        float* pprobs_dst = reinterpret_cast<float*>(static_cast<uintptr_t>(dProbsOutPtr));
        CUDA_CHECK_X(cudaMemcpyAsync(
                pprobs_dst, d_probs, bytesProb, cudaMemcpyDeviceToDevice, kTensorCudaStream));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_scaledDotProductAttentionBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gradOut, jfloatArray h_probs, jfloatArray h_q, jfloatArray h_k,
    jfloatArray h_v, jfloatArray h_gradQ, jfloatArray h_gradK, jfloatArray h_gradV,
    jint batch, jint seqLen, jint dK, jint dV, jfloat scale, jfloatArray h_mask, jboolean useFp16Softmax) {
    (void) clazz;
    (void) h_mask;
    if (batch <= 0 || seqLen <= 0 || dK <= 0 || dV <= 0) {
        return;
    }
    size_t bytesProb = (size_t) batch * (size_t) seqLen * (size_t) seqLen * sizeof(float);
    size_t bytesQK = (size_t) batch * (size_t) seqLen * (size_t) dK * sizeof(float);
    size_t bytesV = (size_t) batch * (size_t) seqLen * (size_t) dV * sizeof(float);
    float *d_go = nullptr, *d_p = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_dp = nullptr, *d_ds = nullptr;
    float *d_gq = nullptr, *d_gk = nullptr, *d_gv = nullptr;
    if (!attn_bwd_host_ensure(bytesProb, bytesQK, bytesV, &d_go, &d_p, &d_q, &d_k, &d_v, &d_dp, &d_ds,
                &d_gq, &d_gk, &d_gv)) {
        return;
    }

    jfloat* pgo = env->GetFloatArrayElements(h_gradOut, nullptr);
    jfloat* pp = env->GetFloatArrayElements(h_probs, nullptr);
    jfloat* pq = env->GetFloatArrayElements(h_q, nullptr);
    jfloat* pk = env->GetFloatArrayElements(h_k, nullptr);
    jfloat* pv = env->GetFloatArrayElements(h_v, nullptr);
    if (!pgo || !pp || !pq || !pk || !pv) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytesV, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_p, pp, bytesProb, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_q, pq, bytesQK, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_k, pk, bytesQK, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_v, pv, bytesV, cudaMemcpyHostToDevice, kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gradOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_probs, pp, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_q, pq, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_k, pk, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_v, pv, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    // go × V^T → dp: transB GEMM (нет явного транспонирования V)
    if (!batched_sgemm_row_major_transB(d_go, d_v, d_dp, batch, seqLen, dV, seqLen, 1.0f, 0.0f)) {
        return;
    }

    /* softmax bwd пишет через +=; scratch свежий — обнуляем. */
    CUDA_CHECK_X(cudaMemset(d_ds, 0, bytesProb));

    int nrows = batch * seqLen;
    /*
     * Раньше при useFp16Softmax пересчитывали QK^T/scores на device и гоняли softmax_last_dim_bwd_from_logits_fp16.
     * Forward softmax_last_dim_kernel_fp16 уже пишет вероятности в FP32 (exp в FP32 — см. комментарий к ядру);
     * сохранённые probs совпадают с softmax(scores), поэтому достаточно стандартного VJP по p — без лишнего GEMM QK.
     */
    (void) useFp16Softmax;
    {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        softmax_last_dim_bwd_block_kernel<<<nrows, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(d_dp, d_p, d_ds, nrows, seqLen);
    }
    CUDA_KERNEL_CHECK();
    // scale перенесён как alpha в GEMMы gQ/gK — отдельный проход scale_inplace устранён
    // P^T × go → gV: transA GEMM
    if (!batched_sgemm_row_major_transA(d_p, d_go, d_gv, batch, seqLen, seqLen, dV, 1.0f, 0.0f)) {
        return;
    }
    if (!batched_sgemm_row_major_extra(d_ds, d_k, d_gq, batch, seqLen, seqLen, dK, scale, 0.0f)) {
        return;
    }
    // ds^T × Q → gK: transA GEMM
    if (!batched_sgemm_row_major_transA(d_ds, d_q, d_gk, batch, seqLen, seqLen, dK, scale, 0.0f)) {
        return;
    }

    jfloat* pgq = env->GetFloatArrayElements(h_gradQ, nullptr);
    jfloat* pgk = env->GetFloatArrayElements(h_gradK, nullptr);
    jfloat* pgv = env->GetFloatArrayElements(h_gradV, nullptr);
    if (!pgq || !pgk || !pgv) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgq, d_gq, bytesQK, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgk, d_gk, bytesQK, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgv, d_gv, bytesV, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gradQ, pgq, 0);
    env->ReleaseFloatArrayElements(h_gradK, pgk, 0);
    env->ReleaseFloatArrayElements(h_gradV, pgv, 0);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_scaledDotProductAttentionBackwardGPUDevice(
    JNIEnv* env, jclass clazz, jlong dGradOut, jlong dProbs, jlong dQ, jlong dK, jlong dV,
    jlong dGradQ, jlong dGradK, jlong dGradV, jint batch, jint seqLen, jint dKDim, jint dVDim, jfloat scale,
    jlong dMask, jboolean useFp16Softmax) {
    (void) env;
    (void) clazz;
    (void) dMask;
    if (dGradOut == 0 || dProbs == 0 || dQ == 0 || dK == 0 || dV == 0 || dGradQ == 0 || dGradK == 0 || dGradV == 0
            || batch <= 0 || seqLen <= 0 || dKDim <= 0 || dVDim <= 0) {
        return;
    }
    size_t bytesProb = (size_t) batch * (size_t) seqLen * (size_t) seqLen * sizeof(float);
    size_t bytesV = (size_t) batch * (size_t) seqLen * (size_t) dVDim * sizeof(float);
    float* go = reinterpret_cast<float*>(static_cast<uintptr_t>(dGradOut));
    float* p = reinterpret_cast<float*>(static_cast<uintptr_t>(dProbs));
    float* q = reinterpret_cast<float*>(static_cast<uintptr_t>(dQ));
    float* k = reinterpret_cast<float*>(static_cast<uintptr_t>(dK));
    float* v = reinterpret_cast<float*>(static_cast<uintptr_t>(dV));
    float* gq = reinterpret_cast<float*>(static_cast<uintptr_t>(dGradQ));
    float* gk = reinterpret_cast<float*>(static_cast<uintptr_t>(dGradK));
    float* gv = reinterpret_cast<float*>(static_cast<uintptr_t>(dGradV));

    float *d_dp = nullptr, *d_ds = nullptr;
    if (!attn_bwd_aux_ensure(bytesProb, bytesV, &d_dp, &d_ds)) {
        return;
    }

    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    int blocksProb = (batch * seqLen * seqLen + threads - 1) / threads;
    (void) blocksProb;
    // go × V^T → dp: transB GEMM
    if (!batched_sgemm_row_major_transB(go, v, d_dp, batch, seqLen, dVDim, seqLen, 1.0f, 0.0f)) {
        return;
    }

    CUDA_CHECK_X(cudaMemset(d_ds, 0, bytesProb));

    int nrows = batch * seqLen;
    (void) useFp16Softmax;
    {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        softmax_last_dim_bwd_block_kernel<<<nrows, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(d_dp, p, d_ds, nrows, seqLen);
    }
    CUDA_KERNEL_CHECK();
    // scale перенесён как alpha в GEMMы gQ/gK — отдельный проход scale_inplace устранён
    // P^T × go → gV: transA GEMM
    if (!batched_sgemm_row_major_transA(p, go, gv, batch, seqLen, seqLen, dVDim, 1.0f, 0.0f)) {
        return;
    }
    if (!batched_sgemm_row_major_extra(d_ds, k, gq, batch, seqLen, seqLen, dKDim, scale, 0.0f)) {
        return;
    }
    // ds^T × Q → gK: transA GEMM
    if (!batched_sgemm_row_major_transA(d_ds, q, gk, batch, seqLen, seqLen, dKDim, scale, 0.0f)) {
        return;
    }
}

/**
 * Сумма квадратов float на device: по чанкам float→double, затем cublasDdot (double).
 * И float cublasSdot, и большой чанк с «тяжёлыми» элементами давали Inf при промежуточном float.
 */
static double sum_squares_float_device_chunked_double_acc(cublasHandle_t h, const float* x, int n) {
    if (n <= 0 || x == nullptr) {
        return 0.0;
    }
    if (h == nullptr) {
        return 0.0;
    }
    const int chunk = 1 << 18;
    double* d_dbl = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_dbl), (size_t) chunk * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "sum_squares_float_device_chunked_double_acc: cudaMalloc double scratch failed\n");
        return 0.0;
    }
    double acc = 0.0;
    int threads = jgpt_cuda_get_optimal_block_size();
    for (int off = 0; off < n; off += chunk) {
        int m = n - off;
        if (m > chunk) {
            m = chunk;
        }
        int blocks = (m + threads - 1) / threads;
        float_to_double_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(x + off, d_dbl, m);
        double part = 0.0;
        cublasStatus_t st = cublasDdot(h, m, d_dbl, 1, d_dbl, 1, &part);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "sum_squares_float_device_chunked_double_acc: cublasDdot failed (status %d)\n", (int) st);
            cudaFree(d_dbl);
            return 0.0;
        }
        acc += part;
    }
    cudaFree(d_dbl);
    return acc;
}

JNIEXPORT jdouble JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sumSquaresGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jint n) {
    (void) clazz;
    if (n <= 0) {
        return 0.0;
    }
    jgpt_cuda_ensure_stream();
    size_t bytes = (size_t) n * sizeof(float);
    float* d_src = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_src), bytes);
    if (e1 != cudaSuccess) {
        cudaFree(d_src);
        return 0.0;
    }
    jfloat* psrc = env->GetFloatArrayElements(h_src, nullptr);
    if (!psrc) {
        cudaFree(d_src);
        return 0.0;
    }
    if (cudaMemcpyAsync(d_src, psrc, bytes, cudaMemcpyHostToDevice, kTensorCudaStream) != cudaSuccess) {
        env->ReleaseFloatArrayElements(h_src, psrc, JNI_ABORT);
        cudaFree(d_src);
        return 0.0;
    }
    if (cudaStreamSynchronize(kTensorCudaStream) != cudaSuccess) {
        env->ReleaseFloatArrayElements(h_src, psrc, JNI_ABORT);
        cudaFree(d_src);
        return 0.0;
    }
    env->ReleaseFloatArrayElements(h_src, psrc, JNI_ABORT);
    cublasHandle_t h = get_extra_cublas_handle();
    if (h == nullptr) {
        cudaFree(d_src);
        return 0.0;
    }
    double acc = sum_squares_float_device_chunked_double_acc(h, d_src, n);
    if (cudaStreamSynchronize(kTensorCudaStream) != cudaSuccess) {
        cudaFree(d_src);
        return 0.0;
    }
    cudaFree(d_src);
    return static_cast<jdouble>(acc);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_scaleInPlaceGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jint n, jfloat scalar) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float* d_src = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_src), bytes));
    jfloat* psrc = env->GetFloatArrayElements(h_src, nullptr);
    if (!psrc) {
        cudaFree(d_src);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_src, psrc, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, psrc, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    scale_inplace_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, n, scalar);
    psrc = env->GetFloatArrayElements(h_src, nullptr);
    if (!psrc) {
        cudaFree(d_src);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(psrc, d_src, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, psrc, 0);
    cudaFree(d_src);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_param, jfloatArray h_grad, jfloatArray h_m, jfloatArray h_v,
    jfloat learningRate, jfloat beta1, jfloat beta2, jfloat epsilon, jfloat weightDecay,
    jfloat invBias1, jfloat invBias2, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    if (!adamw_pool_ensure(bytes)) {
        fprintf(stderr, "adamWStepGPU: adamw_pool_ensure(%zu bytes) failed\n", bytes);
        return;
    }
    float* d_param = tl_adamw_d_param;
    float* d_grad = tl_adamw_d_grad;
    float* d_m = tl_adamw_d_m;
    float* d_v = tl_adamw_d_v;
    jfloat* pparam = env->GetFloatArrayElements(h_param, nullptr);
    jfloat* pgrad = env->GetFloatArrayElements(h_grad, nullptr);
    jfloat* pm = env->GetFloatArrayElements(h_m, nullptr);
    jfloat* pv = env->GetFloatArrayElements(h_v, nullptr);
    if (!pparam || !pgrad || !pm || !pv) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_param, pparam, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_grad, pgrad, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_m, pm, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_v, pv, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_param, pparam, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_grad, pgrad, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_m, pm, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_v, pv, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    adamw_step_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
        d_param, d_grad, d_m, d_v, learningRate, beta1, beta2, epsilon, weightDecay, invBias1, invBias2, n);
    pparam = env->GetFloatArrayElements(h_param, nullptr);
    pm = env->GetFloatArrayElements(h_m, nullptr);
    pv = env->GetFloatArrayElements(h_v, nullptr);
    if (!pparam || !pm || !pv) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pparam, d_param, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pm, d_m, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pv, d_v, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_param, pparam, 0);
    env->ReleaseFloatArrayElements(h_m, pm, 0);
    env->ReleaseFloatArrayElements(h_v, pv, 0);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepFusedGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_param, jfloatArray h_grad, jfloatArray h_m, jfloatArray h_v,
    jfloat learningRate, jfloat beta1, jfloat beta2, jfloat epsilon, jfloat weightDecay,
    jfloat invBias1, jfloat invBias2, jint n) {
    // "Fused" = one JNI + one contiguous host buffer after Java stitching; same adamw kernel as adamWStepGPU.
    // Multi-buffer device fusion without host packing is adamWStepGPUDeviceFused.
    Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepGPU(
        env,
        clazz,
        h_param,
        h_grad,
        h_m,
        h_v,
        learningRate,
        beta1,
        beta2,
        epsilon,
        weightDecay,
        invBias1,
        invBias2,
        n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepGPUDevice(
    JNIEnv* env,
    jclass clazz,
    jlong dParam,
    jlong dGrad,
    jlong dM,
    jlong dV,
    jfloat learningRate,
    jfloat beta1,
    jfloat beta2,
    jfloat epsilon,
    jfloat weightDecay,
    jfloat invBias1,
    jfloat invBias2,
    jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dParam == 0 || dGrad == 0 || dM == 0 || dV == 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    float* pParam = reinterpret_cast<float*>(static_cast<uintptr_t>(dParam));
    const float* pGrad = reinterpret_cast<const float*>(static_cast<uintptr_t>(dGrad));
    float* pM = reinterpret_cast<float*>(static_cast<uintptr_t>(dM));
    float* pV = reinterpret_cast<float*>(static_cast<uintptr_t>(dV));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    adamw_step_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
        pParam, pGrad, pM, pV, learningRate, beta1, beta2, epsilon, weightDecay, invBias1, invBias2, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "adamWStepGPUDevice: %s\n", cudaGetErrorString(err));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepGPUDeviceFused(
    JNIEnv* env,
    jclass clazz,
    jlongArray h_param_ptrs,
    jlongArray h_grad_ptrs,
    jlongArray h_m_ptrs,
    jlongArray h_v_ptrs,
    jintArray h_lengths,
    jfloat learningRate,
    jfloat beta1,
    jfloat beta2,
    jfloat epsilon,
    jfloat weightDecay,
    jfloat invBias1,
    jfloat invBias2) {
    (void) clazz;
    if (h_param_ptrs == nullptr || h_grad_ptrs == nullptr || h_m_ptrs == nullptr || h_v_ptrs == nullptr
        || h_lengths == nullptr) {
        return;
    }
    jsize num_segments = env->GetArrayLength(h_param_ptrs);
    if (num_segments <= 0) {
        return;
    }
    if (env->GetArrayLength(h_grad_ptrs) != num_segments || env->GetArrayLength(h_m_ptrs) != num_segments
        || env->GetArrayLength(h_v_ptrs) != num_segments || env->GetArrayLength(h_lengths) != num_segments) {
        return;
    }
    jlong* pj = env->GetLongArrayElements(h_param_ptrs, nullptr);
    jlong* gj = env->GetLongArrayElements(h_grad_ptrs, nullptr);
    jlong* mj = env->GetLongArrayElements(h_m_ptrs, nullptr);
    jlong* vj = env->GetLongArrayElements(h_v_ptrs, nullptr);
    jint* lens = env->GetIntArrayElements(h_lengths, nullptr);
    if (!pj || !gj || !mj || !vj || !lens) {
        if (pj) {
            env->ReleaseLongArrayElements(h_param_ptrs, pj, JNI_ABORT);
        }
        if (gj) {
            env->ReleaseLongArrayElements(h_grad_ptrs, gj, JNI_ABORT);
        }
        if (mj) {
            env->ReleaseLongArrayElements(h_m_ptrs, mj, JNI_ABORT);
        }
        if (vj) {
            env->ReleaseLongArrayElements(h_v_ptrs, vj, JNI_ABORT);
        }
        if (lens) {
            env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);
        }
        return;
    }
    for (jsize i = 0; i < num_segments; i++) {
        if (lens[i] <= 0 || pj[i] == 0 || gj[i] == 0 || mj[i] == 0 || vj[i] == 0) {
            env->ReleaseLongArrayElements(h_param_ptrs, pj, JNI_ABORT);
            env->ReleaseLongArrayElements(h_grad_ptrs, gj, JNI_ABORT);
            env->ReleaseLongArrayElements(h_m_ptrs, mj, JNI_ABORT);
            env->ReleaseLongArrayElements(h_v_ptrs, vj, JNI_ABORT);
            env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);
            return;
        }
    }

    size_t ptr_bytes = (size_t) num_segments * sizeof(uintptr_t);
    size_t len_bytes = (size_t) num_segments * sizeof(int);
    uintptr_t* h_pp = (uintptr_t*) malloc(ptr_bytes);
    uintptr_t* h_gp = (uintptr_t*) malloc(ptr_bytes);
    uintptr_t* h_mp = (uintptr_t*) malloc(ptr_bytes);
    uintptr_t* h_vp = (uintptr_t*) malloc(ptr_bytes);
    if (!h_pp || !h_gp || !h_mp || !h_vp) {
        free(h_pp);
        free(h_gp);
        free(h_mp);
        free(h_vp);
        env->ReleaseLongArrayElements(h_param_ptrs, pj, JNI_ABORT);
        env->ReleaseLongArrayElements(h_grad_ptrs, gj, JNI_ABORT);
        env->ReleaseLongArrayElements(h_m_ptrs, mj, JNI_ABORT);
        env->ReleaseLongArrayElements(h_v_ptrs, vj, JNI_ABORT);
        env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);
        return;
    }
    for (jsize i = 0; i < num_segments; i++) {
        h_pp[i] = static_cast<uintptr_t>(pj[i]);
        h_gp[i] = static_cast<uintptr_t>(gj[i]);
        h_mp[i] = static_cast<uintptr_t>(mj[i]);
        h_vp[i] = static_cast<uintptr_t>(vj[i]);
    }
    env->ReleaseLongArrayElements(h_param_ptrs, pj, JNI_ABORT);
    env->ReleaseLongArrayElements(h_grad_ptrs, gj, JNI_ABORT);
    env->ReleaseLongArrayElements(h_m_ptrs, mj, JNI_ABORT);
    env->ReleaseLongArrayElements(h_v_ptrs, vj, JNI_ABORT);

    uintptr_t *d_pp = nullptr, *d_gp = nullptr, *d_mp = nullptr, *d_vp = nullptr;
    int* d_lens = nullptr;
    cudaError_t alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_pp), ptr_bytes);
    if (alloc_err == cudaSuccess) {
        alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_gp), ptr_bytes);
    }
    if (alloc_err == cudaSuccess) {
        alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_mp), ptr_bytes);
    }
    if (alloc_err == cudaSuccess) {
        alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_vp), ptr_bytes);
    }
    if (alloc_err == cudaSuccess) {
        alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_lens), len_bytes);
    }
    if (alloc_err != cudaSuccess) {
        cudaFree(d_pp);
        cudaFree(d_gp);
        cudaFree(d_mp);
        cudaFree(d_vp);
        cudaFree(d_lens);
        free(h_pp);
        free(h_gp);
        free(h_mp);
        free(h_vp);
        env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);
        return;
    }
    jgpt_cuda_ensure_stream();
    CUDA_CHECK_X(cudaMemcpyAsync(d_pp, h_pp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gp, h_gp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_mp, h_mp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_vp, h_vp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_lens, lens, len_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    free(h_pp);
    free(h_gp);
    free(h_mp);
    free(h_vp);
    env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (int) num_segments;
    adamw_step_kernel_segments<<<blocks, threads, 0, kTensorCudaStream>>>(
        d_pp,
        d_gp,
        d_mp,
        d_vp,
        d_lens,
        (int) num_segments,
        learningRate,
        beta1,
        beta2,
        epsilon,
        weightDecay,
        invBias1,
        invBias2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "adamWStepGPUDeviceFused: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_pp);
    cudaFree(d_gp);
    cudaFree(d_mp);
    cudaFree(d_vp);
    cudaFree(d_lens);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_softmaxLastDimBackward3DGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_probs, jfloatArray h_gIn, jint batch,
    jint mid, jint inner) {
    (void) clazz;
    int nrows = batch * mid;
    if (nrows <= 0 || inner <= 0) {
        return;
    }
    size_t bytes = (size_t) nrows * (size_t) inner * sizeof(float);
    float *d_go = nullptr, *d_p = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_p), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pp = env->GetFloatArrayElements(h_probs, nullptr);
    jfloat* pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgo || !pp || !pgi) {
        cudaFree(d_go);
        cudaFree(d_p);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_p, pp, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, pgi, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_probs, pp, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gIn, pgi, JNI_ABORT);

    launch_softmax_last_dim_bwd(d_go, d_p, d_gi, nrows, inner);
    pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgi) {
        cudaFree(d_go);
        cudaFree(d_p);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgi, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gIn, pgi, 0);

    cudaFree(d_go);
    cudaFree(d_p);
    cudaFree(d_gi);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_layerNormBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_x, jfloatArray h_gamma, jfloat eps,
    jfloatArray h_gX, jfloatArray h_gGamma, jfloatArray h_gBeta, jint outer, jint lastDim) {
    (void) clazz;
    if (outer <= 0 || lastDim <= 0) {
        return;
    }
    size_t bytes_xy = (size_t) outer * (size_t) lastDim * sizeof(float);
    size_t bytes_g = (size_t) lastDim * sizeof(float);
    float *d_go = nullptr, *d_x = nullptr, *d_g = nullptr, *d_gx = nullptr, *d_gg = nullptr, *d_gb = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_g), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gx), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gg), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gb), bytes_g));

    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* px = env->GetFloatArrayElements(h_x, nullptr);
    jfloat* pg = env->GetFloatArrayElements(h_gamma, nullptr);
    jfloat* pgx = env->GetFloatArrayElements(h_gX, nullptr);
    jfloat* pgg = env->GetFloatArrayElements(h_gGamma, nullptr);
    jfloat* pgb = env->GetFloatArrayElements(h_gBeta, nullptr);
    if (!pgo || !px || !pg || !pgx || !pgg || !pgb) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, pg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gx, pgx, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gg, pgg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gb, pgb, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gamma, pg, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gX, pgx, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gGamma, pgg, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gBeta, pgb, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (outer + threads - 1) / threads;
    layer_norm_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_x, d_g, eps, d_gx, d_gg, d_gb, outer, lastDim);
    CUDA_KERNEL_CHECK();
    pgx = env->GetFloatArrayElements(h_gX, nullptr);
    pgg = env->GetFloatArrayElements(h_gGamma, nullptr);
    pgb = env->GetFloatArrayElements(h_gBeta, nullptr);
    if (!pgx || !pgg || !pgb) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgx, d_gx, bytes_xy, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgg, d_gg, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgb, d_gb, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gX, pgx, 0);
    env->ReleaseFloatArrayElements(h_gGamma, pgg, 0);
    env->ReleaseFloatArrayElements(h_gBeta, pgb, 0);

    cudaFree(d_go);
    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_gx);
    cudaFree(d_gg);
    cudaFree(d_gb);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_rmsNormBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_x, jfloatArray h_gamma, jfloat eps,
    jfloatArray h_gX, jfloatArray h_gGamma, jint outer, jint lastDim) {
    (void) clazz;
    if (outer <= 0 || lastDim <= 0) {
        return;
    }
    size_t bytes_xy = (size_t) outer * (size_t) lastDim * sizeof(float);
    size_t bytes_g = (size_t) lastDim * sizeof(float);
    float *d_go = nullptr, *d_x = nullptr, *d_g = nullptr, *d_gx = nullptr, *d_gg = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_g), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gx), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gg), bytes_g));

    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* px = env->GetFloatArrayElements(h_x, nullptr);
    jfloat* pg = env->GetFloatArrayElements(h_gamma, nullptr);
    jfloat* pgx = env->GetFloatArrayElements(h_gX, nullptr);
    jfloat* pgg = env->GetFloatArrayElements(h_gGamma, nullptr);
    if (!pgo || !px || !pg || !pgx || !pgg) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, pg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gx, pgx, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gg, pgg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gamma, pg, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gX, pgx, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gGamma, pgg, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_bwd(d_go, d_x, d_g, eps, d_gx, d_gg, outer, lastDim);
    pgx = env->GetFloatArrayElements(h_gX, nullptr);
    pgg = env->GetFloatArrayElements(h_gGamma, nullptr);
    if (!pgx || !pgg) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgx, d_gx, bytes_xy, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgg, d_gg, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gX, pgx, 0);
    env->ReleaseFloatArrayElements(h_gGamma, pgg, 0);

    cudaFree(d_go);
    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_gx);
    cudaFree(d_gg);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_rmsNormBackwardGPUDevice(
    JNIEnv* env, jclass clazz, jlong dGOut, jlong dX, jlong dGamma, jfloat eps, jlong dGX, jlong dGGamma,
    jint outer, jint lastDim) {
    (void) env;
    (void) clazz;
    if (outer <= 0 || lastDim <= 0 || dGOut == 0 || dX == 0 || dGamma == 0 || dGX == 0 || dGGamma == 0) {
        return;
    }
    float* go = reinterpret_cast<float*>(static_cast<uintptr_t>(dGOut));
    float* x = reinterpret_cast<float*>(static_cast<uintptr_t>(dX));
    float* gamma = reinterpret_cast<float*>(static_cast<uintptr_t>(dGamma));
    float* gx = reinterpret_cast<float*>(static_cast<uintptr_t>(dGX));
    float* gg = reinterpret_cast<float*>(static_cast<uintptr_t>(dGGamma));
    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_bwd(go, x, gamma, eps, gx, gg, outer, lastDim);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_multiplyBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_a, jfloatArray h_b, jfloatArray h_gA,
    jfloatArray h_gB, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_a = nullptr, *d_b = nullptr, *d_ga = nullptr, *d_gb = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_ga), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gb), bytes));

    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pa = env->GetFloatArrayElements(h_a, nullptr);
    jfloat* pb = env->GetFloatArrayElements(h_b, nullptr);
    jfloat* pga = env->GetFloatArrayElements(h_gA, nullptr);
    jfloat* pgb = env->GetFloatArrayElements(h_gB, nullptr);
    if (!pgo || !pa || !pb || !pga || !pgb) {
        cudaFree(d_go);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_ga);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_b, pb, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_ga, pga, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gb, pgb, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_a, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_b, pb, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gA, pga, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gB, pgb, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    multiply_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_a, d_b, d_ga, d_gb, n);
    pga = env->GetFloatArrayElements(h_gA, nullptr);
    pgb = env->GetFloatArrayElements(h_gB, nullptr);
    if (!pga || !pgb) {
        cudaFree(d_go);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_ga);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pga, d_ga, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgb, d_gb, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gA, pga, 0);
    env->ReleaseFloatArrayElements(h_gB, pgb, 0);

    cudaFree(d_go);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ga);
    cudaFree(d_gb);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_multiplyBackwardGPUDevice(
    JNIEnv* env, jclass clazz, jlong dGOut, jlong dA, jlong dB, jlong dGA, jlong dGB, jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dGOut == 0 || dA == 0 || dB == 0 || dGA == 0 || dGB == 0) {
        return;
    }
    float* go = reinterpret_cast<float*>(static_cast<uintptr_t>(dGOut));
    float* a = reinterpret_cast<float*>(static_cast<uintptr_t>(dA));
    float* b = reinterpret_cast<float*>(static_cast<uintptr_t>(dB));
    float* ga = reinterpret_cast<float*>(static_cast<uintptr_t>(dGA));
    float* gb = reinterpret_cast<float*>(static_cast<uintptr_t>(dGB));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    multiply_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(go, a, b, ga, gb, n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_geluBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_inp, jfloatArray h_gIn, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pin = env->GetFloatArrayElements(h_inp, nullptr);
    jfloat* pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgo || !pin || !pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, pin, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, pgi, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_inp, pin, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gIn, pgi, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    gelu_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgi, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gIn, pgi, 0);
    cudaFree(d_go);
    cudaFree(d_in);
    cudaFree(d_gi);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sigmoidBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_inp, jfloatArray h_gIn, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pin = env->GetFloatArrayElements(h_inp, nullptr);
    jfloat* pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgo || !pin || !pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, pin, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, pgi, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_inp, pin, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gIn, pgi, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgi, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gIn, pgi, 0);
    cudaFree(d_go);
    cudaFree(d_in);
    cudaFree(d_gi);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sigmoidBackwardGPUDevice(
    JNIEnv* env, jclass clazz, jlong dGOut, jlong dInp, jlong dGIn, jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dGOut == 0 || dInp == 0 || dGIn == 0) {
        return;
    }
    float* go = reinterpret_cast<float*>(static_cast<uintptr_t>(dGOut));
    float* inp = reinterpret_cast<float*>(static_cast<uintptr_t>(dInp));
    float* gi = reinterpret_cast<float*>(static_cast<uintptr_t>(dGIn));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(go, inp, gi, n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_reluBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_inp, jfloatArray h_gIn, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pin = env->GetFloatArrayElements(h_inp, nullptr);
    jfloat* pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgo || !pin || !pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, pin, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, pgi, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_inp, pin, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gIn, pgi, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    relu_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgi, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gIn, pgi, 0);
    cudaFree(d_go);
    cudaFree(d_in);
    cudaFree(d_gi);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_accumulateAddGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_acc, jfloatArray h_delta, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_d = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_d), bytes));
    jfloat* pa = env->GetFloatArrayElements(h_acc, nullptr);
    jfloat* pd = env->GetFloatArrayElements(h_delta, nullptr);
    if (!pa || !pd) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_d, pd, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_acc, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_delta, pd, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    accumulate_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_d, n);
    pa = env->GetFloatArrayElements(h_acc, nullptr);
    if (!pa) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pa, d_a, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_acc, pa, 0);
    cudaFree(d_a);
    cudaFree(d_d);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_accumulateScaledAddGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_acc, jfloatArray h_delta, jfloat scale, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_d = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_d), bytes));
    jfloat* pa = env->GetFloatArrayElements(h_acc, nullptr);
    jfloat* pd = env->GetFloatArrayElements(h_delta, nullptr);
    if (!pa || !pd) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_d, pd, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_acc, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_delta, pd, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    accumulate_scaled_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_d, scale, n);
    pa = env->GetFloatArrayElements(h_acc, nullptr);
    if (!pa) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pa, d_a, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_acc, pa, 0);
    cudaFree(d_a);
    cudaFree(d_d);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_accumulateAddGPUDevice(
    JNIEnv* env, jclass clazz, jlong dAcc, jlong dDelta, jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dAcc == 0 || dDelta == 0) {
        return;
    }
    float* acc = reinterpret_cast<float*>(static_cast<uintptr_t>(dAcc));
    float* delta = reinterpret_cast<float*>(static_cast<uintptr_t>(dDelta));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    accumulate_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(acc, delta, n);
}

/* ========== Device-pointer JNI wrappers for full-GPU training ========== */

#define CUDA_CHECK_RV(call, rv) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return (rv); \
        } \
    } while (0)

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_scaleInPlaceGPUDevice(
    JNIEnv* env, jclass clazz, jlong dSrc, jint n, jfloat scalar) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dSrc == 0) return;
    jgpt_cuda_ensure_stream();
    float* src = reinterpret_cast<float*>(static_cast<uintptr_t>(dSrc));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    scale_inplace_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(src, n, scalar);
}

JNIEXPORT jdouble JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sumSquaresGPUDevice(
    JNIEnv* env, jclass clazz, jlong dSrc, jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dSrc == 0) return 0.0;
    jgpt_cuda_ensure_stream();
    float* x = reinterpret_cast<float*>(static_cast<uintptr_t>(dSrc));
    cublasHandle_t h = get_extra_cublas_handle();
    if (h == nullptr) {
        return 0.0;
    }
    double acc = sum_squares_float_device_chunked_double_acc(h, x, n);
    CUDA_CHECK_RV(cudaStreamSynchronize(kTensorCudaStream), 0.0);
    return static_cast<jdouble>(acc);
}

JNIEXPORT jdouble JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sumSquaresGPUDeviceFused(
    JNIEnv* env, jclass clazz, jlongArray dPtrs, jintArray lens, jint nBufs) {
    (void) clazz;
    if (nBufs <= 0) {
        return 0.0;
    }
    jlong* pp = env->GetLongArrayElements(dPtrs, nullptr);
    jint* ll = env->GetIntArrayElements(lens, nullptr);
    if (!pp || !ll) {
        if (pp) {
            env->ReleaseLongArrayElements(dPtrs, pp, JNI_ABORT);
        }
        if (ll) {
            env->ReleaseIntArrayElements(lens, ll, JNI_ABORT);
        }
        return 0.0;
    }
    jgpt_cuda_ensure_stream();
    cublasHandle_t h = get_extra_cublas_handle();
    if (h == nullptr) {
        env->ReleaseLongArrayElements(dPtrs, pp, JNI_ABORT);
        env->ReleaseIntArrayElements(lens, ll, JNI_ABORT);
        return 0.0;
    }
    double acc = 0.0;
    for (jint i = 0; i < nBufs; i++) {
        jint n = ll[i];
        if (n <= 0) {
            continue;
        }
        uintptr_t up = static_cast<uintptr_t>(pp[i]);
        if (up == 0) {
            continue;
        }
        float* x = reinterpret_cast<float*>(up);
        acc += sum_squares_float_device_chunked_double_acc(h, x, n);
    }
    env->ReleaseLongArrayElements(dPtrs, pp, JNI_ABORT);
    env->ReleaseIntArrayElements(lens, ll, JNI_ABORT);
    CUDA_CHECK_RV(cudaStreamSynchronize(kTensorCudaStream), 0.0);
    return static_cast<jdouble>(acc);
}

static std::mutex g_any_nonfinite_alloc_mu;
static unsigned int* g_any_nonfinite_flag = nullptr;

static bool any_nonfinite_flag_ensure() {
    if (g_any_nonfinite_flag != nullptr) {
        return true;
    }
    std::lock_guard<std::mutex> lock(g_any_nonfinite_alloc_mu);
    if (g_any_nonfinite_flag != nullptr) {
        return true;
    }
    unsigned int* p = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&p), sizeof(unsigned int));
    if (e != cudaSuccess) {
        fprintf(stderr, "any_nonfinite_flag_ensure: cudaMalloc failed (%d)\n", static_cast<int>(e));
        return false;
    }
    g_any_nonfinite_flag = p;
    return true;
}

__global__ void any_nonfinite_kernel(const float* src, unsigned int* flag, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(src[i])) {
        atomicOr(flag, 1u);
    }
}

/* Фьюзированная проверка NaN/Inf по нескольким device-буферам: один cudaStreamSynchronize в конце.
 * Каждый буфер обрабатывается отдельным запуском any_nonfinite_kernel на том же стриме (async).
 * Флаг обнуляется один раз перед первым ядром. */
JNIEXPORT jboolean JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_anyNonFiniteGPUDeviceMulti(
    JNIEnv* env, jclass clazz, jlongArray dPtrs, jintArray lens, jint nBufs) {
    (void) clazz;
    if (nBufs <= 0) {
        return JNI_FALSE;
    }
    jlong* pp = env->GetLongArrayElements(dPtrs, nullptr);
    jint*  ll = env->GetIntArrayElements(lens, nullptr);
    if (!pp || !ll) {
        if (pp) env->ReleaseLongArrayElements(dPtrs, pp, JNI_ABORT);
        if (ll) env->ReleaseIntArrayElements(lens,  ll, JNI_ABORT);
        return JNI_FALSE;
    }

    jgpt_cuda_ensure_stream();
    if (!any_nonfinite_flag_ensure()) {
        env->ReleaseLongArrayElements(dPtrs, pp, JNI_ABORT);
        env->ReleaseIntArrayElements(lens, ll, JNI_ABORT);
        return JNI_FALSE;
    }
    /* Единственный memset перед всеми ядрами. */
    CUDA_CHECK_RV(cudaMemsetAsync(g_any_nonfinite_flag, 0, sizeof(unsigned int), kTensorCudaStream), JNI_FALSE);

    int threads = jgpt_cuda_get_optimal_block_size();
    for (jint i = 0; i < nBufs; i++) {
        jint n = ll[i];
        if (n <= 0) continue;
        uintptr_t up = static_cast<uintptr_t>(pp[i]);
        if (up == 0) continue;
        float* src = reinterpret_cast<float*>(up);
        int blocks = (n + threads - 1) / threads;
        any_nonfinite_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(src, g_any_nonfinite_flag, n);
    }

    env->ReleaseLongArrayElements(dPtrs, pp, JNI_ABORT);
    env->ReleaseIntArrayElements(lens,  ll, JNI_ABORT);

    unsigned int h_flag = 0;
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_flag, g_any_nonfinite_flag, sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost, kTensorCudaStream), JNI_FALSE);
    CUDA_CHECK_RV(cudaStreamSynchronize(kTensorCudaStream), JNI_FALSE);
    return h_flag != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_anyNonFiniteGPUDevice(
    JNIEnv* env, jclass clazz, jlong dSrc, jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dSrc == 0) return JNI_FALSE;
    jgpt_cuda_ensure_stream();
    if (!any_nonfinite_flag_ensure()) {
        return JNI_FALSE;
    }
    float* src = reinterpret_cast<float*>(static_cast<uintptr_t>(dSrc));
    CUDA_CHECK_RV(cudaMemsetAsync(g_any_nonfinite_flag, 0, sizeof(unsigned int), kTensorCudaStream), JNI_FALSE);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    any_nonfinite_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(src, g_any_nonfinite_flag, n);
    unsigned int h_flag = 0;
    CUDA_CHECK_RV(
            cudaMemcpyAsync(
                    &h_flag,
                    g_any_nonfinite_flag,
                    sizeof(unsigned int),
                    cudaMemcpyDeviceToHost,
                    kTensorCudaStream),
            JNI_FALSE);
    CUDA_CHECK_RV(cudaStreamSynchronize(kTensorCudaStream), JNI_FALSE);
    return h_flag != 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jfloat JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_crossEntropySoftmaxGradLossGPUDevice(
    JNIEnv* env, jclass clazz, jlong dLogits, jfloatArray hTargets, jlong dGrad,
    jint batch, jint seqLen, jint vocab, jfloat gradScale, jboolean useFp16) {
    (void) clazz;
    int nrows = batch * seqLen;
    if (nrows <= 0 || vocab <= 0) return 0.f;
    jgpt_cuda_ensure_stream();

    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    float* grad   = reinterpret_cast<float*>(static_cast<uintptr_t>(dGrad));

    size_t bytes_tgt = (size_t) nrows * sizeof(float);
    if (bytes_tgt > tl_ce_targets_bytes) {
        cudaFree(tl_ce_d_targets);
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_targets), bytes_tgt), 0.f);
        tl_ce_targets_bytes = bytes_tgt;
    }
    if (tl_ce_d_loss_sum == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_loss_sum), sizeof(float)), 0.f);
    }
    if (tl_ce_d_valid == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_valid), sizeof(unsigned int)), 0.f);
    }

    jfloat* ptgt = env->GetFloatArrayElements(hTargets, nullptr);
    if (!ptgt) return 0.f;
    CUDA_CHECK_RV(cudaMemcpyAsync(tl_ce_d_targets, ptgt, bytes_tgt, cudaMemcpyHostToDevice, kTensorCudaStream), 0.f);
    env->ReleaseFloatArrayElements(hTargets, ptgt, JNI_ABORT);
    /* No cudaStreamSynchronize here: same-stream order queues memset/kernel after H2D; one sync below before D2H loss. */

    CUDA_CHECK_RV(cudaMemsetAsync(tl_ce_d_loss_sum, 0, sizeof(float), kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemsetAsync(tl_ce_d_valid, 0, sizeof(unsigned int), kTensorCudaStream), 0.f);

    launch_cross_entropy(logits, tl_ce_d_targets, grad, gradScale, tl_ce_d_loss_sum,
            tl_ce_d_valid, nrows, vocab, useFp16 == JNI_TRUE);
    CUDA_KERNEL_CHECK_RV(0.f);

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_loss_sum, tl_ce_d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_valid, tl_ce_d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaStreamSynchronize(kTensorCudaStream), 0.f);

    return h_valid > 0 ? h_loss_sum / (float) h_valid : 0.f;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_crossEntropySoftmaxGradLossGPUDeviceHostFloatTargetsAsync(
    JNIEnv* env, jclass clazz, jlong dLogits, jfloatArray hTargets, jlong dGrad,
    jint batch, jint seqLen, jint vocab, jfloat gradScale, jboolean useFp16) {
    (void) clazz;
    int nrows = batch * seqLen;
    if (nrows <= 0 || vocab <= 0) {
        return;
    }
    jgpt_cuda_ensure_stream();

    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    float* grad = reinterpret_cast<float*>(static_cast<uintptr_t>(dGrad));

    size_t bytes_tgt = (size_t) nrows * sizeof(float);
    if (bytes_tgt > tl_ce_targets_bytes) {
        cudaFree(tl_ce_d_targets);
        tl_ce_d_targets = nullptr;
        tl_ce_targets_bytes = 0;
        if (cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_targets), bytes_tgt) != cudaSuccess) {
            return;
        }
        tl_ce_targets_bytes = bytes_tgt;
    }
    if (tl_ce_d_loss_sum == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_loss_sum), sizeof(float)));
    }
    if (tl_ce_d_valid == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_valid), sizeof(unsigned int)));
    }
    ce_async_host_ensure();
    ce_pinned_flt_targets_ensure((size_t) nrows);
    if (tl_ce_h_async_loss == nullptr || tl_ce_h_async_valid == nullptr || tl_ce_h_pinned_flt_targets == nullptr) {
        return;
    }

    jfloat* ptgt = env->GetFloatArrayElements(hTargets, nullptr);
    if (!ptgt) {
        return;
    }
    std::memcpy(tl_ce_h_pinned_flt_targets, ptgt, bytes_tgt);
    env->ReleaseFloatArrayElements(hTargets, ptgt, JNI_ABORT);

    CUDA_CHECK_X(cudaMemcpyAsync(
            tl_ce_d_targets,
            tl_ce_h_pinned_flt_targets,
            bytes_tgt,
            cudaMemcpyHostToDevice,
            kTensorCudaStream));

    CUDA_CHECK_X(cudaMemsetAsync(tl_ce_d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(tl_ce_d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy(logits, tl_ce_d_targets, grad, gradScale, tl_ce_d_loss_sum, tl_ce_d_valid, nrows, vocab, useFp16 == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK_X(cudaMemcpyAsync(
            tl_ce_h_async_loss,
            tl_ce_d_loss_sum,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(
            tl_ce_h_async_valid,
            tl_ce_d_valid,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost,
            kTensorCudaStream));
}

JNIEXPORT jfloat JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_crossEntropySoftmaxGradLossGPUDeviceTargetsDevice(
    JNIEnv* env, jclass clazz, jlong dLogits, jlong dTargetsInt, jlong dGrad,
    jint batch, jint seqLen, jint vocab, jfloat gradScale, jboolean useFp16) {
    (void) env;
    (void) clazz;
    int nrows = batch * seqLen;
    if (nrows <= 0 || vocab <= 0 || dTargetsInt == 0) return 0.f;
    jgpt_cuda_ensure_stream();

    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    float* grad   = reinterpret_cast<float*>(static_cast<uintptr_t>(dGrad));
    int* targets_i = reinterpret_cast<int*>(static_cast<uintptr_t>(dTargetsInt));

    if (tl_ce_d_loss_sum == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_loss_sum), sizeof(float)), 0.f);
    }
    if (tl_ce_d_valid == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_valid), sizeof(unsigned int)), 0.f);
    }

    CUDA_CHECK_RV(cudaMemsetAsync(tl_ce_d_loss_sum, 0, sizeof(float), kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemsetAsync(tl_ce_d_valid, 0, sizeof(unsigned int), kTensorCudaStream), 0.f);

    launch_cross_entropy_i32(logits, targets_i, grad, gradScale, tl_ce_d_loss_sum,
            tl_ce_d_valid, nrows, vocab, useFp16 == JNI_TRUE);
    CUDA_KERNEL_CHECK_RV(0.f);

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_loss_sum, tl_ce_d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_valid, tl_ce_d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaStreamSynchronize(kTensorCudaStream), 0.f);

    return h_valid > 0 ? h_loss_sum / (float) h_valid : 0.f;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_crossEntropySoftmaxGradLossGPUDeviceTargetsDeviceAsync(
    JNIEnv* env, jclass clazz, jlong dLogits, jlong dTargetsInt, jlong dGrad,
    jint batch, jint seqLen, jint vocab, jfloat gradScale, jboolean useFp16) {
    (void) env;
    (void) clazz;
    int nrows = batch * seqLen;
    if (nrows <= 0 || vocab <= 0 || dTargetsInt == 0) {
        return;
    }
    jgpt_cuda_ensure_stream();

    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    float* grad = reinterpret_cast<float*>(static_cast<uintptr_t>(dGrad));
    int* targets_i = reinterpret_cast<int*>(static_cast<uintptr_t>(dTargetsInt));

    if (tl_ce_d_loss_sum == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_loss_sum), sizeof(float)));
    }
    if (tl_ce_d_valid == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_valid), sizeof(unsigned int)));
    }
    ce_async_host_ensure();
    if (tl_ce_h_async_loss == nullptr || tl_ce_h_async_valid == nullptr) {
        return;
    }

    CUDA_CHECK_X(cudaMemsetAsync(tl_ce_d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(tl_ce_d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy_i32(logits, targets_i, grad, gradScale, tl_ce_d_loss_sum, tl_ce_d_valid, nrows, vocab, useFp16 == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK_X(cudaMemcpyAsync(
            tl_ce_h_async_loss,
            tl_ce_d_loss_sum,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(
            tl_ce_h_async_valid,
            tl_ce_d_valid,
            sizeof(unsigned int),
            cudaMemcpyDeviceToHost,
            kTensorCudaStream));
}

/* Чтение после cudaMemcpyAsync(D2H) для скаляра: нужен предшествующий TensorOpsGPU.synchronizeStream() (до backward,
 * который читает ∂logits с device после того же CE-launch). При ошибке sync в JVM pinned-буферы CE не освобождаются —
 * thread-local, переиспользуются следующим launch (без утечки процесса). */
JNIEXPORT jfloat JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_crossEntropySoftmaxGradLossGPUDeviceReadPendingFromHost(
    JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    if (tl_ce_h_async_loss == nullptr || tl_ce_h_async_valid == nullptr) {
        return 0.f;
    }
    float lossSum = *tl_ce_h_async_loss;
    unsigned int v = *tl_ce_h_async_valid;
    return v > 0 ? lossSum / (float) v : 0.f;
}

/** Для каждой пары (строка, k): logits_out = dot(normedHidden[row,:], W[:,cid]).
 *  W [dModel x vocab] row-major как в {@code matmul(rows x dModel) @ (dModel x vocab)}:
 *  W[d, v] по адресу {@code lmHead + d * vocab + v}. */
__global__ void lm_head_candidate_logits_kernel(
    const float* normedHidden,
    const float* lmHead,
    const int* candidateIds,
    float* candidateLogits,
    int rows,
    int dModel,
    int vocab,
    int candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * candidates;
    if (idx >= total) {
        return;
    }
    int row = idx / candidates;
    int cid = candidateIds[idx];
    if (cid < 0 || cid >= vocab) {
        candidateLogits[idx] = -INFINITY;
        return;
    }
    const float* hRow = normedHidden + row * dModel;
    const float* col = lmHead + cid;
    float acc = 0.f;
    for (int d = 0; d < dModel; d++) {
        acc += hRow[d] * col[d * vocab];
    }
    candidateLogits[idx] = acc;
}

__global__ void gather_logits_by_ids_kernel(
    const float* logits, const int* candidateIds, float* candidateLogits, int rows, int vocab, int candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * candidates;
    if (idx >= total) {
        return;
    }
    int row = idx / candidates;
    int cid = candidateIds[idx];
    if (cid < 0 || cid >= vocab) {
        candidateLogits[idx] = -INFINITY;
        return;
    }
    candidateLogits[idx] = logits[row * vocab + cid];
}

__global__ void sampled_ce_first_slot_kernel(
    const float* candidateLogits,
    const int* candidateIds,
    float* candidateGrad,
    float* lossSum,
    unsigned int* validCount,
    int rows,
    int candidates,
    float gradScale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int base = row * candidates;
    if (candidateIds[base] < 0) {
        for (int j = 0; j < candidates; j++) {
            candidateGrad[base + j] = 0.f;
        }
        return;
    }

    float maxLogit = candidateLogits[base];
    for (int j = 1; j < candidates; j++) {
        float v = candidateLogits[base + j];
        if (v > maxLogit) {
            maxLogit = v;
        }
    }

    float sumExp = 0.f;
    for (int j = 0; j < candidates; j++) {
        sumExp += expf(candidateLogits[base + j] - maxLogit);
    }
    if (!(sumExp > 0.f) || !isfinite(sumExp)) {
        for (int j = 0; j < candidates; j++) {
            candidateGrad[base + j] = NAN;
        }
        atomicAdd(lossSum, NAN);
        atomicAdd(validCount, 1u);
        return;
    }

    float invSum = 1.f / sumExp;
    float targetProb = 0.f;
    for (int j = 0; j < candidates; j++) {
        float p = expf(candidateLogits[base + j] - maxLogit) * invSum;
        candidateGrad[base + j] = ((j == 0) ? (p - 1.f) : p) * gradScale;
        if (j == 0) {
            targetProb = p;
        }
    }
    atomicAdd(lossSum, -logf(targetProb));
    atomicAdd(validCount, 1u);
}

__global__ void sampled_lm_head_dhidden_kernel(
    const int* candidateIds,
    const float* candidateGrad,
    const float* lmHeadWeights,
    float* dHidden,
    int rows,
    int dModel,
    int vocab,
    int candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dModel;
    if (idx >= total) {
        return;
    }
    int row = idx / dModel;
    int d = idx % dModel;
    int base = row * candidates;
    int wBase = d * vocab;
    float acc = 0.f;
    for (int j = 0; j < candidates; j++) {
        int cid = candidateIds[base + j];
        if (cid < 0 || cid >= vocab) {
            continue;
        }
        acc += candidateGrad[base + j] * lmHeadWeights[wBase + cid];
    }
    dHidden[idx] = acc;
}

__global__ void sampled_lm_head_dw_kernel(
    const int* candidateIds,
    const float* candidateGrad,
    const float* normedHidden,
    float* dLmHead,
    int rows,
    int dModel,
    int vocab,
    int candidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * candidates;
    if (idx >= total) {
        return;
    }
    int row = idx / candidates;
    int cid = candidateIds[idx];
    if (cid < 0 || cid >= vocab) {
        return;
    }
    float grad = candidateGrad[idx];
    if (grad == 0.f) {
        return;
    }
    const float* hiddenRow = normedHidden + row * dModel;
    for (int d = 0; d < dModel; d++) {
        atomicAdd(&dLmHead[d * vocab + cid], hiddenRow[d] * grad);
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_gatherLogitsByIdsGPUDevice(
    JNIEnv* env, jclass clazz, jlong dLogits, jlong dCandidateIds, jlong dCandidateLogits,
    jint rows, jint vocab, jint candidates) {
    (void) env;
    (void) clazz;
    if (rows <= 0 || vocab <= 0 || candidates <= 0) {
        return;
    }
    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    int* candidateIds = reinterpret_cast<int*>(static_cast<uintptr_t>(dCandidateIds));
    float* candidateLogits = reinterpret_cast<float*>(static_cast<uintptr_t>(dCandidateLogits));
    if (logits == nullptr || candidateIds == nullptr || candidateLogits == nullptr) {
        return;
    }
    int total = rows * candidates;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    gather_logits_by_ids_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            logits, candidateIds, candidateLogits, rows, vocab, candidates);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_lmHeadCandidateLogitsGPUDevice(
    JNIEnv* env, jclass clazz, jlong dNormedHidden, jlong dLmHead, jlong dCandidateIds, jlong dCandidateLogits,
    jint rows, jint dModel, jint vocab, jint candidates) {
    (void) env;
    (void) clazz;
    if (rows <= 0 || dModel <= 0 || vocab <= 0 || candidates <= 0) {
        return;
    }
    float* normedHidden = reinterpret_cast<float*>(static_cast<uintptr_t>(dNormedHidden));
    float* lmHead = reinterpret_cast<float*>(static_cast<uintptr_t>(dLmHead));
    int* candidateIds = reinterpret_cast<int*>(static_cast<uintptr_t>(dCandidateIds));
    float* candidateLogits = reinterpret_cast<float*>(static_cast<uintptr_t>(dCandidateLogits));
    if (normedHidden == nullptr || lmHead == nullptr || candidateIds == nullptr || candidateLogits == nullptr) {
        return;
    }
    int total = rows * candidates;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    lm_head_candidate_logits_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            normedHidden, lmHead, candidateIds, candidateLogits, rows, dModel, vocab, candidates);
    CUDA_KERNEL_CHECK();
}

JNIEXPORT jfloat JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sampledCrossEntropyGradLossGPUDeviceFirstSlot(
    JNIEnv* env, jclass clazz, jlong dCandidateLogits, jlong dCandidateIds, jlong dCandidateGrad,
    jint rows, jint candidates, jfloat gradScale) {
    (void) env;
    (void) clazz;
    if (rows <= 0 || candidates <= 0) {
        return 0.f;
    }
    float* candidateLogits = reinterpret_cast<float*>(static_cast<uintptr_t>(dCandidateLogits));
    int* candidateIds = reinterpret_cast<int*>(static_cast<uintptr_t>(dCandidateIds));
    float* candidateGrad = reinterpret_cast<float*>(static_cast<uintptr_t>(dCandidateGrad));
    if (candidateLogits == nullptr || candidateIds == nullptr || candidateGrad == nullptr) {
        return 0.f;
    }
    if (tl_ce_d_loss_sum == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_loss_sum), sizeof(float)), 0.f);
    }
    if (tl_ce_d_valid == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&tl_ce_d_valid), sizeof(unsigned int)), 0.f);
    }
    CUDA_CHECK_RV(cudaMemsetAsync(tl_ce_d_loss_sum, 0, sizeof(float), kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemsetAsync(tl_ce_d_valid, 0, sizeof(unsigned int), kTensorCudaStream), 0.f);
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    sampled_ce_first_slot_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            candidateLogits, candidateIds, candidateGrad, tl_ce_d_loss_sum, tl_ce_d_valid, rows, candidates, gradScale);
    CUDA_KERNEL_CHECK_RV(0.f);

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_loss_sum, tl_ce_d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_valid, tl_ce_d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaStreamSynchronize(kTensorCudaStream), 0.f);
    return h_valid > 0 ? h_loss_sum / (float) h_valid : 0.f;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sampledLmHeadBackwardGPUDevice(
    JNIEnv* env, jclass clazz, jlong dCandidateIds, jlong dCandidateGrad, jlong dNormedHidden,
    jlong dLmHeadWeights, jlong dHidden, jlong dLmHeadGrad, jint rows, jint dModel, jint vocab, jint candidates) {
    (void) env;
    (void) clazz;
    if (rows <= 0 || dModel <= 0 || vocab <= 0 || candidates <= 0) {
        return;
    }
    int* candidateIds = reinterpret_cast<int*>(static_cast<uintptr_t>(dCandidateIds));
    float* candidateGrad = reinterpret_cast<float*>(static_cast<uintptr_t>(dCandidateGrad));
    float* normedHidden = reinterpret_cast<float*>(static_cast<uintptr_t>(dNormedHidden));
    float* lmHeadWeights = reinterpret_cast<float*>(static_cast<uintptr_t>(dLmHeadWeights));
    float* dHiddenBuf = reinterpret_cast<float*>(static_cast<uintptr_t>(dHidden));
    float* dLmHeadBuf = reinterpret_cast<float*>(static_cast<uintptr_t>(dLmHeadGrad));
    if (candidateIds == nullptr || candidateGrad == nullptr || normedHidden == nullptr
            || lmHeadWeights == nullptr || dHiddenBuf == nullptr || dLmHeadBuf == nullptr) {
        return;
    }
    int threads = jgpt_cuda_get_optimal_block_size();
    int dhTotal = rows * dModel;
    int dhBlocks = (dhTotal + threads - 1) / threads;
    sampled_lm_head_dhidden_kernel<<<dhBlocks, threads, 0, kTensorCudaStream>>>(
            candidateIds, candidateGrad, lmHeadWeights, dHiddenBuf, rows, dModel, vocab, candidates);
    CUDA_KERNEL_CHECK();
    int gradTotal = rows * candidates;
    int gradBlocks = (gradTotal + threads - 1) / threads;
    sampled_lm_head_dw_kernel<<<gradBlocks, threads, 0, kTensorCudaStream>>>(
            candidateIds, candidateGrad, normedHidden, dLmHeadBuf, rows, dModel, vocab, candidates);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_accumulateAddFromHostGPUDevice(
    JNIEnv* env, jclass clazz, jlong dAcc, jfloatArray hDelta, jint off, jint len) {
    (void) clazz;
    if (len <= 0 || dAcc == 0) return;
    float* acc = reinterpret_cast<float*>(static_cast<uintptr_t>(dAcc));
    jfloat* pd = env->GetFloatArrayElements(hDelta, nullptr);
    if (!pd) return;
    size_t bytes = (size_t) len * sizeof(float);
    float* d_tmp = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tmp), bytes));
    CUDA_CHECK_X(cudaMemcpyAsync(d_tmp, pd + off, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(hDelta, pd, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (len + threads - 1) / threads;
    accumulate_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(acc, d_tmp, len);
    cudaFree(d_tmp);
}

static thread_local void* tl_graph_sdpa_warmup = nullptr;
static thread_local size_t tl_graph_sdpa_warmup_bytes = 0;

extern "C" void jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas(int bAttn, int seqLen, int dK, int dV) {
    if (bAttn <= 0 || seqLen <= 0 || dK <= 0 || dV <= 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    if (check_size_overflow((size_t) bAttn, (size_t) seqLen, sizeof(float)) ||
        check_size_overflow((size_t) bAttn * (size_t) seqLen, (size_t) dK, sizeof(float)) ||
        check_size_overflow((size_t) bAttn * (size_t) seqLen, (size_t) seqLen, sizeof(float))) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: size overflow (QK/prob)\n");
        return;
    }
    const size_t bytesQK = (size_t) bAttn * (size_t) seqLen * (size_t) dK * sizeof(float);
    const size_t bytesProb = (size_t) bAttn * (size_t) seqLen * (size_t) seqLen * sizeof(float);
    float *d_kt = nullptr, *d_scores = nullptr, *d_probs = nullptr;
    if (!attn_fwd_aux_ensure_qk_probs_only(bytesQK, bytesProb, &d_scores, &d_probs)) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: attn_fwd_aux_ensure_qk_probs_only failed\n");
        return;
    }
    (void) d_kt;

    const size_t outElems = (size_t) bAttn * (size_t) seqLen * (size_t) dV;
    if (check_size_overflow(outElems, sizeof(float), 1)) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: size overflow (out)\n");
        return;
    }
    const size_t gemm1Elems = (2U * bytesQK + bytesProb) / sizeof(float);
    const size_t gemm2Elems = bytesProb / sizeof(float) + 2U * outElems;
    const size_t needBytes = (gemm1Elems + gemm2Elems) * sizeof(float);
    if (needBytes > tl_graph_sdpa_warmup_bytes) {
        if (tl_graph_sdpa_warmup != nullptr) {
            cudaFree(tl_graph_sdpa_warmup);
            tl_graph_sdpa_warmup = nullptr;
        }
        tl_graph_sdpa_warmup_bytes = 0;
        if (cudaMalloc(&tl_graph_sdpa_warmup, needBytes) != cudaSuccess) {
            fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: cudaMalloc warmup\n");
            return;
        }
        tl_graph_sdpa_warmup_bytes = needBytes;
    }

    float* w = reinterpret_cast<float*>(tl_graph_sdpa_warmup);
    float* g1a = w;
    float* g1b = g1a + (bytesQK / sizeof(float));
    float* g1c = g1b + (bytesQK / sizeof(float));
    if (!batched_sgemm_row_major_extra(g1a, g1b, g1c, bAttn, seqLen, dK, seqLen, 1.0f, 0.0f)) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: SDPA batched gemm1 warmup failed\n");
        return;
    }

    float* g2a = g1c + (bytesProb / sizeof(float));
    float* g2b = g2a + (bytesProb / sizeof(float));
    float* g2c = g2b + outElems;
    (void) batched_sgemm_row_major_extra(g2a, g2b, g2c, bAttn, seqLen, seqLen, dV, 1.0f, 0.0f);
}

void jgpt_cuda_extra_cleanup(void) {
    destroy_extra_cublas_handle();
    {
        std::lock_guard<std::mutex> lock(g_any_nonfinite_alloc_mu);
        if (g_any_nonfinite_flag != nullptr) {
            cudaFree(g_any_nonfinite_flag);
            g_any_nonfinite_flag = nullptr;
        }
    }
    if (tl_graph_sdpa_warmup != nullptr) {
        cudaFree(tl_graph_sdpa_warmup);
        tl_graph_sdpa_warmup = nullptr;
        tl_graph_sdpa_warmup_bytes = 0;
    }
    if (tl_attn_fwd_aux != nullptr) {
        cudaFree(tl_attn_fwd_aux);
        tl_attn_fwd_aux = nullptr;
        tl_attn_fwd_aux_total = 0;
    }
    {
        std::lock_guard<std::mutex> lock(g_attn_fwd_graph_aux_mu);
        if (g_attn_fwd_graph_aux != nullptr) {
            cudaFree(g_attn_fwd_graph_aux);
            g_attn_fwd_graph_aux = nullptr;
            g_attn_fwd_graph_aux_total = 0;
        }
    }
    ce_free_cached();
    softmax_pair_free_cached();
    adamw_pool_free_cached();
    attn_bwd_host_free_cached();
    attn_bwd_aux_free_cached();
    if (tl_fa_D   != nullptr) { cudaFree(tl_fa_D);   tl_fa_D   = nullptr; tl_fa_D_bytes   = 0; }
}

// ----------------------------------------------------------------
//  JNI: Flash Attention Forward (GPU-resident, causal)
//  Q/K/V/O = [BH, S, Dh=kFaDh].  dLSEPtr = device float[BH*S].
//  S must be divisible by kFaBr (= 64); for padding caller truncates.
// ----------------------------------------------------------------
JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_decoderGraphNativeStabilityToken0(
        JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    uintptr_t fp = 0;
    uintptr_t gp = 0;
    size_t fsz = 0;
    size_t gsz = 0;
    jgpt_cuda_decoder_graph_debug_aux_snapshot(&fp, &gp, &fsz, &gsz);
    uintptr_t wp = 0;
    uintptr_t cp = 0;
    unsigned long long cwe = 0;
    unsigned long long cce = 0;
    int ov = 0;
    jgpt_cuda_decoder_graph_pack_snapshot(&wp, &cp, &cwe, &cce, &ov);
    const uintptr_t warm = reinterpret_cast<uintptr_t>(tl_graph_sdpa_warmup);
    const size_t warm_sz = tl_graph_sdpa_warmup_bytes;
    const uintptr_t fad = reinterpret_cast<uintptr_t>(tl_fa_D);
    const size_t fad_sz = tl_fa_D_bytes;

    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t x) {
        h ^= x;
        h *= 1099511628211ULL;
    };
    mix(static_cast<uint64_t>(fp));
    mix(static_cast<uint64_t>(gp));
    mix(static_cast<uint64_t>(fsz));
    mix(static_cast<uint64_t>(gsz));
    mix(static_cast<uint64_t>(wp));
    mix(static_cast<uint64_t>(cp));
    mix(static_cast<uint64_t>(cwe));
    mix(static_cast<uint64_t>(cce));
    mix(static_cast<uint64_t>(static_cast<unsigned int>(ov)));
    mix(static_cast<uint64_t>(warm));
    mix(static_cast<uint64_t>(warm_sz));
    mix(static_cast<uint64_t>(fad));
    mix(static_cast<uint64_t>(fad_sz));
    return static_cast<jlong>(static_cast<int64_t>(h));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_flashAttentionForwardGPUDeviceResident(
    JNIEnv* env, jclass clazz,
    jlong dQPtr, jlong dKPtr, jlong dVPtr, jlong dOutPtr, jlong dLSEPtr,
    jint BH, jint S, jint dHead, jfloat scale)
{
    (void) env; (void) clazz;
    if (!dQPtr || !dKPtr || !dVPtr || !dOutPtr || !dLSEPtr || BH <= 0 || S <= 0) return;
    if (dHead != static_cast<jint>(kFaDh)) {
        fprintf(
                stderr,
                "FlashAttention forward: compiled for d_head=%d, got d_head=%d\n",
                kFaDh,
                static_cast<int>(dHead));
        return;
    }

    /* dLSEPtr — буфер вызывающего на устройстве [BH*S] float. */
    jgpt_cuda_ensure_stream();
    if (!flash_attn_fwd_run(
                reinterpret_cast<float*>(static_cast<uintptr_t>(dQPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dKPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dVPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dOutPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dLSEPtr)),
                BH,
                S,
                scale)) {
        fprintf(stderr, "flashAttentionForwardGPUDeviceResident: flash_attn_fwd_run failed\n");
    }
}

// ----------------------------------------------------------------
//  JNI: Flash Attention Backward (GPU-resident, causal)
//  dO  = upstream gradient of O  [BH, S, Dh]
//  O   = forward output           [BH, S, Dh]  (cached, for D computation)
//  LSE = log-sum-exp from fwd     [BH, S]
//  Outputs: dQ, dK, dV           [BH, S, Dh]
// ----------------------------------------------------------------
JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_flashAttentionBackwardGPUDeviceResident(
    JNIEnv* env, jclass clazz,
    jlong dQPtr, jlong dKPtr, jlong dVPtr,
    jlong dOPtr, jlong dOGradPtr, jlong dLSEPtr,
    jlong dGradQPtr, jlong dGradKPtr, jlong dGradVPtr,
    jint BH, jint S, jint dHead, jfloat scale)
{
    (void) env; (void) clazz;
    if (!dQPtr || !dKPtr || !dVPtr || !dOPtr || !dOGradPtr || !dLSEPtr
            || !dGradQPtr || !dGradKPtr || !dGradVPtr || BH <= 0 || S <= 0) return;
    if (dHead != static_cast<jint>(kFaDh)) {
        fprintf(
                stderr,
                "FlashAttention backward: compiled for d_head=%d, got d_head=%d\n",
                kFaDh,
                static_cast<int>(dHead));
        return;
    }

    size_t D_bytes = (size_t)BH * S * sizeof(float);
    if (!fa_ensure_D(D_bytes)) return;

    jgpt_cuda_ensure_stream();
    if (!flash_attn_bwd_run(
                reinterpret_cast<float*>(static_cast<uintptr_t>(dQPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dKPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dVPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dOPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dOGradPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dLSEPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dGradQPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dGradKPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dGradVPtr)),
                BH,
                S,
                scale)) {
        fprintf(stderr, "flashAttentionBackwardGPUDeviceResident: flash_attn_bwd_run failed\n");
    }
}

#ifdef __cplusplus
}
#endif
