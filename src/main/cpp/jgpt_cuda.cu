#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <library_types.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "jgpt_cuda_stream.cuh"
#include "jgpt_cuda_ffn_link.h"
#include "jgpt_cuda_graph_prewarm.h"

static std::mutex g_stream_init_mutex;
static std::mutex g_cuda_device_caps_mutex;
static int g_cached_max_threads_per_block = 0;

cudaStream_t g_jgpt_cuda_stream = nullptr;

#define kTensorCudaStream (g_jgpt_cuda_stream)

void jgpt_cuda_ensure_stream(void) {
    if (g_jgpt_cuda_stream != nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_stream_init_mutex);
    if (g_jgpt_cuda_stream != nullptr) {
        return;
    }
    if (cudaStreamCreateWithFlags(&g_jgpt_cuda_stream, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "[TensorOpsGPU] cudaStreamCreateWithFlags failed\n");
        g_jgpt_cuda_stream = nullptr;
    }
}

void jgpt_cuda_destroy_stream(void) {
    std::lock_guard<std::mutex> lock(g_stream_init_mutex);
    if (g_jgpt_cuda_stream != nullptr) {
        cudaError_t s = cudaStreamSynchronize(g_jgpt_cuda_stream);
        if (s != cudaSuccess) {
            fprintf(stderr, "[TensorOpsGPU] cudaStreamSynchronize before destroy: %s\n", cudaGetErrorString(s));
        }
        cudaStreamDestroy(g_jgpt_cuda_stream);
        g_jgpt_cuda_stream = nullptr;
    }
}

int jgpt_cuda_max_threads_per_block(void) {
    int v = g_cached_max_threads_per_block;
    if (v > 0) {
        return v;
    }
    std::lock_guard<std::mutex> cap_lock(g_cuda_device_caps_mutex);
    if (g_cached_max_threads_per_block > 0) {
        return g_cached_max_threads_per_block;
    }
    int maxThreads = 256;
    cudaError_t e = cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, 0);
    if (e != cudaSuccess || maxThreads <= 0) {
        maxThreads = 256;
    }
    g_cached_max_threads_per_block = maxThreads;
    return maxThreads;
}

int jgpt_cuda_get_optimal_block_size(void) {
    int m = jgpt_cuda_max_threads_per_block();
    return (m >= 256) ? 256 : m;
}

/* ========== Thread-safe cuBLAS handle per thread ========== */
static thread_local cublasHandle_t tl_cublas_handle = nullptr;

static cublasHandle_t get_cublas_handle() {
    if (tl_cublas_handle == nullptr) {
        jgpt_cuda_ensure_stream();
        cublasCreate(&tl_cublas_handle);
        cublasSetMathMode(tl_cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
        cublasSetStream(tl_cublas_handle, g_jgpt_cuda_stream);
    }
    return tl_cublas_handle;
}

/* Thread-local staging for strided-batched GEMMs sharing operand B (X): QKV batch 3, FFN W1+W3 batch 2. */
static thread_local float* tl_qkv_w_pack_d = nullptr;
static thread_local float* tl_qkv_c_pack_d = nullptr;
static thread_local long long tl_qkv_cap_w_elems = 0;
static thread_local long long tl_qkv_cap_c_elems = 0;

/* Внешние pack-буферы (Java GpuFloatBuffer в GPTModel): cudaGraphExec держит указатели — без cudaFree tl_qkv. */
static thread_local bool tl_qkv_pack_override_active = false;
static thread_local float* tl_qkv_pack_override_w = nullptr;
static thread_local float* tl_qkv_pack_override_c = nullptr;
static thread_local long long tl_qkv_pack_override_cap_w = 0;
static thread_local long long tl_qkv_pack_override_cap_c = 0;

static inline float* jgpt_qkv_w_pack_ptr(void) {
    return tl_qkv_pack_override_active ? tl_qkv_pack_override_w : tl_qkv_w_pack_d;
}

static inline float* jgpt_qkv_c_pack_ptr(void) {
    return tl_qkv_pack_override_active ? tl_qkv_pack_override_c : tl_qkv_c_pack_d;
}

static bool shared_x_strided_batched_packs_ensure(long long wElems, long long cElems) {
    if (wElems <= 0 || cElems <= 0) {
        return false;
    }
    if (tl_qkv_pack_override_active) {
        if (wElems > tl_qkv_pack_override_cap_w || cElems > tl_qkv_pack_override_cap_c) {
            return false;
        }
        return tl_qkv_pack_override_w != nullptr && tl_qkv_pack_override_c != nullptr;
    }
    if (wElems > tl_qkv_cap_w_elems) {
        if (tl_qkv_w_pack_d != nullptr) {
            cudaFree(tl_qkv_w_pack_d);
            tl_qkv_w_pack_d = nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&tl_qkv_w_pack_d), static_cast<size_t>(wElems) * sizeof(float)) !=
            cudaSuccess) {
            tl_qkv_cap_w_elems = 0;
            return false;
        }
        tl_qkv_cap_w_elems = wElems;
    }
    if (cElems > tl_qkv_cap_c_elems) {
        if (tl_qkv_c_pack_d != nullptr) {
            cudaFree(tl_qkv_c_pack_d);
            tl_qkv_c_pack_d = nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&tl_qkv_c_pack_d), static_cast<size_t>(cElems) * sizeof(float)) !=
            cudaSuccess) {
            tl_qkv_cap_c_elems = 0;
            return false;
        }
        tl_qkv_cap_c_elems = cElems;
    }
    return tl_qkv_w_pack_d != nullptr && tl_qkv_c_pack_d != nullptr;
}

extern "C" void jgpt_cuda_graph_prewarm_qkv_ffn_strided_and_wo(int M, int dModel, int dInt) {
    if (M <= 0 || dModel <= 0 || dInt <= 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    const long long kn_qkv = (long long) dModel * (long long) dModel;
    const long long mn_qkv = (long long) M * (long long) dModel;
    const long long kn_ffn = (long long) dModel * (long long) dInt;
    const long long mn_ffn = (long long) M * (long long) dInt;
    if (kn_qkv > INT_MAX / 3 || mn_qkv > INT_MAX / 3 || kn_ffn > INT_MAX / 2 || mn_ffn > INT_MAX / 2) {
        return;
    }
    long long wNeed = 3LL * kn_qkv;
    if (2LL * kn_ffn > wNeed) {
        wNeed = 2LL * kn_ffn;
    }
    long long cNeed = 3LL * mn_qkv;
    if (2LL * mn_ffn > cNeed) {
        cNeed = 2LL * mn_ffn;
    }
    if (!shared_x_strided_batched_packs_ensure(wNeed, cNeed)) {
        return;
    }

    float* xNC = jgpt_qkv_c_pack_ptr();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();

    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            dModel,
            M,
            dModel,
            &alpha,
            jgpt_qkv_w_pack_ptr(),
            dModel,
            kn_qkv,
            xNC,
            dModel,
            0LL,
            &beta,
            jgpt_qkv_c_pack_ptr(),
            dModel,
            mn_qkv,
            3);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(
                stderr,
                "jgpt_cuda_graph_prewarm_qkv_ffn_strided_and_wo: QKV strided status %d\n",
                (int) st);
        return;
    }

    st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            dInt,
            M,
            dModel,
            &alpha,
            jgpt_qkv_w_pack_ptr(),
            dInt,
            kn_ffn,
            xNC,
            dModel,
            0LL,
            &beta,
            jgpt_qkv_c_pack_ptr(),
            dInt,
            mn_ffn,
            2);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(
                stderr,
                "jgpt_cuda_graph_prewarm_qkv_ffn_strided_and_wo: FFN strided status %d\n",
                (int) st);
        return;
    }

    st = cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            dModel,
            M,
            dModel,
            &alpha,
            jgpt_qkv_w_pack_ptr(),
            dModel,
            xNC,
            dModel,
            &beta,
            jgpt_qkv_c_pack_ptr(),
            dModel);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_qkv_ffn_strided_and_wo: Wo Sgemm status %d\n", (int) st);
    }
}

static void destroy_cublas_handle() {
    if (tl_cublas_handle) {
        cublasDestroy(tl_cublas_handle);
        tl_cublas_handle = nullptr;
    }
}

/* ========== Pinned host staging with automatic cleanup ========== */
static thread_local float* tl_pin_A = nullptr;
static thread_local float* tl_pin_B = nullptr;
static thread_local float* tl_pin_C = nullptr;
static thread_local size_t tl_pin_capA = 0;
static thread_local size_t tl_pin_capB = 0;
static thread_local size_t tl_pin_capC = 0;
static thread_local bool tl_pin_heapA = false;
static thread_local bool tl_pin_heapB = false;
static thread_local bool tl_pin_heapC = false;

static void host_pin_release_one(float* p, bool heap) {
    if (p == nullptr) {
        return;
    }
    if (heap) {
        std::free(p);
    } else {
        cudaFreeHost(p);
    }
}

static void gpufb_pin_free();
static bool gpufb_pin_ensure(size_t need_bytes);

static void host_pin_free_all() {
    host_pin_release_one(tl_pin_A, tl_pin_heapA);
    host_pin_release_one(tl_pin_B, tl_pin_heapB);
    host_pin_release_one(tl_pin_C, tl_pin_heapC);
    tl_pin_A = tl_pin_B = tl_pin_C = nullptr;
    tl_pin_capA = tl_pin_capB = tl_pin_capC = 0;
    tl_pin_heapA = tl_pin_heapB = tl_pin_heapC = false;
    gpufb_pin_free();
}

/* ========== Pinned staging for GpuFloatBuffer H2D/D2H (grow-only; freed in host_pin_free_all) ========== */
static thread_local unsigned char* tl_gpufb_pin = nullptr;
static thread_local size_t tl_gpufb_cap = 0;

static void gpufb_pin_free() {
    if (tl_gpufb_pin != nullptr) {
        cudaFreeHost(tl_gpufb_pin);
        tl_gpufb_pin = nullptr;
        tl_gpufb_cap = 0;
    }
}

/** @return false if cudaHostAlloc failed */
static bool gpufb_pin_ensure(size_t need_bytes) {
    if (need_bytes == 0U) {
        return true;
    }
    if (tl_gpufb_cap >= need_bytes) {
        return true;
    }
    gpufb_pin_free();
    cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(&tl_gpufb_pin), need_bytes, cudaHostAllocDefault);
    if (e != cudaSuccess || tl_gpufb_pin == nullptr) {
        tl_gpufb_pin = nullptr;
        tl_gpufb_cap = 0;
        return false;
    }
    tl_gpufb_cap = need_bytes;
    return true;
}

/** Staging host buffers: 0 = pinned, 1 = heap (use blocking cudaMemcpy), -1 = failure */
static int host_pin_ensure(size_t szA, size_t szB, size_t szC,
                           float** outA, float** outB, float** outC) {
    const size_t ALIGN = 64;
    szA = (szA + ALIGN - 1) & ~(ALIGN - 1);
    szB = (szB + ALIGN - 1) & ~(ALIGN - 1);
    szC = (szC + ALIGN - 1) & ~(ALIGN - 1);

    auto grow_one = [](float** pp, size_t* cap, bool* heap, size_t need) -> bool {
        if (*cap >= need) {
            return true;
        }
        host_pin_release_one(*pp, *heap);
        *pp = nullptr;
        *cap = 0;
        *heap = false;
        cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(pp), need, cudaHostAllocDefault);
        if (e != cudaSuccess || *pp == nullptr) {
            *pp = static_cast<float*>(std::malloc(need));
            if (*pp == nullptr) {
                return false;
            }
            *heap = true;
        } else {
            *heap = false;
        }
        *cap = need;
        return true;
    };

    if (!grow_one(&tl_pin_A, &tl_pin_capA, &tl_pin_heapA, szA)) {
        host_pin_free_all();
        return -1;
    }
    if (!grow_one(&tl_pin_B, &tl_pin_capB, &tl_pin_heapB, szB)) {
        host_pin_free_all();
        return -1;
    }
    if (!grow_one(&tl_pin_C, &tl_pin_capC, &tl_pin_heapC, szC)) {
        host_pin_free_all();
        return -1;
    }

    if ((szA > 0U && tl_pin_A == nullptr) || (szB > 0U && tl_pin_B == nullptr) ||
        (szC > 0U && tl_pin_C == nullptr)) {
        host_pin_free_all();
        return -1;
    }

    if (outA) {
        *outA = tl_pin_A;
    }
    if (outB) {
        *outB = tl_pin_B;
    }
    if (outC) {
        *outC = tl_pin_C;
    }

    const bool any_heap = (szA > 0U && tl_pin_heapA) || (szB > 0U && tl_pin_heapB) || (szC > 0U && tl_pin_heapC);
    return any_heap ? 1 : 0;
}

/* ========== Device buffer caching with overflow protection ========== */
static thread_local float* tl_mm_dBias = nullptr;
static thread_local size_t tl_mm_biasCap = 0;

static int ensure_bias_device(size_t biasBytes) {
    if (tl_mm_dBias && tl_mm_biasCap >= biasBytes) return 0;
    cudaFree(tl_mm_dBias);
    tl_mm_dBias = nullptr;
    tl_mm_biasCap = 0;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&tl_mm_dBias), biasBytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA ensure_bias_device: %s\n", cudaGetErrorString(e));
        return -1;
    }
    tl_mm_biasCap = biasBytes;
    return 0;
}

// 🔧 FIX: добавить проверку переполнения перед вычислением размера
static inline bool check_size_overflow(size_t a, size_t b, size_t elem_size) {
    if (a == 0 || b == 0) return false;
    if (a > SIZE_MAX / b) return true;
    size_t prod = a * b;
    return prod > SIZE_MAX / elem_size;
}

static thread_local float* tl_mm_dA = nullptr;
static thread_local float* tl_mm_dB = nullptr;
static thread_local float* tl_mm_dC = nullptr;
static thread_local size_t tl_mm_szA = 0;
static thread_local size_t tl_mm_szB = 0;
static thread_local size_t tl_mm_szC = 0;

static thread_local float* tl_mb_dA = nullptr;  // batched
static thread_local float* tl_mb_dB = nullptr;
static thread_local float* tl_mb_dC = nullptr;
static thread_local size_t tl_mb_szA = 0;
static thread_local size_t tl_mb_szB = 0;
static thread_local size_t tl_mb_szC = 0;

static void mm_free_cached() {
    cudaFree(tl_mm_dA); cudaFree(tl_mm_dB); cudaFree(tl_mm_dC);
    tl_mm_dA = tl_mm_dB = tl_mm_dC = nullptr;
    tl_mm_szA = tl_mm_szB = tl_mm_szC = 0;
}

static void mb_free_cached() {
    cudaFree(tl_mb_dA); cudaFree(tl_mb_dB); cudaFree(tl_mb_dC);
    tl_mb_dA = tl_mb_dB = tl_mb_dC = nullptr;
    tl_mb_szA = tl_mb_szB = tl_mb_szC = 0;
}

// 🔧 FIX: Half-precision buffers
static thread_local __half* tl_mm_dAh = nullptr;
static thread_local __half* tl_mm_dBh = nullptr;
static thread_local size_t tl_mm_nelemAh = 0;
static thread_local size_t tl_mm_nelemBh = 0;

static thread_local __half* tl_mb_dAh = nullptr;
static thread_local __half* tl_mb_dBh = nullptr;
static thread_local size_t tl_mb_nelemAh = 0;
static thread_local size_t tl_mb_nelemBh = 0;

static void mm_free_half_cached() {
    cudaFree(tl_mm_dAh); cudaFree(tl_mm_dBh);
    tl_mm_dAh = tl_mm_dBh = nullptr;
    tl_mm_nelemAh = tl_mm_nelemBh = 0;
}

static void mb_free_half_cached() {
    cudaFree(tl_mb_dAh); cudaFree(tl_mb_dBh);
    tl_mb_dAh = tl_mb_dBh = nullptr;
    tl_mb_nelemAh = tl_mb_nelemBh = 0;
}

extern void jgpt_cuda_extra_cleanup(void);

void jgpt_cuda_cleanup_thread_resources(void) {
    if (g_jgpt_cuda_stream != nullptr) {
        cudaError_t s = cudaStreamSynchronize(g_jgpt_cuda_stream);
        if (s != cudaSuccess) {
            fprintf(stderr, "[TensorOpsGPU] jgpt_cuda_cleanup_thread_resources: %s\n", cudaGetErrorString(s));
        }
    }
    jgpt_cuda_extra_cleanup();
    destroy_cublas_handle();
    host_pin_free_all();
    mm_free_cached();
    mb_free_cached();
    mm_free_half_cached();
    mb_free_half_cached();
}

static int mm_ensure_half_AB(size_t nelemA, size_t nelemB, __half** outA, __half** outB) {
    if (tl_mm_dAh && tl_mm_nelemAh == nelemA && tl_mm_dBh && tl_mm_nelemBh == nelemB) {
        *outA = tl_mm_dAh; *outB = tl_mm_dBh; return 0;
    }
    mm_free_half_cached();
    if (check_size_overflow(nelemA, sizeof(__half), 1) ||
        check_size_overflow(nelemB, sizeof(__half), 1)) {
        fprintf(stderr, "CUDA mm_ensure_half_AB: size overflow\n");
        return -1;
    }
    size_t szA = nelemA * sizeof(__half);
    size_t szB = nelemB * sizeof(__half);
    __half *a = nullptr, *b = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&a), szA);
    cudaError_t e2 = (e1 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&b), szB) : cudaErrorMemoryAllocation;
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        cudaFree(a); cudaFree(b);
        fprintf(stderr, "CUDA mm_ensure_half_AB malloc: %s / %s\n",
                cudaGetErrorString(e1), cudaGetErrorString(e2));
        return -1;
    }
    tl_mm_dAh = a; tl_mm_dBh = b;
    tl_mm_nelemAh = nelemA; tl_mm_nelemBh = nelemB;
    *outA = a; *outB = b;
    return 0;
}

static int mb_ensure_half_AB(size_t nelemA, size_t nelemB, __half** outA, __half** outB) {
    if (tl_mb_dAh && tl_mb_nelemAh == nelemA && tl_mb_dBh && tl_mb_nelemBh == nelemB) {
        *outA = tl_mb_dAh; *outB = tl_mb_dBh; return 0;
    }
    mb_free_half_cached();
    if (check_size_overflow(nelemA, sizeof(__half), 1) ||
        check_size_overflow(nelemB, sizeof(__half), 1)) {
        fprintf(stderr, "CUDA mb_ensure_half_AB: size overflow\n");
        return -1;
    }
    size_t szA = nelemA * sizeof(__half);
    size_t szB = nelemB * sizeof(__half);
    __half *a = nullptr, *b = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&a), szA);
    cudaError_t e2 = (e1 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&b), szB) : cudaErrorMemoryAllocation;
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        cudaFree(a); cudaFree(b);
        fprintf(stderr, "CUDA mb_ensure_half_AB malloc: %s / %s\n",
                cudaGetErrorString(e1), cudaGetErrorString(e2));
        return -1;
    }
    tl_mb_dAh = a; tl_mb_dBh = b;
    tl_mb_nelemAh = nelemA; tl_mb_nelemBh = nelemB;
    *outA = a; *outB = b;
    return 0;
}

static int mm_ensure_buffers(size_t szA, size_t szB, size_t szC,
                             float** outA, float** outB, float** outC) {
    if (tl_mm_dA && tl_mm_szA == szA && tl_mm_szB == szB && tl_mm_szC == szC) {
        *outA = tl_mm_dA; *outB = tl_mm_dB; *outC = tl_mm_dC; return 0;
    }
    mm_free_cached();
    float *a = nullptr, *b = nullptr, *c = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&a), szA);
    cudaError_t e2 = (e1 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&b), szB) : cudaErrorMemoryAllocation;
    cudaError_t e3 = (e2 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&c), szC) : cudaErrorMemoryAllocation;
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        cudaFree(a); cudaFree(b); cudaFree(c);
        fprintf(stderr, "CUDA mm_ensure_buffers malloc: %s / %s / %s\n",
                cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3));
        return -1;
    }
    tl_mm_dA = a; tl_mm_dB = b; tl_mm_dC = c;
    tl_mm_szA = szA; tl_mm_szB = szB; tl_mm_szC = szC;
    *outA = a; *outB = b; *outC = c;
    return 0;
}

static int mb_ensure_buffers(size_t szA, size_t szB, size_t szC,
                             float** outA, float** outB, float** outC) {
    if (tl_mb_dA && tl_mb_szA == szA && tl_mb_szB == szB && tl_mb_szC == szC) {
        *outA = tl_mb_dA; *outB = tl_mb_dB; *outC = tl_mb_dC; return 0;
    }
    mb_free_cached();
    float *a = nullptr, *b = nullptr, *c = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&a), szA);
    cudaError_t e2 = (e1 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&b), szB) : cudaErrorMemoryAllocation;
    cudaError_t e3 = (e2 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&c), szC) : cudaErrorMemoryAllocation;
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        cudaFree(a); cudaFree(b); cudaFree(c);
        fprintf(stderr, "CUDA mb_ensure_buffers malloc: %s / %s / %s\n",
                cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3));
        return -1;
    }
    tl_mb_dA = a; tl_mb_dB = b; tl_mb_dC = c;
    tl_mb_szA = szA; tl_mb_szB = szB; tl_mb_szC = szC;
    *outA = a; *outB = b; *outC = c;
    return 0;
}

// ========== Kernels with error checking ==========

__global__ void vec_add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vec_sub_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void relu_kernel(const float* __restrict__ a, float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = a[i]; b[i] = x > 0.f ? x : 0.f; }
}

// 🔧 FIX: адаптивный выбор blockSize + проверка запуска
static void launch_vec_add(const float* d_a, const float* d_b, float* d_c, int n) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, d_c, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_vec_add error: %s\n", cudaGetErrorString(err));
}

static void launch_vec_sub(const float* d_a, const float* d_b, float* d_c, int n) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    vec_sub_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, d_c, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_vec_sub error: %s\n", cudaGetErrorString(err));
}

static void launch_relu(const float* d_a, float* d_b, int n) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_relu error: %s\n", cudaGetErrorString(err));
}

__global__ void float_to_half_kernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

static void launch_float_to_half(const float* d_src, __half* d_dst, int n) {
    if (n <= 0) return;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, d_dst, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_float_to_half error: %s\n", cudaGetErrorString(err));
}

__global__ void half_to_float_kernel(const __half* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

static void launch_half_to_float(const __half* d_src, float* d_dst, int n) {
    if (n <= 0) return;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    half_to_float_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, d_dst, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_half_to_float error: %s\n", cudaGetErrorString(err));
}

__global__ void bias_relu_inplace_kernel(float* __restrict__ C, const float* __restrict__ bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int col = idx % N;
    float x = C[idx] + bias[col];
    C[idx] = x > 0.f ? x : 0.f;
}

static void launch_bias_relu_inplace(float* d_C, const float* d_bias, int M, int N) {
    int total = M * N;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    bias_relu_inplace_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_C, d_bias, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_bias_relu_inplace error: %s\n", cudaGetErrorString(err));
}

__global__ void bias_add_inplace_kernel(float* __restrict__ C, const float* __restrict__ bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int col = idx % N;
    C[idx] += bias[col];
}

static void launch_bias_add_inplace(float* d_C, const float* d_bias, int M, int N) {
    int total = M * N;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    bias_add_inplace_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_C, d_bias, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_bias_add_inplace error: %s\n", cudaGetErrorString(err));
}

__global__ void sum_columns_kernel(const float* __restrict__ src, float* __restrict__ dst, int M, int N, float beta) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float s = 0.f;
    for (int i = 0; i < M; ++i) {
        s += src[i * N + j];
    }
    dst[j] = beta * dst[j] + s;
}

static void launch_sum_columns(const float* d_src, float* d_dst, int M, int N, float beta) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (N + threads - 1) / threads;
    sum_columns_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, d_dst, M, N, beta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_sum_columns error: %s\n", cudaGetErrorString(err));
}

// ========== Error macros ==========
#define CUDA_CHECK_VOID(call) \
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

// ========== JNI ==========
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    (void) vm; (void) reserved;
    printf("[TensorOpsGPU] cuBLAS will be initialized per-thread with TF32 tensor ops\n");
    return JNI_VERSION_1_8;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    (void) vm; (void) reserved;
    jgpt_cuda_cleanup_thread_resources();
    jgpt_cuda_destroy_stream();
}

// ========== MATMUL (SGEMM + TF32) ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_A, jfloatArray h_B, jfloatArray h_C,
    jint M, jint K, jint N) {

    if (check_size_overflow(M, K, sizeof(float)) ||
        check_size_overflow(K, N, sizeof(float)) ||
        check_size_overflow(M, N, sizeof(float))) {
        fprintf(stderr, "matmulGPU: size overflow M=%d K=%d N=%d\n", M, K, N);
        return;
    }

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    jfloat* A_ptr = env->GetFloatArrayElements(h_A, nullptr);
    jfloat* B_ptr = env->GetFloatArrayElements(h_B, nullptr);
    if (!A_ptr || !B_ptr) {
        if (A_ptr) env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        if (B_ptr) env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        return;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (mm_ensure_buffers(size_A, size_B, size_C, &d_A, &d_B, &d_C) != 0) {
        env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        return;
    }

    float *pA = nullptr, *pB = nullptr, *pC = nullptr;
    int pin_ok = host_pin_ensure(size_A, size_B, size_C, &pA, &pB, &pC);

    if ((pin_ok == 0 || pin_ok == 1) && pA != nullptr && pB != nullptr && pC != nullptr) {
        std::memcpy(pA, A_ptr, size_A);
        std::memcpy(pB, B_ptr, size_B);
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, pA, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, pB, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, A_ptr, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, B_ptr, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Sgemm error: status %d\n", (int)st);
        return;
    }

    jfloat* C_ptr = env->GetFloatArrayElements(h_C, nullptr);
    if (!C_ptr) return;

    if ((pin_ok == 0 || pin_ok == 1) && pC != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpyAsync(pC, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
        std::memcpy(C_ptr, pC, size_C);
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(C_ptr, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_C, C_ptr, 0);
}

// ========== FP16 MATMUL ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulGPUFp16(
    JNIEnv* env, jclass clazz,
    jfloatArray h_A, jfloatArray h_B, jfloatArray h_C,
    jint M, jint K, jint N) {

    if (check_size_overflow(M, K, sizeof(float)) ||
        check_size_overflow(K, N, sizeof(float)) ||
        check_size_overflow(M, N, sizeof(float))) {
        fprintf(stderr, "matmulGPUFp16: size overflow\n");
        return;
    }

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    jfloat* A_ptr = env->GetFloatArrayElements(h_A, nullptr);
    jfloat* B_ptr = env->GetFloatArrayElements(h_B, nullptr);
    if (!A_ptr || !B_ptr) {
        if (A_ptr) env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        if (B_ptr) env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        return;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (mm_ensure_buffers(size_A, size_B, size_C, &d_A, &d_B, &d_C) != 0) {
        env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        return;
    }

    float *pA = nullptr, *pB = nullptr;
    int pin_ok = host_pin_ensure(size_A, size_B, 0, &pA, &pB, nullptr);

    if ((pin_ok == 0 || pin_ok == 1) && pA != nullptr && pB != nullptr) {
        std::memcpy(pA, A_ptr, size_A);
        std::memcpy(pB, B_ptr, size_B);
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, pA, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, pB, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, A_ptr, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, B_ptr, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);

    size_t nelemA = (size_t)M * K;
    size_t nelemB = (size_t)K * N;
    if (nelemA > INT_MAX || nelemB > INT_MAX) {
        fprintf(stderr, "matmulGPUFp16: buffer too large for kernel\n");
        return;
    }

    __half *d_Ah = nullptr, *d_Bh = nullptr;
    if (mm_ensure_half_AB(nelemA, nelemB, &d_Ah, &d_Bh) != 0) return;

    launch_float_to_half(d_A, d_Ah, (int)nelemA);
    launch_float_to_half(d_B, d_Bh, (int)nelemB);
    CUDA_KERNEL_CHECK();

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                     d_Bh, CUDA_R_16F, N, d_Ah, CUDA_R_16F, K, &beta,
                                     d_C, CUDA_R_32F, N,
                                     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GemmEx (FP16) error: status %d\n", (int)st);
        return;
    }

    jfloat* C_ptr = env->GetFloatArrayElements(h_C, nullptr);
    if (!C_ptr) return;

    float *pC = nullptr;
    int pin_c = host_pin_ensure(0, 0, size_C, nullptr, nullptr, &pC);
    if ((pin_c == 0 || pin_c == 1) && pC != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpyAsync(pC, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
        std::memcpy(C_ptr, pC, size_C);
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(C_ptr, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_C, C_ptr, 0);
}

// ========== BATCHED MATMUL (FULL VERSION) ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulBatchedGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_A, jfloatArray h_B, jfloatArray h_C,
    jint M, jint K, jint N, jint batchCount) {
    (void) clazz;
    if (batchCount <= 0 || M <= 0 || K <= 0 || N <= 0) return;

    jsize lenA = env->GetArrayLength(h_A);
    jsize lenB = env->GetArrayLength(h_B);
    jsize lenC = env->GetArrayLength(h_C);
    long long expectedA = (long long) batchCount * (long long) M * (long long) K;
    long long expectedB = (long long) batchCount * (long long) K * (long long) N;
    long long expectedC = (long long) batchCount * (long long) M * (long long) N;
    if ((long long) lenA != expectedA || (long long) lenB != expectedB || (long long) lenC != expectedC) {
        fprintf(stderr, "[TensorOpsGPU] matmulBatchedGPU: buffer size mismatch\n");
        return;
    }

    size_t numA = (size_t) batchCount * (size_t) M * (size_t) K;
    size_t numB = (size_t) batchCount * (size_t) K * (size_t) N;
    size_t numC = (size_t) batchCount * (size_t) M * (size_t) N;

    if (check_size_overflow(numA, sizeof(float), 1) ||
        check_size_overflow(numB, sizeof(float), 1) ||
        check_size_overflow(numC, sizeof(float), 1)) {
        fprintf(stderr, "matmulBatchedGPU: size overflow\n");
        return;
    }

    size_t size_A = numA * sizeof(float);
    size_t size_B = numB * sizeof(float);
    size_t size_C = numC * sizeof(float);

    jfloat* A_ptr = env->GetFloatArrayElements(h_A, nullptr);
    jfloat* B_ptr = env->GetFloatArrayElements(h_B, nullptr);
    if (!A_ptr || !B_ptr) {
        if (A_ptr) env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        if (B_ptr) env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        return;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (mb_ensure_buffers(size_A, size_B, size_C, &d_A, &d_B, &d_C) != 0) {
        env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        return;
    }

    float *pA = nullptr, *pB = nullptr, *pC = nullptr;
    int pin_ok = host_pin_ensure(size_A, size_B, size_C, &pA, &pB, &pC);

    if ((pin_ok == 0 || pin_ok == 1) && pA != nullptr && pB != nullptr && pC != nullptr) {
        std::memcpy(pA, A_ptr, size_A);
        std::memcpy(pB, B_ptr, size_B);
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, pA, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, pB, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, A_ptr, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, B_ptr, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    long long strideA_elems = (long long) M * (long long) K;
    long long strideB_elems = (long long) K * (long long) N;
    long long strideC_elems = (long long) M * (long long) N;

    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        d_B, N, strideB_elems, d_A, K, strideA_elems, &beta, d_C, N, strideC_elems, batchCount);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS SgemmStridedBatched error: status %d\n", (int)st);
        return;
    }

    jfloat* C_ptr = env->GetFloatArrayElements(h_C, nullptr);
    if (!C_ptr) return;

    if ((pin_ok == 0 || pin_ok == 1) && pC != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpyAsync(pC, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
        std::memcpy(C_ptr, pC, size_C);
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(C_ptr, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_C, C_ptr, 0);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulBatchedGPUFp16(
    JNIEnv* env, jclass clazz,
    jfloatArray h_A, jfloatArray h_B, jfloatArray h_C,
    jint M, jint K, jint N, jint batchCount) {
    (void) clazz;
    if (batchCount <= 0 || M <= 0 || K <= 0 || N <= 0) return;

    jsize lenA = env->GetArrayLength(h_A);
    jsize lenB = env->GetArrayLength(h_B);
    jsize lenC = env->GetArrayLength(h_C);
    long long expectedA = (long long) batchCount * (long long) M * (long long) K;
    long long expectedB = (long long) batchCount * (long long) K * (long long) N;
    long long expectedC = (long long) batchCount * (long long) M * (long long) N;
    if ((long long) lenA != expectedA || (long long) lenB != expectedB || (long long) lenC != expectedC) {
        fprintf(stderr, "[TensorOpsGPU] matmulBatchedGPUFp16: buffer size mismatch\n");
        return;
    }

    size_t numA = (size_t) batchCount * (size_t) M * (size_t) K;
    size_t numB = (size_t) batchCount * (size_t) K * (size_t) N;
    size_t numC = (size_t) batchCount * (size_t) M * (size_t) N;

    if (check_size_overflow(numA, sizeof(float), 1) ||
        check_size_overflow(numB, sizeof(float), 1) ||
        check_size_overflow(numC, sizeof(float), 1)) {
        fprintf(stderr, "matmulBatchedGPUFp16: size overflow\n");
        return;
    }

    size_t size_A = numA * sizeof(float);
    size_t size_B = numB * sizeof(float);
    size_t size_C = numC * sizeof(float);

    jfloat* A_ptr = env->GetFloatArrayElements(h_A, nullptr);
    jfloat* B_ptr = env->GetFloatArrayElements(h_B, nullptr);
    if (!A_ptr || !B_ptr) {
        if (A_ptr) env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        if (B_ptr) env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        return;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (mb_ensure_buffers(size_A, size_B, size_C, &d_A, &d_B, &d_C) != 0) {
        env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        return;
    }

    float *pA = nullptr, *pB = nullptr;
    int pin_ok = host_pin_ensure(size_A, size_B, 0, &pA, &pB, nullptr);

    if ((pin_ok == 0 || pin_ok == 1) && pA != nullptr && pB != nullptr) {
        std::memcpy(pA, A_ptr, size_A);
        std::memcpy(pB, B_ptr, size_B);
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, pA, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, pB, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, A_ptr, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, B_ptr, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);

    if (numA > (size_t) INT_MAX || numB > (size_t) INT_MAX) {
        fprintf(stderr, "[TensorOpsGPU] matmulBatchedGPUFp16: buffer too large for int kernel count\n");
        return;
    }

    __half *d_Ah = nullptr, *d_Bh = nullptr;
    if (mb_ensure_half_AB(numA, numB, &d_Ah, &d_Bh) != 0) return;

    launch_float_to_half(d_A, d_Ah, (int) numA);
    launch_float_to_half(d_B, d_Bh, (int) numB);
    CUDA_KERNEL_CHECK();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    long long strideA_elems = (long long) M * (long long) K;
    long long strideB_elems = (long long) K * (long long) N;
    long long strideC_elems = (long long) M * (long long) N;

    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        d_Bh, CUDA_R_16F, N, strideB_elems, d_Ah, CUDA_R_16F, K, strideA_elems, &beta,
        d_C, CUDA_R_32F, N, strideC_elems, batchCount,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GemmStridedBatchedEx (FP16) error: status %d\n", (int) st);
        return;
    }

    jfloat* C_ptr = env->GetFloatArrayElements(h_C, nullptr);
    if (!C_ptr) return;

    float *pC = nullptr;
    int pin_c = host_pin_ensure(0, 0, size_C, nullptr, nullptr, &pC);
    if ((pin_c == 0 || pin_c == 1) && pC != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpyAsync(pC, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
        std::memcpy(C_ptr, pC, size_C);
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(C_ptr, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_C, C_ptr, 0);
}

// ========== MATMUL + ADD + RELU ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulAddReluGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_A, jfloatArray h_B, jfloatArray h_bias, jfloatArray h_C,
    jint M, jint K, jint N) {
    (void) clazz;
    if (M <= 0 || K <= 0 || N <= 0) return;
    if (env->GetArrayLength(h_bias) != N) {
        fprintf(stderr, "[TensorOpsGPU] matmulAddReluGPU: bias length must be N=%d\n", (int) N);
        return;
    }

    if (check_size_overflow(M, K, sizeof(float)) ||
        check_size_overflow(K, N, sizeof(float)) ||
        check_size_overflow(M, N, sizeof(float))) {
        fprintf(stderr, "matmulAddReluGPU: size overflow\n");
        return;
    }

    size_t size_A = (size_t) M * (size_t) K * sizeof(float);
    size_t size_B = (size_t) K * (size_t) N * sizeof(float);
    size_t size_C = (size_t) M * (size_t) N * sizeof(float);
    size_t biasBytes = (size_t) N * sizeof(float);

    jfloat* A_ptr = env->GetFloatArrayElements(h_A, nullptr);
    jfloat* B_ptr = env->GetFloatArrayElements(h_B, nullptr);
    jfloat* bias_ptr = env->GetFloatArrayElements(h_bias, nullptr);
    if (!A_ptr || !B_ptr || !bias_ptr) {
        if (A_ptr) env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        if (B_ptr) env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        if (bias_ptr) env->ReleaseFloatArrayElements(h_bias, bias_ptr, JNI_ABORT);
        return;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (mm_ensure_buffers(size_A, size_B, size_C, &d_A, &d_B, &d_C) != 0) {
        env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_bias, bias_ptr, JNI_ABORT);
        return;
    }
    if (ensure_bias_device(biasBytes) != 0) {
        env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_bias, bias_ptr, JNI_ABORT);
        return;
    }

    CUDA_CHECK_VOID(cudaMemcpyAsync(tl_mm_dBias, bias_ptr, biasBytes, cudaMemcpyHostToDevice, kTensorCudaStream));

    float *pA = nullptr, *pB = nullptr, *pC = nullptr;
    int pin_ok = host_pin_ensure(size_A, size_B, size_C, &pA, &pB, &pC);
    if ((pin_ok == 0 || pin_ok == 1) && pA != nullptr && pB != nullptr && pC != nullptr) {
        std::memcpy(pA, A_ptr, size_A);
        std::memcpy(pB, B_ptr, size_B);
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, pA, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, pB, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, A_ptr, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, B_ptr, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_bias, bias_ptr, JNI_ABORT);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Sgemm (matmulAddRelu) error: status %d\n", (int) st);
        return;
    }
    launch_bias_relu_inplace(d_C, tl_mm_dBias, M, N);
    CUDA_KERNEL_CHECK();

    jfloat* C_ptr = env->GetFloatArrayElements(h_C, nullptr);
    if (!C_ptr) return;
    if ((pin_ok == 0 || pin_ok == 1) && pC != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpyAsync(pC, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
        std::memcpy(C_ptr, pC, size_C);
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(C_ptr, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_C, C_ptr, 0);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulAddReluGPUFp16(
    JNIEnv* env, jclass clazz,
    jfloatArray h_A, jfloatArray h_B, jfloatArray h_bias, jfloatArray h_C,
    jint M, jint K, jint N) {
    (void) clazz;
    if (M <= 0 || K <= 0 || N <= 0) return;
    if (env->GetArrayLength(h_bias) != N) {
        fprintf(stderr, "[TensorOpsGPU] matmulAddReluGPUFp16: bias length must be N=%d\n", (int) N);
        return;
    }

    if (check_size_overflow(M, K, sizeof(float)) ||
        check_size_overflow(K, N, sizeof(float)) ||
        check_size_overflow(M, N, sizeof(float))) {
        fprintf(stderr, "matmulAddReluGPUFp16: size overflow\n");
        return;
    }

    size_t size_A = (size_t) M * (size_t) K * sizeof(float);
    size_t size_B = (size_t) K * (size_t) N * sizeof(float);
    size_t size_C = (size_t) M * (size_t) N * sizeof(float);
    size_t biasBytes = (size_t) N * sizeof(float);

    jfloat* A_ptr = env->GetFloatArrayElements(h_A, nullptr);
    jfloat* B_ptr = env->GetFloatArrayElements(h_B, nullptr);
    jfloat* bias_ptr = env->GetFloatArrayElements(h_bias, nullptr);
    if (!A_ptr || !B_ptr || !bias_ptr) {
        if (A_ptr) env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        if (B_ptr) env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        if (bias_ptr) env->ReleaseFloatArrayElements(h_bias, bias_ptr, JNI_ABORT);
        return;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    if (mm_ensure_buffers(size_A, size_B, size_C, &d_A, &d_B, &d_C) != 0) {
        env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_bias, bias_ptr, JNI_ABORT);
        return;
    }
    if (ensure_bias_device(biasBytes) != 0) {
        env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
        env->ReleaseFloatArrayElements(h_bias, bias_ptr, JNI_ABORT);
        return;
    }

    CUDA_CHECK_VOID(cudaMemcpyAsync(tl_mm_dBias, bias_ptr, biasBytes, cudaMemcpyHostToDevice, kTensorCudaStream));

    float *pA = nullptr, *pB = nullptr;
    int pin_ok = host_pin_ensure(size_A, size_B, 0, &pA, &pB, nullptr);
    if ((pin_ok == 0 || pin_ok == 1) && pA != nullptr && pB != nullptr) {
        std::memcpy(pA, A_ptr, size_A);
        std::memcpy(pB, B_ptr, size_B);
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, pA, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, pB, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_A, A_ptr, size_A, cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaMemcpyAsync(d_B, B_ptr, size_B, cudaMemcpyHostToDevice, kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_A, A_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_B, B_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_bias, bias_ptr, JNI_ABORT);

    size_t nelemA = (size_t) M * (size_t) K;
    size_t nelemB = (size_t) K * (size_t) N;
    if (nelemA > (size_t) INT_MAX || nelemB > (size_t) INT_MAX) {
        fprintf(stderr, "[TensorOpsGPU] matmulAddReluGPUFp16: buffer too large for int kernel count\n");
        return;
    }
    __half *d_Ah = nullptr, *d_Bh = nullptr;
    if (mm_ensure_half_AB(nelemA, nelemB, &d_Ah, &d_Bh) != 0) return;

    launch_float_to_half(d_A, d_Ah, (int) nelemA);
    launch_float_to_half(d_B, d_Bh, (int) nelemB);
    CUDA_KERNEL_CHECK();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                     d_Bh, CUDA_R_16F, N, d_Ah, CUDA_R_16F, K, &beta,
                                     d_C, CUDA_R_32F, N,
                                     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GemmEx (matmulAddRelu FP16) error: status %d\n", (int) st);
        return;
    }
    launch_bias_relu_inplace(d_C, tl_mm_dBias, M, N);
    CUDA_KERNEL_CHECK();

    jfloat* C_ptr = env->GetFloatArrayElements(h_C, nullptr);
    if (!C_ptr) return;
    float *pC = nullptr;
    int pin_c = host_pin_ensure(0, 0, size_C, nullptr, nullptr, &pC);
    if ((pin_c == 0 || pin_c == 1) && pC != nullptr) {
        CUDA_CHECK_VOID(cudaMemcpyAsync(pC, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
        std::memcpy(C_ptr, pC, size_C);
    } else {
        CUDA_CHECK_VOID(cudaMemcpyAsync(C_ptr, d_C, size_C, cudaMemcpyDeviceToHost, kTensorCudaStream));
        CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    }
    env->ReleaseFloatArrayElements(h_C, C_ptr, 0);
}

// ========== DEVICE-POINTER MATMUL (zero-copy) ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_addBiasBroadcastGPUDevice(
    JNIEnv* env, jclass clazz, jlong dC, jlong dBias, jint M, jint N) {
    (void) env; (void) clazz;
    if (dC == 0 || dBias == 0 || M <= 0 || N <= 0) return;
    jgpt_cuda_ensure_stream();
    float* pC = reinterpret_cast<float*>(static_cast<uintptr_t>(dC));
    const float* pB = reinterpret_cast<const float*>(static_cast<uintptr_t>(dBias));
    launch_bias_add_inplace(pC, pB, M, N);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sumColumnsGPUDevice(
    JNIEnv* env, jclass clazz, jlong dSrc, jlong dDst, jint M, jint N, jfloat beta) {
    (void) env; (void) clazz;
    if (dSrc == 0 || dDst == 0 || M <= 0 || N <= 0) return;
    jgpt_cuda_ensure_stream();
    const float* pS = reinterpret_cast<const float*>(static_cast<uintptr_t>(dSrc));
    float* pD = reinterpret_cast<float*>(static_cast<uintptr_t>(dDst));
    launch_sum_columns(pS, pD, M, N, static_cast<float>(beta));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulGPUDevice(
    JNIEnv* env, jclass clazz,
    jlong dA, jlong dB, jlong dC, jint M, jint K, jint N) {
    (void) env; (void) clazz;
    if (M <= 0 || K <= 0 || N <= 0 || dA == 0 || dB == 0 || dC == 0) return;

    float* pdA = reinterpret_cast<float*>(static_cast<uintptr_t>(dA));
    float* pdB = reinterpret_cast<float*>(static_cast<uintptr_t>(dB));
    float* pdC = reinterpret_cast<float*>(static_cast<uintptr_t>(dC));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, pdB, N, pdA, K, &beta, pdC, N);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Sgemm (device) error: status %d\n", (int) st);
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulGPUDeviceEx(
    JNIEnv* env, jclass clazz,
    jlong dA, jlong dB, jlong dC, jint M, jint K, jint N, jboolean transposeA, jboolean transposeB, jfloat betaIn) {
    (void) env; (void) clazz;
    if (M <= 0 || K <= 0 || N <= 0 || dA == 0 || dB == 0 || dC == 0) return;

    float* pdA = reinterpret_cast<float*>(static_cast<uintptr_t>(dA));
    float* pdB = reinterpret_cast<float*>(static_cast<uintptr_t>(dB));
    float* pdC = reinterpret_cast<float*>(static_cast<uintptr_t>(dC));

    const float alpha = 1.0f;
    const float beta = static_cast<float>(betaIn);
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transposeA ? M : K;
    int ldb = transposeB ? K : N;

    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasSgemm(handle, opB, opA, N, M, K, &alpha, pdB, ldb, pdA, lda, &beta, pdC, N);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS SgemmEx (device flags) error: status %d\n", (int) st);
    }
}

JNIEXPORT jboolean JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_ensureStridedBatchedPackScratch0(
    JNIEnv* env,
    jclass clazz,
    jlong rows,
    jint dModel,
    jint dIntermediate) {
    (void) env;
    (void) clazz;
    if (rows <= 0 || dModel <= 0 || dIntermediate <= 0) {
        return JNI_FALSE;
    }
    jgpt_cuda_ensure_stream();
    /* cublasCreate / привязка к потоку недопустимы внутри cudaStreamBeginCapture; разогрев до захвата графа. */
    (void) get_cublas_handle();
    jgpt_cuda_extra_warmup_cublas();
    const long long knQkv = (long long) dModel * (long long) dModel;
    const long long mnQkv = (long long) rows * (long long) dModel;
    const long long knFfn = (long long) dModel * (long long) dIntermediate;
    const long long mnFfn = (long long) rows * (long long) dIntermediate;
    if (knQkv > INT_MAX / 4LL || mnQkv > INT_MAX / 4LL || knFfn > INT_MAX / 4LL || mnFfn > INT_MAX / 4LL) {
        fprintf(stderr, "ensureStridedBatchedPackScratch0: size overflow\n");
        return JNI_FALSE;
    }
    long long wNeed = 3LL * knQkv;
    if (2LL * knFfn > wNeed) {
        wNeed = 2LL * knFfn;
    }
    long long cNeed = 3LL * mnQkv;
    if (2LL * mnFfn > cNeed) {
        cNeed = 2LL * mnFfn;
    }
    if (!shared_x_strided_batched_packs_ensure(wNeed, cNeed)) {
        fprintf(stderr, "ensureStridedBatchedPackScratch0: allocation failed (w=%lld c=%lld)\n", wNeed, cNeed);
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_setStridedBatchedPackOverride0(
    JNIEnv* env, jclass clazz, jlong wPtr, jlong cPtr, jlong capWElems, jlong capCElems) {
    (void) env;
    (void) clazz;
    if (wPtr == 0 || cPtr == 0 || capWElems <= 0 || capCElems <= 0) {
        fprintf(stderr, "setStridedBatchedPackOverride0: invalid args\n");
        return;
    }
    tl_qkv_pack_override_w = reinterpret_cast<float*>(static_cast<uintptr_t>(wPtr));
    tl_qkv_pack_override_c = reinterpret_cast<float*>(static_cast<uintptr_t>(cPtr));
    tl_qkv_pack_override_cap_w = capWElems;
    tl_qkv_pack_override_cap_c = capCElems;
    tl_qkv_pack_override_active = true;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_clearStridedBatchedPackOverride0(
    JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    tl_qkv_pack_override_active = false;
    tl_qkv_pack_override_w = nullptr;
    tl_qkv_pack_override_c = nullptr;
    tl_qkv_pack_override_cap_w = 0;
    tl_qkv_pack_override_cap_c = 0;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_decoderGraphPrewarmDeviceOps0(
    JNIEnv* env,
    jclass clazz,
    jint batch,
    jint seqLen,
    jint dModel,
    jint numHeads,
    jint dIntermediate) {
    (void) env;
    (void) clazz;
    if (batch <= 0 || seqLen <= 0 || dModel <= 0 || numHeads <= 0 || dIntermediate <= 0) {
        return;
    }
    if (dModel % numHeads != 0) {
        return;
    }
    const int dHead = dModel / numHeads;
    const int rows = batch * seqLen;
    const int bAttn = batch * numHeads;
    jgpt_cuda_ensure_stream();
    jgpt_cuda_graph_prewarm_qkv_ffn_strided_and_wo(rows, dModel, dIntermediate);
    jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas(bAttn, seqLen, dHead, dHead);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulGpuDeviceQkvProjections0(
    JNIEnv* env, jclass clazz,
    jlong dXnorm,
    jlong dWq,
    jlong dWk,
    jlong dWv,
    jlong dQ,
    jlong dK,
    jlong dV,
    jint M,
    jint K,
    jint N) {
    (void) (env);
    (void) (clazz);
    if (M <= 0 || K <= 0 || N <= 0) {
        return;
    }
    if (dXnorm == 0 || dWq == 0 || dWk == 0 || dWv == 0 || dQ == 0 || dK == 0 || dV == 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    const float* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(dXnorm));
    const float* wq = reinterpret_cast<const float*>(static_cast<uintptr_t>(dWq));
    const float* wk = reinterpret_cast<const float*>(static_cast<uintptr_t>(dWk));
    const float* wv = reinterpret_cast<const float*>(static_cast<uintptr_t>(dWv));
    float* q = reinterpret_cast<float*>(static_cast<uintptr_t>(dQ));
    float* kout = reinterpret_cast<float*>(static_cast<uintptr_t>(dK));
    float* v = reinterpret_cast<float*>(static_cast<uintptr_t>(dV));

    const long long kn = (long long) K * (long long) N;
    const long long mn = (long long) M * (long long) N;
    if (kn > INT_MAX / 3 || mn > INT_MAX / 3) {
        fprintf(stderr, "matmulGpuDeviceQkvProjections0: size overflow\n");
        return;
    }
    if (!shared_x_strided_batched_packs_ensure(3LL * kn, 3LL * mn)) {
        fprintf(stderr, "matmulGpuDeviceQkvProjections0: pack scratch allocation failed\n");
        return;
    }

    float* wPack = jgpt_qkv_w_pack_ptr();
    float* cPack = jgpt_qkv_c_pack_ptr();
    CUDA_CHECK_VOID(cudaMemcpyAsync(
            wPack, wq, static_cast<size_t>(kn) * sizeof(float), cudaMemcpyDeviceToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaMemcpyAsync(
            wPack + kn,
            wk,
            static_cast<size_t>(kn) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            kTensorCudaStream));
    CUDA_CHECK_VOID(cudaMemcpyAsync(
            wPack + 2LL * kn,
            wv,
            static_cast<size_t>(kn) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            kTensorCudaStream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* xNC = const_cast<float*>(x);
    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            wPack,
            N,
            kn,
            xNC,
            K,
            0LL,
            &beta,
            cPack,
            N,
            mn,
            3);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS SgemmStridedBatched (QKV device) error: status %d\n", (int) st);
        return;
    }

    CUDA_CHECK_VOID(cudaMemcpyAsync(q, cPack, static_cast<size_t>(mn) * sizeof(float), cudaMemcpyDeviceToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaMemcpyAsync(
            kout,
            cPack + mn,
            static_cast<size_t>(mn) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            kTensorCudaStream));
    CUDA_CHECK_VOID(cudaMemcpyAsync(
            v,
            cPack + 2LL * mn,
            static_cast<size_t>(mn) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            kTensorCudaStream));
}

extern "C" int jgpt_cuda_ffn_w1w3_strided_batched_device(
    float* xnorm,
    const float* w1,
    const float* w3,
    float* h1,
    float* gate,
    int M,
    int K,
    int N) {
    if (M <= 0 || K <= 0 || N <= 0 || xnorm == nullptr || w1 == nullptr || w3 == nullptr || h1 == nullptr
            || gate == nullptr) {
        return 0;
    }
    jgpt_cuda_ensure_stream();

    const long long kn = (long long) K * (long long) N;
    const long long mn = (long long) M * (long long) N;
    if (kn > INT_MAX / 2 || mn > INT_MAX / 2) {
        fprintf(stderr, "jgpt_cuda_ffn_w1w3_strided_batched_device: size overflow\n");
        return 0;
    }
    if (!shared_x_strided_batched_packs_ensure(2LL * kn, 2LL * mn)) {
        fprintf(stderr, "jgpt_cuda_ffn_w1w3_strided_batched_device: pack scratch allocation failed\n");
        return 0;
    }

    auto memcpy_ok = [](cudaError_t e) {
        if (e == cudaSuccess) {
            return true;
        }
        fprintf(stderr, "jgpt_cuda_ffn_w1w3_strided_batched_device: %s\n", cudaGetErrorString(e));
        return false;
    };
    float* wPack = jgpt_qkv_w_pack_ptr();
    float* cPack = jgpt_qkv_c_pack_ptr();
    if (!memcpy_ok(cudaMemcpyAsync(
                wPack,
                w1,
                static_cast<size_t>(kn) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                kTensorCudaStream))) {
        return 0;
    }
    if (!memcpy_ok(cudaMemcpyAsync(
                wPack + kn,
                w3,
                static_cast<size_t>(kn) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                kTensorCudaStream))) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            wPack,
            N,
            kn,
            xnorm,
            K,
            0LL,
            &beta,
            cPack,
            N,
            mn,
            2);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS SgemmStridedBatched (FFN W1+W3 device) error: status %d\n", (int) st);
        return 0;
    }

    if (!memcpy_ok(cudaMemcpyAsync(
                h1,
                cPack,
                static_cast<size_t>(mn) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                kTensorCudaStream))) {
        return 0;
    }
    if (!memcpy_ok(cudaMemcpyAsync(
                gate,
                cPack + mn,
                static_cast<size_t>(mn) * sizeof(float),
                cudaMemcpyDeviceToDevice,
                kTensorCudaStream))) {
        return 0;
    }
    return 1;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_matmulGpuDeviceFfnW1W3Projections0(
    JNIEnv* env,
    jclass clazz,
    jlong dXnorm,
    jlong dW1,
    jlong dW3,
    jlong dH1,
    jlong dGate,
    jint M,
    jint K,
    jint N) {
    (void) env;
    (void) clazz;
    if (dXnorm == 0 || dW1 == 0 || dW3 == 0 || dH1 == 0 || dGate == 0) {
        return;
    }
    float* x = reinterpret_cast<float*>(static_cast<uintptr_t>(dXnorm));
    const float* w1 = reinterpret_cast<const float*>(static_cast<uintptr_t>(dW1));
    const float* w3 = reinterpret_cast<const float*>(static_cast<uintptr_t>(dW3));
    float* h1 = reinterpret_cast<float*>(static_cast<uintptr_t>(dH1));
    float* gate = reinterpret_cast<float*>(static_cast<uintptr_t>(dGate));
    (void) jgpt_cuda_ffn_w1w3_strided_batched_device(x, w1, w3, h1, gate, M, K, N);
}

// ========== GPU FLOAT BUFFER NATIVE ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_nativeConvertFloatDeviceToHalfDevice(
    JNIEnv* env, jclass clazz, jlong srcFloat, jlong dstHalf, jint n) {
    (void) env;
    (void) clazz;
    if (srcFloat == 0 || dstHalf == 0 || n <= 0) return;
    jgpt_cuda_ensure_stream();
    launch_float_to_half(
        reinterpret_cast<const float*>(static_cast<uintptr_t>(srcFloat)),
        reinterpret_cast<__half*>(static_cast<uintptr_t>(dstHalf)),
        n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_nativeConvertHalfDeviceToFloatDevice(
    JNIEnv* env, jclass clazz, jlong srcHalf, jlong dstFloat, jint n) {
    (void) env;
    (void) clazz;
    if (srcHalf == 0 || dstFloat == 0 || n <= 0) return;
    jgpt_cuda_ensure_stream();
    launch_half_to_float(
        reinterpret_cast<const __half*>(static_cast<uintptr_t>(srcHalf)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(dstFloat)),
        n);
}

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_GpuHalfBuffer_nativeAlloc(JNIEnv* env, jclass clazz, jlong numHalfs) {
    (void) env;
    (void) clazz;
    if (numHalfs <= 0 || check_size_overflow(numHalfs, sizeof(__half), 1)) return 0;
    size_t bytes = static_cast<size_t>(numHalfs) * sizeof(__half);
    __half* p = nullptr;
    cudaError_t e;
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    if (g_jgpt_cuda_stream != nullptr) {
        e = cudaMallocAsync(reinterpret_cast<void**>(&p), bytes, g_jgpt_cuda_stream);
    } else {
        e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
    }
#else
    e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
#endif
    if (e != cudaSuccess) {
        fprintf(stderr, "GpuHalfBuffer.nativeAlloc: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuHalfBuffer_nativeFree(JNIEnv* env, jclass clazz, jlong ptr) {
    (void) env;
    (void) clazz;
    if (ptr == 0) return;
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    if (g_jgpt_cuda_stream != nullptr) {
        cudaError_t e = cudaFreeAsync(p, g_jgpt_cuda_stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuHalfBuffer.nativeFree cudaFreeAsync: %s\n", cudaGetErrorString(e));
        }
        return;
    }
#endif
    cudaFree(p);
}

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeAlloc(JNIEnv* env, jclass clazz, jlong numFloats) {
    (void) env; (void) clazz;
    if (numFloats <= 0 || check_size_overflow(numFloats, sizeof(float), 1)) return 0;
    size_t bytes = static_cast<size_t>(numFloats) * sizeof(float);
    float* p = nullptr;
    cudaError_t e;
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    if (g_jgpt_cuda_stream != nullptr) {
        e = cudaMallocAsync(reinterpret_cast<void**>(&p), bytes, g_jgpt_cuda_stream);
    } else {
        e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
    }
#else
    e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
#endif
    if (e != cudaSuccess) {
        fprintf(stderr, "GpuFloatBuffer.nativeAlloc: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeFree(JNIEnv* env, jclass clazz, jlong ptr) {
    (void) env; (void) clazz;
    if (ptr == 0) return;
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    if (g_jgpt_cuda_stream != nullptr) {
        cudaError_t e = cudaFreeAsync(p, g_jgpt_cuda_stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuFloatBuffer.nativeFree cudaFreeAsync: %s\n", cudaGetErrorString(e));
        }
        return;
    }
#endif
    cudaFree(p);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyHtoD(
    JNIEnv* env, jclass clazz, jlong devicePtr, jfloatArray src, jint offset, jint length) {
    (void) clazz;
    if (devicePtr == 0 || length <= 0) return;
    jgpt_cuda_ensure_stream();
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t bytes = static_cast<size_t>(length) * sizeof(float);
    jfloat* p = env->GetFloatArrayElements(src, nullptr);
    if (!p) return;
    cudaError_t e = cudaSuccess;
    /* Blocking H2D: avoids ordering issues between legacy default stream and kTensorCudaStream. */
    if (gpufb_pin_ensure(bytes)) {
        std::memcpy(tl_gpufb_pin, p + offset, bytes);
        env->ReleaseFloatArrayElements(src, p, JNI_ABORT);
        e = cudaMemcpy(d, tl_gpufb_pin, bytes, cudaMemcpyHostToDevice);
    } else {
        e = cudaMemcpy(d, p + offset, bytes, cudaMemcpyHostToDevice);
        env->ReleaseFloatArrayElements(src, p, JNI_ABORT);
    }
    if (e == cudaSuccess && g_jgpt_cuda_stream != nullptr) {
        e = cudaStreamSynchronize(g_jgpt_cuda_stream);
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoD: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyHtoDOffset(
    JNIEnv* env, jclass clazz, jlong devicePtr, jint deviceFloatOffset, jfloatArray src, jint srcOff, jint len) {
    (void) clazz;
    if (devicePtr == 0 || len <= 0) return;
    jgpt_cuda_ensure_stream();
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr)) + deviceFloatOffset;
    size_t bytes = static_cast<size_t>(len) * sizeof(float);
    jfloat* p = env->GetFloatArrayElements(src, nullptr);
    if (!p) return;
    cudaError_t e = cudaSuccess;
    if (gpufb_pin_ensure(bytes)) {
        std::memcpy(tl_gpufb_pin, p + srcOff, bytes);
        env->ReleaseFloatArrayElements(src, p, JNI_ABORT);
        e = cudaMemcpy(d, tl_gpufb_pin, bytes, cudaMemcpyHostToDevice);
    } else {
        e = cudaMemcpy(d, p + srcOff, bytes, cudaMemcpyHostToDevice);
        env->ReleaseFloatArrayElements(src, p, JNI_ABORT);
    }
    if (e == cudaSuccess && g_jgpt_cuda_stream != nullptr) {
        e = cudaStreamSynchronize(g_jgpt_cuda_stream);
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDOffset: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoH(
    JNIEnv* env, jclass clazz, jlong devicePtr, jfloatArray dst, jint offset, jint length) {
    (void) clazz;
    if (devicePtr == 0 || length <= 0) return;
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t bytes = static_cast<size_t>(length) * sizeof(float);
    cudaError_t e = cudaSuccess;
    if (gpufb_pin_ensure(bytes)) {
        e = cudaMemcpyAsync(tl_gpufb_pin, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoH: %s\n", cudaGetErrorString(e));
            return;
        }
        jfloat* p = env->GetFloatArrayElements(dst, nullptr);
        if (!p) return;
        std::memcpy(p + offset, tl_gpufb_pin, bytes);
        env->ReleaseFloatArrayElements(dst, p, 0);
    } else {
        jfloat* p = env->GetFloatArrayElements(dst, nullptr);
        if (!p) return;
        e = cudaMemcpyAsync(p + offset, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        env->ReleaseFloatArrayElements(dst, p, 0);
        if (e != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoH: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoHOffset(
    JNIEnv* env, jclass clazz, jlong devicePtr, jint deviceFloatOffset, jfloatArray dst, jint dstOff, jint len) {
    (void) clazz;
    if (devicePtr == 0 || len <= 0) return;
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr)) + deviceFloatOffset;
    size_t bytes = static_cast<size_t>(len) * sizeof(float);
    cudaError_t e = cudaSuccess;
    if (gpufb_pin_ensure(bytes)) {
        e = cudaMemcpyAsync(tl_gpufb_pin, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHOffset: %s\n", cudaGetErrorString(e));
            return;
        }
        jfloat* p = env->GetFloatArrayElements(dst, nullptr);
        if (!p) return;
        std::memcpy(p + dstOff, tl_gpufb_pin, bytes);
        env->ReleaseFloatArrayElements(dst, p, 0);
    } else {
        jfloat* p = env->GetFloatArrayElements(dst, nullptr);
        if (!p) return;
        e = cudaMemcpyAsync(p + dstOff, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        env->ReleaseFloatArrayElements(dst, p, 0);
        if (e != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHOffset: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyHtoDDirect(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject directBuf, jlong byteOffset, jlong numBytes) {
    (void) clazz;
    if (devicePtr == 0 || directBuf == nullptr || numBytes <= 0) return;
    void* host = env->GetDirectBufferAddress(directBuf);
    if (!host) { fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDDirect: not a direct buffer\n"); return; }
    char* base = static_cast<char*>(host) + byteOffset;
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t nb = static_cast<size_t>(numBytes);
    cudaError_t err = cudaSuccess;
    if (gpufb_pin_ensure(nb)) {
        std::memcpy(tl_gpufb_pin, base, nb);
        err = cudaMemcpyAsync(d, tl_gpufb_pin, nb, cudaMemcpyHostToDevice, kTensorCudaStream);
    } else {
        err = cudaMemcpyAsync(d, base, nb, cudaMemcpyHostToDevice, kTensorCudaStream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDDirect: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoHDirect(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject directBuf, jlong byteOffset, jlong numBytes) {
    (void) clazz;
    if (devicePtr == 0 || directBuf == nullptr || numBytes <= 0) return;
    void* host = env->GetDirectBufferAddress(directBuf);
    if (!host) { fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHDirect: not a direct buffer\n"); return; }
    char* base = static_cast<char*>(host) + byteOffset;
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t nb = static_cast<size_t>(numBytes);
    cudaError_t err = cudaSuccess;
    if (gpufb_pin_ensure(nb)) {
        err = cudaMemcpyAsync(tl_gpufb_pin, d, nb, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
        if (err == cudaSuccess) std::memcpy(base, tl_gpufb_pin, nb);
    } else {
        err = cudaMemcpyAsync(base, d, nb, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHDirect: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyHtoDFloatBuffer(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject floatBuf, jlong floatOffset, jlong numFloats) {
    (void) clazz;
    if (devicePtr == 0 || floatBuf == nullptr || numFloats <= 0) return;
    void* host = env->GetDirectBufferAddress(floatBuf);
    if (!host) { fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDFloatBuffer: not a direct buffer\n"); return; }
    char* base = static_cast<char*>(host) + static_cast<size_t>(floatOffset) * sizeof(float);
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t nb = static_cast<size_t>(numFloats) * sizeof(float);
    cudaError_t err = cudaSuccess;
    if (gpufb_pin_ensure(nb)) {
        std::memcpy(tl_gpufb_pin, base, nb);
        err = cudaMemcpyAsync(d, tl_gpufb_pin, nb, cudaMemcpyHostToDevice, kTensorCudaStream);
    } else {
        err = cudaMemcpyAsync(d, base, nb, cudaMemcpyHostToDevice, kTensorCudaStream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDFloatBuffer: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoHFloatBuffer(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject floatBuf, jlong floatOffset, jlong numFloats) {
    (void) clazz;
    if (devicePtr == 0 || floatBuf == nullptr || numFloats <= 0) return;
    void* host = env->GetDirectBufferAddress(floatBuf);
    if (!host) { fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHFloatBuffer: not a direct buffer\n"); return; }
    char* base = static_cast<char*>(host) + static_cast<size_t>(floatOffset) * sizeof(float);
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t nb = static_cast<size_t>(numFloats) * sizeof(float);
    cudaError_t err = cudaSuccess;
    if (gpufb_pin_ensure(nb)) {
        err = cudaMemcpyAsync(tl_gpufb_pin, d, nb, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
        if (err == cudaSuccess) std::memcpy(base, tl_gpufb_pin, nb);
    } else {
        err = cudaMemcpyAsync(base, d, nb, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHFloatBuffer: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyHtoDAddress(
    JNIEnv* env, jclass clazz, jlong devicePtr, jlong hostAddress, jlong numBytes) {
    (void) env;
    (void) clazz;
    if (devicePtr == 0 || hostAddress == 0 || numBytes <= 0) return;
    void* base = reinterpret_cast<void*>(static_cast<uintptr_t>(hostAddress));
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t nb = static_cast<size_t>(numBytes);
    cudaError_t err = cudaSuccess;
    if (gpufb_pin_ensure(nb)) {
        std::memcpy(tl_gpufb_pin, base, nb);
        err = cudaMemcpyAsync(d, tl_gpufb_pin, nb, cudaMemcpyHostToDevice, kTensorCudaStream);
    } else {
        err = cudaMemcpyAsync(d, base, nb, cudaMemcpyHostToDevice, kTensorCudaStream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDAddress: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoHAddress(
    JNIEnv* env, jclass clazz, jlong devicePtr, jlong hostAddress, jlong numBytes) {
    (void) env;
    (void) clazz;
    if (devicePtr == 0 || hostAddress == 0 || numBytes <= 0) return;
    void* base = reinterpret_cast<void*>(static_cast<uintptr_t>(hostAddress));
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t nb = static_cast<size_t>(numBytes);
    cudaError_t err = cudaSuccess;
    if (gpufb_pin_ensure(nb)) {
        err = cudaMemcpyAsync(tl_gpufb_pin, d, nb, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
        if (err == cudaSuccess) std::memcpy(base, tl_gpufb_pin, nb);
    } else {
        err = cudaMemcpyAsync(base, d, nb, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHAddress: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeClear(
    JNIEnv* env, jclass clazz, jlong devicePtr, jlong numFloats) {
    (void) env; (void) clazz;
    if (devicePtr == 0 || numFloats <= 0) return;
    jgpt_cuda_ensure_stream();
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t nbytes = static_cast<size_t>(numFloats) * sizeof(float);
    /* Same stream as D2H async paths — avoids races between cudaMemset on default stream and memcpy on kTensorCudaStream. */
    cudaError_t err = cudaSuccess;
    if (g_jgpt_cuda_stream != nullptr) {
        err = cudaMemsetAsync(d, 0, nbytes, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(g_jgpt_cuda_stream);
    } else {
        err = cudaMemset(d, 0, nbytes);
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeClear: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoD(
    JNIEnv* env, jclass clazz, jlong srcDevicePtr, jlong dstDevicePtr, jint length) {
    (void) env; (void) clazz;
    if (srcDevicePtr == 0 || dstDevicePtr == 0 || length <= 0) return;
    const float* src = reinterpret_cast<const float*>(static_cast<uintptr_t>(srcDevicePtr));
    float* dst = reinterpret_cast<float*>(static_cast<uintptr_t>(dstDevicePtr));
    cudaError_t err = cudaMemcpyAsync(dst, src, static_cast<size_t>(length) * sizeof(float), cudaMemcpyDeviceToDevice, kTensorCudaStream);
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoD: %s\n", cudaGetErrorString(err));
}

// ========== GPU INT BUFFER (int32 targets / indices) ==========

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeAlloc(JNIEnv* env, jclass clazz, jlong numInts) {
    (void) env; (void) clazz;
    if (numInts <= 0 || check_size_overflow(static_cast<size_t>(numInts), sizeof(int), 1)) return 0;
    size_t bytes = static_cast<size_t>(numInts) * sizeof(int);
    int* p = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "GpuIntBuffer.nativeAlloc: %s\n", cudaGetErrorString(e));
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeFree(JNIEnv* env, jclass clazz, jlong ptr) {
    (void) env; (void) clazz;
    if (ptr == 0) return;
    cudaFree(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeCopyHtoD(
    JNIEnv* env, jclass clazz, jlong devicePtr, jintArray src, jint offset, jint length) {
    (void) clazz;
    if (devicePtr == 0 || length <= 0) return;
    jgpt_cuda_ensure_stream();
    int* d = reinterpret_cast<int*>(static_cast<uintptr_t>(devicePtr));
    size_t bytes = static_cast<size_t>(length) * sizeof(int);
    jint* p = env->GetIntArrayElements(src, nullptr);
    if (!p) return;
    cudaError_t e = cudaSuccess;
    /* Blocking H2D (same policy as GpuFloatBuffer.nativeCopyHtoD): avoid ordering issues vs kTensorCudaStream. */
    if (gpufb_pin_ensure(bytes)) {
        std::memcpy(tl_gpufb_pin, p + offset, bytes);
        env->ReleaseIntArrayElements(src, p, JNI_ABORT);
        e = cudaMemcpy(d, tl_gpufb_pin, bytes, cudaMemcpyHostToDevice);
    } else {
        e = cudaMemcpy(d, p + offset, bytes, cudaMemcpyHostToDevice);
        env->ReleaseIntArrayElements(src, p, JNI_ABORT);
    }
    if (e == cudaSuccess && g_jgpt_cuda_stream != nullptr) {
        e = cudaStreamSynchronize(g_jgpt_cuda_stream);
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "GpuIntBuffer.nativeCopyHtoD: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeCopyDtoH(
    JNIEnv* env, jclass clazz, jlong devicePtr, jintArray dst, jint offset, jint length) {
    (void) clazz;
    if (devicePtr == 0 || length <= 0) return;
    jgpt_cuda_ensure_stream();
    int* d = reinterpret_cast<int*>(static_cast<uintptr_t>(devicePtr));
    size_t bytes = static_cast<size_t>(length) * sizeof(int);
    cudaError_t e = cudaSuccess;
    if (gpufb_pin_ensure(bytes)) {
        e = cudaMemcpyAsync(tl_gpufb_pin, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuIntBuffer.nativeCopyDtoH: %s\n", cudaGetErrorString(e));
            return;
        }
        jint* p = env->GetIntArrayElements(dst, nullptr);
        if (!p) return;
        std::memcpy(p + offset, tl_gpufb_pin, bytes);
        env->ReleaseIntArrayElements(dst, p, 0);
    } else {
        jint* p = env->GetIntArrayElements(dst, nullptr);
        if (!p) return;
        e = cudaMemcpyAsync(p + offset, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        env->ReleaseIntArrayElements(dst, p, 0);
        if (e != cudaSuccess) fprintf(stderr, "GpuIntBuffer.nativeCopyDtoH: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeCopyHtoDDirect(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject directBuf, jlong byteOffset, jlong numBytes) {
    (void) clazz;
    if (devicePtr == 0 || directBuf == nullptr || numBytes <= 0) return;
    void* host = env->GetDirectBufferAddress(directBuf);
    if (!host) { fprintf(stderr, "GpuIntBuffer.nativeCopyHtoDDirect: not a direct buffer\n"); return; }
    char* base = static_cast<char*>(host) + byteOffset;
    int* d = reinterpret_cast<int*>(static_cast<uintptr_t>(devicePtr));
    size_t nb = static_cast<size_t>(numBytes);
    cudaError_t err = cudaSuccess;
    if (gpufb_pin_ensure(nb)) {
        std::memcpy(tl_gpufb_pin, base, nb);
        err = cudaMemcpyAsync(d, tl_gpufb_pin, nb, cudaMemcpyHostToDevice, kTensorCudaStream);
    } else {
        err = cudaMemcpyAsync(d, base, nb, cudaMemcpyHostToDevice, kTensorCudaStream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
    if (err != cudaSuccess) fprintf(stderr, "GpuIntBuffer.nativeCopyHtoDDirect: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeCopyDtoHDirect(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject directBuf, jlong byteOffset, jlong numBytes) {
    (void) clazz;
    if (devicePtr == 0 || directBuf == nullptr || numBytes <= 0) return;
    void* host = env->GetDirectBufferAddress(directBuf);
    if (!host) { fprintf(stderr, "GpuIntBuffer.nativeCopyDtoHDirect: not a direct buffer\n"); return; }
    char* base = static_cast<char*>(host) + byteOffset;
    int* d = reinterpret_cast<int*>(static_cast<uintptr_t>(devicePtr));
    size_t nb = static_cast<size_t>(numBytes);
    cudaError_t err = cudaSuccess;
    if (gpufb_pin_ensure(nb)) {
        err = cudaMemcpyAsync(tl_gpufb_pin, d, nb, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
        if (err == cudaSuccess) std::memcpy(base, tl_gpufb_pin, nb);
    } else {
        err = cudaMemcpyAsync(base, d, nb, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(kTensorCudaStream);
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuIntBuffer.nativeCopyDtoHDirect: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeClear(
    JNIEnv* env, jclass clazz, jlong devicePtr, jlong numInts) {
    (void) env; (void) clazz;
    if (devicePtr == 0 || numInts <= 0) return;
    jgpt_cuda_ensure_stream();
    int* d = reinterpret_cast<int*>(static_cast<uintptr_t>(devicePtr));
    size_t nbytes = static_cast<size_t>(numInts) * sizeof(int);
    cudaError_t err = cudaSuccess;
    if (g_jgpt_cuda_stream != nullptr) {
        err = cudaMemsetAsync(d, 0, nbytes, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(g_jgpt_cuda_stream);
    } else {
        err = cudaMemset(d, 0, nbytes);
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuIntBuffer.nativeClear: %s\n", cudaGetErrorString(err));
}

// ========== ADD ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_addGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_a, jfloatArray h_b, jfloatArray h_c, jint n) {
    (void) clazz;
    if (n <= 0) return;
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&d_a, bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_b, bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_c, bytes));

    jfloat* pa = env->GetFloatArrayElements(h_a, nullptr);
    jfloat* pb = env->GetFloatArrayElements(h_b, nullptr);
    if (!pa || !pb) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return;
    }
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_b, pb, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_a, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_b, pb, JNI_ABORT);

    launch_vec_add(d_a, d_b, d_c, n);
    CUDA_KERNEL_CHECK();

    jfloat* pc = env->GetFloatArrayElements(h_c, nullptr);
    if (!pc) { cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return; }
    CUDA_CHECK_VOID(cudaMemcpyAsync(pc, d_c, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_c, pc, 0);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_subtractGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_a, jfloatArray h_b, jfloatArray h_c, jint n) {
    (void) clazz;
    if (n <= 0) return;
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&d_a, bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_b, bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_c, bytes));

    jfloat* pa = env->GetFloatArrayElements(h_a, nullptr);
    jfloat* pb = env->GetFloatArrayElements(h_b, nullptr);
    if (!pa || !pb) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return;
    }
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_b, pb, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_a, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_b, pb, JNI_ABORT);

    launch_vec_sub(d_a, d_b, d_c, n);
    CUDA_KERNEL_CHECK();

    jfloat* pc = env->GetFloatArrayElements(h_c, nullptr);
    if (!pc) { cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return; }
    CUDA_CHECK_VOID(cudaMemcpyAsync(pc, d_c, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_c, pc, 0);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

// ========== ReLU ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_reluGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_a, jfloatArray h_b, jint n) {
    (void) clazz;
    if (n <= 0) return;
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK_VOID(cudaMalloc(&d_a, bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_b, bytes));

    jfloat* pa = env->GetFloatArrayElements(h_a, nullptr);
    if (!pa) { cudaFree(d_a); cudaFree(d_b); return; }
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_a, pa, JNI_ABORT);

    launch_relu(d_a, d_b, n);
    CUDA_KERNEL_CHECK();

    jfloat* pb = env->GetFloatArrayElements(h_b, nullptr);
    if (!pb) { cudaFree(d_a); cudaFree(d_b); return; }
    CUDA_CHECK_VOID(cudaMemcpyAsync(pb, d_b, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_b, pb, 0);

    cudaFree(d_a); cudaFree(d_b);
}

// ========== GPU Info ==========

JNIEXPORT jboolean JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_initGPU(JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount <= 0) {
        return JNI_FALSE;
    }
    jgpt_cuda_ensure_stream();
    return (g_jgpt_cuda_stream != nullptr) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_getGPUName(JNIEnv* env, jclass clazz) {
    (void) clazz;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return env->NewStringUTF(prop.name);
}

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_getGPUMemory(JNIEnv* env, jclass clazz) {
    (void) env; (void) clazz;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return (jlong) (prop.totalGlobalMem / (1024U * 1024U));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_synchronizeStream0(JNIEnv* env, jclass clazz) {
    (void) clazz;
    if (g_jgpt_cuda_stream == nullptr) {
        return;
    }
    cudaError_t err = cudaStreamSynchronize(g_jgpt_cuda_stream);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        char buf[384];
        snprintf(
                buf,
                sizeof(buf),
                "cudaStreamSynchronize failed: %s (code %d)",
                cudaGetErrorString(err),
                static_cast<int>(err));
        jclass jex = env->FindClass("java/lang/IllegalStateException");
        if (jex != nullptr) {
            env->ThrowNew(jex, buf);
            env->DeleteLocalRef(jex);
        } else {
            fprintf(stderr, "%s\n", buf);
        }
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_synchronizeDevice0(JNIEnv* env, jclass clazz) {
    (void) clazz;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        char buf[384];
        snprintf(
                buf,
                sizeof(buf),
                "cudaDeviceSynchronize failed: %s (code %d)",
                cudaGetErrorString(err),
                static_cast<int>(err));
        jclass jex = env->FindClass("java/lang/IllegalStateException");
        if (jex != nullptr) {
            env->ThrowNew(jex, buf);
            env->DeleteLocalRef(jex);
        } else {
            fprintf(stderr, "%s\n", buf);
        }
    }
}

JNIEXPORT jboolean JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_cudaStreamBeginCapture0(JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    jgpt_cuda_ensure_stream();
    cudaError_t e = cudaStreamBeginCapture(kTensorCudaStream, cudaStreamCaptureModeGlobal);
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaStreamBeginCapture failed: %s\n", cudaGetErrorString(e));
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_abortCudaStreamCaptureIfActive0(
    JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    if (g_jgpt_cuda_stream == nullptr) {
        return;
    }
    cudaStreamCaptureStatus capStatus = cudaStreamCaptureStatusNone;
    cudaError_t q = cudaStreamGetCaptureInfo(kTensorCudaStream, &capStatus, nullptr);
    if (q != cudaSuccess || capStatus != cudaStreamCaptureStatusActive) {
        return;
    }
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaStreamEndCapture(kTensorCudaStream, &graph);
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "abortCudaStreamCaptureIfActive: cudaStreamEndCapture failed: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_cudaStreamEndCaptureAndInstantiate0(
    JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    cudaGraph_t graph = nullptr;
    cudaError_t e1 = cudaStreamEndCapture(kTensorCudaStream, &graph);
    if (e1 != cudaSuccess) {
        fprintf(stderr, "cudaStreamEndCapture failed: %s\n", cudaGetErrorString(e1));
        return 0;
    }
    if (graph == nullptr) {
        return 0;
    }
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t errNode = nullptr;
    char log[512];
    log[0] = '\0';
    cudaError_t e2 = cudaGraphInstantiate(&exec, graph, &errNode, log, sizeof(log));
    cudaGraphDestroy(graph);
    if (e2 != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed: %s (log=%s)\n", cudaGetErrorString(e2), log);
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(exec));
}

JNIEXPORT jboolean JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_cudaGraphExecLaunch0(
    JNIEnv* env, jclass clazz, jlong execPtr) {
    (void) env;
    (void) clazz;
    if (execPtr == 0) {
        return JNI_TRUE;
    }
    jgpt_cuda_ensure_stream();
    cudaGraphExec_t exec = reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(execPtr));
    cudaError_t e = cudaGraphLaunch(exec, kTensorCudaStream);
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaGraphLaunch failed: %s\n", cudaGetErrorString(e));
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_cudaGraphExecDestroy0(
    JNIEnv* env, jclass clazz, jlong execPtr) {
    (void) env;
    (void) clazz;
    if (execPtr == 0) {
        return;
    }
    cudaGraphExec_t exec = reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(execPtr));
    cudaError_t e = cudaGraphExecDestroy(exec);
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaGraphExecDestroy failed: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_getGpuMemoryAllocated0(JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }
    return (jlong)(total_bytes - free_bytes);
}

/** Второе число — объём VRAM устройства (total из cudaMemGetInfo), не кэш PyTorch. */
JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_getGpuMemoryReserved0(JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }
    return (jlong) total_bytes;
}

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_cuda_CudaPinnedHost_allocBytes(JNIEnv* env, jclass clazz, jlong numBytes) {
    (void) env;
    (void) clazz;
    if (numBytes <= 0) {
        return 0;
    }
    void* p = nullptr;
    cudaError_t e = cudaHostAlloc(&p, static_cast<size_t>(numBytes), cudaHostAllocDefault);
    if (e != cudaSuccess || p == nullptr) {
        return 0;
    }
    return reinterpret_cast<jlong>(p);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_cuda_CudaPinnedHost_free(JNIEnv* env, jclass clazz, jlong ptr) {
    (void) env;
    (void) clazz;
    if (ptr != 0) {
        (void) cudaFreeHost(reinterpret_cast<void*>(ptr));
    }
}

JNIEXPORT jobject JNICALL Java_com_veles_llm_jgpt_cuda_CudaPinnedHost_directBuffer(JNIEnv* env, jclass clazz, jlong ptr, jlong numBytes) {
    (void) clazz;
    if (ptr == 0 || numBytes <= 0) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(reinterpret_cast<void*>(ptr), numBytes);
}

#ifdef __cplusplus
}
#endif