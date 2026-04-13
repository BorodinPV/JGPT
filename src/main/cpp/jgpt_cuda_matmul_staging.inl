/* Matmul TLS / cuBLAS / QKV pack / pinned & device scratch
 * Thread-local buffers, jgpt_cuda_cleanup_thread_resources, strided-batched helpers.
 * Included only from jgpt_cuda.cu (single translation unit).
 */

#include "jgpt_cuda_cublas_common.cuh"
#include "jgpt_cuda_tls_blob.cuh"

static int alloc_device_float_triple(
        size_t szA, size_t szB, size_t szC, float** outA, float** outB, float** outC, const char* ctx) {
    float *a = nullptr, *b = nullptr, *c = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&a), szA);
    cudaError_t e2 = (e1 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&b), szB) : cudaErrorMemoryAllocation;
    cudaError_t e3 = (e2 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&c), szC) : cudaErrorMemoryAllocation;
    if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        fprintf(stderr, "%s: %s / %s / %s\n", ctx, cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3));
        return -1;
    }
    *outA = a;
    *outB = b;
    *outC = c;
    return 0;
}

static int alloc_device_half_pair(size_t nelemA, size_t nelemB, __half** outA, __half** outB, const char* ctx) {
    if (check_size_overflow(nelemA, sizeof(__half), 1) || check_size_overflow(nelemB, sizeof(__half), 1)) {
        fprintf(stderr, "%s: size overflow\n", ctx);
        return -1;
    }
    size_t szA = nelemA * sizeof(__half);
    size_t szB = nelemB * sizeof(__half);
    __half *a = nullptr, *b = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&a), szA);
    cudaError_t e2 = (e1 == cudaSuccess) ? cudaMalloc(reinterpret_cast<void**>(&b), szB) : cudaErrorMemoryAllocation;
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        cudaFree(a);
        cudaFree(b);
        fprintf(stderr, "%s: %s / %s\n", ctx, cudaGetErrorString(e1), cudaGetErrorString(e2));
        return -1;
    }
    *outA = a;
    *outB = b;
    return 0;
}

/* ========== Thread-safe cuBLAS handle per thread ========== */
static thread_local cublasHandle_t tl_cublas_handle = nullptr;

static cublasHandle_t get_cublas_handle() {
    if (tl_cublas_handle != nullptr) {
        return tl_cublas_handle;
    }
    tl_cublas_handle = jgpt_cuda_detail::create_cublas_for_jgpt_stream("TensorOpsGPU");
    return tl_cublas_handle;
}

/* Thread-local staging for strided-batched GEMMs sharing operand B (X): QKV batch 3, FFN W1+W3 batch 2. */
static thread_local jgpt_cuda_tls::TlsDeviceBlob tl_qkv_w_pack_blob;
static thread_local jgpt_cuda_tls::TlsDeviceBlob tl_qkv_c_pack_blob;

/* Внешние pack-буферы (Java GpuFloatBuffer в GPTModel): cudaGraphExec держит указатели — без cudaFree tl_qkv. */
static thread_local bool tl_qkv_pack_override_active = false;
static thread_local float* tl_qkv_pack_override_w = nullptr;
static thread_local float* tl_qkv_pack_override_c = nullptr;
static thread_local long long tl_qkv_pack_override_cap_w = 0;
static thread_local long long tl_qkv_pack_override_cap_c = 0;

static inline float* jgpt_qkv_w_pack_ptr(void) {
    return tl_qkv_pack_override_active ? tl_qkv_pack_override_w
                                       : static_cast<float*>(tl_qkv_w_pack_blob.ptr);
}

static inline float* jgpt_qkv_c_pack_ptr(void) {
    return tl_qkv_pack_override_active ? tl_qkv_pack_override_c
                                       : static_cast<float*>(tl_qkv_c_pack_blob.ptr);
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
    const size_t wu = static_cast<size_t>(wElems);
    const size_t cu = static_cast<size_t>(cElems);
    if (jgpt_alloc_n_elem_overflows(wu, sizeof(float)) || jgpt_alloc_n_elem_overflows(cu, sizeof(float))) {
        return false;
    }
    if (!tl_qkv_w_pack_blob.grow_to_fit(wu * sizeof(float))) {
        return false;
    }
    if (!tl_qkv_c_pack_blob.grow_to_fit(cu * sizeof(float))) {
        return false;
    }
    return tl_qkv_w_pack_blob.ptr != nullptr && tl_qkv_c_pack_blob.ptr != nullptr;
}

extern "C" void jgpt_cuda_decoder_graph_pack_snapshot(
        uintptr_t* w_ptr,
        uintptr_t* c_ptr,
        unsigned long long* cap_w_elems,
        unsigned long long* cap_c_elems,
        int* override_active) {
    float* w = jgpt_qkv_w_pack_ptr();
    float* c = jgpt_qkv_c_pack_ptr();
    if (w_ptr != nullptr) {
        *w_ptr = reinterpret_cast<uintptr_t>(w);
    }
    if (c_ptr != nullptr) {
        *c_ptr = reinterpret_cast<uintptr_t>(c);
    }
    if (tl_qkv_pack_override_active) {
        if (cap_w_elems != nullptr) {
            *cap_w_elems = static_cast<unsigned long long>(tl_qkv_pack_override_cap_w);
        }
        if (cap_c_elems != nullptr) {
            *cap_c_elems = static_cast<unsigned long long>(tl_qkv_pack_override_cap_c);
        }
        if (override_active != nullptr) {
            *override_active = 1;
        }
    } else {
        if (cap_w_elems != nullptr) {
            *cap_w_elems = static_cast<unsigned long long>(tl_qkv_w_pack_blob.bytes / sizeof(float));
        }
        if (cap_c_elems != nullptr) {
            *cap_c_elems = static_cast<unsigned long long>(tl_qkv_c_pack_blob.bytes / sizeof(float));
        }
        if (override_active != nullptr) {
            *override_active = 0;
        }
    }
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
    if (handle == nullptr) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_qkv_ffn: cuBLAS handle unavailable\n");
        return;
    }

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

extern "C" void jgpt_cuda_extra_cleanup(void);

void jgpt_cuda_cleanup_thread_resources(void) {
    cudaStream_t stream = jgpt_cuda_stream_handle();
    if (stream != nullptr) {
        cudaError_t s = cudaStreamSynchronize(stream);
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
    jgpt_cuda_ensure_stream();
    if (tl_mm_dAh && tl_mm_nelemAh == nelemA && tl_mm_dBh && tl_mm_nelemBh == nelemB) {
        *outA = tl_mm_dAh;
        *outB = tl_mm_dBh;
        return 0;
    }
    mm_free_half_cached();
    __half *a = nullptr, *b = nullptr;
    if (alloc_device_half_pair(nelemA, nelemB, &a, &b, "CUDA mm_ensure_half_AB malloc") != 0) {
        return -1;
    }
    tl_mm_dAh = a;
    tl_mm_dBh = b;
    tl_mm_nelemAh = nelemA;
    tl_mm_nelemBh = nelemB;
    *outA = a;
    *outB = b;
    return 0;
}

static int mb_ensure_half_AB(size_t nelemA, size_t nelemB, __half** outA, __half** outB) {
    jgpt_cuda_ensure_stream();
    if (tl_mb_dAh && tl_mb_nelemAh == nelemA && tl_mb_dBh && tl_mb_nelemBh == nelemB) {
        *outA = tl_mb_dAh;
        *outB = tl_mb_dBh;
        return 0;
    }
    mb_free_half_cached();
    __half *a = nullptr, *b = nullptr;
    if (alloc_device_half_pair(nelemA, nelemB, &a, &b, "CUDA mb_ensure_half_AB malloc") != 0) {
        return -1;
    }
    tl_mb_dAh = a;
    tl_mb_dBh = b;
    tl_mb_nelemAh = nelemA;
    tl_mb_nelemBh = nelemB;
    *outA = a;
    *outB = b;
    return 0;
}

static int mm_ensure_buffers(size_t szA, size_t szB, size_t szC,
                             float** outA, float** outB, float** outC) {
    jgpt_cuda_ensure_stream();
    if (tl_mm_dA && tl_mm_szA == szA && tl_mm_szB == szB && tl_mm_szC == szC) {
        *outA = tl_mm_dA;
        *outB = tl_mm_dB;
        *outC = tl_mm_dC;
        return 0;
    }
    mm_free_cached();
    float *a = nullptr, *b = nullptr, *c = nullptr;
    if (alloc_device_float_triple(szA, szB, szC, &a, &b, &c, "CUDA mm_ensure_buffers malloc") != 0) {
        return -1;
    }
    tl_mm_dA = a;
    tl_mm_dB = b;
    tl_mm_dC = c;
    tl_mm_szA = szA;
    tl_mm_szB = szB;
    tl_mm_szC = szC;
    *outA = a;
    *outB = b;
    *outC = c;
    return 0;
}

static int mb_ensure_buffers(size_t szA, size_t szB, size_t szC,
                             float** outA, float** outB, float** outC) {
    jgpt_cuda_ensure_stream();
    if (tl_mb_dA && tl_mb_szA == szA && tl_mb_szB == szB && tl_mb_szC == szC) {
        *outA = tl_mb_dA;
        *outB = tl_mb_dB;
        *outC = tl_mb_dC;
        return 0;
    }
    mb_free_cached();
    float *a = nullptr, *b = nullptr, *c = nullptr;
    if (alloc_device_float_triple(szA, szB, szC, &a, &b, &c, "CUDA mb_ensure_buffers malloc") != 0) {
        return -1;
    }
    tl_mb_dA = a;
    tl_mb_dB = b;
    tl_mb_dC = c;
    tl_mb_szA = szA;
    tl_mb_szB = szB;
    tl_mb_szC = szC;
    *outA = a;
    *outB = b;
    *outC = c;
    return 0;
}
