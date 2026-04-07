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
#include <cstdio>
#include <chrono>
#include <cstring>
#include <mutex>
#include <atomic>
#include <unordered_set>

#include "jgpt_cuda_stream.cuh"
#include "jgpt_cuda_ffn_link.h"
#include "jgpt_cuda_graph_prewarm.h"
#include "jgpt_cuda_size_check.cuh"

static std::mutex g_stream_init_mutex;
static std::mutex g_cuda_device_caps_mutex;
/** Указатели, выделенные {@code cudaMalloc} (fallback), чтобы {@code nativeFree} вызывал {@code cudaFree}, а не {@code cudaFreeAsync}. */
static std::mutex g_jgpt_sync_device_alloc_mu;
static std::unordered_set<uintptr_t> g_jgpt_sync_device_alloc_ptrs;

static void jgpt_cuda_sync_device_alloc_register(uintptr_t p) {
    if (p == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_jgpt_sync_device_alloc_mu);
    g_jgpt_sync_device_alloc_ptrs.insert(p);
}

/** @return true если указатель был в множестве (снимается). */
static bool jgpt_cuda_sync_device_alloc_consume(uintptr_t p) {
    if (p == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_jgpt_sync_device_alloc_mu);
    auto it = g_jgpt_sync_device_alloc_ptrs.find(p);
    if (it == g_jgpt_sync_device_alloc_ptrs.end()) {
        return false;
    }
    g_jgpt_sync_device_alloc_ptrs.erase(it);
    return true;
}

static int g_cached_max_threads_per_block = 0;

static std::atomic<cudaStream_t> g_jgpt_cuda_stream_atomic{nullptr};

extern "C" cudaStream_t jgpt_cuda_stream_handle(void) {
    return g_jgpt_cuda_stream_atomic.load(std::memory_order_acquire);
}

/** Последний код {@code cudaGraphLaunch} на этом потоке (0 при успехе последнего вызова с ненулевым exec). */
static thread_local int g_jgpt_last_cuda_graph_launch_err = 0;


/** Env {@code JGPT_DECODER_CUDA_GRAPH_NO_PRELAUNCH_SYNC=1}: не вызывать {@code cudaStreamSynchronize} перед {@code cudaGraphLaunch} (быстрее; риск cudaErrorInvalidValue). */
static bool jgpt_cuda_env_decoder_graph_no_prelaunch_sync(void) {
    const char* v = std::getenv("JGPT_DECODER_CUDA_GRAPH_NO_PRELAUNCH_SYNC");
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return strcmp(v, "1") == 0;
}

/** Env {@code JGPT_DECODER_GRAPH_MEM_PROBE=1}: stderr + NDJSON {@code cudaMemGetInfo} в точках capture/launch графа декодера. */
static bool jgpt_env_decoder_graph_mem_probe(void) {
    const char* v = std::getenv("JGPT_DECODER_GRAPH_MEM_PROBE");
    return v != nullptr && strcmp(v, "1") == 0;
}

/** NDJSON в файл только если задан {@code JGPT_DEBUG_NDJSON_LOG} (путь для append). Иначе nullptr — без I/O. */
static FILE* jgpt_debug_ndjson_fopen(void) {
    const char* p = std::getenv("JGPT_DEBUG_NDJSON_LOG");
    if (p == nullptr || p[0] == '\0') {
        return nullptr;
    }
    return std::fopen(p, "a");
}

static void jgpt_decoder_graph_mem_probe_log(const char* tag) {
    if (!jgpt_env_decoder_graph_mem_probe()) {
        return;
    }
    size_t mf = 0;
    size_t mt = 0;
    (void) cudaMemGetInfo(&mf, &mt);
    fprintf(stderr, "[JGPT_DECODER_GRAPH_MEM_PROBE] %s cudaMemGetInfo free=%zu total=%zu\n", tag, mf, mt);
    // #region agent log
    FILE* df = jgpt_debug_ndjson_fopen();
    if (df != nullptr) {
        fprintf(
                df,
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H-memProbe\",\"location\":\"jgpt_cuda.cu:memProbe\","
                "\"message\":\"%s\",\"data\":{\"free\":%zu,\"total\":%zu}}\n",
                tag,
                mf,
                mt);
        fclose(df);
    }
    // #endregion
}

/*
 * При забитом async memory pool cudaMemGetInfo показывает заметный «free», но cudaMallocAsync
 * даёт cudaErrorMemoryAllocation. Trim возвращает неиспользуемое из пула драйверу (CUDART >= 11.5).
 */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11050)
static void jgpt_cuda_trim_mem_pools_best_effort(const char* ctx_tag) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return;
    }
    size_t free_before = 0;
    size_t total_b = 0;
    (void) cudaMemGetInfo(&free_before, &total_b);
    int trimmed_mask = 0;
    cudaMemPool_t def_pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&def_pool, dev) == cudaSuccess && def_pool != nullptr) {
        if (cudaMemPoolTrimTo(def_pool, 0) == cudaSuccess) {
            trimmed_mask |= 1;
        }
    }
    /*
     * cudaStreamGetMemPool есть в полном CUDA Toolkit; в некоторых distro-заголовках отсутствует — используем
     * cudaDeviceGetMemPool (пул устройства / последний set), плюс default выше.
     */
    cudaMemPool_t dev_pool = nullptr;
    if (cudaDeviceGetMemPool(&dev_pool, dev) == cudaSuccess && dev_pool != nullptr && dev_pool != def_pool) {
        if (cudaMemPoolTrimTo(dev_pool, 0) == cudaSuccess) {
            trimmed_mask |= 2;
        }
    }
    size_t free_after = 0;
    (void) cudaMemGetInfo(&free_after, &total_b);
    // #region agent log
    FILE* df = jgpt_debug_ndjson_fopen();
    if (df != nullptr) {
        fprintf(
                df,
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H6-memPoolTrim\",\"location\":\"jgpt_cuda.cu:trim\","
                "\"message\":\"trim\",\"data\":{\"ctx\":\"%s\",\"trimmedMask\":%d,\"freeBefore\":%zu,\"freeAfter\":%zu}}\n",
                ctx_tag != nullptr ? ctx_tag : "",
                trimmed_mask,
                free_before,
                free_after);
        fclose(df);
    }
    // #endregion
}
#else
static void jgpt_cuda_trim_mem_pools_best_effort(const char* ctx_tag) {
    (void) ctx_tag;
}
#endif

/**
 * Полная синхронизация устройства, trim graph memory (CUDA 12+), затем trim async memory pools — тот же путь, что
 * {@code TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort}. Нужен на пути cudaMalloc fallback: без graph trim
 * {@code cudaMemGetInfo} может показывать заметный free, а {@code cudaMalloc} всё равно получает OOM.
 */
static void jgpt_cuda_trim_device_memory_full_best_effort(const char* ctx_tag) {
    (void) cudaDeviceSynchronize();
    /* Сброс «липкой» ошибки после sync — иначе последующие вызовы могут сразу видеть прошлый cudaError. */
    (void) cudaGetLastError();
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess) {
            (void) cudaDeviceGraphMemTrim(dev);
        }
    }
#endif
    jgpt_cuda_trim_mem_pools_best_effort(ctx_tag);
}

/** NDJSON (debug session b39372): фиксирует контекст при cudaGraphLaunch → ошибка. */
static void jgpt_b39372_append_native_graph_launch_fail(
        cudaStreamCaptureStatus cap_before,
        cudaStreamCaptureStatus cap_after,
        cudaError_t launch_err,
        cudaError_t sticky_err,
        int syn_pre_code,
        cudaGraphExec_t exec,
        uintptr_t aux_graph,
        size_t aux_graph_sz) {
    FILE* f = jgpt_debug_ndjson_fopen();
    if (f == nullptr) {
        return;
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    int dev = -1;
    (void) cudaGetDevice(&dev);
    int driver = 0;
    int runtime = 0;
    (void) cudaDriverGetVersion(&driver);
    (void) cudaRuntimeGetVersion(&runtime);
    cudaError_t q = cudaStreamQuery(kTensorCudaStream);
    size_t mem_free = 0;
    size_t mem_total = 0;
    (void) cudaMemGetInfo(&mem_free, &mem_total);
    const unsigned long long exec_ull =
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(exec));
    const unsigned long long stream_ull =
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(kTensorCudaStream));
    long long exec_flags = -1;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11040)
    if (exec != nullptr) {
        unsigned long long gf = 0ULL;
        cudaError_t gfe = cudaGraphExecGetFlags(exec, &gf);
        exec_flags = (gfe == cudaSuccess) ? static_cast<long long>(gf)
                                          : (-1000LL - static_cast<long long>(gfe));
    }
#endif
    fprintf(
            f,
            "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H1-H4\",\"location\":\"jgpt_cuda.cu:cudaGraphExecLaunch0\","
            "\"message\":\"cudaGraphLaunch failed\",\"data\":{\"dev\":%d,\"driverVer\":%d,\"runtimeVer\":%d,"
            "\"execULL\":\"0x%llx\",\"streamULL\":\"0x%llx\",\"capBefore\":%d,\"capAfter\":%d,"
            "\"launchErr\":%d,\"stickyErr\":%d,\"synPre\":%d,\"streamQueryErr\":%d,\"execFlags\":%lld,"
            "\"auxGraph\":\"0x%llx\",\"auxGraphSz\":%zu,\"memFree\":%zu,\"memTotal\":%zu}}\n",
            dev,
            driver,
            runtime,
            exec_ull,
            stream_ull,
            static_cast<int>(cap_before),
            static_cast<int>(cap_after),
            static_cast<int>(launch_err),
            static_cast<int>(sticky_err),
            syn_pre_code,
            static_cast<int>(q),
            exec_flags,
            static_cast<unsigned long long>(aux_graph),
            aux_graph_sz,
            mem_free,
            mem_total);
    fclose(f);
}

void jgpt_cuda_ensure_stream(void) {
    cudaStream_t s = g_jgpt_cuda_stream_atomic.load(std::memory_order_acquire);
    if (s != nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_stream_init_mutex);
    s = g_jgpt_cuda_stream_atomic.load(std::memory_order_relaxed);
    if (s != nullptr) {
        return;
    }
    cudaStream_t created = nullptr;
    if (cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking) != cudaSuccess) {
        fprintf(stderr, "[TensorOpsGPU] cudaStreamCreateWithFlags failed\n");
        return;
    }
    g_jgpt_cuda_stream_atomic.store(created, std::memory_order_release);
}

void jgpt_cuda_destroy_stream(void) {
    std::lock_guard<std::mutex> lock(g_stream_init_mutex);
    cudaStream_t s = g_jgpt_cuda_stream_atomic.load(std::memory_order_relaxed);
    if (s != nullptr) {
        cudaError_t syn = cudaStreamSynchronize(s);
        if (syn != cudaSuccess) {
            fprintf(stderr, "[TensorOpsGPU] cudaStreamSynchronize before destroy: %s\n", cudaGetErrorString(syn));
        }
        cudaStreamDestroy(s);
        g_jgpt_cuda_stream_atomic.store(nullptr, std::memory_order_release);
    }
}

int jgpt_cuda_sync_stream_unless_capturing(const char* ctx) {
    cudaStream_t s = jgpt_cuda_stream_handle();
    if (s == nullptr) {
        return 1;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaError_t gi = cudaStreamGetCaptureInfo(s, &cap, nullptr);
    if (gi != cudaSuccess) {
        fprintf(stderr, "%s: cudaStreamGetCaptureInfo: %s\n", ctx, cudaGetErrorString(gi));
        return 0;
    }
    if (cap == cudaStreamCaptureStatusActive) {
        return 1;
    }
    cudaError_t e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: cudaStreamSynchronize: %s\n", ctx, cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

void jgpt_cuda_abort_stream_capture_discard_graph(void) {
    cudaStream_t s = jgpt_cuda_stream_handle();
    if (s == nullptr) {
        return;
    }
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaError_t q = cudaStreamGetCaptureInfo(s, &st, nullptr);
    if (q != cudaSuccess || st != cudaStreamCaptureStatusActive) {
        return;
    }
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaStreamEndCapture(s, &graph);
    if (graph != nullptr) {
        cudaGraphDestroy(graph);
    }
    if (e != cudaSuccess) {
        fprintf(
                stderr,
                "jgpt_cuda_abort_stream_capture_discard_graph: cudaStreamEndCapture: %s\n",
                cudaGetErrorString(e));
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
    if (tl_cublas_handle != nullptr) {
        return tl_cublas_handle;
    }
    jgpt_cuda_ensure_stream();
    cublasHandle_t h = nullptr;
    cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "TensorOpsGPU: cublasCreate failed: %d\n", static_cast<int>(st));
        return nullptr;
    }
    st = cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "TensorOpsGPU: cublasSetMathMode failed: %d\n", static_cast<int>(st));
        cublasDestroy(h);
        return nullptr;
    }
    cudaStream_t stream = jgpt_cuda_stream_handle();
    if (stream == nullptr) {
        fprintf(stderr, "TensorOpsGPU: CUDA stream unavailable after jgpt_cuda_ensure_stream (cublas)\n");
        cublasDestroy(h);
        return nullptr;
    }
    st = cublasSetStream(h, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "TensorOpsGPU: cublasSetStream failed: %d\n", static_cast<int>(st));
        cublasDestroy(h);
        return nullptr;
    }
    tl_cublas_handle = h;
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
            *cap_w_elems = static_cast<unsigned long long>(tl_qkv_cap_w_elems);
        }
        if (cap_c_elems != nullptr) {
            *cap_c_elems = static_cast<unsigned long long>(tl_qkv_cap_c_elems);
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
    jgpt_cuda_ensure_stream();
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
    jgpt_cuda_ensure_stream();
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
    jgpt_cuda_ensure_stream();
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

/** Как {@code CUDA_CHECK_VOID}, но при ошибке делает {@code ReleaseFloatArrayElements(..., JNI_ABORT)} для выходного jfloatArray. */
#define CUDA_CHECK_VOID_JFLOAT_OUT(env, h_C, C_ptr, call) \
    do { \
        cudaError_t err_jfloat_out_ = (call); \
        if (err_jfloat_out_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_jfloat_out_)); \
            if ((C_ptr) != nullptr) (env)->ReleaseFloatArrayElements((h_C), (C_ptr), JNI_ABORT); \
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
    if (handle == nullptr) {
        fprintf(stderr, "TensorOpsGPU: cuBLAS handle unavailable\n");
        return;
    }
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
    if (handle == nullptr) {
        fprintf(stderr, "TensorOpsGPU: cuBLAS handle unavailable\n");
        return;
    }
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
    if (handle == nullptr) {
        fprintf(stderr, "TensorOpsGPU: cuBLAS handle unavailable\n");
        return;
    }
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
    if (handle == nullptr) {
        fprintf(stderr, "TensorOpsGPU: cuBLAS handle unavailable\n");
        return;
    }
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
    if (handle == nullptr) {
        fprintf(stderr, "TensorOpsGPU: cuBLAS handle unavailable\n");
        return;
    }
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
    if (handle == nullptr) {
        fprintf(stderr, "TensorOpsGPU: cuBLAS handle unavailable\n");
        return;
    }
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

    jgpt_cuda_ensure_stream();
    float* pdA = reinterpret_cast<float*>(static_cast<uintptr_t>(dA));
    float* pdB = reinterpret_cast<float*>(static_cast<uintptr_t>(dB));
    float* pdC = reinterpret_cast<float*>(static_cast<uintptr_t>(dC));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = get_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "TensorOpsGPU: cuBLAS handle unavailable (matmulGPUDevice)\n");
        return;
    }
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

    jgpt_cuda_ensure_stream();
    float* pdA = reinterpret_cast<float*>(static_cast<uintptr_t>(dA));
    float* pdB = reinterpret_cast<float*>(static_cast<uintptr_t>(dB));
    float* pdC = reinterpret_cast<float*>(static_cast<uintptr_t>(dC));

    const float alpha = 1.0f;
    const float beta = static_cast<float>(betaIn);
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
    /*
     * В row-major Java A[M×K], B[K×N] вызов ниже — это Sgemm(opB, opA) с переставленными аргументами (pdB, pdA).
     * lda/ldb — ведущие размерности **хранения** операндов pdA/pdB в соглашении пары (opB,opA,B,A), как у рабочего
     * matmulGPUDevice при NT,NT (там lda=K, ldb=N). Для комбинаций с T согласованы с Linear/GPT backward;
     * менять формулы без численных тестов на всех (transposeA,transposeB) нельзя.
     */
    int lda = transposeA ? M : K;
    int ldb = transposeB ? K : N;

    cublasHandle_t handle = get_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "TensorOpsGPU: cuBLAS handle unavailable (matmulGPUDeviceEx)\n");
        return;
    }
    /*
     * Row-major C[M×N] = op(A)*op(B) через column-major cuBLAS: первым операндом идёт B, вторым A, (m,n,k)=(N,M,K),
     * transa/transb = (opB,opA). Вариант (opA,opB,pdA,pdb) ломает согласование с {@link #matmulGPUDevice} при NT,NT.
     */
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
    if (get_cublas_handle() == nullptr) {
        fprintf(stderr, "ensureStridedBatchedPackScratch0: cuBLAS handle unavailable\n");
        return JNI_FALSE;
    }
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
    if (handle == nullptr) {
        fprintf(stderr, "matmulGpuDeviceQkvProjections0: cuBLAS handle unavailable\n");
        return;
    }
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
    if (handle == nullptr) {
        fprintf(stderr, "jgpt_cuda_ffn_w1w3_strided_batched_device: cuBLAS handle unavailable\n");
        return 0;
    }
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
    /*
     * cuBLAS и memcpy идут на kTensorCudaStream (см. get_cublas_handle). Без синхронизации вызывающий код мог бы
     * прочитать h1/gate на CPU до завершения D2D — гонка. При cudaStreamBeginCapture на этом stream sync запрещён.
     */
    if (!jgpt_cuda_sync_stream_unless_capturing("jgpt_cuda_ffn_w1w3_strided_batched_device")) {
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

static void jgpt_log_gpu_alloc_failure(
        const char* ctx, long long num_elems, size_t elem_bytes, cudaError_t e) {
    size_t free_b = 0;
    size_t total_b = 0;
    (void) cudaMemGetInfo(&free_b, &total_b);
    const size_t req = (num_elems > 0 && elem_bytes > 0)
            ? (static_cast<size_t>(num_elems) * elem_bytes)
            : 0;
    fprintf(
            stderr,
            "%s: %s (code=%d) numElems=%lld elemBytes=%zu reqBytes=%zu cudaMemGetInfo free=%zu total=%zu\n",
            ctx,
            cudaGetErrorString(e),
            static_cast<int>(e),
            num_elems,
            elem_bytes,
            req,
            free_b,
            total_b);
    FILE* df = jgpt_debug_ndjson_fopen();
    if (df != nullptr) {
        fprintf(
                df,
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"OOM-gpuAlloc\",\"location\":\"%s\",\"message\":\"alloc failed\","
                "\"data\":{\"cudaErr\":%d,\"numElems\":%lld,\"elemBytes\":%zu,\"reqBytes\":%zu,\"free\":%zu,\"total\":%zu}}\n",
                ctx,
                static_cast<int>(e),
                num_elems,
                elem_bytes,
                req,
                free_b,
                total_b);
        fclose(df);
    }
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
    cudaStream_t tensor_stream = jgpt_cuda_stream_handle();
    if (tensor_stream != nullptr) {
        e = cudaMallocAsync(reinterpret_cast<void**>(&p), bytes, tensor_stream);
        if (e != cudaSuccess) {
            /* Дождаться отложенных cudaFreeAsync на том же stream — иначе пул часто даёт ложный OOM. */
            (void) cudaStreamSynchronize(tensor_stream);
            (void) cudaGetLastError();
            p = nullptr;
            e = cudaMallocAsync(reinterpret_cast<void**>(&p), bytes, tensor_stream);
        }
        if (e != cudaSuccess) {
            /* Пул async часто отказывает при фрагментации при большом cudaMemGetInfo free — пробуем sync malloc. */
            jgpt_cuda_trim_device_memory_full_best_effort("halfSyncMalloc");
            size_t mf0 = 0;
            size_t mt0 = 0;
            (void) cudaMemGetInfo(&mf0, &mt0);
            // #region agent log
            {
                FILE* df = jgpt_debug_ndjson_fopen();
                if (df != nullptr) {
                    fprintf(
                            df,
                            "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H10-preSyncMalloc\",\"location\":\"GpuHalfBuffer.nativeAlloc\","
                            "\"message\":\"afterFullTrim\",\"data\":{\"attempt\":1,\"bytes\":%zu,\"free\":%zu,\"total\":%zu}}\n",
                            bytes,
                            mf0,
                            mt0);
                    fclose(df);
                }
            }
            // #endregion
            p = nullptr;
            cudaError_t e_sync1 = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
            cudaError_t e_sync2 = e_sync1;
            if (e_sync1 != cudaSuccess) {
                jgpt_cuda_trim_device_memory_full_best_effort("halfSyncMallocRetry2");
                size_t mf1 = 0;
                size_t mt1 = 0;
                (void) cudaMemGetInfo(&mf1, &mt1);
                // #region agent log
                {
                    FILE* df = jgpt_debug_ndjson_fopen();
                    if (df != nullptr) {
                        fprintf(
                                df,
                                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H10-preSyncMalloc\",\"location\":\"GpuHalfBuffer.nativeAlloc\","
                                "\"message\":\"afterFullTrim\",\"data\":{\"attempt\":2,\"bytes\":%zu,\"free\":%zu,\"total\":%zu}}\n",
                                bytes,
                                mf1,
                                mt1);
                        fclose(df);
                    }
                }
                // #endregion
                p = nullptr;
                e_sync2 = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
            }
            if (e_sync2 == cudaSuccess) {
                jgpt_cuda_sync_device_alloc_register(reinterpret_cast<uintptr_t>(p));
                fprintf(
                        stderr,
                        "GpuHalfBuffer.nativeAlloc: cudaMallocAsync failed (%s), using cudaMalloc fallback bytes=%zu\n",
                        cudaGetErrorString(e),
                        bytes);
                FILE* df = jgpt_debug_ndjson_fopen();
                if (df != nullptr) {
                    fprintf(
                            df,
                            "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H-syncMallocFallback\",\"location\":\"GpuHalfBuffer.nativeAlloc\","
                            "\"message\":\"fallback cudaMalloc\",\"data\":{\"bytes\":%zu,\"asyncErr\":%d}}\n",
                            bytes,
                            static_cast<int>(e));
                    fclose(df);
                }
                return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
            }
            fprintf(
                    stderr,
                    "GpuHalfBuffer.nativeAlloc: cudaMalloc fallback failed: attempts %s / %s (async %s)\n",
                    cudaGetErrorString(e_sync1),
                    cudaGetErrorString(e_sync2),
                    cudaGetErrorString(e));
            jgpt_log_gpu_alloc_failure("GpuHalfBuffer.nativeAlloc", numHalfs, sizeof(__half), e_sync2);
            return 0;
        }
    } else {
        e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
        if (e == cudaSuccess) {
            jgpt_cuda_sync_device_alloc_register(reinterpret_cast<uintptr_t>(p));
        }
    }
#else
    e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
#endif
    if (e != cudaSuccess) {
        jgpt_log_gpu_alloc_failure("GpuHalfBuffer.nativeAlloc", numHalfs, sizeof(__half), e);
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuHalfBuffer_nativeFree(JNIEnv* env, jclass clazz, jlong ptr) {
    (void) env;
    (void) clazz;
    if (ptr == 0) return;
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    uintptr_t up = static_cast<uintptr_t>(ptr);
    if (jgpt_cuda_sync_device_alloc_consume(up)) {
        cudaError_t fe = cudaFree(p);
        if (fe != cudaSuccess) {
            fprintf(stderr, "GpuHalfBuffer.nativeFree cudaFree: %s\n", cudaGetErrorString(fe));
        }
        return;
    }
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    if (g_jgpt_cuda_stream != nullptr) {
        cudaError_t e = cudaFreeAsync(p, g_jgpt_cuda_stream);
        if (e == cudaSuccess) {
            return;
        }
        fprintf(stderr, "GpuHalfBuffer.nativeFree cudaFreeAsync: %s\n", cudaGetErrorString(e));
        (void) cudaGetLastError();
        cudaError_t fe = cudaFree(p);
        if (fe != cudaSuccess) {
            fprintf(stderr, "GpuHalfBuffer.nativeFree cudaFree fallback: %s\n", cudaGetErrorString(fe));
        }
        return;
    }
#endif
    cudaFree(p);
}

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeAlloc(JNIEnv* env, jclass clazz, jlong numFloats) {
    (void) env;
    (void) clazz;
    if (numFloats <= 0 || check_size_overflow(numFloats, sizeof(float), 1)) return 0;
    size_t bytes = static_cast<size_t>(numFloats) * sizeof(float);
    float* p = nullptr;
    cudaError_t e;
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    cudaStream_t tensor_stream_f = jgpt_cuda_stream_handle();
    if (tensor_stream_f != nullptr) {
        e = cudaMallocAsync(reinterpret_cast<void**>(&p), bytes, tensor_stream_f);
        if (e != cudaSuccess) {
            (void) cudaStreamSynchronize(tensor_stream_f);
            (void) cudaGetLastError();
            p = nullptr;
            e = cudaMallocAsync(reinterpret_cast<void**>(&p), bytes, tensor_stream_f);
        }
        if (e != cudaSuccess) {
            jgpt_cuda_trim_device_memory_full_best_effort("floatSyncMalloc");
            size_t mf0 = 0;
            size_t mt0 = 0;
            (void) cudaMemGetInfo(&mf0, &mt0);
            // #region agent log
            {
                FILE* df = jgpt_debug_ndjson_fopen();
                if (df != nullptr) {
                    fprintf(
                            df,
                            "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H10-preSyncMalloc\",\"location\":\"GpuFloatBuffer.nativeAlloc\","
                            "\"message\":\"afterFullTrim\",\"data\":{\"attempt\":1,\"bytes\":%zu,\"free\":%zu,\"total\":%zu}}\n",
                            bytes,
                            mf0,
                            mt0);
                    fclose(df);
                }
            }
            // #endregion
            p = nullptr;
            cudaError_t e_sync1f = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
            cudaError_t e_sync2f = e_sync1f;
            if (e_sync1f != cudaSuccess) {
                jgpt_cuda_trim_device_memory_full_best_effort("floatSyncMallocRetry2");
                size_t mf1 = 0;
                size_t mt1 = 0;
                (void) cudaMemGetInfo(&mf1, &mt1);
                // #region agent log
                {
                    FILE* df = jgpt_debug_ndjson_fopen();
                    if (df != nullptr) {
                        fprintf(
                                df,
                                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H10-preSyncMalloc\",\"location\":\"GpuFloatBuffer.nativeAlloc\","
                                "\"message\":\"afterFullTrim\",\"data\":{\"attempt\":2,\"bytes\":%zu,\"free\":%zu,\"total\":%zu}}\n",
                                bytes,
                                mf1,
                                mt1);
                        fclose(df);
                    }
                }
                // #endregion
                p = nullptr;
                e_sync2f = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
            }
            if (e_sync2f == cudaSuccess) {
                jgpt_cuda_sync_device_alloc_register(reinterpret_cast<uintptr_t>(p));
                fprintf(
                        stderr,
                        "GpuFloatBuffer.nativeAlloc: cudaMallocAsync failed (%s), using cudaMalloc fallback bytes=%zu\n",
                        cudaGetErrorString(e),
                        bytes);
                FILE* df = jgpt_debug_ndjson_fopen();
                if (df != nullptr) {
                    fprintf(
                            df,
                            "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H-syncMallocFallback\",\"location\":\"GpuFloatBuffer.nativeAlloc\","
                            "\"message\":\"fallback cudaMalloc\",\"data\":{\"bytes\":%zu,\"asyncErr\":%d}}\n",
                            bytes,
                            static_cast<int>(e));
                    fclose(df);
                }
                return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
            }
            fprintf(
                    stderr,
                    "GpuFloatBuffer.nativeAlloc: cudaMalloc fallback failed: attempts %s / %s (async %s)\n",
                    cudaGetErrorString(e_sync1f),
                    cudaGetErrorString(e_sync2f),
                    cudaGetErrorString(e));
            jgpt_log_gpu_alloc_failure("GpuFloatBuffer.nativeAlloc", numFloats, sizeof(float), e_sync2f);
            return 0;
        }
    } else {
        e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
        if (e == cudaSuccess) {
            jgpt_cuda_sync_device_alloc_register(reinterpret_cast<uintptr_t>(p));
        }
    }
#else
    e = cudaMalloc(reinterpret_cast<void**>(&p), bytes);
#endif
    if (e != cudaSuccess) {
        jgpt_log_gpu_alloc_failure("GpuFloatBuffer.nativeAlloc", numFloats, sizeof(float), e);
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeFree(JNIEnv* env, jclass clazz, jlong ptr) {
    (void) env; (void) clazz;
    if (ptr == 0) return;
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    uintptr_t up = static_cast<uintptr_t>(ptr);
    if (jgpt_cuda_sync_device_alloc_consume(up)) {
        cudaError_t fe = cudaFree(p);
        if (fe != cudaSuccess) {
            fprintf(stderr, "GpuFloatBuffer.nativeFree cudaFree: %s\n", cudaGetErrorString(fe));
        }
        return;
    }
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    if (g_jgpt_cuda_stream != nullptr) {
        cudaError_t e = cudaFreeAsync(p, g_jgpt_cuda_stream);
        if (e == cudaSuccess) {
            return;
        }
        fprintf(stderr, "GpuFloatBuffer.nativeFree cudaFreeAsync: %s\n", cudaGetErrorString(e));
        (void) cudaGetLastError();
        cudaError_t fe = cudaFree(p);
        if (fe != cudaSuccess) {
            fprintf(stderr, "GpuFloatBuffer.nativeFree cudaFree fallback: %s\n", cudaGetErrorString(fe));
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
        if (jgpt_cuda_sync_stream_unless_capturing("GpuFloatBuffer.nativeCopyHtoD") == 0) {
            e = cudaErrorUnknown;
        }
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
        if (jgpt_cuda_sync_stream_unless_capturing("GpuFloatBuffer.nativeCopyHtoDOffset") == 0) {
            e = cudaErrorUnknown;
        }
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDOffset: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoH(
    JNIEnv* env, jclass clazz, jlong devicePtr, jfloatArray dst, jint offset, jint length) {
    (void) clazz;
    if (devicePtr == 0 || length <= 0) return;
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
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
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
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
    jgpt_cuda_ensure_stream();
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
    if (err == cudaSuccess && g_jgpt_cuda_stream != nullptr) {
        if (jgpt_cuda_sync_stream_unless_capturing("GpuFloatBuffer.nativeCopyHtoDDirect") == 0) {
            err = cudaErrorUnknown;
        }
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDDirect: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoHDirect(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject directBuf, jlong byteOffset, jlong numBytes) {
    (void) clazz;
    if (devicePtr == 0 || directBuf == nullptr || numBytes <= 0) return;
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
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
    jgpt_cuda_ensure_stream();
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
    if (err == cudaSuccess && g_jgpt_cuda_stream != nullptr) {
        if (jgpt_cuda_sync_stream_unless_capturing("GpuFloatBuffer.nativeCopyHtoDFloatBuffer") == 0) {
            err = cudaErrorUnknown;
        }
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDFloatBuffer: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoHFloatBuffer(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject floatBuf, jlong floatOffset, jlong numFloats) {
    (void) clazz;
    if (devicePtr == 0 || floatBuf == nullptr || numFloats <= 0) return;
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
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
    jgpt_cuda_ensure_stream();
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
    if (err == cudaSuccess && g_jgpt_cuda_stream != nullptr) {
        if (jgpt_cuda_sync_stream_unless_capturing("GpuFloatBuffer.nativeCopyHtoDAddress") == 0) {
            err = cudaErrorUnknown;
        }
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyHtoDAddress: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoHAddress(
    JNIEnv* env, jclass clazz, jlong devicePtr, jlong hostAddress, jlong numBytes) {
    (void) env;
    (void) clazz;
    if (devicePtr == 0 || hostAddress == 0 || numBytes <= 0) return;
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
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
        if (err == cudaSuccess) {
            if (jgpt_cuda_sync_stream_unless_capturing("GpuFloatBuffer.nativeClear") == 0) {
                err = cudaErrorUnknown;
            }
        }
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
        if (jgpt_cuda_sync_stream_unless_capturing("GpuIntBuffer.nativeCopyHtoD") == 0) {
            e = cudaErrorUnknown;
        }
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
    jgpt_cuda_abort_stream_capture_discard_graph();
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
    jgpt_cuda_ensure_stream();
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
    if (err == cudaSuccess && g_jgpt_cuda_stream != nullptr) {
        if (jgpt_cuda_sync_stream_unless_capturing("GpuIntBuffer.nativeCopyHtoDDirect") == 0) {
            err = cudaErrorUnknown;
        }
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuIntBuffer.nativeCopyHtoDDirect: %s\n", cudaGetErrorString(err));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeCopyDtoHDirect(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject directBuf, jlong byteOffset, jlong numBytes) {
    (void) clazz;
    if (devicePtr == 0 || directBuf == nullptr || numBytes <= 0) return;
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
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
    /* Иначе cudaStreamSynchronize при активном захвате графа даёт «operation not permitted when stream is capturing». */
    jgpt_cuda_abort_stream_capture_discard_graph();
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

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_cleanupCudaThreadResources0(JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    jgpt_cuda_cleanup_thread_resources();
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

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_cudaTrimDeviceMemoryPoolsBestEffort0(
        JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    jgpt_cuda_trim_device_memory_full_best_effort("jniTrimPoolsAfterGraph");
}

JNIEXPORT jboolean JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_cudaStreamBeginCapture0(JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    jgpt_cuda_ensure_stream();
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    {
        int dev_cap = 0;
        if (cudaGetDevice(&dev_cap) == cudaSuccess) {
            (void) cudaDeviceSynchronize();
            (void) cudaDeviceGraphMemTrim(dev_cap);
            if (kTensorCudaStream != nullptr) {
                (void) cudaStreamSynchronize(kTensorCudaStream);
            }
        }
    }
#endif
    jgpt_decoder_graph_mem_probe_log("preBeginCapture");
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
    jgpt_cuda_abort_stream_capture_discard_graph();
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
    cudaError_t e2 = cudaSuccess;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    /*
     * Без флага драйвер держит внутренние graph/cuBLAS аллокации между launch'ами — в логах видно ступень ~32 MiB
     * на postGraphLaunchOk и OOM на cudaGraphLaunch при ещё «зелёном» cudaMemGetInfo free.
     * AutoFreeOnLaunch: после каждого launch освобождать неосвобождённые mem-alloc узлы графа (см. CUDA Runtime docs).
     */
    e2 = cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch);
    if (e2 != cudaSuccess) {
        fprintf(
                stderr,
                "cudaGraphInstantiateWithFlags(AutoFreeOnLaunch) failed: %s; retry cudaGraphInstantiate\n",
                cudaGetErrorString(e2));
        exec = nullptr;
        e2 = cudaGraphInstantiate(&exec, graph, &errNode, log, sizeof(log));
    } else {
        // #region agent log
        {
            FILE* df = jgpt_debug_ndjson_fopen();
            if (df != nullptr) {
                fprintf(
                        df,
                        "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H-autofreeInst\","
                        "\"location\":\"jgpt_cuda.cu:cudaStreamEndCaptureAndInstantiate0\","
                        "\"message\":\"instantiate AutoFreeOnLaunch ok\"}\n");
                fclose(df);
            }
        }
        // #endregion
    }
#else
    e2 = cudaGraphInstantiate(&exec, graph, &errNode, log, sizeof(log));
#endif
    cudaGraphDestroy(graph);
    if (e2 != cudaSuccess) {
        fprintf(stderr, "cudaGraphInstantiate failed: %s (log=%s)\n", cudaGetErrorString(e2), log);
        return 0;
    }
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    {
        int dev_trim = 0;
        if (cudaGetDevice(&dev_trim) == cudaSuccess) {
            (void) cudaDeviceGraphMemTrim(dev_trim);
        }
    }
#endif
    jgpt_decoder_graph_mem_probe_log("postInstantiate");
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(exec));
}

JNIEXPORT jboolean JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_cudaGraphExecLaunch0(
    JNIEnv* env, jclass clazz, jlong execPtr) {
    (void) env;
    (void) clazz;
    if (execPtr == 0) {
        return JNI_TRUE;
    }
    g_jgpt_last_cuda_graph_launch_err = 0;
    jgpt_cuda_ensure_stream();
    cudaStreamCaptureStatus capBefore = cudaStreamCaptureStatusNone;
    (void) cudaStreamGetCaptureInfo(kTensorCudaStream, &capBefore, nullptr);
    /*
     * Синхронизация потока перед replay: без неё отложенная ошибка с предыдущих kernel/cuBLAS на том же stream
     * иногда проявляется как cudaGraphLaunch → cudaErrorInvalidValue (code 1), хотя streamCapture=none.
     * Отключить: env JGPT_DECODER_CUDA_GRAPH_NO_PRELAUNCH_SYNC=1 (после проверки на своём стенде).
     */
    int syn_pre_code = -1;
    if (!jgpt_cuda_env_decoder_graph_no_prelaunch_sync()) {
        cudaError_t syn = cudaStreamSynchronize(kTensorCudaStream);
        syn_pre_code = static_cast<int>(syn);
        if (syn != cudaSuccess) {
            fprintf(
                    stderr,
                    "cudaStreamSynchronize before cudaGraphLaunch: %s (code=%d)\n",
                    cudaGetErrorString(syn),
                    static_cast<int>(syn));
        }
    }
    cudaGraphExec_t exec = reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(execPtr));
    jgpt_decoder_graph_mem_probe_log("preGraphLaunch");
    cudaError_t e = cudaGraphLaunch(exec, kTensorCudaStream);
    if (e == cudaErrorMemoryAllocation) {
        /*
         * Первый launch после instantiate иногда даёт OOM при забитом пуле/отложенных free на других stream —
         * полная синхронизация устройства освобождает память для повторной попытки.
         */
        fprintf(stderr, "cudaGraphLaunch OOM (code=2): retry after cudaDeviceSynchronize\n");
        (void) cudaDeviceSynchronize();
        (void) cudaGetLastError();
        jgpt_cuda_trim_mem_pools_best_effort("graphLaunchOOMRetry");
        e = cudaGraphLaunch(exec, kTensorCudaStream);
    }
    if (e != cudaSuccess) {
        cudaStreamCaptureStatus capAfter = cudaStreamCaptureStatusNone;
        (void) cudaStreamGetCaptureInfo(kTensorCudaStream, &capAfter, nullptr);
        cudaError_t sticky = cudaGetLastError();
        uintptr_t fwd_p = 0, graph_p = 0;
        size_t fwd_sz = 0, graph_sz = 0;
        jgpt_cuda_decoder_graph_debug_aux_snapshot(&fwd_p, &graph_p, &fwd_sz, &graph_sz);
        jgpt_b39372_append_native_graph_launch_fail(
                capBefore, capAfter, e, sticky, syn_pre_code, exec, graph_p, graph_sz);
        size_t mf = 0;
        size_t mt = 0;
        (void) cudaMemGetInfo(&mf, &mt);
        fprintf(
                stderr,
                "cudaGraphLaunch failed: %s (code=%d) cudaGetLastError=%s (code=%d) streamCapture before=%d after=%d "
                "(exec=%p auxNonGraph=0x%llx sz=%zu auxGraph=0x%llx sz=%zu) cudaMemGetInfo free=%zu total=%zu\n",
                cudaGetErrorString(e),
                static_cast<int>(e),
                cudaGetErrorString(sticky),
                static_cast<int>(sticky),
                static_cast<int>(capBefore),
                static_cast<int>(capAfter),
                static_cast<void*>(exec),
                static_cast<unsigned long long>(fwd_p),
                fwd_sz,
                static_cast<unsigned long long>(graph_p),
                graph_sz,
                mf,
                mt);
        g_jgpt_last_cuda_graph_launch_err = static_cast<int>(e);
        jgpt_decoder_graph_mem_probe_log("postGraphLaunchFail");
        return JNI_FALSE;
    }
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    {
        int dev_la = 0;
        if (cudaGetDevice(&dev_la) == cudaSuccess) {
            (void) cudaDeviceGraphMemTrim(dev_la);
        }
    }
#endif
    jgpt_cuda_trim_mem_pools_best_effort("postGraphLaunchOk");
    jgpt_decoder_graph_mem_probe_log("postGraphLaunchOk");
    return JNI_TRUE;
}

JNIEXPORT jint JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_decoderGraphExecLaunchLastCudaError0(
        JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    return static_cast<jint>(g_jgpt_last_cuda_graph_launch_err);
}

JNIEXPORT jlongArray JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_decoderGraphLaunchProbe0(
        JNIEnv* env, jclass clazz, jlong execPtr) {
    (void) clazz;
    jgpt_cuda_ensure_stream();
    int dev = -1;
    (void) cudaGetDevice(&dev);
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    (void) cudaStreamGetCaptureInfo(kTensorCudaStream, &cap, nullptr);
    cudaError_t q = cudaStreamQuery(kTensorCudaStream);
    int driver = 0;
    int runtime = 0;
    (void) cudaDriverGetVersion(&driver);
    (void) cudaRuntimeGetVersion(&runtime);
    int no_sync_env = jgpt_cuda_env_decoder_graph_no_prelaunch_sync() ? 1 : 0;
    jlong exec_flags = -1;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11040)
    if (execPtr != 0) {
        cudaGraphExec_t ex = reinterpret_cast<cudaGraphExec_t>(static_cast<uintptr_t>(execPtr));
        unsigned long long gf = 0ULL;
        cudaError_t gfe = cudaGraphExecGetFlags(ex, &gf);
        exec_flags = (gfe == cudaSuccess) ? static_cast<jlong>(gf)
                                          : (-1000LL - static_cast<jlong>(gfe));
    }
#endif
    const jlong stream_j = static_cast<jlong>(reinterpret_cast<uintptr_t>(kTensorCudaStream));
    const jlong tmp[9] = {
            static_cast<jlong>(dev),
            static_cast<jlong>(cap),
            static_cast<jlong>(q),
            static_cast<jlong>(no_sync_env),
            static_cast<jlong>(driver),
            static_cast<jlong>(runtime),
            exec_flags,
            stream_j,
            execPtr};
    jlongArray out = env->NewLongArray(9);
    if (out == nullptr) {
        return nullptr;
    }
    env->SetLongArrayRegion(out, 0, 9, tmp);
    return out;
}

JNIEXPORT jlongArray JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_decoderGraphDebugNativeAuxSnapshot0(
    JNIEnv* env, jclass clazz) {
    (void) clazz;
    uintptr_t fwd_p = 0;
    uintptr_t graph_p = 0;
    size_t fwd_sz = 0;
    size_t graph_sz = 0;
    jgpt_cuda_decoder_graph_debug_aux_snapshot(&fwd_p, &graph_p, &fwd_sz, &graph_sz);
    jlongArray arr = env->NewLongArray(4);
    if (arr == nullptr) {
        return nullptr;
    }
    jlong tmp[4] = {
            static_cast<jlong>(fwd_p),
            static_cast<jlong>(graph_p),
            static_cast<jlong>(fwd_sz),
            static_cast<jlong>(graph_sz)};
    env->SetLongArrayRegion(arr, 0, 4, tmp);
    return arr;
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
    /*
     * Без trim пул памяти графов (CUDA 12+) и default mem pool часто удерживают десятки МиБ на каждый уничтоженный
     * exec — в логах memProbe видно ровно ~32 MiB просадки free после каждого postGraphLaunchOk, пока не кончится
     * запас и cudaGraphLaunch на следующем слое не даст OOM при том же auxGraph.
     */
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
    {
        int dev_trim = 0;
        if (cudaGetDevice(&dev_trim) == cudaSuccess) {
            (void) cudaDeviceGraphMemTrim(dev_trim);
        }
    }
#endif
    jgpt_cuda_trim_mem_pools_best_effort("cudaGraphExecDestroy");
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