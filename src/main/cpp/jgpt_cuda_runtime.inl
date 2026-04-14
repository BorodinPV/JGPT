/* Runtime / stream / graph debug / memory trim
 * Global mutexes, stream atomics, decoder graph probes, device caps, jgpt_cuda_ensure_stream.
 * Included only from jgpt_cuda.cu (single translation unit).
 */

#include <cstdarg>

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
static FILE* jgpt_debug_ndjson_fopen(void);

/** NDJSON (debug session b39372): пишет одну строку в файл из {@code JGPT_DEBUG_NDJSON_LOG}. */
static void jgpt_ndjson_log(const char* hypothesisId, const char* location, const char* message, const char* fmt, ...) {
    FILE* f = jgpt_debug_ndjson_fopen();
    if (f == nullptr) {
        return;
    }
    fprintf(f, "{\"sessionId\":\"b39372\",\"hypothesisId\":\"%s\",\"location\":\"%s\",\"message\":\"%s\",",
            hypothesisId, location, message);
    if (fmt != nullptr && fmt[0] != '\0') {
        fprintf(f, "\"data\":{");
        va_list args;
        va_start(args, fmt);
        vfprintf(f, fmt, args);
        va_end(args);
        fprintf(f, "}");
    }
    fprintf(f, "}\n");
    fclose(f);
}

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
    jgpt_ndjson_log("H-memProbe", "jgpt_cuda.cu:memProbe", tag,
                    "\"free\":%zu,\"total\":%zu", mf, mt);
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
    jgpt_ndjson_log("H6-memPoolTrim", "jgpt_cuda.cu:trim", "trim",
                    "\"ctx\":\"%s\",\"trimmedMask\":%d,\"freeBefore\":%zu,\"freeAfter\":%zu",
                    ctx_tag != nullptr ? ctx_tag : "", trimmed_mask, free_before, free_after);
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

