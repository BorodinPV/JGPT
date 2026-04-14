/* JNI: add/sub/relu, init, graph exec, memory stats, CudaPinnedHost
 *
 * Included only from jgpt_cuda.cu (single translation unit).
 * RAII helpers are provided by jgpt_cuda_jni_raii.cuh (included before extern "C").
 */

// ========== ADD ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_addGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_a, jfloatArray h_b, jfloatArray h_c, jint n) {
    (void) clazz;
    if (n <= 0) return;
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);

    DeviceFloatPtr d_a, d_b, d_c;
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_a.ptr), bytes));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_b.ptr), bytes));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_c.ptr), bytes));

    JniFloatArrayScope pa(env, h_a, JNI_ABORT);
    JniFloatArrayScope pb(env, h_b, JNI_ABORT);
    if (!pa || !pb) return;
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_a, pa.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_b, pb.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));

    launch_vec_add(d_a, d_b, d_c, n);
    JGPT_KERNEL_LAUNCH_CHECK();

    JniFloatArrayScope pc(env, h_c, 0);
    if (!pc) return;
    CUDA_CHECK_VOID(cudaMemcpyAsync(pc.ptr, d_c, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_subtractGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_a, jfloatArray h_b, jfloatArray h_c, jint n) {
    (void) clazz;
    if (n <= 0) return;
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);

    DeviceFloatPtr d_a, d_b, d_c;
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_a.ptr), bytes));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_b.ptr), bytes));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_c.ptr), bytes));

    JniFloatArrayScope pa(env, h_a, JNI_ABORT);
    JniFloatArrayScope pb(env, h_b, JNI_ABORT);
    if (!pa || !pb) return;
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_a, pa.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_b, pb.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));

    launch_vec_sub(d_a, d_b, d_c, n);
    JGPT_KERNEL_LAUNCH_CHECK();

    JniFloatArrayScope pc(env, h_c, 0);
    if (!pc) return;
    CUDA_CHECK_VOID(cudaMemcpyAsync(pc.ptr, d_c, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
}

// ========== ReLU ==========

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_reluGPU(
    JNIEnv* env, jclass clazz,
    jfloatArray h_a, jfloatArray h_b, jint n) {
    (void) clazz;
    if (n <= 0) return;
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);

    DeviceFloatPtr d_a, d_b;
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_a.ptr), bytes));
    CUDA_CHECK_VOID(cudaMalloc(reinterpret_cast<void**>(&d_b.ptr), bytes));

    JniFloatArrayScope pa(env, h_a, JNI_ABORT);
    if (!pa) return;
    CUDA_CHECK_VOID(cudaMemcpyAsync(d_a, pa.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));

    launch_relu(d_a, d_b, n);
    JGPT_KERNEL_LAUNCH_CHECK();

    JniFloatArrayScope pb(env, h_b, 0);
    if (!pb) return;
    CUDA_CHECK_VOID(cudaMemcpyAsync(pb.ptr, d_b, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(kTensorCudaStream));
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
    return (jgpt_cuda_stream_handle() != nullptr) ? JNI_TRUE : JNI_FALSE;
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
    if (jgpt_cuda_stream_handle() == nullptr) {
        return;
    }
    /* Иначе cudaStreamSynchronize при активном захвате графа даёт «operation not permitted when stream is capturing». */
    jgpt_cuda_abort_stream_capture_discard_graph();
    cudaError_t err = cudaStreamSynchronize(jgpt_cuda_stream_handle());
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
    if (jgpt_jni_long_bytes_invalid(numBytes)) {
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
    if (jgpt_jni_long_bytes_invalid(numBytes)) {
        return nullptr;
    }
    return env->NewDirectByteBuffer(reinterpret_cast<void*>(ptr), numBytes);
}
