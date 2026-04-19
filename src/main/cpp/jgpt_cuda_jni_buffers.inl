/* JNI: GpuHalf/Float/Int buffers + H2D/D2H helpers
 *
 * Included only from jgpt_cuda.cu (single translation unit).
 */

// Forward declaration (defined later in this file)
static void jgpt_log_gpu_alloc_failure(const char* ctx, long long num_elems, size_t elem_bytes, cudaError_t e);

#include "jgpt_cuda_jni_raii.cuh"

// ========== Common device alloc/free helpers ==========

/** Универсальный alloc с async→sync fallback, trim, logging. */
static jlong jgpt_cuda_device_buffer_alloc(JNIEnv* env, jclass clazz,
        jlong numElems, size_t elemBytes, const char* ctx) {
    (void) env;
    (void) clazz;
    if (jgpt_jni_long_elems_invalid(numElems, elemBytes)) {
        return 0;
    }
    size_t bytes = static_cast<size_t>(numElems) * elemBytes;
    void* p = nullptr;
    cudaError_t e;
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    cudaStream_t tensor_stream = jgpt_cuda_stream_handle();
    if (tensor_stream != nullptr) {
        e = cudaMallocAsync(&p, bytes, tensor_stream);
        if (e != cudaSuccess) {
            (void) cudaStreamSynchronize(tensor_stream);
            (void) cudaGetLastError();
            p = nullptr;
            e = cudaMallocAsync(&p, bytes, tensor_stream);
        }
        if (e != cudaSuccess) {
            jgpt_cuda_trim_device_memory_full_best_effort("syncMalloc");
            size_t mf0 = 0; size_t mt0 = 0;
            (void) cudaMemGetInfo(&mf0, &mt0);
            jgpt_ndjson_log("H10-preSyncMalloc", ctx, "afterFullTrim",
                            "\"attempt\":%d,\"bytes\":%zu,\"free\":%zu,\"total\":%zu",
                            1, bytes, mf0, mt0);
            p = nullptr;
            cudaError_t e_sync1 = cudaMalloc(&p, bytes);
            cudaError_t e_sync2 = e_sync1;
            if (e_sync1 != cudaSuccess) {
                jgpt_cuda_trim_device_memory_full_best_effort("syncMallocRetry2");
                size_t mf1 = 0; size_t mt1 = 0;
                (void) cudaMemGetInfo(&mf1, &mt1);
                jgpt_ndjson_log("H10-preSyncMalloc", ctx, "afterFullTrim",
                                "\"attempt\":%d,\"bytes\":%zu,\"free\":%zu,\"total\":%zu",
                                2, bytes, mf1, mt1);
                p = nullptr;
                e_sync2 = cudaMalloc(&p, bytes);
            }
            if (e_sync2 == cudaSuccess) {
                jgpt_cuda_sync_device_alloc_register(reinterpret_cast<uintptr_t>(p));
                fprintf(stderr, "%s: cudaMallocAsync failed (%s), using cudaMalloc fallback bytes=%zu\n",
                        ctx, cudaGetErrorString(e), bytes);
                jgpt_ndjson_log("H-syncMallocFallback", ctx, "fallback cudaMalloc",
                                "\"bytes\":%zu,\"asyncErr\":%d", bytes, static_cast<int>(e));
                return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
            }
            fprintf(stderr, "%s: cudaMalloc fallback failed: attempts %s / %s (async %s)\n",
                    ctx, cudaGetErrorString(e_sync1), cudaGetErrorString(e_sync2), cudaGetErrorString(e));
            jgpt_log_gpu_alloc_failure(ctx, numElems, elemBytes, e_sync2);
            return 0;
        }
    } else {
        e = cudaMalloc(&p, bytes);
        if (e == cudaSuccess) {
            jgpt_cuda_sync_device_alloc_register(reinterpret_cast<uintptr_t>(p));
        }
    }
#else
    e = cudaMalloc(&p, bytes);
#endif
    if (e != cudaSuccess) {
        jgpt_log_gpu_alloc_failure(ctx, numElems, elemBytes, e);
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(p));
}

/** Универсальный free: async если sync-malloc не использовался, иначе sync cudaFree. */
static void jgpt_cuda_device_buffer_free(JNIEnv* env, jclass clazz, jlong ptr, const char* ctx) {
    (void) env;
    (void) clazz;
    if (ptr == 0) return;
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    uintptr_t up = static_cast<uintptr_t>(ptr);
    if (jgpt_cuda_sync_device_alloc_consume(up)) {
        cudaError_t fe = cudaFree(p);
        if (fe != cudaSuccess) {
            fprintf(stderr, "%s cudaFree: %s\n", ctx, cudaGetErrorString(fe));
        }
        return;
    }
#if CUDART_VERSION >= 11020
    jgpt_cuda_ensure_stream();
    if (jgpt_cuda_stream_handle() != nullptr) {
        cudaError_t e = cudaFreeAsync(p, jgpt_cuda_stream_handle());
        if (e == cudaSuccess) return;
        fprintf(stderr, "%s cudaFreeAsync: %s\n", ctx, cudaGetErrorString(e));
        (void) cudaGetLastError();
        cudaError_t fe = cudaFree(p);
        if (fe != cudaSuccess) {
            fprintf(stderr, "%s cudaFree fallback: %s\n", ctx, cudaGetErrorString(fe));
        }
        return;
    }
#endif
    cudaFree(p);
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
    return jgpt_cuda_device_buffer_alloc(env, clazz, numHalfs, sizeof(__half), "GpuHalfBuffer.nativeAlloc");
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuHalfBuffer_nativeFree(JNIEnv* env, jclass clazz, jlong ptr) {
    jgpt_cuda_device_buffer_free(env, clazz, ptr, "GpuHalfBuffer.nativeFree");
}

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeAlloc(JNIEnv* env, jclass clazz, jlong numFloats) {
    return jgpt_cuda_device_buffer_alloc(env, clazz, numFloats, sizeof(float), "GpuFloatBuffer.nativeAlloc");
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeFree(JNIEnv* env, jclass clazz, jlong ptr) {
    jgpt_cuda_device_buffer_free(env, clazz, ptr, "GpuFloatBuffer.nativeFree");
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyHtoD(
    JNIEnv* env, jclass clazz, jlong devicePtr, jfloatArray src, jint offset, jint length) {
    (void) clazz;
    if (devicePtr == 0 || length <= 0) return;
    JGPT_CUDA_GUARD_1D(length, sizeof(float), return;);
    jgpt_cuda_ensure_stream();
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t bytes = static_cast<size_t>(length) * sizeof(float);
    JniFloatArrayScope scope(env, src, JNI_ABORT);
    if (!scope) return;
    cudaError_t e = cudaSuccess;
    /* Blocking H2D: avoids ordering issues between legacy default stream and kTensorCudaStream. */
    if (gpufb_pin_ensure(bytes)) {
        std::memcpy(tl_gpufb_pin, scope.ptr + offset, bytes);
        e = cudaMemcpy(d, tl_gpufb_pin, bytes, cudaMemcpyHostToDevice);
    } else {
        e = cudaMemcpy(d, scope.ptr + offset, bytes, cudaMemcpyHostToDevice);
    }
    if (e == cudaSuccess && jgpt_cuda_stream_handle() != nullptr) {
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
    JGPT_CUDA_GUARD_1D(len, sizeof(float), return;);
    jgpt_cuda_ensure_stream();
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr)) + deviceFloatOffset;
    size_t bytes = static_cast<size_t>(len) * sizeof(float);
    JniFloatArrayScope scope(env, src, JNI_ABORT);
    if (!scope) return;
    cudaError_t e = cudaSuccess;
    if (gpufb_pin_ensure(bytes)) {
        std::memcpy(tl_gpufb_pin, scope.ptr + srcOff, bytes);
        e = cudaMemcpy(d, tl_gpufb_pin, bytes, cudaMemcpyHostToDevice);
    } else {
        e = cudaMemcpy(d, scope.ptr + srcOff, bytes, cudaMemcpyHostToDevice);
    }
    if (e == cudaSuccess && jgpt_cuda_stream_handle() != nullptr) {
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
    JGPT_CUDA_GUARD_1D(length, sizeof(float), return;);
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t bytes = static_cast<size_t>(length) * sizeof(float);
    JniFloatArrayScope scope(env, dst, 0);
    if (!scope) return;
    cudaError_t e = cudaSuccess;
    if (gpufb_pin_ensure(bytes)) {
        e = cudaMemcpyAsync(tl_gpufb_pin, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoH: %s\n", cudaGetErrorString(e));
            return;
        }
        std::memcpy(scope.ptr + offset, tl_gpufb_pin, bytes);
    } else {
        e = cudaMemcpyAsync(scope.ptr + offset, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoH: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyDtoHOffset(
    JNIEnv* env, jclass clazz, jlong devicePtr, jint deviceFloatOffset, jfloatArray dst, jint dstOff, jint len) {
    (void) clazz;
    if (devicePtr == 0 || len <= 0) return;
    JGPT_CUDA_GUARD_1D(len, sizeof(float), return;);
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr)) + deviceFloatOffset;
    size_t bytes = static_cast<size_t>(len) * sizeof(float);
    JniFloatArrayScope scope(env, dst, 0);
    if (!scope) return;
    cudaError_t e = cudaSuccess;
    if (gpufb_pin_ensure(bytes)) {
        e = cudaMemcpyAsync(tl_gpufb_pin, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHOffset: %s\n", cudaGetErrorString(e));
            return;
        }
        std::memcpy(scope.ptr + dstOff, tl_gpufb_pin, bytes);
    } else {
        e = cudaMemcpyAsync(scope.ptr + dstOff, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoHOffset: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuFloatBuffer_nativeCopyHtoDDirect(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject directBuf, jlong byteOffset, jlong numBytes) {
    (void) clazz;
    if (devicePtr == 0 || directBuf == nullptr || numBytes <= 0) return;
    if (jgpt_jni_long_bytes_invalid(numBytes)) {
        return;
    }
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
    if (err == cudaSuccess && jgpt_cuda_stream_handle() != nullptr) {
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
    if (jgpt_jni_long_bytes_invalid(numBytes)) {
        return;
    }
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
    if (floatOffset < 0 || jgpt_jni_long_elems_invalid(numFloats, sizeof(float))) {
        return;
    }
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
    if (err == cudaSuccess && jgpt_cuda_stream_handle() != nullptr) {
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
    if (floatOffset < 0 || jgpt_jni_long_elems_invalid(numFloats, sizeof(float))) {
        return;
    }
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
    if (jgpt_jni_long_bytes_invalid(numBytes)) {
        return;
    }
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
    if (err == cudaSuccess && jgpt_cuda_stream_handle() != nullptr) {
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
    if (jgpt_jni_long_bytes_invalid(numBytes)) {
        return;
    }
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
    if (jgpt_jni_long_elems_invalid(numFloats, sizeof(float))) {
        return;
    }
    jgpt_cuda_ensure_stream();
    float* d = reinterpret_cast<float*>(static_cast<uintptr_t>(devicePtr));
    size_t nbytes = static_cast<size_t>(numFloats) * sizeof(float);
    /* Same stream as D2H async paths — avoids races between cudaMemset on default stream and memcpy on kTensorCudaStream. */
    cudaError_t err = cudaSuccess;
    if (jgpt_cuda_stream_handle() != nullptr) {
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
    JGPT_CUDA_GUARD_1D(length, sizeof(float), return;);
    const float* src = reinterpret_cast<const float*>(static_cast<uintptr_t>(srcDevicePtr));
    float* dst = reinterpret_cast<float*>(static_cast<uintptr_t>(dstDevicePtr));
    cudaError_t err = cudaMemcpyAsync(dst, src, static_cast<size_t>(length) * sizeof(float), cudaMemcpyDeviceToDevice, kTensorCudaStream);
    if (err != cudaSuccess) fprintf(stderr, "GpuFloatBuffer.nativeCopyDtoD: %s\n", cudaGetErrorString(err));
}

// ========== GPU INT BUFFER (int32 targets / indices) ==========

JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeAlloc(JNIEnv* env, jclass clazz, jlong numInts) {
    (void) env; (void) clazz;
    if (jgpt_jni_long_elems_invalid(numInts, sizeof(int))) {
        return 0;
    }
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
    JGPT_CUDA_GUARD_1D(length, sizeof(int), return;);
    jgpt_cuda_ensure_stream();
    int* d = reinterpret_cast<int*>(static_cast<uintptr_t>(devicePtr));
    size_t bytes = static_cast<size_t>(length) * sizeof(int);
    JniIntArrayScope scope(env, src, JNI_ABORT);
    if (!scope) return;
    cudaError_t e = cudaSuccess;
    /* Blocking H2D (same policy as GpuFloatBuffer.nativeCopyHtoD): avoid ordering issues vs kTensorCudaStream. */
    if (gpufb_pin_ensure(bytes)) {
        std::memcpy(tl_gpufb_pin, scope.ptr + offset, bytes);
        e = cudaMemcpy(d, tl_gpufb_pin, bytes, cudaMemcpyHostToDevice);
    } else {
        e = cudaMemcpy(d, scope.ptr + offset, bytes, cudaMemcpyHostToDevice);
    }
    if (e == cudaSuccess && jgpt_cuda_stream_handle() != nullptr) {
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
    JGPT_CUDA_GUARD_1D(length, sizeof(int), return;);
    jgpt_cuda_ensure_stream();
    jgpt_cuda_abort_stream_capture_discard_graph();
    int* d = reinterpret_cast<int*>(static_cast<uintptr_t>(devicePtr));
    size_t bytes = static_cast<size_t>(length) * sizeof(int);
    JniIntArrayScope scope(env, dst, 0);
    if (!scope) return;
    cudaError_t e = cudaSuccess;
    if (gpufb_pin_ensure(bytes)) {
        e = cudaMemcpyAsync(tl_gpufb_pin, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuIntBuffer.nativeCopyDtoH: %s\n", cudaGetErrorString(e));
            return;
        }
        std::memcpy(scope.ptr + offset, tl_gpufb_pin, bytes);
    } else {
        e = cudaMemcpyAsync(scope.ptr + offset, d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(kTensorCudaStream);
        if (e != cudaSuccess) fprintf(stderr, "GpuIntBuffer.nativeCopyDtoH: %s\n", cudaGetErrorString(e));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_GpuIntBuffer_nativeCopyHtoDDirect(
    JNIEnv* env, jclass clazz, jlong devicePtr, jobject directBuf, jlong byteOffset, jlong numBytes) {
    (void) clazz;
    if (devicePtr == 0 || directBuf == nullptr || numBytes <= 0) return;
    if (jgpt_jni_long_bytes_invalid(numBytes)) {
        return;
    }
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
    if (err == cudaSuccess && jgpt_cuda_stream_handle() != nullptr) {
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
    if (jgpt_jni_long_bytes_invalid(numBytes)) {
        return;
    }
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
    if (jgpt_jni_long_elems_invalid(numInts, sizeof(int))) {
        return;
    }
    jgpt_cuda_ensure_stream();
    int* d = reinterpret_cast<int*>(static_cast<uintptr_t>(devicePtr));
    size_t nbytes = static_cast<size_t>(numInts) * sizeof(int);
    cudaError_t err = cudaSuccess;
    if (jgpt_cuda_stream_handle() != nullptr) {
        err = cudaMemsetAsync(d, 0, nbytes, kTensorCudaStream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(jgpt_cuda_stream_handle());
    } else {
        err = cudaMemset(d, 0, nbytes);
    }
    if (err != cudaSuccess) fprintf(stderr, "GpuIntBuffer.nativeClear: %s\n", cudaGetErrorString(err));
}

