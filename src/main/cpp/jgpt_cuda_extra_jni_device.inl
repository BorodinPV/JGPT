/* JNI device pointers: CUDA_CHECK_RV, full-GPU CE, lm-head kernels, anyNonFinite
 * Include only from jgpt_cuda_extra.cu inside extern "C" { }.
 */


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
    JGPT_CUDA_GUARD_1D(nrows, sizeof(float), return 0.f;);
    jgpt_cuda_ensure_stream();

    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    float* grad   = reinterpret_cast<float*>(static_cast<uintptr_t>(dGrad));

    size_t bytes_tgt = (size_t) nrows * sizeof(float);
    if (bytes_tgt > jgpt_extra::jgpt_extra_tls().ce.targets_bytes) {
        cudaFree(jgpt_extra::jgpt_extra_tls().ce.d_targets);
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_targets), bytes_tgt), 0.f);
        jgpt_extra::jgpt_extra_tls().ce.targets_bytes = bytes_tgt;
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_loss_sum == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_loss_sum), sizeof(float)), 0.f);
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_valid == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_valid), sizeof(unsigned int)), 0.f);
    }

    jfloat* ptgt = env->GetFloatArrayElements(hTargets, nullptr);
    if (!ptgt) return 0.f;
    CUDA_CHECK_RV(cudaMemcpyAsync(jgpt_extra::jgpt_extra_tls().ce.d_targets, ptgt, bytes_tgt, cudaMemcpyHostToDevice, kTensorCudaStream), 0.f);
    env->ReleaseFloatArrayElements(hTargets, ptgt, JNI_ABORT);
    /* No cudaStreamSynchronize here: same-stream order queues memset/kernel after H2D; one sync below before D2H loss. */

    CUDA_CHECK_RV(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream), 0.f);

    launch_cross_entropy(logits, jgpt_extra::jgpt_extra_tls().ce.d_targets, grad, gradScale, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum,
            jgpt_extra::jgpt_extra_tls().ce.d_valid, nrows, vocab, useFp16 == JNI_TRUE);
    CUDA_KERNEL_CHECK_RV(0.f);

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_valid, jgpt_extra::jgpt_extra_tls().ce.d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
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
    JGPT_CUDA_GUARD_1D(nrows, sizeof(float), return;);
    jgpt_cuda_ensure_stream();

    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    float* grad = reinterpret_cast<float*>(static_cast<uintptr_t>(dGrad));

    size_t bytes_tgt = (size_t) nrows * sizeof(float);
    if (bytes_tgt > jgpt_extra::jgpt_extra_tls().ce.targets_bytes) {
        cudaFree(jgpt_extra::jgpt_extra_tls().ce.d_targets);
        jgpt_extra::jgpt_extra_tls().ce.d_targets = nullptr;
        jgpt_extra::jgpt_extra_tls().ce.targets_bytes = 0;
        if (cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_targets), bytes_tgt) != cudaSuccess) {
            return;
        }
        jgpt_extra::jgpt_extra_tls().ce.targets_bytes = bytes_tgt;
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_loss_sum == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_loss_sum), sizeof(float)));
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_valid == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_valid), sizeof(unsigned int)));
    }
    ce_async_host_ensure();
    ce_pinned_flt_targets_ensure((size_t) nrows);
    if (jgpt_extra::jgpt_extra_tls().ce.h_async_loss == nullptr || jgpt_extra::jgpt_extra_tls().ce.h_async_valid == nullptr || jgpt_extra::jgpt_extra_tls().ce.h_pinned_flt_targets == nullptr) {
        return;
    }

    jfloat* ptgt = env->GetFloatArrayElements(hTargets, nullptr);
    if (!ptgt) {
        return;
    }
    std::memcpy(jgpt_extra::jgpt_extra_tls().ce.h_pinned_flt_targets, ptgt, bytes_tgt);
    env->ReleaseFloatArrayElements(hTargets, ptgt, JNI_ABORT);

    CUDA_CHECK_X(cudaMemcpyAsync(
            jgpt_extra::jgpt_extra_tls().ce.d_targets,
            jgpt_extra::jgpt_extra_tls().ce.h_pinned_flt_targets,
            bytes_tgt,
            cudaMemcpyHostToDevice,
            kTensorCudaStream));

    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy(logits, jgpt_extra::jgpt_extra_tls().ce.d_targets, grad, gradScale, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_valid, nrows, vocab, useFp16 == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK_X(cudaMemcpyAsync(
            jgpt_extra::jgpt_extra_tls().ce.h_async_loss,
            jgpt_extra::jgpt_extra_tls().ce.d_loss_sum,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(
            jgpt_extra::jgpt_extra_tls().ce.h_async_valid,
            jgpt_extra::jgpt_extra_tls().ce.d_valid,
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
    JGPT_CUDA_GUARD_1D(nrows, sizeof(float), return 0.f;);
    jgpt_cuda_ensure_stream();

    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    float* grad   = reinterpret_cast<float*>(static_cast<uintptr_t>(dGrad));
    int* targets_i = reinterpret_cast<int*>(static_cast<uintptr_t>(dTargetsInt));

    if (jgpt_extra::jgpt_extra_tls().ce.d_loss_sum == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_loss_sum), sizeof(float)), 0.f);
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_valid == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_valid), sizeof(unsigned int)), 0.f);
    }

    CUDA_CHECK_RV(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream), 0.f);

    launch_cross_entropy_i32(logits, targets_i, grad, gradScale, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum,
            jgpt_extra::jgpt_extra_tls().ce.d_valid, nrows, vocab, useFp16 == JNI_TRUE);
    CUDA_KERNEL_CHECK_RV(0.f);

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_valid, jgpt_extra::jgpt_extra_tls().ce.d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
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
    JGPT_CUDA_GUARD_1D(nrows, sizeof(float), return;);
    jgpt_cuda_ensure_stream();

    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));
    float* grad = reinterpret_cast<float*>(static_cast<uintptr_t>(dGrad));
    int* targets_i = reinterpret_cast<int*>(static_cast<uintptr_t>(dTargetsInt));

    if (jgpt_extra::jgpt_extra_tls().ce.d_loss_sum == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_loss_sum), sizeof(float)));
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_valid == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_valid), sizeof(unsigned int)));
    }
    ce_async_host_ensure();
    if (jgpt_extra::jgpt_extra_tls().ce.h_async_loss == nullptr || jgpt_extra::jgpt_extra_tls().ce.h_async_valid == nullptr) {
        return;
    }

    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy_i32(logits, targets_i, grad, gradScale, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_valid, nrows, vocab, useFp16 == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK_X(cudaMemcpyAsync(
            jgpt_extra::jgpt_extra_tls().ce.h_async_loss,
            jgpt_extra::jgpt_extra_tls().ce.d_loss_sum,
            sizeof(float),
            cudaMemcpyDeviceToHost,
            kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(
            jgpt_extra::jgpt_extra_tls().ce.h_async_valid,
            jgpt_extra::jgpt_extra_tls().ce.d_valid,
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
    if (jgpt_extra::jgpt_extra_tls().ce.h_async_loss == nullptr || jgpt_extra::jgpt_extra_tls().ce.h_async_valid == nullptr) {
        return 0.f;
    }
    float lossSum = *jgpt_extra::jgpt_extra_tls().ce.h_async_loss;
    unsigned int v = *jgpt_extra::jgpt_extra_tls().ce.h_async_valid;
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
    if (jgpt_extra::jgpt_extra_tls().ce.d_loss_sum == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_loss_sum), sizeof(float)), 0.f);
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_valid == nullptr) {
        CUDA_CHECK_RV(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_valid), sizeof(unsigned int)), 0.f);
    }
    CUDA_CHECK_RV(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream), 0.f);
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    sampled_ce_first_slot_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            candidateLogits, candidateIds, candidateGrad, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_valid, rows, candidates, gradScale);
    CUDA_KERNEL_CHECK_RV(0.f);

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
    CUDA_CHECK_RV(cudaMemcpyAsync(&h_valid, jgpt_extra::jgpt_extra_tls().ce.d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream), 0.f);
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
    JGPT_CUDA_GUARD_1D(len, sizeof(float), return;);
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
