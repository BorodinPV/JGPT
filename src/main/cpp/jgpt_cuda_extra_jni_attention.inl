/* JNI attention: SDPA forward/backward (device + resident)
 * Include only from jgpt_cuda_extra.cu inside extern "C" { }.
 */


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
    if (jgpt_size_mul_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen))) {
        return;
    }
    if (jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(dKDim))
            || jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(seqLen))
            || jgpt_alloc_matrix_bytes_overflows(static_cast<size_t>(seqLen), static_cast<size_t>(seqLen), sizeof(float))) {
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
    if (jgpt_size_mul_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen))) {
        return;
    }
    if (jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(dKDim))
            || jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(seqLen))) {
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
    if (jgpt_size_mul_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen))) {
        return;
    }
    if (jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(seqLen))
            || jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(dK))
            || jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(dV))) {
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

    /*
     * Раньше при useFp16Softmax пересчитывали QK^T/scores на device и гоняли softmax_last_dim_bwd_from_logits_fp16.
     * Forward softmax_last_dim_kernel_fp16 уже пишет вероятности в FP32 (exp в FP32 — см. комментарий к ядру);
     * сохранённые probs совпадают с softmax(scores), поэтому достаточно стандартного VJP по p — без лишнего GEMM QK.
     */
    (void) useFp16Softmax;
    {
        const long long nrows_ll = static_cast<long long>(batch) * static_cast<long long>(seqLen);
        launch_softmax_last_dim_bwd_block_chunked(d_dp, d_p, d_ds, nrows_ll, seqLen, kTensorCudaStream);
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
    if (jgpt_size_mul_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen))) {
        return;
    }
    if (jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(seqLen))
            || jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(batch), static_cast<size_t>(seqLen), static_cast<size_t>(dVDim))) {
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

    // go × V^T → dp: transB GEMM
    if (!batched_sgemm_row_major_transB(go, v, d_dp, batch, seqLen, dVDim, seqLen, 1.0f, 0.0f)) {
        return;
    }

    CUDA_CHECK_X(cudaMemset(d_ds, 0, bytesProb));

    (void) useFp16Softmax;
    {
        const long long nrows_ll = static_cast<long long>(batch) * static_cast<long long>(seqLen);
        launch_softmax_last_dim_bwd_block_chunked(d_dp, p, d_ds, nrows_ll, seqLen, kTensorCudaStream);
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
