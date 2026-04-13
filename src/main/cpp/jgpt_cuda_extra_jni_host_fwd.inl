/* JNI host forward: Host arrays: softmax, CE, norms, activations, RoPE, embedding, pos-embed
 * Include only from jgpt_cuda_extra.cu inside extern "C" { }.
 */

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_softmaxLastDimGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jfloatArray h_dst, jint batch, jint mid, jint inner,
    jboolean useFp16Softmax) {
    (void) clazz;
    if (batch <= 0 || mid <= 0 || inner <= 0) {
        return;
    }
    const long long nrows_ll = static_cast<long long>(batch) * static_cast<long long>(mid);
    if (nrows_ll <= 0) {
        return;
    }
    if (check_size_overflow(static_cast<size_t>(nrows_ll), static_cast<size_t>(inner), sizeof(float))) {
        fprintf(stderr, "softmaxLastDimGPU: size overflow\n");
        return;
    }
    const size_t bytes = static_cast<size_t>(nrows_ll) * static_cast<size_t>(inner) * sizeof(float);
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

    launch_softmax_last_dim(d_src, d_dst, nrows_ll, inner, useFp16Softmax == JNI_TRUE);
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
        jgpt_jni_loss_out_zero(env, h_lossOut);
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
        jgpt_jni_loss_out_zero(env, h_lossOut);
        return;
    }
    if (bytes_tgt > jgpt_extra::jgpt_extra_tls().ce.targets_bytes) {
        cudaFree(jgpt_extra::jgpt_extra_tls().ce.d_targets);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_targets), bytes_tgt));
        jgpt_extra::jgpt_extra_tls().ce.targets_bytes = bytes_tgt;
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_loss_sum == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_loss_sum), sizeof(float)));
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_valid == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_valid), sizeof(unsigned int)));
    }

    jfloat* plog = env->GetFloatArrayElements(h_logits, nullptr);
    jfloat* ptgt = env->GetFloatArrayElements(h_targets, nullptr);
    if (!plog || !ptgt) {
        jgpt_jni_loss_out_zero(env, h_lossOut);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(jgpt_extra::jgpt_extra_tls().ce.d_logits, plog, bytes_logits, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(jgpt_extra::jgpt_extra_tls().ce.d_targets, ptgt, bytes_tgt, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_logits, plog, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_targets, ptgt, JNI_ABORT);

    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy(jgpt_extra::jgpt_extra_tls().ce.d_logits, jgpt_extra::jgpt_extra_tls().ce.d_targets, jgpt_extra::jgpt_extra_tls().ce.d_grad, gradScaleOverTotalTokens, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum,
            jgpt_extra::jgpt_extra_tls().ce.d_valid, nrows, vocab, useFp16Softmax == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    jfloat* pgrad = env->GetFloatArrayElements(h_grad, nullptr);
    if (!pgrad) {
        jgpt_jni_loss_out_zero(env, h_lossOut);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgrad, jgpt_extra::jgpt_extra_tls().ce.d_grad, bytes_logits, cudaMemcpyDeviceToHost, kTensorCudaStream));

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_X(cudaMemcpyAsync(&h_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(&h_valid, jgpt_extra::jgpt_extra_tls().ce.d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream));
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
        jgpt_jni_loss_out_zero(env, h_loss_out);
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
        jgpt_jni_loss_out_zero(env, h_loss_out);
        return;
    }

    if (!ce_ensure_logits_grad_buffers(bytes_logits)) {
        jgpt_jni_loss_out_zero(env, h_loss_out);
        return;
    }
    if (bytes_tgt > jgpt_extra::jgpt_extra_tls().ce.targets_bytes) {
        cudaFree(jgpt_extra::jgpt_extra_tls().ce.d_targets);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_targets), bytes_tgt));
        jgpt_extra::jgpt_extra_tls().ce.targets_bytes = bytes_tgt;
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_loss_sum == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_loss_sum), sizeof(float)));
    }
    if (jgpt_extra::jgpt_extra_tls().ce.d_valid == nullptr) {
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_valid), sizeof(unsigned int)));
    }

    CUDA_CHECK_X(cudaMemcpyAsync(jgpt_extra::jgpt_extra_tls().ce.d_logits, plog, bytes_logits, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(jgpt_extra::jgpt_extra_tls().ce.d_targets, ptgt, bytes_tgt, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy(jgpt_extra::jgpt_extra_tls().ce.d_logits, jgpt_extra::jgpt_extra_tls().ce.d_targets, jgpt_extra::jgpt_extra_tls().ce.d_grad, grad_scale_over_total_tokens, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum,
            jgpt_extra::jgpt_extra_tls().ce.d_valid, nrows, vocab, use_fp16_softmax == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    jfloat* pgrad = env->GetFloatArrayElements(h_grad, nullptr);
    if (!pgrad) {
        jgpt_jni_loss_out_zero(env, h_loss_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgrad, jgpt_extra::jgpt_extra_tls().ce.d_grad, bytes_logits, cudaMemcpyDeviceToHost, kTensorCudaStream));

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_X(cudaMemcpyAsync(&h_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(&h_valid, jgpt_extra::jgpt_extra_tls().ce.d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream));
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
    JGPT_CUDA_GUARD_1D(n_tokens, sizeof(float), return;);
    size_t bytes_f = (size_t) n_tokens * sizeof(float);
    void* pf = jni_direct_ptr(env, float_buf, float_byte_off, (jlong) bytes_f, "copyFloatToInt targets");
    if (!pf) {
        return;
    }
    if (bytes_f > jgpt_extra::jgpt_extra_tls().ce.targets_bytes) {
        cudaFree(jgpt_extra::jgpt_extra_tls().ce.d_targets);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&jgpt_extra::jgpt_extra_tls().ce.d_targets), bytes_f));
        jgpt_extra::jgpt_extra_tls().ce.targets_bytes = bytes_f;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(jgpt_extra::jgpt_extra_tls().ce.d_targets, pf, bytes_f, cudaMemcpyHostToDevice, kTensorCudaStream));
    int* ddst = reinterpret_cast<int*>(static_cast<uintptr_t>(d_dst_int));
    launch_float_to_int32_targets(jgpt_extra::jgpt_extra_tls().ce.d_targets, ddst, n_tokens);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_layerNormGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_x, jfloatArray h_gamma, jfloatArray h_beta, jfloatArray h_out,
    jint outer, jint lastDim, jfloat eps) {
    (void) clazz;
    if (outer <= 0 || lastDim <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_MATF(outer, lastDim, return;);
    JGPT_CUDA_GUARD_1D(lastDim, sizeof(float), return;);
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
    JGPT_CUDA_GUARD_MATF(outer, lastDim, return;);
    JGPT_CUDA_GUARD_1D(lastDim, sizeof(float), return;);
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
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
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
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
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
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
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
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
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
    if (batch <= 0 || seqLen <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, seqLen, return;);
    JGPT_CUDA_GUARD_MATF(seqLen, seqLen, return;);
    const long long total_ll = static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(seqLen);
    if (total_ll > static_cast<long long>(INT_MAX)) {
        fprintf(stderr, "applyCausalMask3DGPU: total element count exceeds INT_MAX\n");
        return;
    }
    const int total = static_cast<int>(total_ll);
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
    if (d0 <= 0 || d1 <= 0 || d2 <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_VOL3_F(d0, d1, d2, return;);
    const long long total_ll = static_cast<long long>(d0) * static_cast<long long>(d1) * static_cast<long long>(d2);
    if (total_ll > static_cast<long long>(INT_MAX)) {
        fprintf(stderr, "transpose2DLastGPU: total element count exceeds INT_MAX\n");
        return;
    }
    const int total = static_cast<int>(total_ll);
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
    if (dSrc == 0 || dDst == 0 || batch <= 0 || seqLen <= 0 || dModel <= 0) {
        return;
    }
    const long long total_ll = static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel);
    if (total_ll <= 0 || total_ll > static_cast<long long>(INT_MAX)) {
        return;
    }
    const int total = static_cast<int>(total_ll);
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
    if (dSrc == 0 || dDst == 0 || batch <= 0 || numHeads <= 0 || seqLen <= 0 || dHead <= 0) {
        return;
    }
    const long long dModel_ll = static_cast<long long>(numHeads) * static_cast<long long>(dHead);
    if (dModel_ll > static_cast<long long>(INT_MAX)) {
        return;
    }
    const int dModel = static_cast<int>(dModel_ll);
    const long long total_ll =
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * dModel_ll;
    if (total_ll <= 0 || total_ll > static_cast<long long>(INT_MAX)) {
        return;
    }
    const int total = static_cast<int>(total_ll);
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
    const long long total_ll =
            static_cast<long long>(numHeads) * static_cast<long long>(seqLen) * static_cast<long long>(dHead);
    if (total_ll <= 0 || total_ll > static_cast<long long>(INT_MAX)) {
        return;
    }
    const int total = static_cast<int>(total_ll);
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
    JGPT_CUDA_GUARD_CHAIN4_F(batch, numHeads, seqLen, dHead, return;);
    const long long total_ll = static_cast<long long>(batch) * static_cast<long long>(numHeads)
            * static_cast<long long>(seqLen) * static_cast<long long>(halfPairs);
    if (total_ll <= 0 || total_ll > static_cast<long long>(INT_MAX)) {
        fprintf(stderr, "applyRoPE4DGPU: total element count exceeds INT_MAX\n");
        return;
    }
    const int total = static_cast<int>(total_ll);
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
        if (!posHost) {
            cudaFree(d_s);
            cudaFree(d_d);
            return;
        }
        JGPT_CUDA_GUARD_JSIZE(posLen, sizeof(int), env->ReleaseIntArrayElements(h_positions, posHost, JNI_ABORT);
                cudaFree(d_s);
                cudaFree(d_d);
                return;);
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
    if (dSrc == 0 || dDst == 0 || dHead % 2 != 0 || batch <= 0 || numHeads <= 0 || seqLen <= 0) {
        return;
    }
    int halfPairs = dHead / 2;
    const long long total_ll = static_cast<long long>(batch) * static_cast<long long>(numHeads)
            * static_cast<long long>(seqLen) * static_cast<long long>(halfPairs);
    if (total_ll <= 0 || total_ll > static_cast<long long>(INT_MAX)) {
        return;
    }
    const int total = static_cast<int>(total_ll);
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
        JGPT_CUDA_GUARD_JSIZE(posLen, sizeof(int), env->ReleaseIntArrayElements(h_positions, posHost, JNI_ABORT);
                return;);
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
    JGPT_CUDA_GUARD_CHAIN4_F(batch, numHeads, seqLen, dHead, return;);
    const long long total_ll = static_cast<long long>(batch) * static_cast<long long>(numHeads)
            * static_cast<long long>(seqLen) * static_cast<long long>(halfPairs);
    if (total_ll <= 0 || total_ll > static_cast<long long>(INT_MAX)) {
        fprintf(stderr, "applyRoPEBackward4DGPU: total element count exceeds INT_MAX\n");
        return;
    }
    const int total = static_cast<int>(total_ll);
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
        if (!posHost) {
            cudaFree(d_gy);
            cudaFree(d_gx);
            return;
        }
        JGPT_CUDA_GUARD_JSIZE(posLen, sizeof(int), env->ReleaseIntArrayElements(h_positions, posHost, JNI_ABORT);
                cudaFree(d_gy);
                cudaFree(d_gx);
                return;);
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
    JGPT_CUDA_GUARD_CHAIN4_F(batch, numHeads, seqLen, dHead, return;);
    const long long total_ll = static_cast<long long>(batch) * static_cast<long long>(numHeads)
            * static_cast<long long>(seqLen) * static_cast<long long>(halfPairs);
    if (total_ll <= 0 || total_ll > static_cast<long long>(INT_MAX)) {
        return;
    }
    const int total = static_cast<int>(total_ll);
    const float* gradY = reinterpret_cast<const float*>(static_cast<uintptr_t>(dGradY));
    float* gradX = reinterpret_cast<float*>(static_cast<uintptr_t>(dGradX));
    jgpt_cuda_ensure_stream();
    const size_t gxBytes = (size_t) batch * (size_t) numHeads * (size_t) seqLen * (size_t) dHead * sizeof(float);
    CUDA_CHECK_X(cudaMemsetAsync(gradX, 0, gxBytes, kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    JGPT_CUDA_GUARD_MATF(vocabSize, dModel, return;);
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    JGPT_CUDA_GUARD_MATF(vocabSize, dModel, return;);
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
        return;
    }
    size_t outBytes = (size_t) batch * (size_t) seqLen * (size_t) dModel * sizeof(float);
    float *d_tok = nullptr, *d_out = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tok_bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_out), outBytes));
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, tok_host, tok_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
        return;
    }
    float* d_tok = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_tok), tok_bytes));
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, tok_host, tok_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    jgpt_cuda_ensure_stream();
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
        return;
    }
    JGPT_CUDA_GUARD_MATF(vocabSize, dModel, return;);
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
        return;
    }
    JGPT_CUDA_GUARD_MATF(seqLen, dModel, return;);
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_MATF(batch, seqLen, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
        return;
    }
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
        return;
    }
    jgpt_cuda_ensure_stream();
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
    JGPT_CUDA_GUARD_VOL3_F(batch, seqLen, dModel, return;);
    if (!jgpt_bsd_product_fits_int(batch, seqLen, dModel)) {
        fprintf(stderr, "TensorOpsGPU JNI: batch*seqLen*dModel exceeds INT_MAX\n");
        return;
    }
    JGPT_CUDA_GUARD_MATF(seqLen, dModel, return;);
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
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
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
