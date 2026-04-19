/* JNI host forward: Host arrays: softmax, CE, norms, activations, RoPE, embedding, pos-embed
 * Include only from jgpt_cuda_extra.cu inside extern "C" { }.
 */

#include "jgpt_cuda_jni_raii.cuh"

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

    JniFloatArrayScope p_src_raii1(env, h_src, JNI_ABORT);
    if (!p_src_raii1) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_src, p_src_raii1.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    /* No sync: kernel launch on same stream is ordered after H2D automatically. */

    launch_softmax_last_dim(d_src, d_dst, nrows_ll, inner, useFp16Softmax == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    JniFloatArrayScope p_dst_raii2(env, h_dst, 0);
    if (!p_dst_raii2) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(p_dst_raii2.ptr, d_dst, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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

    JniFloatArrayScope plog_raii3(env, h_logits, JNI_ABORT);
    JniFloatArrayScope ptgt_raii4(env, h_targets, JNI_ABORT);
    if (!plog_raii3 || !ptgt_raii4) {
        jgpt_jni_loss_out_zero(env, h_lossOut);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(jgpt_extra::jgpt_extra_tls().ce.d_logits, plog_raii3.ptr, bytes_logits, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(jgpt_extra::jgpt_extra_tls().ce.d_targets, ptgt_raii4.ptr, bytes_tgt, cudaMemcpyHostToDevice, kTensorCudaStream));
    /* No sync: kernel launch on same stream is ordered after H2D automatically. */

    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy(jgpt_extra::jgpt_extra_tls().ce.d_logits, jgpt_extra::jgpt_extra_tls().ce.d_targets, jgpt_extra::jgpt_extra_tls().ce.d_grad, gradScaleOverTotalTokens, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum,
            jgpt_extra::jgpt_extra_tls().ce.d_valid, nrows, vocab, useFp16Softmax == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    JniFloatArrayScope pgrad_raii5(env, h_grad, 0);
    if (!pgrad_raii5) {
        jgpt_jni_loss_out_zero(env, h_lossOut);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgrad_raii5.ptr, jgpt_extra::jgpt_extra_tls().ce.d_grad, bytes_logits, cudaMemcpyDeviceToHost, kTensorCudaStream));

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_X(cudaMemcpyAsync(&h_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(&h_valid, jgpt_extra::jgpt_extra_tls().ce.d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));


    JniFloatArrayScope plo_raii6(env, h_lossOut, 0);
    if (plo_raii6.ptr) {
        plo_raii6.ptr[0] = (h_valid == 0U) ? 0.f : (h_loss_sum / (float)h_valid);
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
    /* No sync: kernel launch on same stream is ordered after H2D automatically. */

    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, 0, sizeof(float), kTensorCudaStream));
    CUDA_CHECK_X(cudaMemsetAsync(jgpt_extra::jgpt_extra_tls().ce.d_valid, 0, sizeof(unsigned int), kTensorCudaStream));

    launch_cross_entropy(jgpt_extra::jgpt_extra_tls().ce.d_logits, jgpt_extra::jgpt_extra_tls().ce.d_targets, jgpt_extra::jgpt_extra_tls().ce.d_grad, grad_scale_over_total_tokens, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum,
            jgpt_extra::jgpt_extra_tls().ce.d_valid, nrows, vocab, use_fp16_softmax == JNI_TRUE);
    CUDA_KERNEL_CHECK();

    JniFloatArrayScope pgrad_raii7(env, h_grad, 0);
    if (!pgrad_raii7) {
        jgpt_jni_loss_out_zero(env, h_loss_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgrad_raii7.ptr, jgpt_extra::jgpt_extra_tls().ce.d_grad, bytes_logits, cudaMemcpyDeviceToHost, kTensorCudaStream));

    float h_loss_sum = 0.f;
    unsigned int h_valid = 0;
    CUDA_CHECK_X(cudaMemcpyAsync(&h_loss_sum, jgpt_extra::jgpt_extra_tls().ce.d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(&h_valid, jgpt_extra::jgpt_extra_tls().ce.d_valid, sizeof(unsigned int), cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));


    JniFloatArrayScope plo_raii8(env, h_loss_out, 0);
    if (plo_raii8.ptr) {
        plo_raii8.ptr[0] = (h_valid == 0U) ? 0.f : (h_loss_sum / (float) h_valid);
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

    JniFloatArrayScope px_raii9(env, h_x, JNI_ABORT);
    JniFloatArrayScope pg_raii10(env, h_gamma, JNI_ABORT);
    JniFloatArrayScope pb_raii11(env, h_beta, JNI_ABORT);
    if (!px_raii9 || !pg_raii10 || !pb_raii11) {
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_b);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px_raii9.ptr, bytes_x, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, pg_raii10.ptr, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_b, pb_raii11.ptr, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    /* No sync: kernel launch on same stream is ordered after H2D automatically. */

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (outer + threads - 1) / threads;
    layer_norm_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_x, d_g, d_b, d_o, outer, lastDim, eps);
    CUDA_KERNEL_CHECK();
    JniFloatArrayScope po_raii12(env, h_out, 0);
    if (!po_raii12) {
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_b);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po_raii12.ptr, d_o, bytes_x, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

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

    JniFloatArrayScope px_raii13(env, h_x, JNI_ABORT);
    JniFloatArrayScope pg_raii14(env, h_gamma, JNI_ABORT);
    if (!px_raii13 || !pg_raii14) {
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px_raii13.ptr, bytes_x, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, pg_raii14.ptr, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    /* No sync: kernel launch on same stream is ordered after H2D automatically. */

    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_fwd(d_x, d_g, d_o, outer, lastDim, eps);
    JniFloatArrayScope po_raii15(env, h_out, 0);
    if (!po_raii15) {
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po_raii15.ptr, d_o, bytes_x, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

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
    (void) dNormOut;  // Ignored - always use cuBLAS path with cached buffer
    if (dX == 0 || dGamma == 0 || dW == 0 || dLogits == 0) {
        return;
    }
    if (rows <= 0 || dModel <= 0 || vocab <= 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    const float* x = reinterpret_cast<const float*>(static_cast<uintptr_t>(dX));
    const float* gamma = reinterpret_cast<const float*>(static_cast<uintptr_t>(dGamma));
    const float* w = reinterpret_cast<const float*>(static_cast<uintptr_t>(dW));
    float* logits = reinterpret_cast<float*>(static_cast<uintptr_t>(dLogits));

    // Always use cuBLAS path: RMSNorm → cached buffer → cuBLAS GEMM
    size_t normOutBytes = (size_t) rows * (size_t) dModel * sizeof(float);
    float* normOut = jgpt_extra::jgpt_extra_tls().lm_head_norm.ensure(normOutBytes);
    if (normOut == nullptr) {
        fprintf(stderr, "rmsNormMatmulLmHeadGPUDevice: failed to allocate temp buffer (%zu bytes)\n", normOutBytes);
        return;
    }
    
    launch_rms_norm_fwd(x, gamma, normOut, rows, dModel, eps);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = get_extra_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "rmsNormMatmulLmHeadGPUDevice: cuBLAS handle unavailable\n");
        return;
    }
    cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, vocab, rows, dModel, &alpha, w, vocab, normOut, dModel, &beta, logits, vocab);
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
    JniFloatArrayScope pa_raii16(env, h_src, JNI_ABORT);
    if (!pa_raii16) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa_raii16.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    /* No sync: kernel launch on same stream is ordered after H2D automatically. */
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, n);
    JniFloatArrayScope pb_raii17(env, h_dst, 0);
    if (!pb_raii17) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pb_raii17.ptr, d_b, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope pa_raii18(env, h_src, JNI_ABORT);
    if (!pa_raii18) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa_raii18.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    /* No sync: kernel launch on same stream is ordered after H2D automatically. */
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, n);
    JniFloatArrayScope pb_raii19(env, h_dst, 0);
    if (!pb_raii19) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pb_raii19.ptr, d_b, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope pa_raii20(env, h_a, JNI_ABORT);
    JniFloatArrayScope pb_raii21(env, h_b, JNI_ABORT);
    if (!pa_raii20 || !pb_raii21) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa_raii20.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_b, pb_raii21.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, d_c, n);
    JniFloatArrayScope pc_raii22(env, h_c, 0);
    if (!pc_raii22) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pc_raii22.ptr, d_c, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope pa_raii23(env, h_a, JNI_ABORT);
    if (!pa_raii23) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa_raii23.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    /* No sync: kernel launch on same stream is ordered after H2D automatically. */
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    mul_scalar_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, n, scalar);
    JniFloatArrayScope pb_raii24(env, h_b, 0);
    if (!pb_raii24) {
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pb_raii24.ptr, d_b, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ps_raii25(env, h_scores, JNI_ABORT);
    JniFloatArrayScope pm_raii26(env, h_mask, JNI_ABORT);
    if (!ps_raii25 || !pm_raii26) {
        cudaFree(d_s);
        cudaFree(d_m);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_s, ps_raii25.ptr, bytes_s, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_m, pm_raii26.ptr, bytes_m, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    apply_causal_mask_3d_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_s, d_m, d_o, batch, seqLen);
    JniFloatArrayScope po_raii27(env, h_out, 0);
    if (!po_raii27) {
        cudaFree(d_s);
        cudaFree(d_m);
        cudaFree(d_o);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po_raii27.ptr, d_o, bytes_s, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ps_raii28(env, h_src, JNI_ABORT);
    if (!ps_raii28) {
        cudaFree(d_s);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_s, ps_raii28.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    transpose_2d_last_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_s, d_d, d0, d1, d2);
    JniFloatArrayScope pd_raii29(env, h_dst, 0);
    if (!pd_raii29) {
        cudaFree(d_s);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pd_raii29.ptr, d_d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ps_raii30(env, h_src, JNI_ABORT);
    if (!ps_raii30) {
        cudaFree(d_s);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_s, ps_raii30.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    int posLen = seqLen;
    JniIntArrayScope posHost_raii70(env, h_positions, JNI_ABORT);
    if (h_positions != nullptr) {
        if (!posHost_raii70) {
            cudaFree(d_s);
            cudaFree(d_d);
            return;
        }
        posLen = env->GetArrayLength(h_positions);
        JGPT_CUDA_GUARD_JSIZE(posLen, sizeof(int),                cudaFree(d_s);
                cudaFree(d_d);
                return;);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_pos), (size_t) posLen * sizeof(int)));
        CUDA_CHECK_X(cudaMemcpyAsync(d_pos, posHost_raii70.ptr, (size_t) posLen * sizeof(int), cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    }

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    rope_4d_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_s, d_d, batch, numHeads, seqLen, dHead, d_pos, posLen, posBaseOffset);
    JniFloatArrayScope pd_raii31(env, h_dst, 0);
    if (!pd_raii31) {
        cudaFree(d_s);
        cudaFree(d_d);
        if (d_pos) {
            cudaFree(d_pos);
        }
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pd_raii31.ptr, d_d, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
        JniIntArrayScope posHost_raii32(env, h_positions, JNI_ABORT);
        if (!posHost_raii32) {
            return;
        }
        JGPT_CUDA_GUARD_JSIZE(posLen, sizeof(int),                return;);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_pos), (size_t) posLen * sizeof(int)));
        CUDA_CHECK_X(cudaMemcpyAsync(d_pos, posHost_raii32.ptr, (size_t) posLen * sizeof(int), cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope pgy_raii33(env, h_gradY, JNI_ABORT);
    JniFloatArrayScope pgx_in_raii35(env, h_gradX, JNI_ABORT);
    if (!pgy_raii33 || !pgx_in_raii35) {
        cudaFree(d_gy);
        cudaFree(d_gx);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_gy, pgy_raii33.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gx, pgx_in_raii35.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    int posLen = seqLen;
    JniIntArrayScope posHost_raii85(env, h_positions, JNI_ABORT);
    if (h_positions != nullptr) {
        if (!posHost_raii85) {
            cudaFree(d_gy);
            cudaFree(d_gx);
            return;
        }
        posLen = env->GetArrayLength(h_positions);
        JGPT_CUDA_GUARD_JSIZE(posLen, sizeof(int),                cudaFree(d_gy);
                cudaFree(d_gx);
                return;);
        CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_pos), (size_t) posLen * sizeof(int)));
        CUDA_CHECK_X(cudaMemcpyAsync(d_pos, posHost_raii85.ptr, (size_t) posLen * sizeof(int), cudaMemcpyHostToDevice, kTensorCudaStream));
        CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    }

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    rope_4d_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
            d_gy, d_gx, batch, numHeads, seqLen, dHead, d_pos, posLen, posBaseOffset);
    JniFloatArrayScope pgx_out_raii35(env, h_gradX, 0);
    if (!pgx_out_raii35) {
        cudaFree(d_gy);
        cudaFree(d_gx);
        if (d_pos) {
            cudaFree(d_pos);
        }
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgx_out_raii35.ptr, d_gx, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ptok_raii36(env, h_tokens, JNI_ABORT);
    JniFloatArrayScope pw_raii37(env, h_weights, JNI_ABORT);
    if (!ptok_raii36 || !pw_raii37) {
        cudaFree(d_tok);
        cudaFree(d_w);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok_raii36.ptr, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_w, pw_raii37.ptr, wBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
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
    JniFloatArrayScope po_raii38(env, h_out, 0);
    if (!po_raii38) {
        cudaFree(d_tok);
        cudaFree(d_w);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po_raii38.ptr, d_out, outBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope pw_raii39(env, h_weights, JNI_ABORT);
    if (!pw_raii39) {
        cudaFree(d_tok);
        cudaFree(d_w);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, tok_host, tok_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_w, pw_raii39.ptr, wBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
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
    JniFloatArrayScope po_raii40(env, h_out, 0);
    if (!po_raii40) {
        cudaFree(d_tok);
        cudaFree(d_w);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po_raii40.ptr, d_out, outBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ptok_raii41(env, h_tokens, JNI_ABORT);
    if (!ptok_raii41) {
        cudaFree(d_tok);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok_raii41.ptr, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
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
    JniFloatArrayScope po_raii42(env, h_out, 0);
    if (!po_raii42) {
        cudaFree(d_tok);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po_raii42.ptr, d_out, outBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ptok_raii43(env, h_tokens, JNI_ABORT);
    if (!ptok_raii43) {
        cudaFree(d_tok);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok_raii43.ptr, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
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
    JniFloatArrayScope po_raii44(env, h_out, 0);
    if (!po_raii44) {
        cudaFree(d_tok);
        cudaFree(d_out);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(po_raii44.ptr, d_out, outBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ptok_raii45(env, h_tokens, JNI_ABORT);
    JniFloatArrayScope pgo_raii46(env, h_gradOut, JNI_ABORT);
    JniFloatArrayScope pgw_in_raii48(env, h_gradWeights, JNI_ABORT);
    if (!ptok_raii45 || !pgo_raii46 || !pgw_in_raii48) {
        cudaFree(d_tok); cudaFree(d_go); cudaFree(d_gw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok_raii45.ptr, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo_raii46.ptr, goBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gw, pgw_in_raii48.ptr, gwBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    embedding_token_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_tok, d_go, d_gw, batch, seqLen, dModel, vocabSize);
    JniFloatArrayScope pgw_out_raii48(env, h_gradWeights, 0);
    if (!pgw_out_raii48) {
        cudaFree(d_tok); cudaFree(d_go); cudaFree(d_gw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgw_out_raii48.ptr, d_gw, gwBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ptok_raii49(env, h_tokens, JNI_ABORT);
    JniFloatArrayScope pgo_raii50(env, h_gradOut, JNI_ABORT);
    if (!ptok_raii49 || !pgo_raii50) {
        cudaFree(d_tok);
        cudaFree(d_go);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok_raii49.ptr, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo_raii50.ptr, goBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope pgc_raii51(env, h_gradCombined, JNI_ABORT);
    JniFloatArrayScope pgw_in_raii53(env, h_gradWeights, JNI_ABORT);
    if (!pgc_raii51 || !pgw_in_raii53) {
        cudaFree(d_gc); cudaFree(d_gw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_gc, pgc_raii51.ptr, gcBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gw, pgw_in_raii53.ptr, gwBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    const int total = static_cast<int>(
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    embedding_position_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_gc, d_gw, batch, seqLen, dModel);
    JniFloatArrayScope pgw_out_raii53(env, h_gradWeights, 0);
    if (!pgw_out_raii53) {
        cudaFree(d_gc); cudaFree(d_gw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgw_out_raii53.ptr, d_gw, gwBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope pgc_raii54(env, h_gradCombined, JNI_ABORT);
    if (!pgc_raii54) {
        cudaFree(d_gc);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_gc, pgc_raii54.ptr, gcBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope ptok_raii55(env, h_tokens, JNI_ABORT);
    if (!ptok_raii55) {
        cudaFree(d_tok);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_tok, ptok_raii55.ptr, tokBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope px_in_raii57(env, h_x, JNI_ABORT);
    if (!px_in_raii57) {
        cudaFree(d_x);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px_in_raii57.ptr, xBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope px_out_raii57(env, h_x, 0);
    if (!px_out_raii57) {
        cudaFree(d_x);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(px_out_raii57.ptr, d_x, xBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope px_in_raii60(env, h_x, JNI_ABORT);
    JniFloatArrayScope ppos_raii59(env, h_pos_rows, JNI_ABORT);
    if (!px_in_raii60 || !ppos_raii59) {
        cudaFree(d_x);
        cudaFree(d_pw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px_in_raii60.ptr, xBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_pw, ppos_raii59.ptr, posBytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
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
    JniFloatArrayScope px_out_raii60(env, h_x, 0);
    if (!px_out_raii60) {
        cudaFree(d_x);
        cudaFree(d_pw);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(px_out_raii60.ptr, d_x, xBytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    cudaFree(d_x);
    cudaFree(d_pw);
}