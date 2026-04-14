/* JNI optim & backward: sumSquares helper, AdamW, backward ops, accumulate
 * Include only from jgpt_cuda_extra.cu inside extern "C" { }.
 */

#include "jgpt_cuda_jni_raii.cuh"


/**
 * Сумма квадратов float на device: по чанкам float→double, затем cublasDdot (double).
 * И float cublasSdot, и большой чанк с «тяжёлыми» элементами давали Inf при промежуточном float.
 */
static double sum_squares_float_device_chunked_double_acc(cublasHandle_t h, const float* x, int n) {
    if (n <= 0 || x == nullptr) {
        return 0.0;
    }
    if (h == nullptr) {
        return 0.0;
    }
    const int chunk = 1 << 18;
    JGPT_CUDA_GUARD_1D(chunk, sizeof(double), return 0.0;);
    double* d_dbl = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_dbl), (size_t) chunk * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "sum_squares_float_device_chunked_double_acc: cudaMalloc double scratch failed\n");
        return 0.0;
    }
    double acc = 0.0;
    int threads = jgpt_cuda_get_optimal_block_size();
    for (int off = 0; off < n; off += chunk) {
        int m = n - off;
        if (m > chunk) {
            m = chunk;
        }
        int blocks = (m + threads - 1) / threads;
        float_to_double_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(x + off, d_dbl, m);
        double part = 0.0;
        cublasStatus_t st = cublasDdot(h, m, d_dbl, 1, d_dbl, 1, &part);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "sum_squares_float_device_chunked_double_acc: cublasDdot failed (status %d)\n", (int) st);
            cudaFree(d_dbl);
            return 0.0;
        }
        acc += part;
    }
    cudaFree(d_dbl);
    return acc;
}

JNIEXPORT jdouble JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sumSquaresGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jint n) {
    (void) clazz;
    if (n <= 0) {
        return 0.0;
    }
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return 0.0;);
    jgpt_cuda_ensure_stream();
    size_t bytes = (size_t) n * sizeof(float);
    float* d_src = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_src), bytes);
    if (e1 != cudaSuccess) {
        cudaFree(d_src);
        return 0.0;
    }
    JniFloatArrayScope src(env, h_src, JNI_ABORT);
    if (!src) {
        cudaFree(d_src);
        return 0.0;
    }
    if (cudaMemcpyAsync(d_src, src.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream) != cudaSuccess) {
        cudaFree(d_src);
        return 0.0;
    }
    if (cudaStreamSynchronize(kTensorCudaStream) != cudaSuccess) {
        cudaFree(d_src);
        return 0.0;
    }
    cublasHandle_t h = get_extra_cublas_handle();
    if (h == nullptr) {
        cudaFree(d_src);
        return 0.0;
    }
    double acc = sum_squares_float_device_chunked_double_acc(h, d_src, n);
    if (cudaStreamSynchronize(kTensorCudaStream) != cudaSuccess) {
        cudaFree(d_src);
        return 0.0;
    }
    cudaFree(d_src);
    return static_cast<jdouble>(acc);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_scaleInPlaceGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_src, jint n, jfloat scalar) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);
    float* d_src = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_src), bytes));
    JniFloatArrayScope src_in(env, h_src, JNI_ABORT);
    if (!src_in) {
        cudaFree(d_src);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_src, src_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    scale_inplace_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, n, scalar);
    JniFloatArrayScope src_out(env, h_src, 0);
    if (!src_out) {
        cudaFree(d_src);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(src_out.ptr, d_src, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    cudaFree(d_src);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_param, jfloatArray h_grad, jfloatArray h_m, jfloatArray h_v,
    jfloat learningRate, jfloat beta1, jfloat beta2, jfloat epsilon, jfloat weightDecay,
    jfloat invBias1, jfloat invBias2, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    if (!adamw_pool_ensure(bytes)) {
        fprintf(stderr, "adamWStepGPU: adamw_pool_ensure(%zu bytes) failed\n", bytes);
        return;
    }
    float* d_param = jgpt_extra::jgpt_extra_tls().adamw.d_param;
    float* d_grad = jgpt_extra::jgpt_extra_tls().adamw.d_grad;
    float* d_m = jgpt_extra::jgpt_extra_tls().adamw.d_m;
    float* d_v = jgpt_extra::jgpt_extra_tls().adamw.d_v;
    JniFloatArrayScope param_in(env, h_param, JNI_ABORT);
    JniFloatArrayScope grad_in(env, h_grad, JNI_ABORT);
    JniFloatArrayScope m_in(env, h_m, JNI_ABORT);
    JniFloatArrayScope v_in(env, h_v, JNI_ABORT);
    if (!param_in || !grad_in || !m_in || !v_in) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_param, param_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_grad, grad_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_m, m_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_v, v_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    adamw_step_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
        d_param, d_grad, d_m, d_v, learningRate, beta1, beta2, epsilon, weightDecay, invBias1, invBias2, n);
    JniFloatArrayScope param_out(env, h_param, 0);
    JniFloatArrayScope m_out(env, h_m, 0);
    JniFloatArrayScope v_out(env, h_v, 0);
    if (!param_out || !m_out || !v_out) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(param_out.ptr, d_param, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(m_out.ptr, d_m, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(v_out.ptr, d_v, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepFusedGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_param, jfloatArray h_grad, jfloatArray h_m, jfloatArray h_v,
    jfloat learningRate, jfloat beta1, jfloat beta2, jfloat epsilon, jfloat weightDecay,
    jfloat invBias1, jfloat invBias2, jint n) {
    // "Fused" = one JNI + one contiguous host buffer after Java stitching; same adamw kernel as adamWStepGPU.
    // Multi-buffer device fusion without host packing is adamWStepGPUDeviceFused.
    Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepGPU(
        env,
        clazz,
        h_param,
        h_grad,
        h_m,
        h_v,
        learningRate,
        beta1,
        beta2,
        epsilon,
        weightDecay,
        invBias1,
        invBias2,
        n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepGPUDevice(
    JNIEnv* env,
    jclass clazz,
    jlong dParam,
    jlong dGrad,
    jlong dM,
    jlong dV,
    jfloat learningRate,
    jfloat beta1,
    jfloat beta2,
    jfloat epsilon,
    jfloat weightDecay,
    jfloat invBias1,
    jfloat invBias2,
    jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dParam == 0 || dGrad == 0 || dM == 0 || dV == 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    float* pParam = reinterpret_cast<float*>(static_cast<uintptr_t>(dParam));
    const float* pGrad = reinterpret_cast<const float*>(static_cast<uintptr_t>(dGrad));
    float* pM = reinterpret_cast<float*>(static_cast<uintptr_t>(dM));
    float* pV = reinterpret_cast<float*>(static_cast<uintptr_t>(dV));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    adamw_step_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
        pParam, pGrad, pM, pV, learningRate, beta1, beta2, epsilon, weightDecay, invBias1, invBias2, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "adamWStepGPUDevice: %s\n", cudaGetErrorString(err));
    }
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_adamWStepGPUDeviceFused(
    JNIEnv* env,
    jclass clazz,
    jlongArray h_param_ptrs,
    jlongArray h_grad_ptrs,
    jlongArray h_m_ptrs,
    jlongArray h_v_ptrs,
    jintArray h_lengths,
    jfloat learningRate,
    jfloat beta1,
    jfloat beta2,
    jfloat epsilon,
    jfloat weightDecay,
    jfloat invBias1,
    jfloat invBias2) {
    (void) clazz;
    if (h_param_ptrs == nullptr || h_grad_ptrs == nullptr || h_m_ptrs == nullptr || h_v_ptrs == nullptr
        || h_lengths == nullptr) {
        return;
    }
    jsize num_segments = env->GetArrayLength(h_param_ptrs);
    if (num_segments <= 0) {
        return;
    }
    if (env->GetArrayLength(h_grad_ptrs) != num_segments || env->GetArrayLength(h_m_ptrs) != num_segments
        || env->GetArrayLength(h_v_ptrs) != num_segments || env->GetArrayLength(h_lengths) != num_segments) {
        return;
    }
    JniLongArrayScope pj(env, h_param_ptrs, JNI_ABORT);
    JniLongArrayScope gj(env, h_grad_ptrs, JNI_ABORT);
    JniLongArrayScope mj(env, h_m_ptrs, JNI_ABORT);
    JniLongArrayScope vj(env, h_v_ptrs, JNI_ABORT);
    JniIntArrayScope lens(env, h_lengths, JNI_ABORT);
    if (!pj || !gj || !mj || !vj || !lens) {
        return;
    }
    for (jsize i = 0; i < num_segments; i++) {
        if (lens.ptr[i] <= 0 || pj.ptr[i] == 0 || gj.ptr[i] == 0 || mj.ptr[i] == 0 || vj.ptr[i] == 0) {
            return;
        }
    }

    if (num_segments < 0
            || jgpt_alloc_n_elem_overflows(static_cast<size_t>(num_segments), sizeof(uintptr_t))
            || jgpt_alloc_n_elem_overflows(static_cast<size_t>(num_segments), sizeof(int))) {
        fprintf(stderr, "adamWStepGPUDeviceFused: alloc size overflow %s:%d\n", __FILE__, __LINE__);
        return;
    }
    size_t ptr_bytes = (size_t) num_segments * sizeof(uintptr_t);
    size_t len_bytes = (size_t) num_segments * sizeof(int);
    uintptr_t* h_pp = (uintptr_t*) malloc(ptr_bytes);
    uintptr_t* h_gp = (uintptr_t*) malloc(ptr_bytes);
    uintptr_t* h_mp = (uintptr_t*) malloc(ptr_bytes);
    uintptr_t* h_vp = (uintptr_t*) malloc(ptr_bytes);
    if (!h_pp || !h_gp || !h_mp || !h_vp) {
        free(h_pp);
        free(h_gp);
        free(h_mp);
        free(h_vp);
        return;
    }
    for (jsize i = 0; i < num_segments; i++) {
        h_pp[i] = static_cast<uintptr_t>(pj.ptr[i]);
        h_gp[i] = static_cast<uintptr_t>(gj.ptr[i]);
        h_mp[i] = static_cast<uintptr_t>(mj.ptr[i]);
        h_vp[i] = static_cast<uintptr_t>(vj.ptr[i]);
    }

    uintptr_t *d_pp = nullptr, *d_gp = nullptr, *d_mp = nullptr, *d_vp = nullptr;
    int* d_lens = nullptr;
    cudaError_t alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_pp), ptr_bytes);
    if (alloc_err == cudaSuccess) {
        alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_gp), ptr_bytes);
    }
    if (alloc_err == cudaSuccess) {
        alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_mp), ptr_bytes);
    }
    if (alloc_err == cudaSuccess) {
        alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_vp), ptr_bytes);
    }
    if (alloc_err == cudaSuccess) {
        alloc_err = cudaMalloc(reinterpret_cast<void**>(&d_lens), len_bytes);
    }
    if (alloc_err != cudaSuccess) {
        cudaFree(d_pp);
        cudaFree(d_gp);
        cudaFree(d_mp);
        cudaFree(d_vp);
        cudaFree(d_lens);
        free(h_pp);
        free(h_gp);
        free(h_mp);
        free(h_vp);
        return;
    }
    jgpt_cuda_ensure_stream();
    CUDA_CHECK_X(cudaMemcpyAsync(d_pp, h_pp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gp, h_gp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_mp, h_mp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_vp, h_vp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_lens, lens.ptr, len_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    free(h_pp);
    free(h_gp);
    free(h_mp);
    free(h_vp);

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (int) num_segments;
    adamw_step_kernel_segments<<<blocks, threads, 0, kTensorCudaStream>>>(
        d_pp,
        d_gp,
        d_mp,
        d_vp,
        d_lens,
        (int) num_segments,
        learningRate,
        beta1,
        beta2,
        epsilon,
        weightDecay,
        invBias1,
        invBias2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "adamWStepGPUDeviceFused: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_pp);
    cudaFree(d_gp);
    cudaFree(d_mp);
    cudaFree(d_vp);
    cudaFree(d_lens);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_softmaxLastDimBackward3DGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_probs, jfloatArray h_gIn, jint batch,
    jint mid, jint inner) {
    (void) clazz;
    if (batch <= 0 || mid <= 0 || inner <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_VOL3_F(batch, mid, inner, return;);
    const long long nrows_ll = static_cast<long long>(batch) * static_cast<long long>(mid);
    if (nrows_ll <= 0) {
        return;
    }
    size_t bytes = static_cast<size_t>(nrows_ll) * static_cast<size_t>(inner) * sizeof(float);
    float *d_go = nullptr, *d_p = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_p), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    JniFloatArrayScope go_in(env, h_gOut, JNI_ABORT);
    JniFloatArrayScope p_in(env, h_probs, JNI_ABORT);
    JniFloatArrayScope gi_in(env, h_gIn, JNI_ABORT);
    if (!go_in || !p_in || !gi_in) {
        cudaFree(d_go);
        cudaFree(d_p);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, go_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_p, p_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, gi_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    launch_softmax_last_dim_bwd(d_go, d_p, d_gi, nrows_ll, inner);
    JniFloatArrayScope gi_out(env, h_gIn, 0);
    if (!gi_out) {
        cudaFree(d_go);
        cudaFree(d_p);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(gi_out.ptr, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    cudaFree(d_go);
    cudaFree(d_p);
    cudaFree(d_gi);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_layerNormBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_x, jfloatArray h_gamma, jfloat eps,
    jfloatArray h_gX, jfloatArray h_gGamma, jfloatArray h_gBeta, jint outer, jint lastDim) {
    (void) clazz;
    if (outer <= 0 || lastDim <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_MATF(outer, lastDim, return;);
    JGPT_CUDA_GUARD_1D(lastDim, sizeof(float), return;);
    size_t bytes_xy = (size_t) outer * (size_t) lastDim * sizeof(float);
    size_t bytes_g = (size_t) lastDim * sizeof(float);
    float *d_go = nullptr, *d_x = nullptr, *d_g = nullptr, *d_gx = nullptr, *d_gg = nullptr, *d_gb = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_g), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gx), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gg), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gb), bytes_g));

    JniFloatArrayScope go_in(env, h_gOut, JNI_ABORT);
    JniFloatArrayScope x_in(env, h_x, JNI_ABORT);
    JniFloatArrayScope g_in(env, h_gamma, JNI_ABORT);
    JniFloatArrayScope gx_in(env, h_gX, JNI_ABORT);
    JniFloatArrayScope gg_in(env, h_gGamma, JNI_ABORT);
    JniFloatArrayScope gb_in(env, h_gBeta, JNI_ABORT);
    if (!go_in || !x_in || !g_in || !gx_in || !gg_in || !gb_in) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, go_in.ptr, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, x_in.ptr, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, g_in.ptr, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gx, gx_in.ptr, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gg, gg_in.ptr, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gb, gb_in.ptr, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (outer + threads - 1) / threads;
    layer_norm_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_x, d_g, eps, d_gx, d_gg, d_gb, outer, lastDim);
    CUDA_KERNEL_CHECK();
    JniFloatArrayScope gx_out(env, h_gX, 0);
    JniFloatArrayScope gg_out(env, h_gGamma, 0);
    JniFloatArrayScope gb_out(env, h_gBeta, 0);
    if (!gx_out || !gg_out || !gb_out) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(gx_out.ptr, d_gx, bytes_xy, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(gg_out.ptr, d_gg, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(gb_out.ptr, d_gb, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    cudaFree(d_go);
    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_gx);
    cudaFree(d_gg);
    cudaFree(d_gb);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_rmsNormBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_x, jfloatArray h_gamma, jfloat eps,
    jfloatArray h_gX, jfloatArray h_gGamma, jint outer, jint lastDim) {
    (void) clazz;
    if (outer <= 0 || lastDim <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_MATF(outer, lastDim, return;);
    JGPT_CUDA_GUARD_1D(lastDim, sizeof(float), return;);
    size_t bytes_xy = (size_t) outer * (size_t) lastDim * sizeof(float);
    size_t bytes_g = (size_t) lastDim * sizeof(float);
    float *d_go = nullptr, *d_x = nullptr, *d_g = nullptr, *d_gx = nullptr, *d_gg = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_g), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gx), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gg), bytes_g));

    JniFloatArrayScope go_in(env, h_gOut, JNI_ABORT);
    JniFloatArrayScope x_in(env, h_x, JNI_ABORT);
    JniFloatArrayScope g_in(env, h_gamma, JNI_ABORT);
    JniFloatArrayScope gx_in(env, h_gX, JNI_ABORT);
    JniFloatArrayScope gg_in(env, h_gGamma, JNI_ABORT);
    if (!go_in || !x_in || !g_in || !gx_in || !gg_in) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, go_in.ptr, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, x_in.ptr, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, g_in.ptr, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gx, gx_in.ptr, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gg, gg_in.ptr, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_bwd(d_go, d_x, d_g, eps, d_gx, d_gg, outer, lastDim);
    JniFloatArrayScope gx_out(env, h_gX, 0);
    JniFloatArrayScope gg_out(env, h_gGamma, 0);
    if (!gx_out || !gg_out) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(gx_out.ptr, d_gx, bytes_xy, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(gg_out.ptr, d_gg, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    cudaFree(d_go);
    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_gx);
    cudaFree(d_gg);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_rmsNormBackwardGPUDevice(
    JNIEnv* env, jclass clazz, jlong dGOut, jlong dX, jlong dGamma, jfloat eps, jlong dGX, jlong dGGamma,
    jint outer, jint lastDim) {
    (void) env;
    (void) clazz;
    if (outer <= 0 || lastDim <= 0 || dGOut == 0 || dX == 0 || dGamma == 0 || dGX == 0 || dGGamma == 0) {
        return;
    }
    float* go = reinterpret_cast<float*>(static_cast<uintptr_t>(dGOut));
    float* x = reinterpret_cast<float*>(static_cast<uintptr_t>(dX));
    float* gamma = reinterpret_cast<float*>(static_cast<uintptr_t>(dGamma));
    float* gx = reinterpret_cast<float*>(static_cast<uintptr_t>(dGX));
    float* gg = reinterpret_cast<float*>(static_cast<uintptr_t>(dGGamma));
    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_bwd(go, x, gamma, eps, gx, gg, outer, lastDim);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_multiplyBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_a, jfloatArray h_b, jfloatArray h_gA,
    jfloatArray h_gB, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_a = nullptr, *d_b = nullptr, *d_ga = nullptr, *d_gb = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_ga), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gb), bytes));

    JniFloatArrayScope go_in(env, h_gOut, JNI_ABORT);
    JniFloatArrayScope a_in(env, h_a, JNI_ABORT);
    JniFloatArrayScope b_in(env, h_b, JNI_ABORT);
    JniFloatArrayScope ga_in(env, h_gA, JNI_ABORT);
    JniFloatArrayScope gb_in(env, h_gB, JNI_ABORT);
    if (!go_in || !a_in || !b_in || !ga_in || !gb_in) {
        cudaFree(d_go);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_ga);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, go_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, a_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_b, b_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_ga, ga_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gb, gb_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    multiply_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_a, d_b, d_ga, d_gb, n);
    JniFloatArrayScope ga_out(env, h_gA, 0);
    JniFloatArrayScope gb_out(env, h_gB, 0);
    if (!ga_out || !gb_out) {
        cudaFree(d_go);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_ga);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(ga_out.ptr, d_ga, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(gb_out.ptr, d_gb, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));

    cudaFree(d_go);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ga);
    cudaFree(d_gb);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_multiplyBackwardGPUDevice(
    JNIEnv* env, jclass clazz, jlong dGOut, jlong dA, jlong dB, jlong dGA, jlong dGB, jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dGOut == 0 || dA == 0 || dB == 0 || dGA == 0 || dGB == 0) {
        return;
    }
    float* go = reinterpret_cast<float*>(static_cast<uintptr_t>(dGOut));
    float* a = reinterpret_cast<float*>(static_cast<uintptr_t>(dA));
    float* b = reinterpret_cast<float*>(static_cast<uintptr_t>(dB));
    float* ga = reinterpret_cast<float*>(static_cast<uintptr_t>(dGA));
    float* gb = reinterpret_cast<float*>(static_cast<uintptr_t>(dGB));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    multiply_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(go, a, b, ga, gb, n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_geluBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_inp, jfloatArray h_gIn, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    JniFloatArrayScope go_in(env, h_gOut, JNI_ABORT);
    JniFloatArrayScope in_in(env, h_inp, JNI_ABORT);
    JniFloatArrayScope gi_in(env, h_gIn, JNI_ABORT);
    if (!go_in || !in_in || !gi_in) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, go_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, in_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, gi_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    gelu_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    JniFloatArrayScope gi_out(env, h_gIn, 0);
    if (!gi_out) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(gi_out.ptr, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    cudaFree(d_go);
    cudaFree(d_in);
    cudaFree(d_gi);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sigmoidBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_inp, jfloatArray h_gIn, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    JniFloatArrayScope go_in(env, h_gOut, JNI_ABORT);
    JniFloatArrayScope in_in(env, h_inp, JNI_ABORT);
    JniFloatArrayScope gi_in(env, h_gIn, JNI_ABORT);
    if (!go_in || !in_in || !gi_in) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, go_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, in_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, gi_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    JniFloatArrayScope gi_out(env, h_gIn, 0);
    if (!gi_out) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(gi_out.ptr, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    cudaFree(d_go);
    cudaFree(d_in);
    cudaFree(d_gi);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_sigmoidBackwardGPUDevice(
    JNIEnv* env, jclass clazz, jlong dGOut, jlong dInp, jlong dGIn, jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dGOut == 0 || dInp == 0 || dGIn == 0) {
        return;
    }
    float* go = reinterpret_cast<float*>(static_cast<uintptr_t>(dGOut));
    float* inp = reinterpret_cast<float*>(static_cast<uintptr_t>(dInp));
    float* gi = reinterpret_cast<float*>(static_cast<uintptr_t>(dGIn));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(go, inp, gi, n);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_reluBackwardGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_gOut, jfloatArray h_inp, jfloatArray h_gIn, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    JniFloatArrayScope go_in(env, h_gOut, JNI_ABORT);
    JniFloatArrayScope in_in(env, h_inp, JNI_ABORT);
    JniFloatArrayScope gi_in(env, h_gIn, JNI_ABORT);
    if (!go_in || !in_in || !gi_in) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, go_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, in_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, gi_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    relu_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    JniFloatArrayScope gi_out(env, h_gIn, 0);
    if (!gi_out) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(gi_out.ptr, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    cudaFree(d_go);
    cudaFree(d_in);
    cudaFree(d_gi);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_accumulateAddGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_acc, jfloatArray h_delta, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_d = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_d), bytes));
    JniFloatArrayScope a_in(env, h_acc, JNI_ABORT);
    JniFloatArrayScope d_in(env, h_delta, JNI_ABORT);
    if (!a_in || !d_in) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, a_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_d, d_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    accumulate_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_d, n);
    JniFloatArrayScope a_out(env, h_acc, 0);
    if (!a_out) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(a_out.ptr, d_a, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    cudaFree(d_a);
    cudaFree(d_d);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_accumulateScaledAddGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_acc, jfloatArray h_delta, jfloat scale, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    JGPT_CUDA_GUARD_1D(n, sizeof(float), return;);
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_d = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_d), bytes));
    JniFloatArrayScope a_in(env, h_acc, JNI_ABORT);
    JniFloatArrayScope d_in(env, h_delta, JNI_ABORT);
    if (!a_in || !d_in) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, a_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_d, d_in.ptr, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    accumulate_scaled_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_d, scale, n);
    JniFloatArrayScope a_out(env, h_acc, 0);
    if (!a_out) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(a_out.ptr, d_a, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    cudaFree(d_a);
    cudaFree(d_d);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_accumulateAddGPUDevice(
    JNIEnv* env, jclass clazz, jlong dAcc, jlong dDelta, jint n) {
    (void) env;
    (void) clazz;
    if (n <= 0 || dAcc == 0 || dDelta == 0) {
        return;
    }
    float* acc = reinterpret_cast<float*>(static_cast<uintptr_t>(dAcc));
    float* delta = reinterpret_cast<float*>(static_cast<uintptr_t>(dDelta));
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    accumulate_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(acc, delta, n);
}
