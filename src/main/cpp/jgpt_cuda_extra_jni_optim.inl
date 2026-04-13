/* JNI optim & backward: sumSquares helper, AdamW, backward ops, accumulate
 * Include only from jgpt_cuda_extra.cu inside extern "C" { }.
 */


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
    jgpt_cuda_ensure_stream();
    size_t bytes = (size_t) n * sizeof(float);
    float* d_src = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(&d_src), bytes);
    if (e1 != cudaSuccess) {
        cudaFree(d_src);
        return 0.0;
    }
    jfloat* psrc = env->GetFloatArrayElements(h_src, nullptr);
    if (!psrc) {
        cudaFree(d_src);
        return 0.0;
    }
    if (cudaMemcpyAsync(d_src, psrc, bytes, cudaMemcpyHostToDevice, kTensorCudaStream) != cudaSuccess) {
        env->ReleaseFloatArrayElements(h_src, psrc, JNI_ABORT);
        cudaFree(d_src);
        return 0.0;
    }
    if (cudaStreamSynchronize(kTensorCudaStream) != cudaSuccess) {
        env->ReleaseFloatArrayElements(h_src, psrc, JNI_ABORT);
        cudaFree(d_src);
        return 0.0;
    }
    env->ReleaseFloatArrayElements(h_src, psrc, JNI_ABORT);
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
    size_t bytes = (size_t) n * sizeof(float);
    float* d_src = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_src), bytes));
    jfloat* psrc = env->GetFloatArrayElements(h_src, nullptr);
    if (!psrc) {
        cudaFree(d_src);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_src, psrc, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, psrc, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    scale_inplace_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, n, scalar);
    psrc = env->GetFloatArrayElements(h_src, nullptr);
    if (!psrc) {
        cudaFree(d_src);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(psrc, d_src, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_src, psrc, 0);
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
    jfloat* pparam = env->GetFloatArrayElements(h_param, nullptr);
    jfloat* pgrad = env->GetFloatArrayElements(h_grad, nullptr);
    jfloat* pm = env->GetFloatArrayElements(h_m, nullptr);
    jfloat* pv = env->GetFloatArrayElements(h_v, nullptr);
    if (!pparam || !pgrad || !pm || !pv) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_param, pparam, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_grad, pgrad, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_m, pm, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_v, pv, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_param, pparam, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_grad, pgrad, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_m, pm, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_v, pv, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    adamw_step_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
        d_param, d_grad, d_m, d_v, learningRate, beta1, beta2, epsilon, weightDecay, invBias1, invBias2, n);
    pparam = env->GetFloatArrayElements(h_param, nullptr);
    pm = env->GetFloatArrayElements(h_m, nullptr);
    pv = env->GetFloatArrayElements(h_v, nullptr);
    if (!pparam || !pm || !pv) {
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pparam, d_param, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pm, d_m, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pv, d_v, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_param, pparam, 0);
    env->ReleaseFloatArrayElements(h_m, pm, 0);
    env->ReleaseFloatArrayElements(h_v, pv, 0);
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
    jlong* pj = env->GetLongArrayElements(h_param_ptrs, nullptr);
    jlong* gj = env->GetLongArrayElements(h_grad_ptrs, nullptr);
    jlong* mj = env->GetLongArrayElements(h_m_ptrs, nullptr);
    jlong* vj = env->GetLongArrayElements(h_v_ptrs, nullptr);
    jint* lens = env->GetIntArrayElements(h_lengths, nullptr);
    if (!pj || !gj || !mj || !vj || !lens) {
        if (pj) {
            env->ReleaseLongArrayElements(h_param_ptrs, pj, JNI_ABORT);
        }
        if (gj) {
            env->ReleaseLongArrayElements(h_grad_ptrs, gj, JNI_ABORT);
        }
        if (mj) {
            env->ReleaseLongArrayElements(h_m_ptrs, mj, JNI_ABORT);
        }
        if (vj) {
            env->ReleaseLongArrayElements(h_v_ptrs, vj, JNI_ABORT);
        }
        if (lens) {
            env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);
        }
        return;
    }
    for (jsize i = 0; i < num_segments; i++) {
        if (lens[i] <= 0 || pj[i] == 0 || gj[i] == 0 || mj[i] == 0 || vj[i] == 0) {
            env->ReleaseLongArrayElements(h_param_ptrs, pj, JNI_ABORT);
            env->ReleaseLongArrayElements(h_grad_ptrs, gj, JNI_ABORT);
            env->ReleaseLongArrayElements(h_m_ptrs, mj, JNI_ABORT);
            env->ReleaseLongArrayElements(h_v_ptrs, vj, JNI_ABORT);
            env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);
            return;
        }
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
        env->ReleaseLongArrayElements(h_param_ptrs, pj, JNI_ABORT);
        env->ReleaseLongArrayElements(h_grad_ptrs, gj, JNI_ABORT);
        env->ReleaseLongArrayElements(h_m_ptrs, mj, JNI_ABORT);
        env->ReleaseLongArrayElements(h_v_ptrs, vj, JNI_ABORT);
        env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);
        return;
    }
    for (jsize i = 0; i < num_segments; i++) {
        h_pp[i] = static_cast<uintptr_t>(pj[i]);
        h_gp[i] = static_cast<uintptr_t>(gj[i]);
        h_mp[i] = static_cast<uintptr_t>(mj[i]);
        h_vp[i] = static_cast<uintptr_t>(vj[i]);
    }
    env->ReleaseLongArrayElements(h_param_ptrs, pj, JNI_ABORT);
    env->ReleaseLongArrayElements(h_grad_ptrs, gj, JNI_ABORT);
    env->ReleaseLongArrayElements(h_m_ptrs, mj, JNI_ABORT);
    env->ReleaseLongArrayElements(h_v_ptrs, vj, JNI_ABORT);

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
        env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);
        return;
    }
    jgpt_cuda_ensure_stream();
    CUDA_CHECK_X(cudaMemcpyAsync(d_pp, h_pp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gp, h_gp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_mp, h_mp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_vp, h_vp, ptr_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_lens, lens, len_bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    free(h_pp);
    free(h_gp);
    free(h_mp);
    free(h_vp);
    env->ReleaseIntArrayElements(h_lengths, lens, JNI_ABORT);

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
    int nrows = batch * mid;
    if (nrows <= 0 || inner <= 0) {
        return;
    }
    size_t bytes = (size_t) nrows * (size_t) inner * sizeof(float);
    float *d_go = nullptr, *d_p = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_p), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pp = env->GetFloatArrayElements(h_probs, nullptr);
    jfloat* pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgo || !pp || !pgi) {
        cudaFree(d_go);
        cudaFree(d_p);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_p, pp, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, pgi, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_probs, pp, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gIn, pgi, JNI_ABORT);

    launch_softmax_last_dim_bwd(d_go, d_p, d_gi, nrows, inner);
    pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgi) {
        cudaFree(d_go);
        cudaFree(d_p);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgi, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gIn, pgi, 0);

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
    size_t bytes_xy = (size_t) outer * (size_t) lastDim * sizeof(float);
    size_t bytes_g = (size_t) lastDim * sizeof(float);
    float *d_go = nullptr, *d_x = nullptr, *d_g = nullptr, *d_gx = nullptr, *d_gg = nullptr, *d_gb = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_g), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gx), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gg), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gb), bytes_g));

    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* px = env->GetFloatArrayElements(h_x, nullptr);
    jfloat* pg = env->GetFloatArrayElements(h_gamma, nullptr);
    jfloat* pgx = env->GetFloatArrayElements(h_gX, nullptr);
    jfloat* pgg = env->GetFloatArrayElements(h_gGamma, nullptr);
    jfloat* pgb = env->GetFloatArrayElements(h_gBeta, nullptr);
    if (!pgo || !px || !pg || !pgx || !pgg || !pgb) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, pg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gx, pgx, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gg, pgg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gb, pgb, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gamma, pg, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gX, pgx, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gGamma, pgg, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gBeta, pgb, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (outer + threads - 1) / threads;
    layer_norm_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_x, d_g, eps, d_gx, d_gg, d_gb, outer, lastDim);
    CUDA_KERNEL_CHECK();
    pgx = env->GetFloatArrayElements(h_gX, nullptr);
    pgg = env->GetFloatArrayElements(h_gGamma, nullptr);
    pgb = env->GetFloatArrayElements(h_gBeta, nullptr);
    if (!pgx || !pgg || !pgb) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgx, d_gx, bytes_xy, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgg, d_gg, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgb, d_gb, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gX, pgx, 0);
    env->ReleaseFloatArrayElements(h_gGamma, pgg, 0);
    env->ReleaseFloatArrayElements(h_gBeta, pgb, 0);

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
    size_t bytes_xy = (size_t) outer * (size_t) lastDim * sizeof(float);
    size_t bytes_g = (size_t) lastDim * sizeof(float);
    float *d_go = nullptr, *d_x = nullptr, *d_g = nullptr, *d_gx = nullptr, *d_gg = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_g), bytes_g));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gx), bytes_xy));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gg), bytes_g));

    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* px = env->GetFloatArrayElements(h_x, nullptr);
    jfloat* pg = env->GetFloatArrayElements(h_gamma, nullptr);
    jfloat* pgx = env->GetFloatArrayElements(h_gX, nullptr);
    jfloat* pgg = env->GetFloatArrayElements(h_gGamma, nullptr);
    if (!pgo || !px || !pg || !pgx || !pgg) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_x, px, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_g, pg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gx, pgx, bytes_xy, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gg, pgg, bytes_g, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_x, px, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gamma, pg, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gX, pgx, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gGamma, pgg, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    (void) threads;
    launch_rms_norm_bwd(d_go, d_x, d_g, eps, d_gx, d_gg, outer, lastDim);
    pgx = env->GetFloatArrayElements(h_gX, nullptr);
    pgg = env->GetFloatArrayElements(h_gGamma, nullptr);
    if (!pgx || !pgg) {
        cudaFree(d_go);
        cudaFree(d_x);
        cudaFree(d_g);
        cudaFree(d_gx);
        cudaFree(d_gg);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgx, d_gx, bytes_xy, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgg, d_gg, bytes_g, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gX, pgx, 0);
    env->ReleaseFloatArrayElements(h_gGamma, pgg, 0);

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
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_a = nullptr, *d_b = nullptr, *d_ga = nullptr, *d_gb = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_ga), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gb), bytes));

    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pa = env->GetFloatArrayElements(h_a, nullptr);
    jfloat* pb = env->GetFloatArrayElements(h_b, nullptr);
    jfloat* pga = env->GetFloatArrayElements(h_gA, nullptr);
    jfloat* pgb = env->GetFloatArrayElements(h_gB, nullptr);
    if (!pgo || !pa || !pb || !pga || !pgb) {
        cudaFree(d_go);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_ga);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_b, pb, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_ga, pga, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gb, pgb, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_a, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_b, pb, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gA, pga, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gB, pgb, JNI_ABORT);

    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    multiply_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_a, d_b, d_ga, d_gb, n);
    pga = env->GetFloatArrayElements(h_gA, nullptr);
    pgb = env->GetFloatArrayElements(h_gB, nullptr);
    if (!pga || !pgb) {
        cudaFree(d_go);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_ga);
        cudaFree(d_gb);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pga, d_ga, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(pgb, d_gb, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gA, pga, 0);
    env->ReleaseFloatArrayElements(h_gB, pgb, 0);

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
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pin = env->GetFloatArrayElements(h_inp, nullptr);
    jfloat* pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgo || !pin || !pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, pin, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, pgi, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_inp, pin, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gIn, pgi, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    gelu_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgi, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gIn, pgi, 0);
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
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pin = env->GetFloatArrayElements(h_inp, nullptr);
    jfloat* pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgo || !pin || !pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, pin, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, pgi, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_inp, pin, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gIn, pgi, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    sigmoid_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgi, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gIn, pgi, 0);
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
    size_t bytes = (size_t) n * sizeof(float);
    float *d_go = nullptr, *d_in = nullptr, *d_gi = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_go), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_gi), bytes));
    jfloat* pgo = env->GetFloatArrayElements(h_gOut, nullptr);
    jfloat* pin = env->GetFloatArrayElements(h_inp, nullptr);
    jfloat* pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgo || !pin || !pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_go, pgo, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_in, pin, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_gi, pgi, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gOut, pgo, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_inp, pin, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_gIn, pgi, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    relu_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_go, d_in, d_gi, n);
    pgi = env->GetFloatArrayElements(h_gIn, nullptr);
    if (!pgi) {
        cudaFree(d_go);
        cudaFree(d_in);
        cudaFree(d_gi);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pgi, d_gi, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_gIn, pgi, 0);
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
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_d = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_d), bytes));
    jfloat* pa = env->GetFloatArrayElements(h_acc, nullptr);
    jfloat* pd = env->GetFloatArrayElements(h_delta, nullptr);
    if (!pa || !pd) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_d, pd, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_acc, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_delta, pd, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    accumulate_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_d, n);
    pa = env->GetFloatArrayElements(h_acc, nullptr);
    if (!pa) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pa, d_a, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_acc, pa, 0);
    cudaFree(d_a);
    cudaFree(d_d);
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_accumulateScaledAddGPU(
    JNIEnv* env, jclass clazz, jfloatArray h_acc, jfloatArray h_delta, jfloat scale, jint n) {
    (void) clazz;
    if (n <= 0) {
        return;
    }
    size_t bytes = (size_t) n * sizeof(float);
    float *d_a = nullptr, *d_d = nullptr;
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK_X(cudaMalloc(reinterpret_cast<void**>(&d_d), bytes));
    jfloat* pa = env->GetFloatArrayElements(h_acc, nullptr);
    jfloat* pd = env->GetFloatArrayElements(h_delta, nullptr);
    if (!pa || !pd) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(d_a, pa, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaMemcpyAsync(d_d, pd, bytes, cudaMemcpyHostToDevice, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_acc, pa, JNI_ABORT);
    env->ReleaseFloatArrayElements(h_delta, pd, JNI_ABORT);
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    accumulate_scaled_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_d, scale, n);
    pa = env->GetFloatArrayElements(h_acc, nullptr);
    if (!pa) {
        cudaFree(d_a);
        cudaFree(d_d);
        return;
    }
    CUDA_CHECK_X(cudaMemcpyAsync(pa, d_a, bytes, cudaMemcpyDeviceToHost, kTensorCudaStream));
    CUDA_CHECK_X(cudaStreamSynchronize(kTensorCudaStream));
    env->ReleaseFloatArrayElements(h_acc, pa, 0);
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
