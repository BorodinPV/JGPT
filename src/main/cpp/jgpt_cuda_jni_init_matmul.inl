/* JNI: OnLoad/Unload + matmul (host, fp16, batched, device, QKV/FFN)
 * 
 * Included only from jgpt_cuda.cu (single translation unit).
 */

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
