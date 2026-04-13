/* Kernels + launches + JNI CUDA macros
 * vec_*, relu, half convert, bias, sum_columns; CUDA_CHECK_* macros.
 * Included only from jgpt_cuda.cu (single translation unit).
 */

// ========== Kernels with error checking ==========

__global__ void vec_add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vec_sub_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void relu_kernel(const float* __restrict__ a, float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = a[i]; b[i] = x > 0.f ? x : 0.f; }
}

// Адаптивный выбор blockSize + проверка запуска
static void launch_vec_add(const float* d_a, const float* d_b, float* d_c, int n) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    vec_add_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, d_c, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_vec_add error: %s\n", cudaGetErrorString(err));
}

static void launch_vec_sub(const float* d_a, const float* d_b, float* d_c, int n) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    vec_sub_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, d_c, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_vec_sub error: %s\n", cudaGetErrorString(err));
}

static void launch_relu(const float* d_a, float* d_b, int n) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_a, d_b, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_relu error: %s\n", cudaGetErrorString(err));
}

__global__ void float_to_half_kernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

static void launch_float_to_half(const float* d_src, __half* d_dst, int n) {
    if (n <= 0) return;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, d_dst, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_float_to_half error: %s\n", cudaGetErrorString(err));
}

__global__ void half_to_float_kernel(const __half* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

static void launch_half_to_float(const __half* d_src, float* d_dst, int n) {
    if (n <= 0) return;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    half_to_float_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, d_dst, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_half_to_float error: %s\n", cudaGetErrorString(err));
}

__global__ void bias_relu_inplace_kernel(float* __restrict__ C, const float* __restrict__ bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int col = idx % N;
    float x = C[idx] + bias[col];
    C[idx] = x > 0.f ? x : 0.f;
}

static void launch_bias_relu_inplace(float* d_C, const float* d_bias, int M, int N) {
    int total = M * N;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    bias_relu_inplace_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_C, d_bias, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_bias_relu_inplace error: %s\n", cudaGetErrorString(err));
}

__global__ void bias_add_inplace_kernel(float* __restrict__ C, const float* __restrict__ bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int col = idx % N;
    C[idx] += bias[col];
}

static void launch_bias_add_inplace(float* d_C, const float* d_bias, int M, int N) {
    int total = M * N;
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (total + threads - 1) / threads;
    bias_add_inplace_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_C, d_bias, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_bias_add_inplace error: %s\n", cudaGetErrorString(err));
}

__global__ void sum_columns_kernel(const float* __restrict__ src, float* __restrict__ dst, int M, int N, float beta) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float s = 0.f;
    for (int i = 0; i < M; ++i) {
        s += src[i * N + j];
    }
    dst[j] = beta * dst[j] + s;
}

static void launch_sum_columns(const float* d_src, float* d_dst, int M, int N, float beta) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (N + threads - 1) / threads;
    sum_columns_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_src, d_dst, M, N, beta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) fprintf(stderr, "launch_sum_columns error: %s\n", cudaGetErrorString(err));
}

// ========== Error macros ==========
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

/** Как {@code CUDA_CHECK_VOID}, но при ошибке делает {@code ReleaseFloatArrayElements(..., JNI_ABORT)} для выходного jfloatArray. */
#define CUDA_CHECK_VOID_JFLOAT_OUT(env, h_C, C_ptr, call) \
    do { \
        cudaError_t err_jfloat_out_ = (call); \
        if (err_jfloat_out_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_jfloat_out_)); \
            if ((C_ptr) != nullptr) (env)->ReleaseFloatArrayElements((h_C), (C_ptr), JNI_ABORT); \
            return; \
        } \
    } while (0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)
