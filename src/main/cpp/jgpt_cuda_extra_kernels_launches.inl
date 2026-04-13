/* Kernels, launch helpers, cuBLAS SDPA glue, CUDA-graph aux, FlashAttention host dispatch.
 * Include only from jgpt_cuda_extra.cu (single translation unit; no CUDA separable compilation).
 */

// ========== Ядра (float32) ==========

__global__ void softmax_last_dim_kernel(const float* src, float* dst, int nrows, int inner) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * inner;
    float maxv = -INFINITY;
    for (int j = 0; j < inner; j++) {
        maxv = fmaxf(maxv, src[base + j]);
    }
    float sum = 0.f;
    for (int j = 0; j < inner; j++) {
        float e = expf(src[base + j] - maxv);
        dst[base + j] = e;
        sum += e;
    }
    float inv = 1.f / sum;
    for (int j = 0; j < inner; j++) {
        dst[base + j] *= inv;
    }
}

/**
 * Softmax по последней оси (ветка «fp16» в API): раньше exp шёл через FP16-округление диффа;
 * это давало sum=0 и Inf/NaN в вероятностях. Считаем exp в FP32, как в softmax_last_dim_kernel.
 */
__global__ void softmax_last_dim_kernel_fp16(const float* src, float* dst, int nrows, int inner) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * inner;
    float maxv = -INFINITY;
    for (int j = 0; j < inner; j++) {
        maxv = fmaxf(maxv, src[base + j]);
    }
    float sum = 0.f;
    for (int j = 0; j < inner; j++) {
        float diff = src[base + j] - maxv;
        float e = expf(diff);
        dst[base + j] = e;
        sum += e;
    }
    float inv = 1.f / fmaxf(sum, 1e-12f);
    for (int j = 0; j < inner; j++) {
        dst[base + j] *= inv;
    }
}

// ========== Block-per-row softmax (оптимизированная версия для inner >= 64) ==========
//
// Проблема «одна нить на строку»: все нити варпа обращаются к разным строкам →
// нет коалесценции. При inner=512 каждая нить делает 3 последовательных цикла по 512 эл.
//
// Решение: один блок на строку (gridDim.x = nrows, blockDim.x = kSoftmaxBlockDim=256).
// Нити блока читают ОДНУ строку с шагом blockDim.x — warp читает 32 смежных float → отличная
// коалесценция. Редукция max/sum — через warp-shuffle + shared memory между warpами.

static constexpr int kSoftmaxBlockDim       = 256;         // нитей на блок
static constexpr int kSoftmaxNumWarps       = kSoftmaxBlockDim / 32;  // 8
/** Префикс dynamic shared для softmax_scaled_masked_* под blockReduce (не пересекается с row-кэшем длиной inner). */
static constexpr int kSoftmaxSharedPrefixFloats = 32;
static constexpr int kSoftmaxBlockThreshold = 64;          // при inner >= этого → block-per-row

/** Макс. gridDim.x для текущего устройства (кэш), не выше INT_MAX для безопасного приведения в launch. */
static unsigned int jgpt_extra_cuda_max_grid_x() {
    static unsigned int cached = 0;
    if (cached == 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
            cached = (1u << 20);
        } else {
            unsigned long x = static_cast<unsigned long>(prop.maxGridSize[0]);
            if (x == 0ul || x > static_cast<unsigned long>(INT_MAX)) {
                x = static_cast<unsigned long>(INT_MAX);
            }
            cached = static_cast<unsigned int>(x);
        }
    }
    return cached;
}

/** true если смещение (nrows-1)*inner + (inner-1) выходит за PTRDIFF_MAX (консервативно). */
static bool jgpt_softmax_row_inner_offset_overflows(long long nrows_total, int inner) {
    if (nrows_total <= 0 || inner <= 0) {
        return false;
    }
    const auto inner_u = static_cast<unsigned long long>(inner);
    const auto last_row = static_cast<unsigned long long>(nrows_total - 1);
    if (last_row > static_cast<unsigned long long>(PTRDIFF_MAX) / inner_u) {
        return true;
    }
    const unsigned long long max_elem_off = last_row * inner_u + (inner_u - 1u);
    return max_elem_off > static_cast<unsigned long long>(PTRDIFF_MAX);
}

__device__ __forceinline__ float warpReduceMax32(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    return v;
}

__device__ __forceinline__ float warpReduceSum32(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

// Вызывается всеми kSoftmaxBlockDim нитями; возвращает broadcast-max блока.
__device__ float blockReduceMax256(float v, float* smem) {
    v = warpReduceMax32(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
    __syncthreads();
    float bv = (threadIdx.x < kSoftmaxNumWarps) ? smem[threadIdx.x] : -INFINITY;
    if (threadIdx.x < 32) bv = warpReduceMax32(bv);
    if (threadIdx.x == 0) smem[0] = bv;
    __syncthreads();
    return smem[0];
}

// Вызывается всеми kSoftmaxBlockDim нитями; возвращает broadcast-sum блока.
__device__ float blockReduceSum256(float v, float* smem) {
    v = warpReduceSum32(v);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
    __syncthreads();
    float bv = (threadIdx.x < kSoftmaxNumWarps) ? smem[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) bv = warpReduceSum32(bv);
    if (threadIdx.x == 0) smem[0] = bv;
    __syncthreads();
    return smem[0];
}

/**
 * Softmax по последней оси, block-per-row.
 * Запуск: gridDim.x=nrows, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void softmax_last_dim_block_kernel(
        const float* __restrict__ src, float* __restrict__ dst, int nrows, int inner) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= nrows) return;
    const int tid = threadIdx.x;
    const float* rowSrc = src + (ptrdiff_t)row * inner;
    float*       rowDst = dst + (ptrdiff_t)row * inner;

    float maxv = -INFINITY;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        maxv = fmaxf(maxv, rowSrc[j]);
    maxv = blockReduceMax256(maxv, smem);

    float sumv = 0.f;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        sumv += expf(rowSrc[j] - maxv);
    sumv = blockReduceSum256(sumv, smem);

    const float inv = 1.f / fmaxf(sumv, 1e-12f);
    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        rowDst[j] = expf(rowSrc[j] - maxv) * inv;
}

/**
 * Слитый (scale + causal-mask + softmax), block-per-row для attention.
 * Одно чтение глобальной строки scores: z[j] кэшируется в shared (dynamic smem = prefix + inner).
 * Для causal mask: row % inner — query-позиция; mask[qPos * inner + j] добавляется к scaled logit.
 * Запуск: gridDim.x=nrows, blockDim.x=kSoftmaxBlockDim, smem=(kSoftmaxSharedPrefixFloats+inner)*sizeof(float).
 */
__global__ void softmax_scaled_masked_block_kernel(
        const float* __restrict__ src, float* __restrict__ dst,
        const float* __restrict__ mask, float scale,
        long long nrows_total, long long row_offset, int inner) {
    extern __shared__ float smem[];
    float* const red = smem;
    float* const z = smem + kSoftmaxSharedPrefixFloats;
    const long long row = row_offset + static_cast<long long>(blockIdx.x);
    if (row >= nrows_total) {
        return;
    }
    const int tid = threadIdx.x;
    const float* rowSrc = src + static_cast<ptrdiff_t>(row) * inner;
    float* rowDst = dst + static_cast<ptrdiff_t>(row) * inner;
    const int qPos = static_cast<int>(row % inner);  // query-позиция в последовательности для causal mask

    float maxv = -INFINITY;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        const float v = rowSrc[j] * scale + (mask ? mask[(ptrdiff_t)qPos * inner + j] : 0.f);
        z[j] = v;
        maxv = fmaxf(maxv, v);
    }
    __syncthreads();
    maxv = blockReduceMax256(maxv, red);

    float sumv = 0.f;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        const float v = z[j];
        sumv += expf(v - maxv);
    }
    sumv = blockReduceSum256(sumv, red);

    const float inv = 1.f / fmaxf(sumv, 1e-12f);
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        const float v = z[j];
        rowDst[j] = expf(v - maxv) * inv;
    }
}

/**
 * In-place: один указатель без src/dst restrict-aliasing (для graph-aux размером bytesProb вместо 2×).
 * Одно чтение глобальной строки на этапе заполнения z в shared.
 */
__global__ void softmax_scaled_masked_inplace_block_kernel(
        float* __restrict__ buf, const float* __restrict__ mask, float scale,
        long long nrows_total, long long row_offset, int inner) {
    extern __shared__ float smem[];
    float* const red = smem;
    float* const z = smem + kSoftmaxSharedPrefixFloats;
    const long long row = row_offset + static_cast<long long>(blockIdx.x);
    if (row >= nrows_total) {
        return;
    }
    const int tid = threadIdx.x;
    float* rowBuf = buf + static_cast<ptrdiff_t>(row) * inner;
    const int qPos = static_cast<int>(row % inner);

    float maxv = -INFINITY;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        const float v = rowBuf[j] * scale + (mask ? mask[(ptrdiff_t)qPos * inner + j] : 0.f);
        z[j] = v;
        maxv = fmaxf(maxv, v);
    }
    __syncthreads();
    maxv = blockReduceMax256(maxv, red);

    float sumv = 0.f;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        const float v = z[j];
        sumv += expf(v - maxv);
    }
    sumv = blockReduceSum256(sumv, red);

    const float inv = 1.f / fmaxf(sumv, 1e-12f);
    for (int j = tid; j < inner; j += kSoftmaxBlockDim) {
        const float v = z[j];
        rowBuf[j] = expf(v - maxv) * inv;
    }
}

static bool launch_softmax_scaled_masked_block_chunked(
        float* d_scores,
        float* d_probs,
        const float* d_mask,
        float scale,
        long long nrows_total,
        int inner,
        cudaStream_t stream) {
    if (inner < 0 || nrows_total <= 0) {
        return true;
    }
    if (jgpt_softmax_row_inner_offset_overflows(nrows_total, inner)) {
        fprintf(stderr, "attn_fwd: softmax masked row*inner offset overflow\n");
        return false;
    }
    const size_t smem = (static_cast<size_t>(kSoftmaxSharedPrefixFloats) + static_cast<size_t>(inner)) * sizeof(float);
    const unsigned int chunk_cap = jgpt_extra_cuda_max_grid_x();
    const long long chunk_ll = static_cast<long long>(chunk_cap);

    for (long long row_off = 0; row_off < nrows_total; row_off += chunk_ll) {
        const long long remain = nrows_total - row_off;
        const unsigned int grid_x = static_cast<unsigned int>(std::min(chunk_ll, remain));
        if (grid_x == 0u) {
            break;
        }
        if (d_probs == d_scores) {
            softmax_scaled_masked_inplace_block_kernel<<<grid_x, kSoftmaxBlockDim, smem, stream>>>(
                    d_scores, d_mask, scale, nrows_total, row_off, inner);
        } else {
            softmax_scaled_masked_block_kernel<<<grid_x, kSoftmaxBlockDim, smem, stream>>>(
                    d_scores, d_probs, d_mask, scale, nrows_total, row_off, inner);
        }
        const cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            fprintf(stderr, "launch_softmax_scaled_masked_block_chunked: %s\n", cudaGetErrorString(le));
            return false;
        }
    }
    return true;
}

/**
 * Одна строка = один позиционный токен [batch*seq]: стабильный softmax, CE loss (atomic),
 * градиент (p - one_hot) * scale. Невалидный target: строка градиента в ноль, в loss не входит.
 */
__global__ void cross_entropy_softmax_grad_loss_kernel(const float* logits, const float* targets_f,
                                                       float* grad, float scale, float* loss_sum,
                                                       unsigned int* valid_count, int nrows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * vocab;
    int t = (int)targets_f[row];
    if (t < 0 || t >= vocab) {
        for (int v = 0; v < vocab; ++v) {
            grad[base + v] = 0.f;
        }
        return;
    }
    float maxv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        maxv = fmaxf(maxv, logits[base + v]);
    }
    float sumExp = 0.f;
    for (int v = 0; v < vocab; ++v) {
        sumExp += expf(logits[base + v] - maxv);
    }
    float logProb = logits[base + t] - maxv - logf(sumExp);
    atomicAdd(loss_sum, -logProb);
    atomicAdd(valid_count, 1U);
    float invSum = 1.f / sumExp;
    for (int v = 0; v < vocab; ++v) {
        float p = expf(logits[base + v] - maxv) * invSum;
        float g = p;
        if (v == t) {
            g -= 1.f;
        }
        grad[base + v] = g * scale;
    }
}

/**
 * Fused CE + grad (ветка use_fp16_softmax в JNI): exp в FP32, без half-округления диффа (sumExp=0 / Inf).
 */
__global__ void cross_entropy_softmax_grad_loss_kernel_fp16(const float* logits, const float* targets_f,
                                                              float* grad, float scale, float* loss_sum,
                                                              unsigned int* valid_count, int nrows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * vocab;
    int t = (int)targets_f[row];
    if (t < 0 || t >= vocab) {
        for (int v = 0; v < vocab; ++v) {
            grad[base + v] = 0.f;
        }
        return;
    }
    float maxv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        maxv = fmaxf(maxv, logits[base + v]);
    }
    float sumExp = 0.f;
    for (int v = 0; v < vocab; ++v) {
        float diff = logits[base + v] - maxv;
        sumExp += expf(diff);
    }
    sumExp = fmaxf(sumExp, 1e-12f);
    float logProb = logits[base + t] - maxv - logf(sumExp);
    atomicAdd(loss_sum, -logProb);
    atomicAdd(valid_count, 1U);
    float invSum = 1.f / sumExp;
    for (int v = 0; v < vocab; ++v) {
        float diff = logits[base + v] - maxv;
        float p = expf(diff) * invSum;
        float g = p;
        if (v == t) {
            g -= 1.f;
        }
        grad[base + v] = g * scale;
    }
}

/** CE + softmax + grad; targets как int32 на строку (как (int)targets_f[row] в float-версии). */
__global__ void cross_entropy_softmax_grad_loss_kernel_i32(const float* logits, const int* targets_i,
                                                           float* grad, float scale, float* loss_sum,
                                                           unsigned int* valid_count, int nrows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * vocab;
    int t = targets_i[row];
    if (t < 0 || t >= vocab) {
        for (int v = 0; v < vocab; ++v) {
            grad[base + v] = 0.f;
        }
        return;
    }
    float maxv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        maxv = fmaxf(maxv, logits[base + v]);
    }
    float sumExp = 0.f;
    for (int v = 0; v < vocab; ++v) {
        sumExp += expf(logits[base + v] - maxv);
    }
    float logProb = logits[base + t] - maxv - logf(sumExp);
    atomicAdd(loss_sum, -logProb);
    atomicAdd(valid_count, 1U);
    float invSum = 1.f / sumExp;
    for (int v = 0; v < vocab; ++v) {
        float p = expf(logits[base + v] - maxv) * invSum;
        float g = p;
        if (v == t) {
            g -= 1.f;
        }
        grad[base + v] = g * scale;
    }
}

__global__ void cross_entropy_softmax_grad_loss_kernel_fp16_i32(const float* logits, const int* targets_i,
                                                                float* grad, float scale, float* loss_sum,
                                                                unsigned int* valid_count, int nrows, int vocab) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * vocab;
    int t = targets_i[row];
    if (t < 0 || t >= vocab) {
        for (int v = 0; v < vocab; ++v) {
            grad[base + v] = 0.f;
        }
        return;
    }
    float maxv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        maxv = fmaxf(maxv, logits[base + v]);
    }
    float sumExp = 0.f;
    for (int v = 0; v < vocab; ++v) {
        float diff = logits[base + v] - maxv;
        sumExp += expf(diff);
    }
    sumExp = fmaxf(sumExp, 1e-12f);
    float logProb = logits[base + t] - maxv - logf(sumExp);
    atomicAdd(loss_sum, -logProb);
    atomicAdd(valid_count, 1U);
    float invSum = 1.f / sumExp;
    for (int v = 0; v < vocab; ++v) {
        float diff = logits[base + v] - maxv;
        float p = expf(diff) * invSum;
        float g = p;
        if (v == t) {
            g -= 1.f;
        }
        grad[base + v] = g * scale;
    }
}

__global__ void layer_norm_fwd_kernel(const float* src, const float* gamma, const float* beta,
                                     float* dst, int outer, int lastDim, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer) {
        return;
    }
    int base = i * lastDim;
    float mean = 0.f;
    for (int j = 0; j < lastDim; j++) {
        mean += src[base + j];
    }
    mean /= (float) lastDim;
    float var = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float d = src[base + j] - mean;
        var += d * d;
    }
    var /= (float) lastDim;
    float invStd = rsqrtf(var + eps);
    for (int j = 0; j < lastDim; j++) {
        float normalized = (src[base + j] - mean) * invStd;
        dst[base + j] = normalized * gamma[j] + beta[j];
    }
}

__global__ void rms_norm_fwd_kernel(const float* src, const float* gamma, float* dst,
                                    int outer, int lastDim, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer) {
        return;
    }
    int base = i * lastDim;
    float sumSq = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float v = src[base + j];
        sumSq += v * v;
    }
    float rms = sqrtf(sumSq / (float) lastDim + eps);
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;
    for (int j = 0; j < lastDim; j++) {
        dst[base + j] = src[base + j] * invRms * gamma[j];
    }
}

/**
 * Ветка «fp16» в API: раньше x/γ округлялись до half → при больших |x| half давал Inf, далее Q/K/V и softmax ломались.
 * Считаем как rms_norm_fwd_kernel (FP32), выход по-прежнему float.
 */
__global__ void rms_norm_fwd_kernel_fp16(const float* src, const float* gamma, float* dst,
                                         int outer, int lastDim, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer) {
        return;
    }
    int base = i * lastDim;
    float sumSq = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float v = src[base + j];
        sumSq += v * v;
    }
    float rms = sqrtf(sumSq / (float) lastDim + eps);
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;
    for (int j = 0; j < lastDim; j++) {
        dst[base + j] = src[base + j] * invRms * gamma[j];
    }
}

/**
 * RMSNorm forward, block-per-row.
 * Один блок kSoftmaxBlockDim нитей обрабатывает одну строку.
 * При lastDim=256: ровно 1 нить на элемент, серийные циклы исчезают.
 * Запуск: gridDim.x=outer, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void rms_norm_fwd_block_kernel(
        const float* __restrict__ src, const float* __restrict__ gamma,
        float* __restrict__ dst, int outer, int lastDim, float eps) {
    extern __shared__ float smem[];
    const int o = blockIdx.x;
    if (o >= outer) return;
    const int tid = threadIdx.x;

    float sumSq = 0.f;
    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim)
        sumSq += src[o * lastDim + j] * src[o * lastDim + j];
    sumSq = blockReduceSum256(sumSq, smem);

    float rms = sqrtf(sumSq / (float) lastDim + eps);
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;

    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim)
        dst[o * lastDim + j] = src[o * lastDim + j] * invRms * gamma[j];
}

static constexpr int kRmsNormFwdBlockThreshold = 64;

#define CUDA_CHECK_X(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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

#define CUDA_KERNEL_CHECK_RV(rv) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return (rv); \
        } \
    } while (0)

static void launch_rms_norm_fwd(const float* src, const float* gamma, float* dst,
        int outer, int lastDim, float eps) {
    if (lastDim >= kRmsNormFwdBlockThreshold) {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        rms_norm_fwd_block_kernel<<<outer, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(
                src, gamma, dst, outer, lastDim, eps);
    } else {
        int threads = jgpt_cuda_get_optimal_block_size();
        int blocks = (outer + threads - 1) / threads;
        rms_norm_fwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
                src, gamma, dst, outer, lastDim, eps);
    }
    CUDA_KERNEL_CHECK();
}

__device__ float gelu_tanh_dev(float x) {
    const float SQRT_2_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    float x3 = x * x * x;
    float tanhArg = SQRT_2_PI * (x + COEF * x3);
    return 0.5f * x * (1.0f + tanhf(tanhArg));
}

__global__ void gelu_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = gelu_tanh_dev(src[i]);
    }
}

__global__ void sigmoid_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = src[i];
        float s;
        if (x >= 20.f) {
            s = 1.f;
        } else if (x <= -20.f) {
            s = 0.f;
        } else {
            s = 1.f / (1.f + expf(-x));
        }
        dst[i] = s;
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void mul_scalar_kernel(const float* a, float* b, int n, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i] * scalar;
    }
}

__global__ void scale_inplace_kernel(float* a, int n, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= scalar;
    }
}

__global__ void sum_squares_kernel(const float* src, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = src[i];
        atomicAdd(out, v * v);
    }
}

/** Для устойчивой суммы квадратов: далее cublasDdot(dst,dst) в double. */
__global__ void float_to_double_kernel(const float* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (double) src[i];
    }
}

__global__ void adamw_step_kernel(
        float* param, const float* grad, float* m, float* v,
        float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, float invBias1, float invBias2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float gi = grad[i];
    float mi = beta1 * m[i] + (1.f - beta1) * gi;
    float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float mHat = mi * invBias1;
    float vHat = vi * invBias2;
    param[i] -= learningRate * (mHat / (sqrtf(vHat) + epsilon) + weightDecay * param[i]);
}

/** Один launch: по блоку на сегмент (разные device-указатели), потоки обходят длину с шагом blockDim.x. */
__global__ void adamw_step_kernel_segments(
        const uintptr_t* d_param_ptrs,
        const uintptr_t* d_grad_ptrs,
        const uintptr_t* d_m_ptrs,
        const uintptr_t* d_v_ptrs,
        const int* d_lengths,
        int num_segments,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        float invBias1,
        float invBias2) {
    int seg = blockIdx.x;
    if (seg >= num_segments) {
        return;
    }
    int n = d_lengths[seg];
    if (n <= 0) {
        return;
    }
    float* param = reinterpret_cast<float*>(d_param_ptrs[seg]);
    const float* grad = reinterpret_cast<const float*>(d_grad_ptrs[seg]);
    float* m = reinterpret_cast<float*>(d_m_ptrs[seg]);
    float* v = reinterpret_cast<float*>(d_v_ptrs[seg]);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float gi = grad[i];
        float mi = beta1 * m[i] + (1.f - beta1) * gi;
        float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float mHat = mi * invBias1;
        float vHat = vi * invBias2;
        param[i] -= learningRate * (mHat / (sqrtf(vHat) + epsilon) + weightDecay * param[i]);
    }
}

__global__ void embedding_token_fwd_kernel(
        const float* tokens, const float* weights, float* out, int batch, int seqLen, int dModel, int vocabSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    int b = t0 / seqLen;
    int token = (int) tokens[b * seqLen + s];
    if (token < 0 || token >= vocabSize) {
        out[idx] = 0.f;
        return;
    }
    out[idx] = weights[(size_t) token * (size_t) dModel + (size_t) j];
}

/**
 * Ветка «fp16» в API: раньше вес строки эмбеддинга прогоняли через half → крупные веса давали Inf в out.
 * Считаем как embedding_token_fwd_kernel (без квантования веса).
 */
__global__ void embedding_token_fwd_kernel_fp16(
        const float* tokens, const float* weights, float* out, int batch, int seqLen, int dModel, int vocabSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    int b = t0 / seqLen;
    int token = (int) tokens[b * seqLen + s];
    if (token < 0 || token >= vocabSize) {
        out[idx] = 0.f;
        return;
    }
    out[idx] = weights[(size_t) token * (size_t) dModel + (size_t) j];
}

__global__ void embedding_token_bwd_kernel(
        const float* tokens, const float* gradOut, float* gradWeights,
        int batch, int seqLen, int dModel, int vocabSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    int b = t0 / seqLen;
    int token = (int) tokens[b * seqLen + s];
    if (token < 0 || token >= vocabSize) {
        return;
    }
    atomicAdd(&gradWeights[token * dModel + j], gradOut[idx]);
}

__global__ void embedding_position_bwd_kernel(
        const float* gradCombined, float* gradWeights,
        int batch, int seqLen, int dModel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    atomicAdd(&gradWeights[s * dModel + j], gradCombined[idx]);
}

/** x[b,s,j] += posWeights[posRowStart + s, j] (таблица [>=posRowStart+seqLen, dModel] row-major). */
__global__ void add_position_embedding_broadcast_kernel(
        float* x, const float* posWeights, int posRowStart, int batch, int seqLen, int dModel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t0 = idx / dModel;
    int s = t0 % seqLen;
    x[idx] += posWeights[(size_t)(posRowStart + s) * (size_t) dModel + (size_t) j];
}

__global__ void apply_causal_mask_3d_kernel(const float* scores, const float* mask, float* out,
                                            int batch, int seqLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * seqLen;
    if (idx >= total) {
        return;
    }
    int j = idx % seqLen;
    int i = idx / seqLen;
    int queryRow = i % seqLen;
    out[idx] = scores[idx] + mask[queryRow * seqLen + j];
}

__global__ void transpose_2d_last_kernel(const float* src, float* dst, int d0, int d1, int d2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2;
    if (idx >= total) {
        return;
    }
    int c = idx % d2;
    int t = idx / d2;
    int b = t % d1;
    int a = t / d1;
    int dstIdx = a * (d2 * d1) + c * d1 + b;
    dst[dstIdx] = src[idx];
}

__global__ void split_heads_kernel(const float* src, float* dst, int batch, int seqLen, int dModel, int numHeads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int dHead = dModel / numHeads;
    int j = idx % dModel;
    int t = idx / dModel;
    int i = t % seqLen;
    int b = t / seqLen;
    int h = j / dHead;
    int jd = j % dHead;
    int r0 = numHeads * seqLen * dHead;
    int r1 = seqLen * dHead;
    int r2 = dHead;
    int dstIdx = b * r0 + h * r1 + i * r2 + jd;
    dst[dstIdx] = src[idx];
}

/** Головы K или V в row-major [batch, H, seq, dHead]: копия среза батча в кэш {@code head * maxSeqLen * dHead + pos * dHead}. */
__global__ void kv_heads4d_to_cache_kernel(
        const float* src, float* dst, int numHeads, int seqLen, int maxSeqLen, int dHead) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numHeads * seqLen * dHead;
    if (idx >= total) {
        return;
    }
    int j = idx % dHead;
    int s = (idx / dHead) % seqLen;
    int h = idx / (seqLen * dHead);
    int srcO = h * (seqLen * dHead) + s * dHead + j;
    int dstO = h * (maxSeqLen * dHead) + s * dHead + j;
    dst[dstO] = src[srcO];
}

__global__ void concat_heads_kernel(const float* src, float* dst, int batch, int numHeads, int seqLen, int dHead) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dModel = numHeads * dHead;
    int total = batch * seqLen * dModel;
    if (idx >= total) {
        return;
    }
    int j = idx % dModel;
    int t = idx / dModel;
    int i = t % seqLen;
    int b = t / seqLen;
    int h = j / dHead;
    int jd = j % dHead;
    int s0 = numHeads * seqLen * dHead;
    int s1 = seqLen * dHead;
    int s2 = dHead;
    int s3 = 1;
    int srcIdx = b * s0 + h * s1 + i * s2 + jd * s3;
    dst[idx] = src[srcIdx];
}

__global__ void rope_4d_kernel(const float* src, float* dst, int batch, int numHeads, int seqLen, int dHead,
                               const int* positions, int posLen, int posBaseOffset) {
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfPairs = dHead / 2;
    int total = batch * numHeads * seqLen * halfPairs;
    if (pairIdx >= total) {
        return;
    }
    int j = pairIdx % halfPairs;
    int tmp = pairIdx / halfPairs;
    int i = tmp % seqLen;
    tmp /= seqLen;
    int h = tmp % numHeads;
    int b = tmp / numHeads;

    int s0 = numHeads * seqLen * dHead;
    int s1 = seqLen * dHead;
    int s2 = dHead;
    int s3 = 1;
    int base = b * s0 + h * s1 + i * s2;
    int idx1 = base + (2 * j) * s3;
    int idx2 = base + (2 * j + 1) * s3;
    int p = (positions != nullptr && i < posLen) ? positions[i] : (i + posBaseOffset);
    float theta = (float) p / powf(10000.f, (2.f * (float) j) / (float) dHead);
    float c = cosf(theta);
    float s = sinf(theta);
    float x1 = src[idx1];
    float x2 = src[idx2];
    dst[idx1] = x1 * c - x2 * s;
    dst[idx2] = x1 * s + x2 * c;
}

__global__ void rope_4d_bwd_kernel(const float* gradY, float* gradX, int batch, int numHeads, int seqLen, int dHead,
                                   const int* positions, int posLen, int posBaseOffset) {
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfPairs = dHead / 2;
    int total = batch * numHeads * seqLen * halfPairs;
    if (pairIdx >= total) {
        return;
    }
    int j = pairIdx % halfPairs;
    int tmp = pairIdx / halfPairs;
    int i = tmp % seqLen;
    tmp /= seqLen;
    int h = tmp % numHeads;
    int b = tmp / numHeads;

    int s0 = numHeads * seqLen * dHead;
    int s1 = seqLen * dHead;
    int s2 = dHead;
    int base = b * s0 + h * s1 + i * s2;
    int idx1 = base + 2 * j;
    int idx2 = base + 2 * j + 1;
    int p = (positions != nullptr && i < posLen) ? positions[i] : (i + posBaseOffset);
    float theta = (float) p / powf(10000.f, (2.f * (float) j) / (float) dHead);
    float c = cosf(theta);
    float s = sinf(theta);
    float gy1 = gradY[idx1];
    float gy2 = gradY[idx2];
    gradX[idx1] += gy1 * c + gy2 * s;
    gradX[idx2] += -gy1 * s + gy2 * c;
}

__global__ void softmax_last_dim_bwd_kernel(const float* gOut, const float* p, float* gIn, int nrows, int inner) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    int base = row * inner;
    float dot = 0.f;
    for (int j = 0; j < inner; j++) {
        dot += p[base + j] * gOut[base + j];
    }
    for (int j = 0; j < inner; j++) {
        gIn[base + j] += p[base + j] * (gOut[base + j] - dot);
    }
}

/**
 * Backward softmax по последней оси, block-per-row. VJP: gIn[j] += p[j]*(gOut[j] - dot(p,gOut)).
 * Запуск: gridDim.x=nrows, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void softmax_last_dim_bwd_block_kernel(
        const float* __restrict__ gOut, const float* __restrict__ p,
        float* __restrict__ gIn, long long nrows_total, long long row_offset, int inner) {
    extern __shared__ float smem[];
    const long long row = row_offset + static_cast<long long>(blockIdx.x);
    if (row >= nrows_total) {
        return;
    }
    const int tid = threadIdx.x;
    const ptrdiff_t base = static_cast<ptrdiff_t>(row) * inner;

    float dot = 0.f;
    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        dot += p[base + j] * gOut[base + j];
    dot = blockReduceSum256(dot, smem);

    for (int j = tid; j < inner; j += kSoftmaxBlockDim)
        gIn[base + j] += p[base + j] * (gOut[base + j] - dot);
}

static void launch_softmax_last_dim_bwd_block_chunked(
        const float* gOut, const float* p, float* gIn, long long nrows_total, int inner, cudaStream_t stream) {
    if (inner < 0 || nrows_total <= 0) {
        return;
    }
    if (jgpt_softmax_row_inner_offset_overflows(nrows_total, inner)) {
        fprintf(stderr, "softmax_last_dim_bwd_block: row*inner offset overflow\n");
        return;
    }
    const size_t smem = static_cast<size_t>(kSoftmaxNumWarps) * sizeof(float);
    const unsigned int chunk_cap = jgpt_extra_cuda_max_grid_x();
    const long long chunk_ll = static_cast<long long>(chunk_cap);

    for (long long row_off = 0; row_off < nrows_total; row_off += chunk_ll) {
        const long long remain = nrows_total - row_off;
        const unsigned int grid_x = static_cast<unsigned int>(std::min(chunk_ll, remain));
        if (grid_x == 0u) {
            break;
        }
        softmax_last_dim_bwd_block_kernel<<<grid_x, kSoftmaxBlockDim, smem, stream>>>(
                gOut, p, gIn, nrows_total, row_off, inner);
        const cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) {
            fprintf(stderr, "launch_softmax_last_dim_bwd_block_chunked: %s\n", cudaGetErrorString(le));
            return;
        }
    }
}

/* ========== Missing helper kernels (были пропущены) ========== */

/** Транспонирование последних двух измерений: [batch, d1, d2] → [batch, d2, d1] */
static __global__ void transpose_last2_3d_kernel(const float* __restrict__ src, float* __restrict__ dst,
        int batch, int d1, int d2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d1 * d2;
    if (idx >= total) {
        return;
    }
    int k = idx % d2;
    int tmp = idx / d2;
    int j = tmp % d1;
    int b = tmp / d1;
    // dst[b][k][j] = src[b][j][k]
    int dstIdx = (b * d2 + k) * d1 + j;
    dst[dstIdx] = src[idx];
}

/** Scale inplace (extra версия для attention) */
static __global__ void scale_inplace_kernel_extra(float* __restrict__ a, int n, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= scalar;
    }
}

/** Добавление causal mask inplace: scores += mask */
static __global__ void add_mask_inplace_kernel(float* __restrict__ scores, const float* __restrict__ mask,
        int batch, int seqLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seqLen * seqLen;
    if (idx >= total) {
        return;
    }
    int j = idx % seqLen;
    int row = idx / seqLen;
    int i = row % seqLen;
    scores[idx] += mask[i * seqLen + j];
}

__global__ void multiply_bwd_kernel(const float* gOut, const float* a, const float* b, float* gA, float* gB, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gA[i] += gOut[i] * b[i];
        gB[i] += gOut[i] * a[i];
    }
}

__global__ void accumulate_add_kernel(float* acc, const float* delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        acc[i] += delta[i];
    }
}

/** acc[i] += delta[i] * scale (scale = -1 для вычитания delta из acc). */
__global__ void accumulate_scaled_add_kernel(float* acc, const float* delta, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        acc[i] += delta[i] * scale;
    }
}

__global__ void gelu_bwd_kernel(const float* gOut, const float* inp, float* gIn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float x = inp[i];
    const float SQRT_2_PI = 0.7978845608f;
    const float COEF = 0.044715f;
    float x3 = x * x * x;
    float tanhArg = SQRT_2_PI * (x + COEF * x3);
    float t = tanhf(tanhArg);
    float sech2 = 1.f - t * t;
    float dtanh = SQRT_2_PI * (1.f + 3.f * COEF * x * x) * sech2;
    float ddx = 0.5f * x * dtanh + 0.5f * (1.f + t);
    gIn[i] += gOut[i] * ddx;
}

__global__ void sigmoid_bwd_kernel(const float* gOut, const float* inp, float* gIn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float x = inp[i];
    float s;
    if (x >= 20.f) {
        s = 1.f;
    } else if (x <= -20.f) {
        s = 0.f;
    } else {
        s = 1.f / (1.f + expf(-x));
    }
    gIn[i] += gOut[i] * s * (1.f - s);
}

__global__ void relu_bwd_kernel(const float* gOut, const float* inp, float* gIn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && inp[i] > 0.f) {
        gIn[i] += gOut[i];
    }
}

__global__ void layer_norm_bwd_kernel(const float* gOut, const float* x, const float* gamma, float eps,
                                      float* gX, float* gGamma, float* gBeta, int outer, int lastDim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outer) {
        return;
    }
    int base = i * lastDim;
    float mean = 0.f;
    for (int j = 0; j < lastDim; j++) {
        mean += x[base + j];
    }
    mean /= (float) lastDim;
    float var = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float d = x[base + j] - mean;
        var += d * d;
    }
    var /= (float) lastDim;
    float invStd = rsqrtf(var + eps);

    float sumDh = 0.f;
    float sumDhXh = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float xh = (x[base + j] - mean) * invStd;
        float dh = gOut[base + j] * gamma[j];
        sumDh += dh;
        sumDhXh += dh * xh;
    }
    float meanDh = sumDh / (float) lastDim;
    float meanDhXh = sumDhXh / (float) lastDim;

    for (int j = 0; j < lastDim; j++) {
        float xh = (x[base + j] - mean) * invStd;
        atomicAdd(&gGamma[j], gOut[base + j] * xh);
        atomicAdd(&gBeta[j], gOut[base + j]);
        float dh = gOut[base + j] * gamma[j];
        float dx = invStd * (dh - meanDh - xh * meanDhXh);
        gX[base + j] += dx;
    }
}

__global__ void rms_norm_bwd_kernel(const float* gOut, const float* x, const float* gamma, float eps,
                                    float* gX, float* gGamma, int outer, int lastDim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= outer) {
        return;
    }
    int base = o * lastDim;
    float sumSq = 0.f;
    for (int j = 0; j < lastDim; j++) {
        float v = x[base + j];
        sumSq += v * v;
    }
    float ms = sumSq / (float) lastDim;
    float rms = sqrtf(ms + eps);
    // Согласовано с rms_norm_fwd_kernel*: при не-конечном rms не даём NaN расползаться по ∂x/∂γ.
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;

    float sumXGrad = 0.f;
    for (int j = 0; j < lastDim; j++) {
        sumXGrad += x[base + j] * gamma[j] * gOut[base + j];
    }
    float meanXGrad = sumXGrad / (float) lastDim;

    for (int j = 0; j < lastDim; j++) {
        float xj = x[base + j];
        int idx = base + j;
        atomicAdd(&gGamma[j], gOut[idx] * xj * invRms);
        gX[idx] += invRms * gamma[j] * gOut[idx] - invRms * invRms * invRms * xj * meanXGrad;
    }
}

/**
 * RMSNorm backward, block-per-row. Одна строка = один блок kSoftmaxBlockDim нитей.
 * При lastDim = d_model = 256: ровно 1 нить на элемент, серийные циклы исчезают.
 * Запуск: gridDim.x=outer, blockDim.x=kSoftmaxBlockDim, smem=kSoftmaxNumWarps*sizeof(float).
 */
__global__ void rms_norm_bwd_block_kernel(
        const float* __restrict__ gOut, const float* __restrict__ x,
        const float* __restrict__ gamma, float eps,
        float* __restrict__ gX, float* __restrict__ gGamma, int outer, int lastDim) {
    extern __shared__ float smem[];
    const int o = blockIdx.x;
    if (o >= outer) return;
    const int tid = threadIdx.x;

    // Вычислить sumSq = sum(x[j]^2) по строке
    float sumSq = 0.f;
    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim)
        sumSq += x[o * lastDim + j] * x[o * lastDim + j];
    sumSq = blockReduceSum256(sumSq, smem);

    float rms = sqrtf(sumSq / (float) lastDim + eps);
    float invRms = (rms > 0.f && isfinite(rms)) ? (1.f / rms) : 0.f;

    // Вычислить sumXGrad = sum(x[j] * gamma[j] * gOut[j])
    float localXG = 0.f;
    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim)
        localXG += x[o * lastDim + j] * gamma[j] * gOut[o * lastDim + j];
    float sumXGrad = blockReduceSum256(localXG, smem);
    float meanXGrad = sumXGrad / (float) lastDim;

    // Записать gX; gGamma накапливается атомарно (across outer строк)
    for (int j = tid; j < lastDim; j += kSoftmaxBlockDim) {
        int idx = o * lastDim + j;
        atomicAdd(&gGamma[j], gOut[idx] * x[idx] * invRms);
        gX[idx] += invRms * gamma[j] * gOut[idx] - invRms * invRms * invRms * x[idx] * meanXGrad;
    }
}

static constexpr int kRmsNormBwdBlockThreshold = 64; // block-per-row при lastDim >= этого

static void launch_rms_norm_bwd(const float* gOut, const float* x, const float* gamma, float eps,
        float* gX, float* gGamma, int outer, int lastDim) {
    if (lastDim >= kRmsNormBwdBlockThreshold) {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        rms_norm_bwd_block_kernel<<<outer, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(
                gOut, x, gamma, eps, gX, gGamma, outer, lastDim);
    } else {
        int threads = jgpt_cuda_get_optimal_block_size();
        int blocks = (outer + threads - 1) / threads;
        rms_norm_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
                gOut, x, gamma, eps, gX, gGamma, outer, lastDim);
    }
    CUDA_KERNEL_CHECK();
}

static void launch_softmax_last_dim(const float* src, float* dst, int nrows, int inner, bool use_fp16_softmax) {
    if (inner >= kSoftmaxBlockThreshold) {
        size_t smem = kSoftmaxNumWarps * sizeof(float);
        softmax_last_dim_block_kernel<<<nrows, kSoftmaxBlockDim, smem, kTensorCudaStream>>>(src, dst, nrows, inner);
    } else {
        int threads = jgpt_cuda_get_optimal_block_size();
        int blocks = (nrows + threads - 1) / threads;
        if (use_fp16_softmax) {
            softmax_last_dim_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(src, dst, nrows, inner);
        } else {
            softmax_last_dim_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(src, dst, nrows, inner);
        }
    }
    CUDA_KERNEL_CHECK();
}

static void launch_softmax_last_dim_bwd(const float* gOut, const float* p, float* gIn, int nrows, int inner) {
    if (inner >= kSoftmaxBlockThreshold) {
        launch_softmax_last_dim_bwd_block_chunked(
                gOut, p, gIn, static_cast<long long>(nrows), inner, kTensorCudaStream);
    } else {
        int threads = jgpt_cuda_get_optimal_block_size();
        int blocks = (nrows + threads - 1) / threads;
        softmax_last_dim_bwd_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(gOut, p, gIn, nrows, inner);
    }
    CUDA_KERNEL_CHECK();
}

static void launch_cross_entropy(const float* logits, const float* targets, float* grad, float scale,
        float* loss_sum, unsigned int* valid, int nrows, int vocab, bool use_fp16_softmax) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (nrows + threads - 1) / threads;
    if (use_fp16_softmax) {
        cross_entropy_softmax_grad_loss_kernel_fp16<<<blocks, threads, 0, kTensorCudaStream>>>(
                logits, targets, grad, scale, loss_sum, valid, nrows, vocab);
    } else {
        cross_entropy_softmax_grad_loss_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(
                logits, targets, grad, scale, loss_sum, valid, nrows, vocab);
    }
    CUDA_KERNEL_CHECK();
}

static void launch_cross_entropy_i32(const float* logits, const int* targets_i, float* grad, float scale,
        float* loss_sum, unsigned int* valid, int nrows, int vocab, bool use_fp16_softmax) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (nrows + threads - 1) / threads;
    if (use_fp16_softmax) {
        cross_entropy_softmax_grad_loss_kernel_fp16_i32<<<blocks, threads, 0, kTensorCudaStream>>>(
                logits, targets_i, grad, scale, loss_sum, valid, nrows, vocab);
    } else {
        cross_entropy_softmax_grad_loss_kernel_i32<<<blocks, threads, 0, kTensorCudaStream>>>(
                logits, targets_i, grad, scale, loss_sum, valid, nrows, vocab);
    }
    CUDA_KERNEL_CHECK();
}

__global__ void float_token_ids_to_int32_kernel(const float* src, int* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = static_cast<int>(src[i]);
    }
}

static void launch_float_to_int32_targets(const float* d_float_src, int* d_int_dst, int n) {
    int threads = jgpt_cuda_get_optimal_block_size();
    int blocks = (n + threads - 1) / threads;
    float_token_ids_to_int32_kernel<<<blocks, threads, 0, kTensorCudaStream>>>(d_float_src, d_int_dst, n);
    CUDA_KERNEL_CHECK();
}

static inline void* jni_direct_ptr(JNIEnv* env, jobject buf, jlong byte_off, jlong need_bytes, const char* ctx) {
    if (buf == nullptr) {
        fprintf(stderr, "%s: null buffer\n", ctx);
        return nullptr;
    }
    void* addr = env->GetDirectBufferAddress(buf);
    if (addr == nullptr) {
        fprintf(stderr, "%s: not a direct buffer\n", ctx);
        return nullptr;
    }
    jlong cap = env->GetDirectBufferCapacity(buf);
    if (byte_off < 0 || need_bytes < 0 || byte_off > cap || need_bytes > cap - byte_off) {
        fprintf(stderr, "%s: range out of capacity (off=%lld need=%lld cap=%lld)\n", ctx, (long long) byte_off,
                (long long) need_bytes, (long long) cap);
        return nullptr;
    }
    return static_cast<void*>(static_cast<char*>(addr) + byte_off);
}

static bool batched_sgemm_row_major_extra(
        const float* d_A, const float* d_B, float* d_C,
        int batchCount, int M, int K, int N, float alpha, float beta) {
    cublasHandle_t handle = get_extra_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm: cuBLAS handle unavailable\n");
        return false;
    }
    long long strideA = (long long) M * (long long) K;
    long long strideB = (long long) K * (long long) N;
    long long strideC = (long long) M * (long long) N;
    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_B,
            N,
            strideB,
            d_A,
            K,
            strideA,
            &beta,
            d_C,
            N,
            strideC,
            batchCount);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm failed: status %d\n", (int) st);
        return false;
    }
    return true;
}

// Row-major C(M×N) = A(M×K) × B^T,  где B хранится как N×K row-major.
// Устраняет необходимость явно транспонировать B перед GEMM.
static bool batched_sgemm_row_major_transB(
        const float* d_A, const float* d_B, float* d_C,
        int batchCount, int M, int K, int N, float alpha, float beta) {
    cublasHandle_t handle = get_extra_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm (transB): cuBLAS handle unavailable\n");
        return false;
    }
    long long strideA = (long long) M * K;
    long long strideB = (long long) N * K;   // B is N×K row-major
    long long strideC = (long long) M * N;
    // cuBLAS col-major: C_col(N×M) = B_col^T(N×K) × A_col(K×M)
    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,   // transpose B (K×N col-major → N×K)
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, K, strideB,    // B: N×K row-major = K×N col-major, ldb=K
            d_A, K, strideA,    // A: M×K row-major = K×M col-major, lda=K
            &beta,
            d_C, N, strideC,
            batchCount);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm (transB) failed: %d\n", (int) st);
        return false;
    }
    return true;
}

// Row-major C(M×N) = A^T × B,  где A хранится как K×M row-major, B — K×N row-major.
// Устраняет необходимость явно транспонировать A перед GEMM.
static bool batched_sgemm_row_major_transA(
        const float* d_A, const float* d_B, float* d_C,
        int batchCount, int M, int K, int N, float alpha, float beta) {
    cublasHandle_t handle = get_extra_cublas_handle();
    if (handle == nullptr) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm (transA): cuBLAS handle unavailable\n");
        return false;
    }
    long long strideA = (long long) K * M;   // A is K×M row-major
    long long strideB = (long long) K * N;   // B is K×N row-major
    long long strideC = (long long) M * N;
    // cuBLAS col-major: C_col(N×M) = B_col(N×K) × A_col^T(K×M)
    cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,   // transpose A (M×K col-major → K×M)
            N, M, K,
            &alpha,
            d_B, N, strideB,    // B: K×N row-major = N×K col-major, ldb=N
            d_A, M, strideA,    // A: K×M row-major = M×K col-major, lda=M
            &beta,
            d_C, N, strideC,
            batchCount);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[TensorOpsGPU] extra batched sgemm (transA) failed: %d\n", (int) st);
        return false;
    }
    return true;
}

/*
 * Отдельный буфер только для CUDA-graph пути (scaledDotProductAttentionForwardGPUDeviceResident
 * и jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas).
 *
 * КРИТИЧЕСКИ ВАЖНО: этот буфер НЕ должен совпадать с буфером jgpt_extra_tls().attn_fwd.
 * При захвате CUDA graph GPU-адреса d_scores/d_probs фиксируются в графе.
 * Если thread-local attn_fwd перевыделяется другими операциями (например, генерацией текста через
 * scaledDotProductAttentionForwardGPUDevice с маской), GPU-адреса в графе устаревают →
 * cudaGraphLaunch падает с "illegal memory access".
 * Использование отдельного буфера (не attn_fwd thread-local) гарантирует, что
 * нe-graph операции не могут изменить указатели, захваченные в графе.
 *
 * Глобальный (процессный) буфер с mutex: один и тот же device-адрес для всех слоёв декодера на одном GPU —
 * иначе thread_local копии не делят память между вызовами и при глубоком стеке растёт пик VRAM.
 *
 * Размер = bytesProb (одна матрица B×S×S): QK^T пишет logits, затем in-place softmax → probs, затем GEMM(P,V).
 * Раньше было 2×bytesProb (отдельные scores/probs); уполовинивание снимает ~100 MiB+ VRAM на типичных S и снижает OOM на cudaGraphLaunch.
 */
static std::mutex g_attn_fwd_graph_aux_mu;
static void* g_attn_fwd_graph_aux = nullptr;
static size_t g_attn_fwd_graph_aux_total = 0;

/** Workspace без слота под mask: mask передаётся отдельным device-указателем (для CUDA graph / без H2D probs). */
static bool attn_fwd_aux_ensure_qk_probs_only(
        size_t bytesQK,
        size_t bytesProb,
        float** d_scores,
        float** d_probs) {
    size_t total = bytesProb;
    std::lock_guard<std::mutex> lock(g_attn_fwd_graph_aux_mu);
    if (total > g_attn_fwd_graph_aux_total
            || (g_attn_fwd_graph_aux != nullptr && total < g_attn_fwd_graph_aux_total)) {
        cudaFree(g_attn_fwd_graph_aux);
        g_attn_fwd_graph_aux = nullptr;
        g_attn_fwd_graph_aux_total = 0;
        if (cudaMalloc(&g_attn_fwd_graph_aux, total) != cudaSuccess) {
            fprintf(
                    stderr,
                    "attn_fwd_aux_ensure_qk_probs_only: cudaMalloc %zu bytes failed: %s\n",
                    total,
                    cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        g_attn_fwd_graph_aux_total = total;
    }
    unsigned char* base = static_cast<unsigned char*>(g_attn_fwd_graph_aux);
    float* p = reinterpret_cast<float*>(base);
    *d_scores = p;
    *d_probs = p;
    (void) bytesQK;
    return true;
}

void jgpt_cuda_decoder_graph_debug_aux_snapshot(
        uintptr_t* fwd_ptr, uintptr_t* graph_ptr, size_t* fwd_sz, size_t* graph_sz) {
    if (fwd_ptr != nullptr) {
        *fwd_ptr = reinterpret_cast<uintptr_t>(jgpt_extra::jgpt_extra_tls().attn_fwd.blob.ptr);
    }
    if (fwd_sz != nullptr) {
        *fwd_sz = jgpt_extra::jgpt_extra_tls().attn_fwd.blob.bytes;
    }
    std::lock_guard<std::mutex> lock(g_attn_fwd_graph_aux_mu);
    if (graph_ptr != nullptr) {
        *graph_ptr = reinterpret_cast<uintptr_t>(g_attn_fwd_graph_aux);
    }
    if (graph_sz != nullptr) {
        *graph_sz = g_attn_fwd_graph_aux_total;
    }
}

static bool attn_fwd_run_core(
        float* d_q,
        float* d_k,
        float* d_v,
        float* d_scores,
        float* d_probs,
        float* d_out,
        float* d_mask,
        int batch,
        int seqLen,
        int dK,
        int dV,
        float scale,
        bool use_fp16_softmax) {
    (void) use_fp16_softmax;
    // Q × K^T → scores: transB GEMM (нет явного транспонирования K)
    if (!batched_sgemm_row_major_transB(d_q, d_k, d_scores, batch, seqLen, dK, seqLen, 1.0f, 0.0f)) {
        return false;
    }
    // Слитый scale + causal-mask + softmax: одно чтение global scores на строку (кэш z в shared).
    const long long nrows_ll = static_cast<long long>(batch) * static_cast<long long>(seqLen);
    if (!launch_softmax_scaled_masked_block_chunked(
                d_scores, d_probs, d_mask, scale, nrows_ll, seqLen, kTensorCudaStream)) {
        return false;
    }
    CUDA_KERNEL_CHECK_RV(false);
    if (!batched_sgemm_row_major_extra(d_probs, d_v, d_out, batch, seqLen, seqLen, dV, 1.0f, 0.0f)) {
        return false;
    }
    return true;
}

// ============================================================
//  FlashAttention-2 (causal, forward + backward)
//  Layout: Q/K/V/O  = [BH, S, Dh]  (BH = batch * numHeads)
//          LSE      = [BH, S]       log-sum-exp per query row
//          D        = [BH, S]       dot(dO, O) per query row (bwd scratch)
//  Tile sizes: Br = Bc = kFaBr.  Head dim: kFaDh (compile-time).
// ============================================================

static constexpr int kFaDh = 16;    // d_head
static constexpr int kFaBr = 64;    // query tile rows  (= block size for fwd / dQ)
static constexpr int kFaBc = 64;    // KV tile rows     (= block size for dKdV)

static_assert(kFaDh > 0 && (kFaDh % 2) == 0, "FlashAttention: d_head must be positive even");

/** Поднимает лимит dynamic shared memory, если плитка (Br/Bc/Dh) требует больше 48 KiB на блок. */
static cudaError_t flash_kernel_ensure_dyn_smem(const void* kernel, size_t dynamic_shmem) {
    constexpr size_t kDefaultDynSmem = 48u * 1024u;
    if (dynamic_shmem <= kDefaultDynSmem) {
        return cudaSuccess;
    }
    if (dynamic_shmem > static_cast<size_t>(INT_MAX)) {
        return cudaErrorInvalidValue;
    }
    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(dynamic_shmem));
}

/* Thread-local D: jgpt_extra_tls().flash_attn (fa_ensure_D в начале .cu). */

// ----------------------------------------------------------------
//  Forward kernel
//  One block per (bh, q_tile).  Block: kFaBr threads.
//  Shared memory layout (floats):
//    q_smem [kFaBr][kFaDh]   — query tile, loaded once
//    k_smem [kFaBc][kFaDh]   — key tile, per KV step
//    v_smem [kFaBc][kFaDh]   — value tile, per KV step
//    s_smem [kFaBc][kFaBr]   — S_ij transposed: s_smem[j][i] = S[i][j]
//                               (transposed layout avoids bank conflicts when
//                                all threads read their own s column)
// ----------------------------------------------------------------
__global__ void __launch_bounds__(kFaBr)
flash_attn_fwd_kernel(
    const float* __restrict__ Q,    // [BH, S, Dh]
    const float* __restrict__ K,    // [BH, S, Dh]
    const float* __restrict__ V,    // [BH, S, Dh]
    float* __restrict__ O,          // [BH, S, Dh]
    float* __restrict__ LSE,        // [BH, S]
    int BH, int S, float scale)
{
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const int bh     = blockIdx.x / num_q_tiles;
    const int q_tile = blockIdx.x % num_q_tiles;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int qi  = q_tile * kFaBr + tid;

    const ptrdiff_t bh_off = (ptrdiff_t)bh * S * kFaDh;
    const float* Qp   = Q   + bh_off;
    const float* Kp   = K   + bh_off;
    const float* Vp   = V   + bh_off;
    float*       Op   = O   + bh_off;
    float*       LSEp = LSE + (ptrdiff_t)bh * S;

    // Smem
    extern __shared__ float smem[];
    float* q_smem = smem;                                         // [kFaBr][kFaDh]
    float* k_smem = q_smem + kFaBr * kFaDh;                      // [kFaBc][kFaDh]
    float* v_smem = k_smem + kFaBc * kFaDh;                      // [kFaBc][kFaDh]
    float* s_smem = v_smem + kFaBc * kFaDh;                      // [kFaBc][kFaBr] transposed

    // Load Q tile once (one row per thread)
    if (qi < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++)
            q_smem[tid * kFaDh + d] = Qp[qi * kFaDh + d];
    } else {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++)
            q_smem[tid * kFaDh + d] = 0.f;
    }
    __syncthreads();

    // Per-thread accumulators
    float o_reg[kFaDh];
    #pragma unroll
    for (int d = 0; d < kFaDh; d++) o_reg[d] = 0.f;
    float mi = -INFINITY;
    float li = 0.f;

    const int num_kv_tiles = (S + kFaBc - 1) / kFaBc;

    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        const int kv_start = kv_t * kFaBc;
        // Causal: skip tiles where ALL keys are beyond this query tile's last row
        if (kv_start > q_tile * kFaBr + kFaBr - 1) break;

        // Collaborative load of K and V tiles (kFaBr threads load kFaBc rows)
        for (int row = tid; row < kFaBc; row += kFaBr) {
            const int kj = kv_start + row;
            if (kj < S) {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[row * kFaDh + d] = Kp[kj * kFaDh + d];
                    v_smem[row * kFaDh + d] = Vp[kj * kFaDh + d];
                }
            } else {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[row * kFaDh + d] = 0.f;
                    v_smem[row * kFaDh + d] = 0.f;
                }
            }
        }
        __syncthreads();

        /* Все потоки блока обязаны выполнять одинаковое число __syncthreads() на итерацию.
         * Нельзя вызывать __syncthreads() только для qi >= S (ветка continue) — иначе UB CUDA. */
        if (qi < S) {
            // Compute S_ij for this thread's query row, store transposed: s_smem[j][tid]
            #pragma unroll
            for (int j = 0; j < kFaBc; j++) {
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dot = __fmaf_rn(q_smem[tid * kFaDh + d], k_smem[j * kFaDh + d], dot);
                const int kj = kv_start + j;
                float s = dot * scale;
                if (qi < kj || kj >= S) s = -INFINITY;  // causal + boundary
                s_smem[j * kFaBr + tid] = s;             // transposed: [j][tid]
            }

            // Online softmax
            float mij = -INFINITY;
            #pragma unroll
            for (int j = 0; j < kFaBc; j++)
                mij = fmaxf(mij, s_smem[j * kFaBr + tid]);

            float lij = 0.f;
            #pragma unroll
            for (int j = 0; j < kFaBc; j++) {
                float p = __expf(s_smem[j * kFaBr + tid] - mij);
                s_smem[j * kFaBr + tid] = p;
                lij += p;
            }

            const float mi_new = fmaxf(mi, mij);
            const float alpha  = __expf(mi  - mi_new);
            const float beta   = __expf(mij - mi_new);
            const float li_new = alpha * li + beta * lij;

            // O update: O_new = alpha*O_old + beta * sum_j(p_j * V_j)
            #pragma unroll
            for (int d = 0; d < kFaDh; d++) {
                float vacc = 0.f;
                #pragma unroll
                for (int j = 0; j < kFaBc; j++)
                    vacc = __fmaf_rn(s_smem[j * kFaBr + tid], v_smem[j * kFaDh + d], vacc);
                o_reg[d] = alpha * o_reg[d] + beta * vacc;
            }

            mi = mi_new;
            li = li_new;
        }
        __syncthreads();
    }

    if (qi < S) {
        const float inv_l = (li > 0.f) ? __frcp_rn(li) : 0.f;
        #pragma unroll
        for (int d = 0; d < kFaDh; d++)
            Op[qi * kFaDh + d] = o_reg[d] * inv_l;
        LSEp[qi] = mi + __logf(fmaxf(li, 1e-12f));
    }
}

// ----------------------------------------------------------------
//  D precompute kernel: D[bh][qi] = dot(dO[bh,qi], O[bh,qi])
//  Grid: ceil(BH*S / 64),  Block: 64.
// ----------------------------------------------------------------
__global__ void flash_attn_compute_D_kernel(
    const float* __restrict__ dO,  // [BH, S, kFaDh]
    const float* __restrict__ O,   // [BH, S, kFaDh]
    float* __restrict__ D,         // [BH, S]
    int BH, int S)
{
    const int idx = blockIdx.x * 64 + threadIdx.x;
    const int bh  = idx / S;
    const int qi  = idx % S;
    if (bh >= BH || qi >= S) return;

    const ptrdiff_t base = (ptrdiff_t)bh * S * kFaDh + (ptrdiff_t)qi * kFaDh;
    float acc = 0.f;
    #pragma unroll
    for (int d = 0; d < kFaDh; d++)
        acc += dO[base + d] * O[base + d];
    D[(ptrdiff_t)bh * S + qi] = acc;
}

// ----------------------------------------------------------------
//  Backward: dK and dV
//  One block per (bh, kv_tile).  Block: kFaBc threads.
//  Each block iterates over all Q tiles, accumulates dK/dV in registers.
//  Shared memory:
//    k_smem [kFaBc][kFaDh]   — key tile for this block (fixed)
//    v_smem [kFaBc][kFaDh]   — value tile (fixed)
//    q_smem [kFaBr][kFaDh]   — query tile (per Q step)
//    do_smem[kFaBr][kFaDh]   — dO tile (per Q step)
// ----------------------------------------------------------------
__global__ void __launch_bounds__(kFaBc)
flash_attn_bwd_dkdv_kernel(
    const float* __restrict__ Q,    // [BH, S, Dh]
    const float* __restrict__ K,    // [BH, S, Dh]
    const float* __restrict__ V,    // [BH, S, Dh]
    const float* __restrict__ dO,   // [BH, S, Dh]
    const float* __restrict__ LSE,  // [BH, S]
    const float* __restrict__ D,    // [BH, S]
    float* __restrict__ dK,         // [BH, S, Dh]
    float* __restrict__ dV,         // [BH, S, Dh]
    int BH, int S, float scale)
{
    const int num_kv_tiles = (S + kFaBc - 1) / kFaBc;
    const int bh      = blockIdx.x / num_kv_tiles;
    const int kv_tile = blockIdx.x % num_kv_tiles;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int kj  = kv_tile * kFaBc + tid;

    const ptrdiff_t bh_off = (ptrdiff_t)bh * S * kFaDh;
    const float* Qp   = Q   + bh_off;
    const float* Kp   = K   + bh_off;
    const float* Vp   = V   + bh_off;
    const float* dOp  = dO  + bh_off;
    const float* LSEp = LSE + (ptrdiff_t)bh * S;
    const float* Dp   = D   + (ptrdiff_t)bh * S;
    float*       dKp  = dK  + bh_off;
    float*       dVp  = dV  + bh_off;

    extern __shared__ float smem[];
    float* k_smem  = smem;                                          // [kFaBc][kFaDh]
    float* v_smem  = k_smem  + kFaBc * kFaDh;                     // [kFaBc][kFaDh]
    float* q_smem  = v_smem  + kFaBc * kFaDh;                     // [kFaBr][kFaDh]
    float* do_smem = q_smem  + kFaBr * kFaDh;                     // [kFaBr][kFaDh]

    // Load K/V tile once for this block
    if (kj < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            k_smem[tid * kFaDh + d] = Kp[kj * kFaDh + d];
            v_smem[tid * kFaDh + d] = Vp[kj * kFaDh + d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            k_smem[tid * kFaDh + d] = 0.f;
            v_smem[tid * kFaDh + d] = 0.f;
        }
    }
    __syncthreads();

    // dK/dV accumulators in registers
    float dk_reg[kFaDh], dv_reg[kFaDh];
    #pragma unroll
    for (int d = 0; d < kFaDh; d++) { dk_reg[d] = 0.f; dv_reg[d] = 0.f; }

    // For causal: only Q tiles where qi_start <= kj (= kv_tile*kFaBc + kFaBc-1 at most)
    // So q_tile_start = kv_tile * kFaBc / kFaBr (integer division — first q_tile with any valid qi)
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const int q_tile_start = (kv_tile * kFaBc) / kFaBr;

    for (int q_tile = q_tile_start; q_tile < num_q_tiles; q_tile++) {
        const int qi_base = q_tile * kFaBr;

        // Load Q and dO tiles (kFaBc threads load kFaBr rows, strided)
        for (int row = tid; row < kFaBr; row += kFaBc) {
            const int qi = qi_base + row;
            if (qi < S) {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    q_smem [row * kFaDh + d] = Qp [qi * kFaDh + d];
                    do_smem[row * kFaDh + d] = dOp[qi * kFaDh + d];
                }
            } else {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    q_smem [row * kFaDh + d] = 0.f;
                    do_smem[row * kFaDh + d] = 0.f;
                }
            }
        }
        __syncthreads();

        // For kFaBc=kFaBr=64: each thread computes its KV column (tid) across all Br query rows.
        // Compute P_ij and dS_ij, accumulate dK[tid] and dV[tid].
        if (kj < S) {
            #pragma unroll
            for (int i = 0; i < kFaBr; i++) {
                const int qi = qi_base + i;
                if (qi >= S) break;

                // S_ij = dot(Q[qi], K[kj]) * scale
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dot = __fmaf_rn(q_smem[i * kFaDh + d], k_smem[tid * kFaDh + d], dot);
                float s = dot * scale;
                if (qi < kj) s = -INFINITY;  // causal

                // P_ij = exp(S_ij - LSE[qi])
                float lse_qi = LSEp[qi];
                float p = (s == -INFINITY) ? 0.f : __expf(s - lse_qi);

                // dV[kj] += P_ij * dO[qi]  → accumulate in dv_reg
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dv_reg[d] = __fmaf_rn(p, do_smem[i * kFaDh + d], dv_reg[d]);

                // dP_ij = dot(dO[qi], V[kj])
                float dp = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dp = __fmaf_rn(do_smem[i * kFaDh + d], v_smem[tid * kFaDh + d], dp);

                // dS_ij = P_ij * (dP_ij - D[qi]) * scale
                float ds = p * (dp - Dp[qi]) * scale;

                // dK[kj] += dS_ij * Q[qi]
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dk_reg[d] = __fmaf_rn(ds, q_smem[i * kFaDh + d], dk_reg[d]);
            }
        }
        __syncthreads();
    }

    // Write dK, dV
    if (kj < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            dKp[kj * kFaDh + d] = dk_reg[d];
            dVp[kj * kFaDh + d] = dv_reg[d];
        }
    }
}

// ----------------------------------------------------------------
//  Backward: dQ
//  One block per (bh, q_tile).  Block: kFaBr threads.
//  Each block iterates over KV tiles (causal: only kv_t with kv_start <= qi).
//  Shared memory:
//    q_smem [kFaBr][kFaDh]   — query tile (fixed)
//    do_smem[kFaBr][kFaDh]   — dO tile (fixed)
//    k_smem [kFaBc][kFaDh]   — key tile (per KV step)
//    v_smem [kFaBc][kFaDh]   — value tile (per KV step)
// ----------------------------------------------------------------
__global__ void __launch_bounds__(kFaBr)
flash_attn_bwd_dq_kernel(
    const float* __restrict__ Q,    // [BH, S, Dh]
    const float* __restrict__ K,    // [BH, S, Dh]
    const float* __restrict__ V,    // [BH, S, Dh]
    const float* __restrict__ dO,   // [BH, S, Dh]
    const float* __restrict__ LSE,  // [BH, S]
    const float* __restrict__ D,    // [BH, S]
    float* __restrict__ dQ,         // [BH, S, Dh]
    int BH, int S, float scale)
{
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const int bh     = blockIdx.x / num_q_tiles;
    const int q_tile = blockIdx.x % num_q_tiles;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int qi  = q_tile * kFaBr + tid;

    const ptrdiff_t bh_off = (ptrdiff_t)bh * S * kFaDh;
    const float* Qp   = Q   + bh_off;
    const float* Kp   = K   + bh_off;
    const float* Vp   = V   + bh_off;
    const float* dOp  = dO  + bh_off;
    const float* LSEp = LSE + (ptrdiff_t)bh * S;
    const float* Dp   = D   + (ptrdiff_t)bh * S;
    float*       dQp  = dQ  + bh_off;

    extern __shared__ float smem[];
    float* q_smem  = smem;                                          // [kFaBr][kFaDh]
    float* do_smem = q_smem  + kFaBr * kFaDh;                     // [kFaBr][kFaDh]
    float* k_smem  = do_smem + kFaBr * kFaDh;                     // [kFaBc][kFaDh]
    float* v_smem  = k_smem  + kFaBc * kFaDh;                     // [kFaBc][kFaDh]

    // Load Q and dO tiles once
    if (qi < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            q_smem [tid * kFaDh + d] = Qp [qi * kFaDh + d];
            do_smem[tid * kFaDh + d] = dOp[qi * kFaDh + d];
        }
    } else {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            q_smem [tid * kFaDh + d] = 0.f;
            do_smem[tid * kFaDh + d] = 0.f;
        }
    }
    __syncthreads();

    float dq_reg[kFaDh];
    #pragma unroll
    for (int d = 0; d < kFaDh; d++) dq_reg[d] = 0.f;

    float mi  = (qi < S) ? LSEp[qi] : 0.f;
    float di  = (qi < S) ? Dp  [qi] : 0.f;

    const int num_kv_tiles = (S + kFaBc - 1) / kFaBc;

    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        const int kv_start = kv_t * kFaBc;
        if (kv_start > q_tile * kFaBr + kFaBr - 1) break;  // causal: no more valid KV tiles

        // Load K and V tiles
        for (int row = tid; row < kFaBc; row += kFaBr) {
            const int kj = kv_start + row;
            if (kj < S) {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[row * kFaDh + d] = Kp[kj * kFaDh + d];
                    v_smem[row * kFaDh + d] = Vp[kj * kFaDh + d];
                }
            } else {
                #pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[row * kFaDh + d] = 0.f;
                    v_smem[row * kFaDh + d] = 0.f;
                }
            }
        }
        __syncthreads();

        if (qi < S) {
            // Each thread computes its query row across all Bc KV positions
            #pragma unroll
            for (int j = 0; j < kFaBc; j++) {
                const int kj = kv_start + j;
                if (kj >= S) break;

                // S_ij = dot(Q[qi], K[kj]) * scale
                float dot = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dot = __fmaf_rn(q_smem[tid * kFaDh + d], k_smem[j * kFaDh + d], dot);
                float s = dot * scale;
                if (qi < kj) s = -INFINITY;

                float p = (s == -INFINITY) ? 0.f : __expf(s - mi);

                // dP_ij = dot(dO[qi], V[kj])
                float dp = 0.f;
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dp = __fmaf_rn(do_smem[tid * kFaDh + d], v_smem[j * kFaDh + d], dp);

                float ds = p * (dp - di) * scale;

                // dQ[qi] += dS_ij * K[kj]
                #pragma unroll
                for (int d = 0; d < kFaDh; d++)
                    dq_reg[d] = __fmaf_rn(ds, k_smem[j * kFaDh + d], dq_reg[d]);
            }
        }
        __syncthreads();
    }

    if (qi < S) {
        #pragma unroll
        for (int d = 0; d < kFaDh; d++)
            dQp[qi * kFaDh + d] = dq_reg[d];
    }
}

// ----------------------------------------------------------------
//  Host-side launchers
// ----------------------------------------------------------------
static bool flash_attn_sync_stream_ok(const char* ctx) {
    return jgpt_cuda_sync_stream_unless_capturing(ctx) != 0;
}

static bool flash_attn_fwd_run(
        const float* d_q, const float* d_k, const float* d_v,
        float* d_o, float* d_lse,
        int BH, int S, float scale)
{
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const long long grid_ll = (long long) BH * (long long) num_q_tiles;
    if (grid_ll <= 0LL || grid_ll > static_cast<long long>(INT_MAX)) {
        fprintf(
                stderr,
                "flash_attn_fwd: grid overflow BH=%d S=%d num_q_tiles=%d (blocks=%lld)\n",
                BH,
                S,
                num_q_tiles,
                static_cast<long long>(grid_ll));
        return false;
    }
    const int grid = static_cast<int>(grid_ll);
    constexpr size_t smem =
        (kFaBr + 2 * kFaBc) * kFaDh * sizeof(float)   // q + k + v tiles
        + kFaBc * kFaBr * sizeof(float);               // s_smem [kFaBc][kFaBr]
    cudaError_t sa = flash_kernel_ensure_dyn_smem((const void*)flash_attn_fwd_kernel, smem);
    if (sa != cudaSuccess) {
        fprintf(stderr, "flash_attn_fwd: cudaFuncSetAttribute smem=%zu: %s\n", smem, cudaGetErrorString(sa));
        return false;
    }
    flash_attn_fwd_kernel<<<grid, kFaBr, smem, kTensorCudaStream>>>(
            d_q, d_k, d_v, d_o, d_lse, BH, S, scale);
    CUDA_KERNEL_CHECK_RV(false);
    return flash_attn_sync_stream_ok("flash_attn_fwd");
}

static bool flash_attn_bwd_run(
        const float* d_q, const float* d_k, const float* d_v,
        const float* d_o, const float* d_do,
        const float* d_lse,
        float* d_dq, float* d_dk, float* d_dv,
        int BH, int S, float scale)
{
    const size_t qkv_bytes = (size_t)BH * (size_t)S * (size_t)kFaDh * sizeof(float);
    if (cudaMemsetAsync(d_dq, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }
    if (cudaMemsetAsync(d_dk, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }
    if (cudaMemsetAsync(d_dv, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }

    // 1. Compute D[bh][qi] = dot(dO[qi], O[qi])
    const long long total_rows_ll = (long long) BH * (long long) S;
    if (total_rows_ll <= 0LL || total_rows_ll > static_cast<long long>(INT_MAX)) {
        fprintf(
                stderr,
                "flash_attn_bwd: total_rows overflow BH=%d S=%d (rows=%lld)\n",
                BH,
                S,
                static_cast<long long>(total_rows_ll));
        return false;
    }
    const int total_rows = static_cast<int>(total_rows_ll);
    const int d_grid = (total_rows + 63) / 64;
    flash_attn_compute_D_kernel<<<d_grid, 64, 0, kTensorCudaStream>>>(
            d_do, d_o, jgpt_extra::jgpt_extra_tls().flash_attn.d_D, BH, S);
    CUDA_KERNEL_CHECK_RV(false);

    // 2. dK, dV kernel (one block per kv_tile)
    constexpr size_t smem_dkdv =
        2 * kFaBc * kFaDh * sizeof(float)                                // k + v fixed
        + 2 * kFaBr * kFaDh * sizeof(float);                             // q + do per q_tile
    const int num_kv_tiles = (S + kFaBc - 1) / kFaBc;
    const long long grid_dkdv_ll = (long long) BH * (long long) num_kv_tiles;
    if (grid_dkdv_ll <= 0LL || grid_dkdv_ll > static_cast<long long>(INT_MAX)) {
        fprintf(
                stderr,
                "flash_attn_bwd dkdv: grid overflow BH=%d S=%d num_kv_tiles=%d (blocks=%lld)\n",
                BH,
                S,
                num_kv_tiles,
                static_cast<long long>(grid_dkdv_ll));
        return false;
    }
    const int grid_dkdv = static_cast<int>(grid_dkdv_ll);
    cudaError_t sb = flash_kernel_ensure_dyn_smem((const void*)flash_attn_bwd_dkdv_kernel, smem_dkdv);
    if (sb != cudaSuccess) {
        fprintf(stderr, "flash_attn_bwd dkdv: cudaFuncSetAttribute smem=%zu: %s\n", smem_dkdv, cudaGetErrorString(sb));
        return false;
    }
    flash_attn_bwd_dkdv_kernel<<<grid_dkdv, kFaBc, smem_dkdv, kTensorCudaStream>>>(
            d_q, d_k, d_v, d_do, d_lse, jgpt_extra::jgpt_extra_tls().flash_attn.d_D,
            d_dk, d_dv, BH, S, scale);
    CUDA_KERNEL_CHECK_RV(false);

    // 3. dQ kernel (one block per q_tile)
    constexpr size_t smem_dq =
        2 * kFaBr * kFaDh * sizeof(float)                                // q + do fixed
        + 2 * kFaBc * kFaDh * sizeof(float);                             // k + v per kv_tile
    const int num_q_tiles = (S + kFaBr - 1) / kFaBr;
    const long long grid_dq_ll = (long long) BH * (long long) num_q_tiles;
    if (grid_dq_ll <= 0LL || grid_dq_ll > static_cast<long long>(INT_MAX)) {
        fprintf(
                stderr,
                "flash_attn_bwd dq: grid overflow BH=%d S=%d num_q_tiles=%d (blocks=%lld)\n",
                BH,
                S,
                num_q_tiles,
                static_cast<long long>(grid_dq_ll));
        return false;
    }
    const int grid_dq = static_cast<int>(grid_dq_ll);
    sb = flash_kernel_ensure_dyn_smem((const void*)flash_attn_bwd_dq_kernel, smem_dq);
    if (sb != cudaSuccess) {
        fprintf(stderr, "flash_attn_bwd dq: cudaFuncSetAttribute smem=%zu: %s\n", smem_dq, cudaGetErrorString(sb));
        return false;
    }
    flash_attn_bwd_dq_kernel<<<grid_dq, kFaBr, smem_dq, kTensorCudaStream>>>(
            d_q, d_k, d_v, d_do, d_lse, jgpt_extra::jgpt_extra_tls().flash_attn.d_D,
            d_dq, BH, S, scale);
    CUDA_KERNEL_CHECK_RV(false);
    return flash_attn_sync_stream_ok("flash_attn_bwd");
}
