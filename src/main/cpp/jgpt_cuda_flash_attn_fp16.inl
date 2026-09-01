/* FP16 I/O + Tensor Core (WMMA m16n16k16) FlashAttention for d_head=16.
 * Softmax / online rescaling stay FP32. Included from jgpt_cuda_extra_kernels_launches.inl.
 */
#include <mma.h>
#include <cuda_fp16.h>

static constexpr int kFaTcBr = 64;
static constexpr int kFaTcBc = 64;
static constexpr int kFaTcThreads = 128;

static constexpr size_t kFaTcFwdSmemBytes =
        (size_t) kFaTcBr * kFaDh * sizeof(__half)     /* Q */
        + (size_t) kFaTcBc * kFaDh * sizeof(__half)   /* K */
        + (size_t) kFaTcBc * kFaDh * sizeof(__half)   /* V */
        + (size_t) kFaTcBr * kFaTcBc * sizeof(float); /* S / P */

static constexpr size_t kFaTcBwdDkdvSmemBytes =
        (size_t) (2 * kFaTcBc + 2 * kFaTcBr) * kFaDh * sizeof(__half);

static constexpr size_t kFaTcBwdDqSmemBytes = kFaTcBwdDkdvSmemBytes;

/** Q @ K^T via Tensor Cores; P @ V in FP32 (N=d_head=16). */
__global__ void __launch_bounds__(kFaTcThreads)
flash_attn_fwd_kernel_tc(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float* __restrict__ O,
        float* __restrict__ LSE,
        int BH,
        int S,
        float scale)
{
    const int num_q_tiles = (S + kFaTcBr - 1) / kFaTcBr;
    const int bh = blockIdx.x / num_q_tiles;
    const int q_tile = blockIdx.x % num_q_tiles;
    if (bh >= BH) {
        return;
    }

    const int tid = threadIdx.x;
    const int qi_base = q_tile * kFaTcBr;

    const ptrdiff_t bh_off = (ptrdiff_t) bh * S * kFaDh;
    const float* Qp = Q + bh_off;
    const float* Kp = K + bh_off;
    const float* Vp = V + bh_off;
    float* Op = O + bh_off;
    float* LSEp = LSE + (ptrdiff_t) bh * S;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* q_h = reinterpret_cast<__half*>(smem_raw);
    __half* k_h = q_h + kFaTcBr * kFaDh;
    __half* v_h = k_h + kFaTcBc * kFaDh;
    float* s_f = reinterpret_cast<float*>(v_h + kFaTcBc * kFaDh);

    for (int idx = tid; idx < kFaTcBr * kFaDh; idx += kFaTcThreads) {
        const int row = idx / kFaDh;
        const int d = idx % kFaDh;
        const int qi = qi_base + row;
        q_h[idx] = (qi < S) ? __float2half_rn(Qp[qi * kFaDh + d]) : __float2half_rn(0.f);
    }
    __syncthreads();

    float o_reg[kFaDh];
#pragma unroll
    for (int d = 0; d < kFaDh; d++) {
        o_reg[d] = 0.f;
    }
    float mi = -INFINITY;
    float li = 0.f;
    const int row = tid; /* threads 0..63 own one query row of the tile */

    const int num_kv_tiles = (S + kFaTcBc - 1) / kFaTcBc;
    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        const int kv_start = kv_t * kFaTcBc;
        if (kv_start > qi_base + kFaTcBr - 1) {
            break;
        }

        for (int idx = tid; idx < kFaTcBc * kFaDh; idx += kFaTcThreads) {
            const int r = idx / kFaDh;
            const int d = idx % kFaDh;
            const int kj = kv_start + r;
            const float kv = (kj < S) ? Kp[kj * kFaDh + d] : 0.f;
            const float vv = (kj < S) ? Vp[kj * kFaDh + d] : 0.f;
            k_h[idx] = __float2half_rn(kv);
            v_h[idx] = __float2half_rn(vv);
        }
        __syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        {
            const int warp = tid / 32;
            if (warp < (kFaTcBr / 16)) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major>
                    a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major>
                    b_frag;
            nvcuda::wmma::load_matrix_sync(a_frag, q_h + warp * 16 * kFaDh, kFaDh);
#pragma unroll
            for (int nt = 0; nt < kFaTcBc / 16; nt++) {
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
                nvcuda::wmma::fill_fragment(c_frag, 0.f);
                nvcuda::wmma::load_matrix_sync(b_frag, k_h + nt * 16 * kFaDh, kFaDh);
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                nvcuda::wmma::store_matrix_sync(
                        s_f + warp * 16 * kFaTcBc + nt * 16,
                        c_frag,
                        kFaTcBc,
                        nvcuda::wmma::mem_row_major);
            }
            }
        }
#else
        for (int idx = tid; idx < kFaTcBr * kFaTcBc; idx += kFaTcThreads) {
            const int r = idx / kFaTcBc;
            const int c = idx % kFaTcBc;
            float dot = 0.f;
#pragma unroll
            for (int d = 0; d < kFaDh; d++) {
                dot = __fmaf_rn(__half2float(q_h[r * kFaDh + d]), __half2float(k_h[c * kFaDh + d]), dot);
            }
            s_f[idx] = dot;
        }
#endif
        __syncthreads();

        if (row < kFaTcBr) {
            const int qi = qi_base + row;
#pragma unroll
            for (int j = 0; j < kFaTcBc; j++) {
                const int kj = kv_start + j;
                float s = s_f[row * kFaTcBc + j] * scale;
                if (qi >= S || kj >= S || qi < kj) {
                    s = -INFINITY;
                }
                s_f[row * kFaTcBc + j] = s;
            }

            float mij = -INFINITY;
#pragma unroll
            for (int j = 0; j < kFaTcBc; j++) {
                mij = fmaxf(mij, s_f[row * kFaTcBc + j]);
            }
            float lij = 0.f;
#pragma unroll
            for (int j = 0; j < kFaTcBc; j++) {
                const float p = __expf(s_f[row * kFaTcBc + j] - mij);
                s_f[row * kFaTcBc + j] = p;
                lij += p;
            }

            const float mi_new = fmaxf(mi, mij);
            const float alpha = __expf(mi - mi_new);
            const float beta = __expf(mij - mi_new);
            const float li_new = alpha * li + beta * lij;

#pragma unroll
            for (int d = 0; d < kFaDh; d++) {
                float vacc = 0.f;
#pragma unroll
                for (int j = 0; j < kFaTcBc; j++) {
                    vacc = __fmaf_rn(s_f[row * kFaTcBc + j], __half2float(v_h[j * kFaDh + d]), vacc);
                }
                o_reg[d] = alpha * o_reg[d] + beta * vacc;
            }
            mi = mi_new;
            li = li_new;
        }
        __syncthreads();
    }

    if (row < kFaTcBr) {
        const int qi = qi_base + row;
        if (qi < S) {
            const float inv_l = (li > 0.f) ? __frcp_rn(li) : 0.f;
#pragma unroll
            for (int d = 0; d < kFaDh; d++) {
                Op[qi * kFaDh + d] = o_reg[d] * inv_l;
            }
            LSEp[qi] = mi + __logf(fmaxf(li, 1e-12f));
        }
    }
}

__global__ void __launch_bounds__(kFaTcBc)
flash_attn_bwd_dkdv_kernel_fp16(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        const float* __restrict__ dO,
        const float* __restrict__ LSE,
        const float* __restrict__ D,
        float* __restrict__ dK,
        float* __restrict__ dV,
        int BH,
        int S,
        float scale)
{
    const int num_kv_tiles = (S + kFaTcBc - 1) / kFaTcBc;
    const int bh = blockIdx.x / num_kv_tiles;
    const int kv_tile = blockIdx.x % num_kv_tiles;
    if (bh >= BH) {
        return;
    }

    const int tid = threadIdx.x;
    const int kj = kv_tile * kFaTcBc + tid;
    const ptrdiff_t bh_off = (ptrdiff_t) bh * S * kFaDh;
    const float* Qp = Q + bh_off;
    const float* Kp = K + bh_off;
    const float* Vp = V + bh_off;
    const float* dOp = dO + bh_off;
    const float* LSEp = LSE + (ptrdiff_t) bh * S;
    const float* Dp = D + (ptrdiff_t) bh * S;
    float* dKp = dK + bh_off;
    float* dVp = dV + bh_off;

    extern __shared__ __align__(16) unsigned char smem_bwd[];
    __half* k_smem = reinterpret_cast<__half*>(smem_bwd);
    __half* v_smem = k_smem + kFaTcBc * kFaDh;
    __half* q_smem = v_smem + kFaTcBc * kFaDh;
    __half* do_smem = q_smem + kFaTcBr * kFaDh;

    if (kj < S) {
#pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            k_smem[tid * kFaDh + d] = __float2half_rn(Kp[kj * kFaDh + d]);
            v_smem[tid * kFaDh + d] = __float2half_rn(Vp[kj * kFaDh + d]);
        }
    } else {
#pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            k_smem[tid * kFaDh + d] = __float2half_rn(0.f);
            v_smem[tid * kFaDh + d] = __float2half_rn(0.f);
        }
    }
    __syncthreads();

    float dk_reg[kFaDh], dv_reg[kFaDh];
#pragma unroll
    for (int d = 0; d < kFaDh; d++) {
        dk_reg[d] = 0.f;
        dv_reg[d] = 0.f;
    }

    const int num_q_tiles = (S + kFaTcBr - 1) / kFaTcBr;
    const int q_tile_start = (kv_tile * kFaTcBc) / kFaTcBr;
    for (int q_tile = q_tile_start; q_tile < num_q_tiles; q_tile++) {
        const int qi_base = q_tile * kFaTcBr;
        for (int r = tid; r < kFaTcBr; r += kFaTcBc) {
            const int qi = qi_base + r;
            if (qi < S) {
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    q_smem[r * kFaDh + d] = __float2half_rn(Qp[qi * kFaDh + d]);
                    do_smem[r * kFaDh + d] = __float2half_rn(dOp[qi * kFaDh + d]);
                }
            } else {
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    q_smem[r * kFaDh + d] = __float2half_rn(0.f);
                    do_smem[r * kFaDh + d] = __float2half_rn(0.f);
                }
            }
        }
        __syncthreads();

        if (kj < S) {
#pragma unroll
            for (int i = 0; i < kFaTcBr; i++) {
                const int qi = qi_base + i;
                if (qi >= S) {
                    break;
                }
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    dot = __fmaf_rn(
                            __half2float(q_smem[i * kFaDh + d]),
                            __half2float(k_smem[tid * kFaDh + d]),
                            dot);
                }
                float s = dot * scale;
                if (qi < kj) {
                    s = -INFINITY;
                }
                const float p = (s == -INFINITY) ? 0.f : __expf(s - LSEp[qi]);
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    dv_reg[d] = __fmaf_rn(p, __half2float(do_smem[i * kFaDh + d]), dv_reg[d]);
                }
                float dp = 0.f;
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    dp = __fmaf_rn(
                            __half2float(do_smem[i * kFaDh + d]),
                            __half2float(v_smem[tid * kFaDh + d]),
                            dp);
                }
                const float ds = p * (dp - Dp[qi]) * scale;
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    dk_reg[d] = __fmaf_rn(ds, __half2float(q_smem[i * kFaDh + d]), dk_reg[d]);
                }
            }
        }
        __syncthreads();
    }

    if (kj < S) {
#pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            dKp[kj * kFaDh + d] = dk_reg[d];
            dVp[kj * kFaDh + d] = dv_reg[d];
        }
    }
}

__global__ void __launch_bounds__(kFaTcBr)
flash_attn_bwd_dq_kernel_fp16(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        const float* __restrict__ dO,
        const float* __restrict__ LSE,
        const float* __restrict__ D,
        float* __restrict__ dQ,
        int BH,
        int S,
        float scale)
{
    const int num_q_tiles = (S + kFaTcBr - 1) / kFaTcBr;
    const int bh = blockIdx.x / num_q_tiles;
    const int q_tile = blockIdx.x % num_q_tiles;
    if (bh >= BH) {
        return;
    }

    const int tid = threadIdx.x;
    const int qi = q_tile * kFaTcBr + tid;
    const ptrdiff_t bh_off = (ptrdiff_t) bh * S * kFaDh;
    const float* Qp = Q + bh_off;
    const float* Kp = K + bh_off;
    const float* Vp = V + bh_off;
    const float* dOp = dO + bh_off;
    const float* LSEp = LSE + (ptrdiff_t) bh * S;
    const float* Dp = D + (ptrdiff_t) bh * S;
    float* dQp = dQ + bh_off;

    extern __shared__ __align__(16) unsigned char smem_dq[];
    __half* q_smem = reinterpret_cast<__half*>(smem_dq);
    __half* do_smem = q_smem + kFaTcBr * kFaDh;
    __half* k_smem = do_smem + kFaTcBr * kFaDh;
    __half* v_smem = k_smem + kFaTcBc * kFaDh;

    if (qi < S) {
#pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            q_smem[tid * kFaDh + d] = __float2half_rn(Qp[qi * kFaDh + d]);
            do_smem[tid * kFaDh + d] = __float2half_rn(dOp[qi * kFaDh + d]);
        }
    } else {
#pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            q_smem[tid * kFaDh + d] = __float2half_rn(0.f);
            do_smem[tid * kFaDh + d] = __float2half_rn(0.f);
        }
    }
    __syncthreads();

    float dq_reg[kFaDh];
#pragma unroll
    for (int d = 0; d < kFaDh; d++) {
        dq_reg[d] = 0.f;
    }
    const float mi = (qi < S) ? LSEp[qi] : 0.f;
    const float di = (qi < S) ? Dp[qi] : 0.f;
    const int num_kv_tiles = (S + kFaTcBc - 1) / kFaTcBc;

    for (int kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        const int kv_start = kv_t * kFaTcBc;
        if (kv_start > q_tile * kFaTcBr + kFaTcBr - 1) {
            break;
        }
        for (int r = tid; r < kFaTcBc; r += kFaTcBr) {
            const int kj = kv_start + r;
            if (kj < S) {
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[r * kFaDh + d] = __float2half_rn(Kp[kj * kFaDh + d]);
                    v_smem[r * kFaDh + d] = __float2half_rn(Vp[kj * kFaDh + d]);
                }
            } else {
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    k_smem[r * kFaDh + d] = __float2half_rn(0.f);
                    v_smem[r * kFaDh + d] = __float2half_rn(0.f);
                }
            }
        }
        __syncthreads();

        if (qi < S) {
#pragma unroll
            for (int j = 0; j < kFaTcBc; j++) {
                const int kj = kv_start + j;
                if (kj >= S) {
                    break;
                }
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    dot = __fmaf_rn(
                            __half2float(q_smem[tid * kFaDh + d]),
                            __half2float(k_smem[j * kFaDh + d]),
                            dot);
                }
                float s = dot * scale;
                if (qi < kj) {
                    s = -INFINITY;
                }
                const float p = (s == -INFINITY) ? 0.f : __expf(s - mi);
                float dp = 0.f;
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    dp = __fmaf_rn(
                            __half2float(do_smem[tid * kFaDh + d]),
                            __half2float(v_smem[j * kFaDh + d]),
                            dp);
                }
                const float ds = p * (dp - di) * scale;
#pragma unroll
                for (int d = 0; d < kFaDh; d++) {
                    dq_reg[d] = __fmaf_rn(ds, __half2float(k_smem[j * kFaDh + d]), dq_reg[d]);
                }
            }
        }
        __syncthreads();
    }

    if (qi < S) {
#pragma unroll
        for (int d = 0; d < kFaDh; d++) {
            dQp[qi * kFaDh + d] = dq_reg[d];
        }
    }
}

static bool flash_attn_fwd_run_tc(
        const float* d_q,
        const float* d_k,
        const float* d_v,
        float* d_o,
        float* d_lse,
        int BH,
        int S,
        float scale)
{
    const int num_q_tiles = (S + kFaTcBr - 1) / kFaTcBr;
    const long long grid_ll = (long long) BH * (long long) num_q_tiles;
    const long long max_grid_ll = static_cast<long long>(jgpt_extra_cuda_max_grid_x());
    if (grid_ll <= 0LL || grid_ll > max_grid_ll) {
        fprintf(stderr, "flash_attn_fwd_tc: grid overflow BH=%d S=%d tiles=%d\n", BH, S, num_q_tiles);
        return false;
    }
    cudaError_t sa = flash_kernel_ensure_dyn_smem((const void*) flash_attn_fwd_kernel_tc, kFaTcFwdSmemBytes);
    if (sa != cudaSuccess) {
        fprintf(stderr, "flash_attn_fwd_tc: smem %zu: %s\n", kFaTcFwdSmemBytes, cudaGetErrorString(sa));
        return false;
    }
    flash_attn_fwd_kernel_tc<<<static_cast<int>(grid_ll), kFaTcThreads, kFaTcFwdSmemBytes, kTensorCudaStream>>>(
            d_q, d_k, d_v, d_o, d_lse, BH, S, scale);
    CUDA_KERNEL_CHECK_RV(false);
    return true;
}

static bool flash_attn_bwd_run_fp16(
        const float* d_q,
        const float* d_k,
        const float* d_v,
        const float* d_o,
        const float* d_do,
        const float* d_lse,
        float* d_dq,
        float* d_dk,
        float* d_dv,
        int BH,
        int S,
        float scale)
{
    const size_t qkv_bytes = (size_t) BH * (size_t) S * (size_t) kFaDh * sizeof(float);
    if (cudaMemsetAsync(d_dq, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }
    if (cudaMemsetAsync(d_dk, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }
    if (cudaMemsetAsync(d_dv, 0, qkv_bytes, kTensorCudaStream) != cudaSuccess) {
        return false;
    }

    const long long total_rows_ll = (long long) BH * (long long) S;
    if (total_rows_ll <= 0LL || total_rows_ll > static_cast<long long>(INT_MAX)) {
        return false;
    }
    const long long max_grid_ll_bwd = static_cast<long long>(jgpt_extra_cuda_max_grid_x());
    const long long d_grid_ll = (total_rows_ll + 63LL) / 64LL;
    if (d_grid_ll > max_grid_ll_bwd) {
        return false;
    }
    flash_attn_compute_D_kernel<<<static_cast<int>(d_grid_ll), 64, 0, kTensorCudaStream>>>(
            d_do, d_o, jgpt_extra::jgpt_extra_tls().flash_attn.d_D, BH, S);
    CUDA_KERNEL_CHECK_RV(false);

    const int num_kv_tiles = (S + kFaTcBc - 1) / kFaTcBc;
    const long long grid_dkdv_ll = (long long) BH * (long long) num_kv_tiles;
    if (grid_dkdv_ll <= 0LL || grid_dkdv_ll > max_grid_ll_bwd) {
        return false;
    }
    cudaError_t sb =
            flash_kernel_ensure_dyn_smem((const void*) flash_attn_bwd_dkdv_kernel_fp16, kFaTcBwdDkdvSmemBytes);
    if (sb != cudaSuccess) {
        return false;
    }
    flash_attn_bwd_dkdv_kernel_fp16<<<static_cast<int>(grid_dkdv_ll), kFaTcBc, kFaTcBwdDkdvSmemBytes, kTensorCudaStream>>>(
            d_q,
            d_k,
            d_v,
            d_do,
            d_lse,
            jgpt_extra::jgpt_extra_tls().flash_attn.d_D,
            d_dk,
            d_dv,
            BH,
            S,
            scale);
    CUDA_KERNEL_CHECK_RV(false);

    const int num_q_tiles = (S + kFaTcBr - 1) / kFaTcBr;
    const long long grid_dq_ll = (long long) BH * (long long) num_q_tiles;
    if (grid_dq_ll <= 0LL || grid_dq_ll > max_grid_ll_bwd) {
        return false;
    }
    sb = flash_kernel_ensure_dyn_smem((const void*) flash_attn_bwd_dq_kernel_fp16, kFaTcBwdDqSmemBytes);
    if (sb != cudaSuccess) {
        return false;
    }
    flash_attn_bwd_dq_kernel_fp16<<<static_cast<int>(grid_dq_ll), kFaTcBr, kFaTcBwdDqSmemBytes, kTensorCudaStream>>>(
            d_q,
            d_k,
            d_v,
            d_do,
            d_lse,
            jgpt_extra::jgpt_extra_tls().flash_attn.d_D,
            d_dq,
            BH,
            S,
            scale);
    CUDA_KERNEL_CHECK_RV(false);
    return true;
}

static void jgpt_cuda_flash_attn_prewarm_kernels(int bAttn, int seqLen)
{
    if (bAttn <= 0 || seqLen <= 0) {
        return;
    }
    const size_t d_bytes = (size_t) bAttn * (size_t) seqLen * sizeof(float);
    (void) fa_ensure_D(d_bytes);
    (void) flash_kernel_ensure_dyn_smem((const void*) flash_attn_fwd_kernel, (kFaBr + 2 * kFaBc) * kFaDh * sizeof(float)
                    + (size_t) kFaBc * kFaBr * sizeof(float));
    (void) flash_kernel_ensure_dyn_smem((const void*) flash_attn_fwd_kernel_tc, kFaTcFwdSmemBytes);
    (void) flash_kernel_ensure_dyn_smem((const void*) flash_attn_bwd_dkdv_kernel_fp16, kFaTcBwdDkdvSmemBytes);
    (void) flash_kernel_ensure_dyn_smem((const void*) flash_attn_bwd_dq_kernel_fp16, kFaTcBwdDqSmemBytes);
}
