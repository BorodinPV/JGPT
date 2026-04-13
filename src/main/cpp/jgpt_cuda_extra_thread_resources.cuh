#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "jgpt_cuda_tls_blob.cuh"
#include "jgpt_cuda_stream.cuh"
#include "jgpt_cuda_cublas_common.cuh"

namespace jgpt_extra {

using TlsDeviceBlob = jgpt_cuda_tls::TlsDeviceBlob;

/** Thread-local cuBLAS for extra strided-batched GEMM / JNI paths. */
struct ExtraCublas {
    cublasHandle_t h = nullptr;

    cublasHandle_t get_handle() {
        if (h != nullptr) {
            return h;
        }
        h = jgpt_cuda_detail::create_cublas_for_jgpt_stream("TensorOpsGPU extra");
        return h;
    }

    void destroy() {
        if (h) {
            cublasDestroy(h);
            h = nullptr;
        }
    }
};

/** Device + pinned host scratch for cross-entropy / softmax+grad JNI (freed in jgpt_cuda_extra_cleanup). */
struct CeLossBuffers {
    float* d_logits = nullptr;
    float* d_targets = nullptr;
    float* d_grad = nullptr;
    float* d_loss_sum = nullptr;
    unsigned int* d_valid = nullptr;
    size_t logits_bytes = 0;
    size_t targets_bytes = 0;

    float* h_async_loss = nullptr;
    unsigned int* h_async_valid = nullptr;

    float* h_pinned_flt_targets = nullptr;
    size_t h_pinned_flt_n = 0;

    void async_host_free() {
        if (h_async_loss != nullptr) {
            cudaFreeHost(h_async_loss);
            h_async_loss = nullptr;
        }
        if (h_async_valid != nullptr) {
            cudaFreeHost(h_async_valid);
            h_async_valid = nullptr;
        }
    }

    void async_host_ensure() {
        if (h_async_loss == nullptr) {
            cudaError_t err =
                    cudaHostAlloc(reinterpret_cast<void**>(&h_async_loss), sizeof(float), cudaHostAllocDefault);
            if (err != cudaSuccess) {
                fprintf(stderr, "ce_async_host_ensure(loss): %s\n", cudaGetErrorString(err));
                return;
            }
        }
        if (h_async_valid == nullptr) {
            cudaError_t err = cudaHostAlloc(
                    reinterpret_cast<void**>(&h_async_valid), sizeof(unsigned int), cudaHostAllocDefault);
            if (err != cudaSuccess) {
                fprintf(stderr, "ce_async_host_ensure(valid): %s\n", cudaGetErrorString(err));
                return;
            }
        }
    }

    void pinned_flt_targets_free() {
        if (h_pinned_flt_targets != nullptr) {
            cudaFreeHost(h_pinned_flt_targets);
            h_pinned_flt_targets = nullptr;
            h_pinned_flt_n = 0;
        }
    }

    void pinned_flt_targets_ensure(size_t nfloats) {
        if (nfloats == 0) {
            return;
        }
        if (nfloats > h_pinned_flt_n) {
            pinned_flt_targets_free();
            cudaError_t err = cudaHostAlloc(
                    reinterpret_cast<void**>(&h_pinned_flt_targets), nfloats * sizeof(float), cudaHostAllocDefault);
            if (err != cudaSuccess) {
                fprintf(stderr, "ce_pinned_flt_targets_ensure: %s\n", cudaGetErrorString(err));
                return;
            }
            h_pinned_flt_n = nfloats;
        }
    }

    void free_cached() {
        cudaFree(d_logits);
        cudaFree(d_targets);
        cudaFree(d_grad);
        cudaFree(d_loss_sum);
        cudaFree(d_valid);
        d_logits = d_targets = d_grad = nullptr;
        d_loss_sum = nullptr;
        d_valid = nullptr;
        logits_bytes = targets_bytes = 0;
        async_host_free();
        pinned_flt_targets_free();
    }

    bool ensure_logits_grad_buffers(size_t bytes_logits) {
        if (bytes_logits <= logits_bytes) {
            return true;
        }
        cudaFree(d_logits);
        cudaFree(d_grad);
        d_logits = nullptr;
        d_grad = nullptr;
        logits_bytes = 0;
        if (!jgpt_cuda_tls::malloc_pair_same_size(
                    &d_logits,
                    &d_grad,
                    bytes_logits,
                    "ce_ensure_logits_grad_buffers: cudaMalloc(logits)",
                    "ce_ensure_logits_grad_buffers: cudaMalloc(grad)")) {
            return false;
        }
        logits_bytes = bytes_logits;
        return true;
    }
};

struct SoftmaxPairScratch {
    TlsDeviceBlob blob;
    size_t bytes_per = 0;

    bool ensure(size_t bytes_per_elem, float** d_src, float** d_dst) {
        jgpt_cuda_ensure_stream();
        if (bytes_per_elem == 0U) {
            return false;
        }
        if (bytes_per_elem > bytes_per) {
            blob.free_cached();
            bytes_per = 0;
            if (!blob.grow_to_fit(bytes_per_elem * 2U)) {
                return false;
            }
            bytes_per = bytes_per_elem;
        }
        *d_src = static_cast<float*>(blob.ptr);
        *d_dst = reinterpret_cast<float*>(static_cast<unsigned char*>(blob.ptr) + bytes_per_elem);
        /* Bounds check: d_dst + bytes_per_elem должен помещаться в blob. */
        size_t total = bytes_per_elem * 2U;
        if (total > blob.bytes) {
            fprintf(stderr,
                    "SoftmaxPairScratch::ensure: layout overflow %zu > %zu bytes\n",
                    total, blob.bytes);
            return false;
        }
        return true;
    }

    void free_cached() {
        blob.free_cached();
        bytes_per = 0;
    }
};

struct AdamwPoolScratch {
    float* d_param = nullptr;
    float* d_grad = nullptr;
    float* d_m = nullptr;
    float* d_v = nullptr;
    size_t bytes = 0;

    bool ensure(size_t need_bytes) {
        jgpt_cuda_ensure_stream();
        if (need_bytes == 0U) {
            return false;
        }
        if (bytes >= need_bytes) {
            return true;
        }
        cudaFree(d_param);
        cudaFree(d_grad);
        cudaFree(d_m);
        cudaFree(d_v);
        d_param = nullptr;
        d_grad = nullptr;
        d_m = nullptr;
        d_v = nullptr;
        bytes = 0;
        float* s[4];
        if (!jgpt_cuda_tls::malloc_n_same_size(s, 4, need_bytes)) {
            return false;
        }
        d_param = s[0];
        d_grad = s[1];
        d_m = s[2];
        d_v = s[3];
        bytes = need_bytes;
        return true;
    }

    void free_cached() {
        cudaFree(d_param);
        cudaFree(d_grad);
        cudaFree(d_m);
        cudaFree(d_v);
        d_param = nullptr;
        d_grad = nullptr;
        d_m = nullptr;
        d_v = nullptr;
        bytes = 0;
    }
};

struct AttnBwdHostScratch {
    TlsDeviceBlob blob;

    bool ensure(size_t bytesProb, size_t bytesQK, size_t bytesV, float** d_go, float** d_p, float** d_q, float** d_k,
            float** d_v, float** d_dp, float** d_ds, float** d_gq, float** d_gk, float** d_gv) {
        size_t need = 3U * bytesProb + 4U * bytesQK + 3U * bytesV;
        if (!blob.grow_to_fit(need)) {
            return false;
        }
        unsigned char* base = static_cast<unsigned char*>(blob.ptr);
        size_t off = 0;
        *d_go = reinterpret_cast<float*>(base + off);
        off += bytesV;
        *d_p = reinterpret_cast<float*>(base + off);
        off += bytesProb;
        *d_q = reinterpret_cast<float*>(base + off);
        off += bytesQK;
        *d_k = reinterpret_cast<float*>(base + off);
        off += bytesQK;
        *d_v = reinterpret_cast<float*>(base + off);
        off += bytesV;
        *d_dp = reinterpret_cast<float*>(base + off);
        off += bytesProb;
        *d_ds = reinterpret_cast<float*>(base + off);
        off += bytesProb;
        *d_gq = reinterpret_cast<float*>(base + off);
        off += bytesQK;
        *d_gk = reinterpret_cast<float*>(base + off);
        off += bytesQK;
        *d_gv = reinterpret_cast<float*>(base + off);
        off += bytesV;
        /* Bounds check: убедимся что итоговое смещение не превышает размер blob. */
        if (off > blob.bytes) {
            fprintf(stderr,
                    "AttnBwdHostScratch::ensure: layout overflow %zu > %zu bytes "
                    "(bytesProb=%zu bytesQK=%zu bytesV=%zu)\n",
                    off, blob.bytes, bytesProb, bytesQK, bytesV);
            return false;
        }
        return true;
    }

    void free_cached() {
        blob.free_cached();
    }
};

struct AttnBwdAuxScratch {
    TlsDeviceBlob blob;

    bool ensure(size_t bytesProb, size_t bytesV, float** d_dp, float** d_ds) {
        (void) bytesV;
        size_t need = 2U * bytesProb;
        if (!blob.grow_to_fit(need)) {
            return false;
        }
        unsigned char* base = static_cast<unsigned char*>(blob.ptr);
        size_t off = 0;
        *d_dp = reinterpret_cast<float*>(base + off);
        off += bytesProb;
        *d_ds = reinterpret_cast<float*>(base + off);
        off += bytesProb;
        if (off > blob.bytes) {
            fprintf(stderr,
                    "AttnBwdAuxScratch::ensure: layout overflow %zu > %zu bytes\n",
                    off, blob.bytes);
            return false;
        }
        return true;
    }

    void free_cached() {
        blob.free_cached();
    }
};

/**
 * Non-graph attention forward scratch (mask slot included). May be reallocated by non-graph ops.
 * Separate from g_attn_fwd_graph_aux (graph path).
 */
struct AttnFwdScratch {
    TlsDeviceBlob blob;

    bool ensure(size_t bytesQK, size_t bytesProb, size_t bytesMask, bool with_mask, float** d_scores,
            float** d_probs, float** d_mask) {
        size_t total_need = bytesProb * 2U;
        if (with_mask) {
            total_need += bytesMask;
        }
        if (!blob.grow_to_fit(total_need)) {
            return false;
        }
        unsigned char* base = static_cast<unsigned char*>(blob.ptr);
        size_t off = 0;
        *d_scores = reinterpret_cast<float*>(base + off);
        off += bytesProb;
        *d_probs = reinterpret_cast<float*>(base + off);
        off += bytesProb;
        if (with_mask) {
            *d_mask = reinterpret_cast<float*>(base + off);
            off += bytesMask;
        } else {
            *d_mask = nullptr;
        }
        if (off > blob.bytes) {
            fprintf(stderr,
                    "AttnFwdScratch::ensure: layout overflow %zu > %zu bytes\n",
                    off, blob.bytes);
            return false;
        }
        (void) bytesQK;
        return true;
    }

    void free_cached() {
        blob.free_cached();
    }
};

/** Thread-local D vector for Flash-Attention backward (BH*S floats). */
struct FlashAttnScratch {
    float* d_D = nullptr;
    size_t d_bytes = 0;

    bool ensure_D(size_t bytes) {
        if (bytes <= d_bytes) {
            return true;
        }
        void* raw = nullptr;
        if (cudaMalloc(&raw, bytes) != cudaSuccess) {
            fprintf(stderr, "fa_ensure_D: cudaMalloc(%zu) failed\n", bytes);
            return false;
        }
        if (d_D != nullptr) {
            (void) cudaFree(d_D);
        }
        d_D = static_cast<float*>(raw);
        d_bytes = bytes;
        return true;
    }

    void free_cached() {
        if (d_D != nullptr) {
            cudaFree(d_D);
            d_D = nullptr;
            d_bytes = 0;
        }
    }
};

/** Workspace for jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas batched-GEMM warmup. */
struct GraphSdpaWarmupScratch {
    TlsDeviceBlob blob;

    float* ensure(size_t need_bytes) {
        if (!blob.grow_to_fit(need_bytes)) {
            fprintf(
                    stderr,
                    "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: warmup alloc failed (%zu bytes): %s\n",
                    need_bytes,
                    cudaGetErrorString(cudaGetLastError()));
            return nullptr;
        }
        return reinterpret_cast<float*>(blob.ptr);
    }

    void free_cached() {
        blob.free_cached();
    }
};

struct ExtraThreadResources {
    ExtraCublas cublas;
    CeLossBuffers ce;
    SoftmaxPairScratch softmax_pair;
    AdamwPoolScratch adamw;
    AttnBwdHostScratch attn_bwd_host;
    AttnBwdAuxScratch attn_bwd_aux;
    AttnFwdScratch attn_fwd;
    FlashAttnScratch flash_attn;
    GraphSdpaWarmupScratch graph_sdpa_warmup;
};

/** One aggregate per host thread; defined in jgpt_cuda_extra.cu. */
ExtraThreadResources& jgpt_extra_tls();

}  // namespace jgpt_extra
