#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <jni.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <algorithm>
#include <climits>
#include <cstring>
#include <cstddef>
#include <mutex>

#include "jgpt_cuda_stream.cuh"
#include "jgpt_cuda_extra_thread_resources.cuh"
#include "jgpt_cuda_ffn_link.h"
#include "jgpt_cuda_graph_prewarm.h"
#include "jgpt_cuda_size_check.cuh"
#include "jgpt_cuda_jni_helpers.cuh"
#include "jgpt_cuda_jni_raii.cuh"

namespace jgpt_extra {
ExtraThreadResources& jgpt_extra_tls() {
    thread_local ExtraThreadResources tr;
    return tr;
}
}  // namespace jgpt_extra

extern "C" void jgpt_cuda_extra_warmup_cublas(void) {
    (void) jgpt_cuda_detail::get_cublas_handle();
}

static cublasHandle_t get_extra_cublas_handle() {
    return jgpt_cuda_detail::get_cublas_handle();
}

static void ce_async_host_free() {
    jgpt_extra::jgpt_extra_tls().ce.async_host_free();
}
static void ce_async_host_ensure() {
    jgpt_extra::jgpt_extra_tls().ce.async_host_ensure();
}
static void ce_pinned_flt_targets_free() {
    jgpt_extra::jgpt_extra_tls().ce.pinned_flt_targets_free();
}
static void ce_pinned_flt_targets_ensure(size_t nfloats) {
    jgpt_extra::jgpt_extra_tls().ce.pinned_flt_targets_ensure(nfloats);
}
static void ce_free_cached() {
    jgpt_extra::jgpt_extra_tls().ce.free_cached();
}
static bool ce_ensure_logits_grad_buffers(size_t bytes_logits) {
    return jgpt_extra::jgpt_extra_tls().ce.ensure_logits_grad_buffers(bytes_logits);
}
static bool softmax_pair_ensure(size_t bytes_per, float** d_src, float** d_dst) {
    return jgpt_extra::jgpt_extra_tls().softmax_pair.ensure(bytes_per, d_src, d_dst);
}
static void softmax_pair_free_cached() {
    jgpt_extra::jgpt_extra_tls().softmax_pair.free_cached();
}
static bool adamw_pool_ensure(size_t bytes) {
    return jgpt_extra::jgpt_extra_tls().adamw.ensure(bytes);
}
static void adamw_pool_free_cached() {
    jgpt_extra::jgpt_extra_tls().adamw.free_cached();
}
static bool attn_bwd_host_ensure(
        size_t bytesProb, size_t bytesQK, size_t bytesV, float** d_go, float** d_p, float** d_q, float** d_k,
        float** d_v, float** d_dp, float** d_ds, float** d_gq, float** d_gk, float** d_gv) {
    return jgpt_extra::jgpt_extra_tls().attn_bwd_host.ensure(
            bytesProb, bytesQK, bytesV, d_go, d_p, d_q, d_k, d_v, d_dp, d_ds, d_gq, d_gk, d_gv);
}
static void attn_bwd_host_free_cached() {
    jgpt_extra::jgpt_extra_tls().attn_bwd_host.free_cached();
}
static bool attn_bwd_aux_ensure(size_t bytesProb, size_t bytesV, float** d_dp, float** d_ds) {
    return jgpt_extra::jgpt_extra_tls().attn_bwd_aux.ensure(bytesProb, bytesV, d_dp, d_ds);
}
static void attn_bwd_aux_free_cached() {
    jgpt_extra::jgpt_extra_tls().attn_bwd_aux.free_cached();
}
static bool attn_fwd_aux_ensure(
        size_t bytesQK, size_t bytesProb, size_t bytesMask, bool with_mask, float** d_scores, float** d_probs,
        float** d_mask) {
    return jgpt_extra::jgpt_extra_tls().attn_fwd.ensure(bytesQK, bytesProb, bytesMask, with_mask, d_scores, d_probs, d_mask);
}
static bool fa_ensure_D(size_t bytes) {
    return jgpt_extra::jgpt_extra_tls().flash_attn.ensure_D(bytes);
}

#include "jgpt_cuda_extra_kernels_launches.inl"


// ========== JNI ==========
#ifdef __cplusplus
extern "C" {
#endif

#include "jgpt_cuda_extra_jni_host_fwd.inl"
#include "jgpt_cuda_extra_jni_attention.inl"
#include "jgpt_cuda_extra_jni_optim.inl"
#include "jgpt_cuda_extra_jni_device.inl"
#include "jgpt_cuda_extra_jni_graph_flash.inl"

#ifdef __cplusplus
}
#endif
