/* JNI graph & flash: graph prewarm, stability token, FlashAttention JNI
 * Include only from jgpt_cuda_extra.cu inside extern "C" { }.
 */


extern "C" void jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas(int bAttn, int seqLen, int dK, int dV) {
    if (bAttn <= 0 || seqLen <= 0 || dK <= 0 || dV <= 0) {
        return;
    }
    jgpt_cuda_ensure_stream();
    if (check_size_overflow((size_t) bAttn, (size_t) seqLen, sizeof(float)) ||
        check_size_overflow((size_t) bAttn * (size_t) seqLen, (size_t) dK, sizeof(float)) ||
        check_size_overflow((size_t) bAttn * (size_t) seqLen, (size_t) seqLen, sizeof(float))) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: size overflow (QK/prob)\n");
        return;
    }
    const size_t bytesQK = (size_t) bAttn * (size_t) seqLen * (size_t) dK * sizeof(float);
    const size_t bytesProb = (size_t) bAttn * (size_t) seqLen * (size_t) seqLen * sizeof(float);
    float *d_kt = nullptr, *d_scores = nullptr, *d_probs = nullptr;
    if (!attn_fwd_aux_ensure_qk_probs_only(bytesQK, bytesProb, &d_scores, &d_probs)) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: attn_fwd_aux_ensure_qk_probs_only failed\n");
        return;
    }
    (void) d_kt;

    const size_t outElems = (size_t) bAttn * (size_t) seqLen * (size_t) dV;
    if (check_size_overflow(outElems, sizeof(float), 1)) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: size overflow (out)\n");
        return;
    }
    const size_t gemm1Elems = (2U * bytesQK + bytesProb) / sizeof(float);
    const size_t gemm2Elems = bytesProb / sizeof(float) + 2U * outElems;
    const size_t needBytes = (gemm1Elems + gemm2Elems) * sizeof(float);
    float* w = jgpt_extra::jgpt_extra_tls().graph_sdpa_warmup.ensure(needBytes);
    if (w == nullptr) {
        return;
    }
    float* g1a = w;
    float* g1b = g1a + (bytesQK / sizeof(float));
    float* g1c = g1b + (bytesQK / sizeof(float));
    if (!batched_sgemm_row_major_extra(g1a, g1b, g1c, bAttn, seqLen, dK, seqLen, 1.0f, 0.0f)) {
        fprintf(stderr, "jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas: SDPA batched gemm1 warmup failed\n");
        return;
    }

    float* g2a = g1c + (bytesProb / sizeof(float));
    float* g2b = g2a + (bytesProb / sizeof(float));
    float* g2c = g2b + outElems;
    (void) batched_sgemm_row_major_extra(g2a, g2b, g2c, bAttn, seqLen, seqLen, dV, 1.0f, 0.0f);
}

void jgpt_cuda_extra_cleanup(void) {
    jgpt_extra::jgpt_extra_tls().cublas.destroy();
    {
        std::lock_guard<std::mutex> lock(g_any_nonfinite_alloc_mu);
        if (g_any_nonfinite_flag != nullptr) {
            cudaFree(g_any_nonfinite_flag);
            g_any_nonfinite_flag = nullptr;
        }
    }
    jgpt_extra::jgpt_extra_tls().graph_sdpa_warmup.free_cached();
    jgpt_extra::jgpt_extra_tls().attn_fwd.free_cached();
    {
        std::lock_guard<std::mutex> lock(g_attn_fwd_graph_aux_mu);
        if (g_attn_fwd_graph_aux != nullptr) {
            cudaFree(g_attn_fwd_graph_aux);
            g_attn_fwd_graph_aux = nullptr;
            g_attn_fwd_graph_aux_total = 0;
        }
    }
    ce_free_cached();
    softmax_pair_free_cached();
    adamw_pool_free_cached();
    attn_bwd_host_free_cached();
    attn_bwd_aux_free_cached();
    jgpt_extra::jgpt_extra_tls().flash_attn.free_cached();
}

// ----------------------------------------------------------------
//  JNI: Flash Attention Forward (GPU-resident, causal)
//  Q/K/V/O = [BH, S, Dh=kFaDh].  dLSEPtr = device float[BH*S].
//  S must be divisible by kFaBr (= 64); for padding caller truncates.
// ----------------------------------------------------------------
JNIEXPORT jlong JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_decoderGraphNativeStabilityToken0(
        JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    uintptr_t fp = 0;
    uintptr_t gp = 0;
    size_t fsz = 0;
    size_t gsz = 0;
    jgpt_cuda_decoder_graph_debug_aux_snapshot(&fp, &gp, &fsz, &gsz);
    uintptr_t wp = 0;
    uintptr_t cp = 0;
    unsigned long long cwe = 0;
    unsigned long long cce = 0;
    int ov = 0;
    jgpt_cuda_decoder_graph_pack_snapshot(&wp, &cp, &cwe, &cce, &ov);
    const uintptr_t warm = reinterpret_cast<uintptr_t>(jgpt_extra::jgpt_extra_tls().graph_sdpa_warmup.blob.ptr);
    const size_t warm_sz = jgpt_extra::jgpt_extra_tls().graph_sdpa_warmup.blob.bytes;
    const uintptr_t fad = reinterpret_cast<uintptr_t>(jgpt_extra::jgpt_extra_tls().flash_attn.d_D);
    const size_t fad_sz = jgpt_extra::jgpt_extra_tls().flash_attn.d_bytes;

    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t x) {
        h ^= x;
        h *= 1099511628211ULL;
    };
    mix(static_cast<uint64_t>(fp));
    mix(static_cast<uint64_t>(gp));
    mix(static_cast<uint64_t>(fsz));
    mix(static_cast<uint64_t>(gsz));
    mix(static_cast<uint64_t>(wp));
    mix(static_cast<uint64_t>(cp));
    mix(static_cast<uint64_t>(cwe));
    mix(static_cast<uint64_t>(cce));
    mix(static_cast<uint64_t>(static_cast<unsigned int>(ov)));
    mix(static_cast<uint64_t>(warm));
    mix(static_cast<uint64_t>(warm_sz));
    mix(static_cast<uint64_t>(fad));
    mix(static_cast<uint64_t>(fad_sz));
    return static_cast<jlong>(static_cast<int64_t>(h));
}

JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_flashAttentionForwardGPUDeviceResident(
    JNIEnv* env, jclass clazz,
    jlong dQPtr, jlong dKPtr, jlong dVPtr, jlong dOutPtr, jlong dLSEPtr,
    jint BH, jint S, jint dHead, jfloat scale)
{
    (void) env; (void) clazz;
    if (!dQPtr || !dKPtr || !dVPtr || !dOutPtr || !dLSEPtr || BH <= 0 || S <= 0) return;
    if (dHead != static_cast<jint>(kFaDh)) {
        fprintf(
                stderr,
                "FlashAttention forward: compiled for d_head=%d, got d_head=%d\n",
                kFaDh,
                static_cast<int>(dHead));
        return;
    }

    /* dLSEPtr — буфер вызывающего на устройстве [BH*S] float. */
    jgpt_cuda_ensure_stream();
    if (!flash_attn_fwd_run(
                reinterpret_cast<float*>(static_cast<uintptr_t>(dQPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dKPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dVPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dOutPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dLSEPtr)),
                BH,
                S,
                scale)) {
        fprintf(stderr, "flashAttentionForwardGPUDeviceResident: flash_attn_fwd_run failed\n");
    }
}

// ----------------------------------------------------------------
//  JNI: Flash Attention Backward (GPU-resident, causal)
//  dO  = upstream gradient of O  [BH, S, Dh]
//  O   = forward output           [BH, S, Dh]  (cached, for D computation)
//  LSE = log-sum-exp from fwd     [BH, S]
//  Outputs: dQ, dK, dV           [BH, S, Dh]
// ----------------------------------------------------------------
JNIEXPORT void JNICALL Java_com_veles_llm_jgpt_TensorOpsGPU_flashAttentionBackwardGPUDeviceResident(
    JNIEnv* env, jclass clazz,
    jlong dQPtr, jlong dKPtr, jlong dVPtr,
    jlong dOPtr, jlong dOGradPtr, jlong dLSEPtr,
    jlong dGradQPtr, jlong dGradKPtr, jlong dGradVPtr,
    jint BH, jint S, jint dHead, jfloat scale)
{
    (void) env; (void) clazz;
    if (!dQPtr || !dKPtr || !dVPtr || !dOPtr || !dOGradPtr || !dLSEPtr
            || !dGradQPtr || !dGradKPtr || !dGradVPtr || BH <= 0 || S <= 0) return;
    if (dHead != static_cast<jint>(kFaDh)) {
        fprintf(
                stderr,
                "FlashAttention backward: compiled for d_head=%d, got d_head=%d\n",
                kFaDh,
                static_cast<int>(dHead));
        return;
    }

    size_t D_bytes = (size_t)BH * S * sizeof(float);
    if (!fa_ensure_D(D_bytes)) return;

    jgpt_cuda_ensure_stream();
    if (!flash_attn_bwd_run(
                reinterpret_cast<float*>(static_cast<uintptr_t>(dQPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dKPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dVPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dOPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dOGradPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dLSEPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dGradQPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dGradKPtr)),
                reinterpret_cast<float*>(static_cast<uintptr_t>(dGradVPtr)),
                BH,
                S,
                scale)) {
        fprintf(stderr, "flashAttentionBackwardGPUDeviceResident: flash_attn_bwd_run failed\n");
    }
}
