#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** 1 если собран с cuDNN и SDPA не выключен через {@code JGPT_CUDNN_SDPA=0}. */
int jgpt_cudnn_sdpa_available(void);

/**
 * Causal FP16 Flash SDPA. Q/K/V/O — float [B*H, S, Dh] ≡ BHSD {B,H,S,Dh};
 * stats/LSE — float [B*H*S]. 1 = ok.
 */
int jgpt_cudnn_sdpa_fwd(
        const float* q,
        const float* k,
        const float* v,
        float* o,
        float* stats,
        int batch,
        int nHeads,
        int seq,
        int dHead,
        float scale);

int jgpt_cudnn_sdpa_bwd(
        const float* q,
        const float* k,
        const float* v,
        const float* o,
        const float* dO,
        const float* stats,
        float* dQ,
        float* dK,
        float* dV,
        int batch,
        int nHeads,
        int seq,
        int dHead,
        float scale);

/** Сборка графов + dummy execute (workspace/half staging) до cudaStreamBeginCapture. */
void jgpt_cudnn_sdpa_prewarm(int batch, int nHeads, int seq, int dHead, float scale);
void jgpt_cudnn_sdpa_cleanup(void);

#ifdef __cplusplus
}
#endif
