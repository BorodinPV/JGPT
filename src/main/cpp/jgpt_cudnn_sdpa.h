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

/**
 * То же, что fwd/bwd, но Q/K/V/O/dO/dQ/dK/dV уже FP16 на device (без f32 staging).
 * Указатели — {@code __half*}; stats/LSE по-прежнему float.
 */
int jgpt_cudnn_sdpa_fwd_half(
        const void* q,
        const void* k,
        const void* v,
        void* o,
        float* stats,
        int batch,
        int nHeads,
        int seq,
        int dHead,
        float scale);

int jgpt_cudnn_sdpa_bwd_half(
        const void* q,
        const void* k,
        const void* v,
        const void* o,
        const void* dO,
        const float* stats,
        void* dQ,
        void* dK,
        void* dV,
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
