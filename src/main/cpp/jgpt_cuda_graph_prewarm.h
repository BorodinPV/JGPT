#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * До cudaStreamBeginCapture: strided-batched QKV (batch 3), FFN W1+W3 (batch 2) и один Wo Sgemm на
 * thread-local pack-буферах — инициализация workspace cuBLAS вне захвата.
 */
void jgpt_cuda_graph_prewarm_qkv_ffn_strided_and_wo(int M, int dModel, int dInt);

/**
 * До cudaStreamBeginCapture: {@code attn_fwd_aux_ensure_qk_probs_only} + два batched Sgemm resident SDPA
 * (get_extra_cublas_handle), чтобы не было cudaMalloc/ленивого workspace внутри графа.
 */
void jgpt_cuda_graph_prewarm_sdpa_aux_and_cublas(int bAttn, int seqLen, int dK, int dV);

#ifdef __cplusplus
}
#endif
