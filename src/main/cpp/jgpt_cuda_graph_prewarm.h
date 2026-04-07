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

/**
 * Отладка CUDA graph: текущие thread-local SDPA aux (non-graph и graph-only) и их размеры в байтах.
 * Указатели могут быть NULL, если ещё не выделялись.
 */
void jgpt_cuda_decoder_graph_debug_aux_snapshot(
    uintptr_t* fwd_ptr, uintptr_t* graph_ptr, size_t* fwd_sz, size_t* graph_sz);

/**
 * Эффективные QKV/FFN strided pack указатели (override из Java или tl_qkv_*_pack_d) и ёмкости — для инвалидации
 * decoder CUDA graph при перевыделении.
 */
void jgpt_cuda_decoder_graph_pack_snapshot(
    uintptr_t* w_ptr,
    uintptr_t* c_ptr,
    unsigned long long* cap_w_elems,
    unsigned long long* cap_c_elems,
    int* override_active);

#ifdef __cplusplus
}
#endif
