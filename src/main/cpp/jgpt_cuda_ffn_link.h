#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Strided-batched GEMM для xNorm×[W1|W3] → h1, gate (общий scratch в jgpt_cuda.cu).
 * @return 1 при успехе, 0 при ошибке
 */
int jgpt_cuda_ffn_w1w3_strided_batched_device(
    float* xnorm, const float* w1, const float* w3, float* h1, float* gate, int M, int K, int N);

#ifdef __cplusplus
}
#endif
