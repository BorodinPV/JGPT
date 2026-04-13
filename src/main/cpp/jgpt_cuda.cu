#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <library_types.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <mutex>
#include <atomic>
#include <unordered_set>

#include "jgpt_cuda_stream.cuh"
#include "jgpt_cuda_ffn_link.h"
#include "jgpt_cuda_graph_prewarm.h"
#include "jgpt_cuda_size_check.cuh"

#include "jgpt_cuda_runtime.inl"
#include "jgpt_cuda_matmul_staging.inl"
#include "jgpt_cuda_kernels_launches_macros.inl"

// ========== JNI ==========
#ifdef __cplusplus
extern "C" {
#endif

#include "jgpt_cuda_jni_init_matmul.inl"
#include "jgpt_cuda_jni_buffers.inl"
#include "jgpt_cuda_jni_ops_graph.inl"

#ifdef __cplusplus
}
#endif
