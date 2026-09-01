#pragma once

#include <cstddef>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

void jgpt_extra_f32_to_f16(const float* src, __half* dst, size_t n);
void jgpt_extra_f16_to_f32(const __half* src, float* dst, size_t n);

#ifdef __cplusplus
}
#endif
