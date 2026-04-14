#pragma once

#include <cstdio>
#include <cuda_runtime.h>

// ========== CUDA error-check macros (единый набор для всех .inl/.cu) ==========

/** Проверка CUDA-вызова; при ошибке — лог + return (void-функции). */
#define JGPT_CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

/** Как {@code JGPT_CUDA_CHECK}, но возвращает заданное значение. */
#define JGPT_CUDA_CHECK_RV(call, rv) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return (rv); \
        } \
    } while (0)

/** Проверка kernel-launch (cudaGetLastError). */
#define JGPT_KERNEL_LAUNCH_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

/** Как {@code JGPT_KERNEL_LAUNCH_CHECK}, но с возвратом значения. */
#define JGPT_KERNEL_LAUNCH_CHECK_RV(rv) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return (rv); \
        } \
    } while (0)

/** Проверка с ReleaseFloatArrayElements для выходного jfloatArray при ошибке. */
#define JGPT_CUDA_CHECK_RELEASE_FLOAT_ARRAY(env, h_arr, ptr, call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            if ((ptr) != nullptr) (env)->ReleaseFloatArrayElements((h_arr), (ptr), JNI_ABORT); \
            return; \
        } \
    } while (0)

// ========== Обратная совместимость (старые имена → новые) ==========

#define CUDA_CHECK_VOID(call)              JGPT_CUDA_CHECK(call)
#define CUDA_CHECK_X(call)                 JGPT_CUDA_CHECK(call)
#define CUDA_CHECK_RV(call, rv)            JGPT_CUDA_CHECK_RV(call, rv)
#define CUDA_KERNEL_CHECK()                JGPT_KERNEL_LAUNCH_CHECK()
#define CUDA_KERNEL_CHECK_RV(rv)           JGPT_KERNEL_LAUNCH_CHECK_RV(rv)
#define CUDA_CHECK_VOID_JFLOAT_OUT(env, h, p, call)  JGPT_CUDA_CHECK_RELEASE_FLOAT_ARRAY(env, h, p, call)
