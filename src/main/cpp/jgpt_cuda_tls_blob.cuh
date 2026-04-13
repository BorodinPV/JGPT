#pragma once

#include <cuda_runtime.h>
#include <cstdio>

namespace jgpt_cuda_tls {

/**
 * Device-буфер с политикой «только растём»: при {@code need <= bytes} указатель не трогаем
 * (меньший запрос переиспользует больший старый блок). Подходит для thread-local кэшей.
 */
struct TlsDeviceBlob {
    void* ptr = nullptr;
    size_t bytes = 0;

    bool grow_to_fit(size_t need) {
        if (need == 0U) {
            return true;
        }
        if (need <= bytes && ptr != nullptr) {
            return true;
        }
        cudaFree(ptr);
        ptr = nullptr;
        bytes = 0;
        if (cudaMalloc(&ptr, need) != cudaSuccess) {
            return false;
        }
        bytes = need;
        return true;
    }

    void free_cached() {
        cudaFree(ptr);
        ptr = nullptr;
        bytes = 0;
    }
};

/** Два {@code cudaMalloc} одинакового размера; при ошибке второго — откат первого. */
inline bool malloc_pair_same_size(float** a, float** b, size_t nbytes, const char* ctx_a, const char* ctx_b) {
    *a = nullptr;
    *b = nullptr;
    cudaError_t e1 = cudaMalloc(reinterpret_cast<void**>(a), nbytes);
    if (e1 != cudaSuccess) {
        fprintf(
                stderr,
                "%s: cudaMalloc %zu bytes: %s\n",
                ctx_a,
                nbytes,
                cudaGetErrorString(e1));
        return false;
    }
    cudaError_t e2 = cudaMalloc(reinterpret_cast<void**>(b), nbytes);
    if (e2 != cudaSuccess) {
        fprintf(
                stderr,
                "%s: cudaMalloc %zu bytes: %s\n",
                ctx_b,
                nbytes,
                cudaGetErrorString(e2));
        cudaFree(*a);
        *a = nullptr;
        return false;
    }
    return true;
}

/**
 * Цепочка {@code cudaMalloc} одинакового размера для {@code n} буферов {@code float*};
 * при сбое — освободить уже выделенные. {@code slots} — массив из {@code n} указателей на {@code float*} (как {@code float* s[4]}).
 */
inline bool malloc_n_same_size(float** slots, int n, size_t nbytes) {
    for (int i = 0; i < n; ++i) {
        slots[i] = nullptr;
    }
    for (int i = 0; i < n; ++i) {
        cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&slots[i]), nbytes);
        if (e != cudaSuccess) {
            fprintf(
                    stderr,
                    "malloc_n_same_size: slot %d cudaMalloc %zu bytes: %s\n",
                    i,
                    nbytes,
                    cudaGetErrorString(e));
            for (int j = 0; j < i; ++j) {
                cudaFree(slots[j]);
                slots[j] = nullptr;
            }
            return false;
        }
    }
    return true;
}

}  // namespace jgpt_cuda_tls
