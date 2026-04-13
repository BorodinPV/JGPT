#pragma once

#include <climits>
#include <cstddef>
#include <cstdio>

/** @return true if {@code a*b} elements of size {@code elem_size} would overflow representable range (or a/b zero). */
static inline bool check_size_overflow(size_t a, size_t b, size_t elem_size) {
    if (a == 0 || b == 0) {
        return false;
    }
    if (a > SIZE_MAX / b) {
        return true;
    }
    size_t prod = a * b;
    return prod > SIZE_MAX / elem_size;
}

/** true if {@code a * b} не помещается в {@code size_t}. */
static inline bool jgpt_size_mul_overflows(size_t a, size_t b) {
    if (a == 0 || b == 0) {
        return false;
    }
    return a > SIZE_MAX / b;
}

/** true если {@code n_elems * elem_size} переполняет {@code size_t}. */
static inline bool jgpt_alloc_n_elem_overflows(size_t n_elems, size_t elem_size) {
    return n_elems > 0 && jgpt_size_mul_overflows(n_elems, elem_size);
}

/**
 * true если выделение матрицы {@code rows * cols} элементов размера {@code elem_size} переполняет {@code size_t}
 * (включая промежуточное {@code rows * cols}).
 */
static inline bool jgpt_alloc_matrix_bytes_overflows(size_t rows, size_t cols, size_t elem_size) {
    if (rows == 0 || cols == 0) {
        return false;
    }
    if (jgpt_size_mul_overflows(rows, cols)) {
        return true;
    }
    return jgpt_size_mul_overflows(rows * cols, elem_size);
}

/** Объём {@code d0×d1×d2} float-элементов: проверка цепочки произведений и ×sizeof(float). */
static inline bool jgpt_alloc_volume3d_float_overflows(size_t d0, size_t d1, size_t d2) {
    if (d0 == 0 || d1 == 0 || d2 == 0) {
        return false;
    }
    if (jgpt_size_mul_overflows(d0, d1)) {
        return true;
    }
    size_t a = d0 * d1;
    if (jgpt_size_mul_overflows(a, d2)) {
        return true;
    }
    return jgpt_size_mul_overflows(a * d2, sizeof(float));
}

/** Цепочка {@code a*b*c*d} float-элементов (все размеры >0). */
static inline bool jgpt_alloc_chain4_float_overflows(size_t a, size_t b, size_t c, size_t d) {
    if (a == 0 || b == 0 || c == 0 || d == 0) {
        return false;
    }
    if (jgpt_size_mul_overflows(a, b)) {
        return true;
    }
    size_t ab = a * b;
    if (jgpt_size_mul_overflows(ab, c)) {
        return true;
    }
    size_t abc = ab * c;
    if (jgpt_size_mul_overflows(abc, d)) {
        return true;
    }
    return jgpt_size_mul_overflows(abc * d, sizeof(float));
}

/**
 * JNI {@code jlong} как число элементов (или смещение в элементах): неположительное значение
 * или произведение с {@code elem_size} не помещается в {@code size_t}.
 */
static inline bool jgpt_jni_long_elems_invalid(long long n_elems, size_t elem_size) {
    if (n_elems <= 0) {
        return true;
    }
    const auto n = static_cast<unsigned long long>(n_elems);
    const auto esz = static_cast<unsigned long long>(elem_size);
    if (esz == 0ULL) {
        return true;
    }
    const auto smax = static_cast<unsigned long long>(SIZE_MAX);
    return n > smax / esz;
}

/**
 * Произведение batch*seqLen*dModel для размеров сетки/итераторов в {@code int}:
 * все сомножители > 0 и произведение не больше {@code INT_MAX}.
 */
static inline bool jgpt_bsd_product_fits_int(int batch, int seqLen, int dModel) {
    if (batch <= 0 || seqLen <= 0 || dModel <= 0) {
        return false;
    }
    const long long p =
            static_cast<long long>(batch) * static_cast<long long>(seqLen) * static_cast<long long>(dModel);
    return p <= static_cast<long long>(INT_MAX);
}

/** Два положительных int: {@code a*b} без переполнения {@code int} (для сеток и линейных индексов). */
static inline bool jgpt_pair_product_fits_int(int a, int b) {
    if (a <= 0 || b <= 0) {
        return false;
    }
    const long long p = static_cast<long long>(a) * static_cast<long long>(b);
    return p <= static_cast<long long>(INT_MAX);
}

/** JNI {@code jlong} как размер в байтах: неположительный или больше {@code SIZE_MAX}. */
static inline bool jgpt_jni_long_bytes_invalid(long long num_bytes) {
    if (num_bytes <= 0) {
        return true;
    }
    const auto nb = static_cast<unsigned long long>(num_bytes);
    return nb > static_cast<unsigned long long>(SIZE_MAX);
}

/**
 * JNI/CUDA: {@code n < 0} или {@code n * elem_size} переполняет {@code size_t}.
 * {@code ret_stmt} — полный оператор (например {@code return 0.f;} или {@code return;}).
 */
#define JGPT_CUDA_GUARD_1D(n, elem_size, ret_stmt)                                                                  \
    do {                                                                                                            \
        if ((n) < 0 || ((n) > 0 && jgpt_alloc_n_elem_overflows(static_cast<size_t>(n), static_cast<size_t>(elem_size)))) { \
            fprintf(stderr, "CUDA JNI alloc overflow (1d) %s:%d\n", __FILE__, __LINE__);                              \
            ret_stmt                                                                                                \
        }                                                                                                           \
    } while (0)

/** Как {@code JGPT_CUDA_GUARD_1D}, но {@code n} уже {@code size_t} (только проверка произведения). */
#define JGPT_CUDA_GUARD_1D_SZ(n, elem_size, ret_stmt)                                                               \
    do {                                                                                                            \
        if ((n) > 0 && jgpt_alloc_n_elem_overflows((n), static_cast<size_t>(elem_size))) {                           \
            fprintf(stderr, "CUDA JNI alloc overflow (1d sz) %s:%d\n", __FILE__, __LINE__);                         \
            ret_stmt                                                                                                \
        }                                                                                                           \
    } while (0)

/** Для {@code jsize} / длины сегмента: {@code n < 0} или переполнение {@code n * elem_size}. */
#define JGPT_CUDA_GUARD_JSIZE(n, elem_size, ret_stmt)                                                             \
    do {                                                                                                          \
        if ((n) < 0 || ((n) > 0 && jgpt_alloc_n_elem_overflows(static_cast<size_t>(n), static_cast<size_t>(elem_size)))) { \
            fprintf(stderr, "CUDA JNI alloc overflow (jsize) %s:%d\n", __FILE__, __LINE__);                        \
            ret_stmt                                                                                              \
        }                                                                                                         \
    } while (0)

#define JGPT_CUDA_GUARD_VOL3_F(d0, d1, d2, ret_stmt)                                                              \
    do {                                                                                                          \
        if ((d0) < 0 || (d1) < 0 || (d2) < 0 ||                                                                     \
                jgpt_alloc_volume3d_float_overflows(static_cast<size_t>(d0), static_cast<size_t>(d1), static_cast<size_t>(d2))) { \
            fprintf(stderr, "CUDA JNI alloc overflow (vol3) %s:%d\n", __FILE__, __LINE__);                        \
            ret_stmt                                                                                              \
        }                                                                                                         \
    } while (0)

#define JGPT_CUDA_GUARD_CHAIN4_F(a, b, c, d, ret_stmt)                                                            \
    do {                                                                                                          \
        if ((a) < 0 || (b) < 0 || (c) < 0 || (d) < 0 ||                                                              \
                jgpt_alloc_chain4_float_overflows(static_cast<size_t>(a), static_cast<size_t>(b), static_cast<size_t>(c), static_cast<size_t>(d))) { \
            fprintf(stderr, "CUDA JNI alloc overflow (chain4) %s:%d\n", __FILE__, __LINE__);                      \
            ret_stmt                                                                                              \
        }                                                                                                         \
    } while (0)

/** Матрица {@code rows×cols} элементов {@code float} на устройстве. */
#define JGPT_CUDA_GUARD_MATF(rows, cols, ret_stmt)                                                                  \
    do {                                                                                                            \
        if ((rows) < 0 || (cols) < 0 ||                                                                               \
                jgpt_alloc_matrix_bytes_overflows(static_cast<size_t>(rows), static_cast<size_t>(cols), sizeof(float))) { \
            fprintf(stderr, "CUDA JNI alloc overflow (mat float) %s:%d\n", __FILE__, __LINE__);                     \
            ret_stmt                                                                                                \
        }                                                                                                           \
    } while (0)

