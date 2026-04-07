#pragma once

#include <climits>
#include <cstddef>

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
