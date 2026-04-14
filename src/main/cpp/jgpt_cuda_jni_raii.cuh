#pragma once

#include <cuda_runtime.h>
#include <jni.h>
#include <cstdio>
#include <cstdarg>

/**
 * RAII-обёртки для JNI-вызовов и CUDA-памяти.
 * Сокращают шаблонный код GetFloatArrayElements/ReleaseFloatArrayElements
 * и cudaMalloc/cudaFree в ~40 JNI-функциях.
 */

/* ========== JNI float array scope ========== */

/**
 * Захватывает jfloatArray через GetFloatArrayElements и освобождает в деструкторе.
 * Использование:
 *   JniFloatArrayScope a(env, h_A);
 *   if (!a) { /* OOM — cleanup + return *\/ }
 *   float* ptr = a.ptr;
 */
struct JniFloatArrayScope {
    JNIEnv* env;
    jfloatArray arr;
    jfloat* ptr;
    jint mode;  /* JNI_ABORT или 0 */

    JniFloatArrayScope(JNIEnv* e, jfloatArray a, jint m = JNI_ABORT)
        : env(e), arr(a), ptr(nullptr), mode(m) {
        if (a != nullptr) {
            ptr = env->GetFloatArrayElements(a, nullptr);
        }
    }

    ~JniFloatArrayScope() {
        if (ptr != nullptr && arr != nullptr) {
            env->ReleaseFloatArrayElements(arr, ptr, mode);
        }
    }

    /* Запрет копирования */
    JniFloatArrayScope(const JniFloatArrayScope&) = delete;
    JniFloatArrayScope& operator=(const JniFloatArrayScope&) = delete;

    /* Move */
    JniFloatArrayScope(JniFloatArrayScope&& o) noexcept
        : env(o.env), arr(o.arr), ptr(o.ptr), mode(o.mode) {
        o.ptr = nullptr; o.arr = nullptr;
    }

    operator bool() const { return ptr != nullptr; }
};

/* ========== Device memory RAII ========== */

/**
 * RAII-обёртка над device-указателем. При уничтожении вызывает cudaFree.
 * Для передачи владения — .release().
 */
struct DevicePtr {
    void* ptr = nullptr;

    DevicePtr() = default;
    explicit DevicePtr(void* p) : ptr(p) {}
    ~DevicePtr() { reset(); }

    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;

    DevicePtr(DevicePtr&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    DevicePtr& operator=(DevicePtr&& o) noexcept {
        if (this != &o) { reset(); ptr = o.ptr; o.ptr = nullptr; }
        return *this;
    }

    void reset() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    /** Освободить и вернуть nullptr (передача владения). */
    void* release() {
        void* p = ptr;
        ptr = nullptr;
        return p;
    }

    template<typename T>
    T* get() const { return static_cast<T*>(ptr); }

    operator bool() const { return ptr != nullptr; }
};

/**
 * Устройство-указатель для float. RAII — cudaFree в деструкторе.
 * Для передачи владения — .release().
 */
struct DeviceFloatPtr {
    float* ptr = nullptr;

    DeviceFloatPtr() = default;
    explicit DeviceFloatPtr(float* p) : ptr(p) {}
    ~DeviceFloatPtr() { reset(); }

    DeviceFloatPtr(const DeviceFloatPtr&) = delete;
    DeviceFloatPtr& operator=(const DeviceFloatPtr&) = delete;

    DeviceFloatPtr(DeviceFloatPtr&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    DeviceFloatPtr& operator=(DeviceFloatPtr&& o) noexcept {
        if (this != &o) { reset(); ptr = o.ptr; o.ptr = nullptr; }
        return *this;
    }

    void reset() {
        if (ptr != nullptr) { cudaFree(ptr); ptr = nullptr; }
    }

    float* release() { float* p = ptr; ptr = nullptr; return p; }
    operator float*() const { return ptr; }
    float& operator[](size_t i) { return ptr[i]; }
    const float& operator[](size_t i) const { return ptr[i]; }
};

/* ========== JNI int array scope ========== */

/**
 * Захватывает jintArray через GetIntArrayElements и освобождает в деструкторе.
 * Использование аналогично JniFloatArrayScope.
 */
struct JniIntArrayScope {
    JNIEnv* env;
    jintArray arr;
    jint* ptr;
    jint mode;  /* JNI_ABORT или 0 */

    JniIntArrayScope(JNIEnv* e, jintArray a, jint m = JNI_ABORT)
        : env(e), arr(a), ptr(nullptr), mode(m) {
        if (a != nullptr) {
            ptr = env->GetIntArrayElements(a, nullptr);
        }
    }

    ~JniIntArrayScope() {
        if (ptr != nullptr && arr != nullptr) {
            env->ReleaseIntArrayElements(arr, ptr, mode);
        }
    }

    /* Запрет копирования */
    JniIntArrayScope(const JniIntArrayScope&) = delete;
    JniIntArrayScope& operator=(const JniIntArrayScope&) = delete;

    /* Move */
    JniIntArrayScope(JniIntArrayScope&& o) noexcept
        : env(o.env), arr(o.arr), ptr(o.ptr), mode(o.mode) {
        o.ptr = nullptr; o.arr = nullptr;
    }

    operator bool() const { return ptr != nullptr; }
};

/* ========== JNI long array scope ========== */

/**
 * Захватывает jlongArray через GetLongArrayElements и освобождает в деструкторе.
 * Использование аналогично JniIntArrayScope.
 */
struct JniLongArrayScope {
    JNIEnv* env;
    jlongArray arr;
    jlong* ptr;
    jint mode;  /* JNI_ABORT или 0 */

    JniLongArrayScope(JNIEnv* e, jlongArray a, jint m = JNI_ABORT)
        : env(e), arr(a), ptr(nullptr), mode(m) {
        if (a != nullptr) {
            ptr = env->GetLongArrayElements(a, nullptr);
        }
    }

    ~JniLongArrayScope() {
        if (ptr != nullptr && arr != nullptr) {
            env->ReleaseLongArrayElements(arr, ptr, mode);
        }
    }

    /* Запрет копирования */
    JniLongArrayScope(const JniLongArrayScope&) = delete;
    JniLongArrayScope& operator=(const JniLongArrayScope&) = delete;

    /* Move */
    JniLongArrayScope(JniLongArrayScope&& o) noexcept
        : env(o.env), arr(o.arr), ptr(o.ptr), mode(o.mode) {
        o.ptr = nullptr; o.arr = nullptr;
    }

    operator bool() const { return ptr != nullptr; }
};

/* ========== Direct buffer scope ========== */

/**
 * RAII-обёртка над GetDirectBufferAddress.
 * Не требует Release (просто адрес), но проверяет nullptr.
 */
struct JniDirectBufferScope {
    JNIEnv* env;
    jobject buf;
    void* addr;

    JniDirectBufferScope(JNIEnv* e, jobject b)
        : env(e), buf(b), addr(b != nullptr ? env->GetDirectBufferAddress(b) : nullptr) {}

    operator bool() const { return addr != nullptr; }
    void* get() const { return addr; }
};

/* ========== Helpers ========== */

/** Проверить несколько JniFloatArrayScope; если хоть один nullptr — cleanup + return. */
#define JGPT_JNI_CHECK_FLOAT_ARRAYS(...) \
    do { \
        if (!_jgpt_jni_all_float_arrays_valid(__VA_ARGS__, nullptr)) { \
            return; \
        } \
    } while (0)

/** Проверить несколько JniIntArrayScope; если хоть один nullptr — cleanup + return. */
#define JGPT_JNI_CHECK_INT_ARRAYS(...) \
    do { \
        if (!_jgpt_jni_all_int_arrays_valid(__VA_ARGS__, nullptr)) { \
            return; \
        } \
    } while (0)

/** Проверить несколько JniLongArrayScope; если хоть один nullptr — cleanup + return. */
#define JGPT_JNI_CHECK_LONG_ARRAYS(...) \
    do { \
        if (!_jgpt_jni_all_long_arrays_valid(__VA_ARGS__, nullptr)) { \
            return; \
        } \
    } while (0)

static inline bool _jgpt_jni_all_float_arrays_valid(JniFloatArrayScope* first, ...) {
    va_list args;
    va_start(args, first);
    JniFloatArrayScope* cur = first;
    while (cur != nullptr) {
        if (!static_cast<bool>(*cur)) {
            va_end(args);
            return false;
        }
        cur = va_arg(args, JniFloatArrayScope*);
    }
    va_end(args);
    return true;
}

static inline bool _jgpt_jni_all_int_arrays_valid(JniIntArrayScope* first, ...) {
    va_list args;
    va_start(args, first);
    JniIntArrayScope* cur = first;
    while (cur != nullptr) {
        if (!static_cast<bool>(*cur)) {
            va_end(args);
            return false;
        }
        cur = va_arg(args, JniIntArrayScope*);
    }
    va_end(args);
    return true;
}

static inline bool _jgpt_jni_all_long_arrays_valid(JniLongArrayScope* first, ...) {
    va_list args;
    va_start(args, first);
    JniLongArrayScope* cur = first;
    while (cur != nullptr) {
        if (!static_cast<bool>(*cur)) {
            va_end(args);
            return false;
        }
        cur = va_arg(args, JniLongArrayScope*);
    }
    va_end(args);
    return true;
}
