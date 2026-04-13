package com.veles.llm.jgpt;

import java.util.Objects;

/** Проверки Gpu* буферов перед JNI {@link TensorOpsGPU}. */
final class TensorOpsGpuBufferChecks {

    private TensorOpsGpuBufferChecks() {}

    static GpuFloatBuffer requireGpu(GpuFloatBuffer b, String name) {
        Objects.requireNonNull(b, name);
        if (b.isClosed()) {
            throw new IllegalStateException(name + ": GpuFloatBuffer is closed");
        }
        return b;
    }

    static void requireMinFloats(GpuFloatBuffer b, long need, String name) {
        if (need < 0) {
            throw new IllegalArgumentException(name + ": invalid required length");
        }
        if (b.numFloats() < need) {
            throw new IllegalArgumentException(
                    name + ": buffer too small, need >= " + need + " floats, have " + b.numFloats());
        }
    }

    static GpuIntBuffer requireGpuInt(GpuIntBuffer b, String name) {
        Objects.requireNonNull(b, name);
        if (b.isClosed()) {
            throw new IllegalStateException(name + ": GpuIntBuffer is closed");
        }
        return b;
    }

    static void requireMinInts(GpuIntBuffer b, long need, String name) {
        if (need < 0) {
            throw new IllegalArgumentException(name + ": invalid required length");
        }
        if (b.numInts() < need) {
            throw new IllegalArgumentException(
                    name + ": buffer too small, need >= " + need + " ints, have " + b.numInts());
        }
    }

    /**
     * Часть JNI/нативных путей индексирует плоские буферы как {@code int}; гарантируем, что размер не выходит за
     * {@link Integer#MAX_VALUE}.
     */
    static void requireJniFlatElementCount(String name, long n) {
        if (n < 0 || n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    name + " element count out of int32 range for JNI: " + n + " (max " + Integer.MAX_VALUE + ")");
        }
    }
}
