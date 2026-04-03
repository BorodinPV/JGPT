package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;

/**
 * Общие операции над {@link GpuFloatBuffer} и хостовыми staging-массивами для GPU workspace.
 *
 * <p><b>Контракт device-буфера:</b> при переиспользовании без новой аллокации память <b>не</b> обнуляется.
 * Ядра CUDA и вызывающий код должны полностью перезаписать логически используемый префикс буфера перед любыми
 * чтениями, где допускается мусор (в т.ч. градиенты с {@code +=} в ядре — только если ядро само задаёт
 * начальное состояние).
 */
final class GpuBufferUtils {

    private GpuBufferUtils() {}

    static long mulExact(String what, long a, long b) {
        try {
            return Math.multiplyExact(a, b);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Size overflow (" + what + "): " + a + " * " + b, e);
        }
    }

    static int intSize(String what, long value) {
        if (value < 0 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Size out of int range (" + what + "): " + value);
        }
        return (int) value;
    }

    /**
     * Держит буфер не короче {@code minFloats}; иначе закрывает и выделяет новый.
     *
     * @see #mulExact
     */
    static GpuFloatBuffer ensure(GpuFloatBuffer buffer, long minFloats) {
        if (buffer != null && !buffer.isClosed() && buffer.numFloats() >= minFloats) {
            return buffer;
        }
        if (buffer != null && !buffer.isClosed()) {
            buffer.close();
        }
        return GpuFloatBuffer.allocate(minFloats);
    }

    static GpuFloatBuffer closeAndNull(GpuFloatBuffer b) {
        if (b != null && !b.isClosed()) {
            b.close();
        }
        return null;
    }

    /** Переиспользовать массив, если длины хватает; иначе новый. */
    static float[] ensureHost(float[] buffer, int minLength) {
        if (buffer != null && buffer.length >= minLength) {
            return buffer;
        }
        return new float[minLength];
    }
}
