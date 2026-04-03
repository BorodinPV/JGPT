package com.veles.llm.jgpt.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * Хранение весов/активаций в int8 ({@code byte[]}) — в ~4× меньше памяти, чем {@link Tensor}.
 * Симметричная квантизация: диапазон {@code [-scale, scale]} отображается в {@code [-127, 127]}.
 * <p>
 * Диапазон квантования намеренно {@code [-127, 127]} (значение {@code -128} не используется), в духе совместимости
 * с некоторыми GEMM/Tensor Core конвенциями.
 * <p>
 * Вычисления в {@link com.veles.llm.jgpt.ops.TensorOps} для квантованных входов обычно выполняются в float32 после
 * деквантования ({@link #toTensor()}, {@linkplain #get(int...) get}).
 * <p>
 * <b>Шкала:</b> при {@code max|x| = 0} в {@link #fromTensor(Tensor)} берётся {@code scale = 1} (все коды 0). Иначе
 * {@code scale = max|x|}. Промежуточные вычисления квантования — в {@code double}.
 */
public final class QuantizedTensor {

    private final byte[] data;
    private final int[] shape;
    private final int[] strides;
    /** Полуамплитуда: {@code |float| ≤ scale} соответствует коду в [-127, 127]. */
    private final float scale;

    /** Создаёт новый тензор с нулевым буфером и {@code scale = 1}. */
    public QuantizedTensor(int[] shape) {
        validateShape(shape);
        this.shape = shape.clone();
        this.strides = calculateStrides(this.shape);
        this.data = new byte[numElementsForShape(this.shape)];
        this.scale = 1.0f;
    }

    /**
     * Внутренний конструктор: {@code short[]} форма всегда клонируется; {@code data} хранится по ссылке
     * (вызывающий обеспечивает длину и согласованность шкалы).
     */
    private QuantizedTensor(int[] shape, float scale, byte[] data) {
        this.shape = shape.clone();
        this.strides = calculateStrides(this.shape);
        if (scale <= 0f || !Float.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite and > 0, got " + scale);
        }
        int n = numElementsForShape(this.shape);
        if (data.length != n) {
            throw new IllegalArgumentException(
                    "data length " + data.length + " != shape product " + n + " for " + Arrays.toString(this.shape));
        }
        this.scale = scale;
        this.data = data;
    }

    /** Квантует float32-тензор; шкала — {@code max|x|} или {@code 1}, если все элементы нулевые. */
    public static QuantizedTensor fromTensor(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        float[] buf = t.internalBuffer();
        float maxAbs = 0f;
        for (float v : buf) {
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                throw new IllegalArgumentException(
                        "Tensor contains NaN or Infinity; quantization expects finite values");
            }
            maxAbs = Math.max(maxAbs, Math.abs(v));
        }
        float s = maxAbs == 0f ? 1f : maxAbs;
        byte[] bytes = new byte[buf.length];
        for (int i = 0; i < buf.length; i++) {
            bytes[i] = quantizeToByte(buf[i], s);
        }
        return new QuantizedTensor(t.getShape(), s, bytes);
    }

    /**
     * Собирает тензор из уже квантованного буфера (данные копируются).
     *
     * @param quantized массив int8-токенов длины, равной произведению размерностей {@code shape}
     * @param scale положительная конечная шкала
     */
    public static QuantizedTensor fromBytes(byte[] quantized, int[] shape, float scale) {
        validateShape(shape);
        Objects.requireNonNull(quantized, "quantized");
        int n = numElementsForShape(shape);
        if (quantized.length != n) {
            throw new IllegalArgumentException(
                    "data length " + quantized.length + " != shape product for " + Arrays.toString(shape));
        }
        if (scale <= 0f || !Float.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite and > 0, got " + scale);
        }
        return new QuantizedTensor(shape, scale, Arrays.copyOf(quantized, quantized.length));
    }

    /**
     * Копия формы; не изменяйте возвращённый массив, если полагаетесь на неизменность тензора.
     */
    public int[] getShape() {
        return shape.clone();
    }

    /**
     * Копия шагов; не изменяйте возвращённый массив.
     */
    public int[] getStrides() {
        return strides.clone();
    }

    public int rank() {
        return shape.length;
    }

    /** Размер измерения {@code dim} (0 … {@link #rank()}{@code -1}). */
    public int dimension(int dim) {
        return shape[dim];
    }

    public float getScale() {
        return scale;
    }

    /** Число элементов (не байт смысловых «каналов» — один элемент = один {@code byte}). */
    public int numElements() {
        return data.length;
    }

    /** Размер буфера в байтах (1 на элемент). */
    public long sizeBytes() {
        return (long) data.length;
    }

    /**
     * Деквантование: {@code (q / 127) * scale}. Не для tight loop — предпочтительнее {@link #internalBuffer()} и
     * плоский индекс при массовом доступе.
     */
    public float get(int... indices) {
        int q = data[flatIndex(indices, true)];
        return (q / 127.0f) * scale;
    }

    /** Квантование в {@code [-127, 127]}; не допускает NaN/Infinity. */
    public void set(float value, int... indices) {
        if (Float.isNaN(value) || Float.isInfinite(value)) {
            throw new IllegalArgumentException("value must be finite");
        }
        data[flatIndex(indices, true)] = quantizeToByte(value, scale);
    }

    /**
     * Прямой доступ к байтам (знаковый int8). Изменения отражаются в тензоре; не раздавать между потоками без
     * явного контракта.
     */
    public byte[] internalBuffer() {
        return data;
    }

    /**
     * Полный float32-тензор (новая аллокация). Для очень больших буферов возможна векторизация (Vector API) вместо
     * скалярного цикла.
     */
    public Tensor toTensor() {
        Tensor t = new Tensor(shape);
        float[] out = t.internalBuffer();
        for (int i = 0; i < data.length; i++) {
            int q = data[i];
            out[i] = (q / 127.0f) * scale;
        }
        return t;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof QuantizedTensor that)) {
            return false;
        }
        return Float.floatToRawIntBits(scale) == Float.floatToRawIntBits(that.scale)
                && Arrays.equals(shape, that.shape)
                && Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(data);
        result = 31 * result + Float.hashCode(scale);
        return result;
    }

    @Override
    public String toString() {
        return "QuantizedTensor{shape="
                + Arrays.toString(shape)
                + ", elements="
                + data.length
                + ", scale="
                + scale
                + "}";
    }

    private static byte quantizeToByte(float value, float scale) {
        if (scale <= 0f || !Float.isFinite(scale)) {
            throw new IllegalArgumentException("scale must be finite and > 0");
        }
        int q = (int) Math.round((double) value / (double) scale * 127.0);
        return (byte) Math.min(127, Math.max(-127, q));
    }

    private int flatIndex(int[] indices, boolean checkBounds) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("wrong number of indices");
        }
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            if (checkBounds && (indices[i] < 0 || indices[i] >= shape[i])) {
                throw new IndexOutOfBoundsException(
                        "index " + indices[i] + " for dimension " + i + " (size " + shape[i] + ")");
            }
            index += indices[i] * strides[i];
        }
        return index;
    }

    private static int[] calculateStrides(int[] shape) {
        int[] s = new int[shape.length];
        s[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            s[i] = s[i + 1] * shape[i + 1];
        }
        return s;
    }

    private static int numElementsForShape(int[] shape) {
        long size = 1L;
        for (int d : shape) {
            size *= d;
            if (size > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "Total elements exceed Integer.MAX_VALUE (max ~2.14B elements), shape "
                                + Arrays.toString(shape));
            }
        }
        return (int) size;
    }

    private static void validateShape(int[] shape) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("shape must be non-empty");
        }
        for (int d : shape) {
            if (d <= 0) {
                throw new IllegalArgumentException("each dimension must be positive");
            }
        }
    }
}
