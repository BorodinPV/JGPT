package com.veles.llm.jgpt.core;

import java.util.Arrays;

/**
 * Тензор в IEEE 754 binary16 (FP16): буфер {@code short[]} — значения half ({@link Float#floatToFloat16(float)}).
 * Память ~2× меньше, чем у {@link Tensor} (float32). Для градиентов и обучения по умолчанию используйте {@link Tensor}.
 * <p>
 * Требует <b>JDK с FP16 API</b> ({@link Float#float16ToFloat(short)} / {@link Float#floatToFloat16(float)}, по сути Java 20+).
 * <p>
 * Согласованность с {@link QuantizedTensor}: {@link #fromFloat16Array(short[], int[])} копирует буферы;
 * {@link #wrap(short[], int[])} — zero-copy по данным (см. javadoc).
 */
public final class Fp16Tensor {

    private final short[] data;
    private final int[] shape;
    private final int[] strides;

    /** Создаёт новый тензор с нулевым буфером. */
    public Fp16Tensor(int[] shape) {
        this.shape = TensorUtils.validateAndCloneShape(shape);
        this.strides = TensorUtils.calculateStrides(this.shape);
        int size = TensorUtils.computeSizeOrThrow(this.shape);
        this.data = new short[size];
    }

    /**
     * Общий приватный конструктор: форма всегда клонируется, чтобы внешние изменения {@code shape} не ломали
     * инварианты; {@code data} хранится по переданной ссылке (см. {@link #wrap} / {@link #fromFloat16Array}).
     */
    private Fp16Tensor(int[] shape, short[] data) {
        this.shape = TensorUtils.validateAndCloneShape(shape);
        this.strides = TensorUtils.calculateStrides(this.shape);
        int expectedSize = TensorUtils.computeSizeOrThrow(this.shape);
        if (data.length != expectedSize) {
            throw new IllegalArgumentException(
                    "data length " + data.length + " != shape product for " + Arrays.toString(this.shape));
        }
        this.data = data;
    }

    /**
     * Оборачивает существующий массив без копирования данных (как {@link Tensor#wrap(float[], int[])}).
     * <p><b>Важно:</b> после вызова буфер считается принадлежащим тензору; внешние записи в {@code data} — такие же
     * опасны, как гонки при разделении массива между потоками.
     */
    public static Fp16Tensor wrap(short[] data, int[] shape) {
        int n = TensorUtils.computeSizeOrThrow(shape);
        if (data.length != n) {
            throw new IllegalArgumentException("data length " + data.length + " != shape product " + n);
        }
        return new Fp16Tensor(shape, data);
    }

    /**
     * Копирует half-биты и форму (аналог {@link QuantizedTensor#fromBytes(byte[], int[], float)}).
     */
    public static Fp16Tensor fromFloat16Array(short[] data, int[] shape) {
        int n = TensorUtils.computeSizeOrThrow(shape);
        if (data.length != n) {
            throw new IllegalArgumentException("data length " + data.length + " != shape product " + n);
        }
        return new Fp16Tensor(shape, Arrays.copyOf(data, data.length));
    }

    /**
     * Конвертация float32 → FP16 с округлением по IEEE.
     *
     * @throws IllegalArgumentException если тензор содержит NaN или Infinity
     */
    public static Fp16Tensor fromTensor(Tensor t) {
        float[] f = t.internalBuffer();
        short[] h = new short[f.length];
        for (int i = 0; i < f.length; i++) {
            float v = f[i];
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                throw new IllegalArgumentException(
                        "Tensor contains NaN or Infinity; FP16 path expects finite values");
            }
            h[i] = Float.floatToFloat16(v);
        }
        return new Fp16Tensor(t.getShape(), h);
    }

    /** Полный float32-тензор (новая аллокация). */
    public Tensor toTensor() {
        Tensor tensor = new Tensor(shape);
        copyToFloats(tensor.internalBuffer(), 0);
        return tensor;
    }

    /** Новый массив float32 той же длины, что и {@link #numElements()} (без обёртки {@link Tensor}). */
    public float[] toFloatArray() {
        float[] o = new float[data.length];
        copyToFloats(o, 0);
        return o;
    }

    /**
     * Записывает float16→float32 в {@code dst}, начиная с {@code dstOffset}.
     * Цикл полагается на интринсики JVM для {@link Float#float16ToFloat(short)}; для очень больших буферов
     * предпочтительны массовые пути (Vector API / натив), а не этот метод в tight loop.
     *
     * @throws IllegalArgumentException если диапазон выходит за пределы {@code dst}
     */
    public void copyToFloats(float[] dst, int dstOffset) {
        if (dstOffset < 0 || dstOffset + data.length > dst.length) {
            throw new IllegalArgumentException(
                    "dst range: need length >= " + (dstOffset + data.length) + ", dst.length=" + dst.length
                            + ", dstOffset=" + dstOffset);
        }
        for (int i = 0; i < data.length; i++) {
            dst[dstOffset + i] = Float.float16ToFloat(data[i]);
        }
    }

    /**
     * Элемент с проверкой границ и размерности. Для hot-loop по плоскому индексу используйте {@link
     * #internalBuffer()} и свой индекс.
     */
    public float get(int... indices) {
        return Float.float16ToFloat(data[TensorUtils.flatIndex(indices, shape, strides, true)]);
    }

    /** Запись с проверкой границ; NaN и Infinity отклоняются. */
    public void set(float value, int... indices) {
        if (Float.isNaN(value) || Float.isInfinite(value)) {
            throw new IllegalArgumentException("value must be finite");
        }
        data[TensorUtils.flatIndex(indices, shape, strides, true)] = Float.floatToFloat16(value);
    }

    /**
     * Прямой доступ к half-битам (zero-copy). Изменения массива отражаются в тензоре; не отдавайте в другие потоки без
     * синхронизации контракта доступа.
     */
    public short[] internalBuffer() {
        return data;
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

    /**
     * Внутренний массив шагов без копирования (как {@link Tensor#stridesInternal()}).
     * <p>
     * <b>Только для внутреннего использования:</b> не изменять возвращённый массив.
     */
    public int[] stridesInternal() {
        return strides;
    }

    public int rank() {
        return shape.length;
    }

    /** Размер измерения {@code dim} (0 … {@link #rank()}{@code -1}). */
    public int dimension(int dim) {
        return shape[dim];
    }

    /** Число элементов (не байт). */
    public int numElements() {
        return data.length;
    }

    /** Размер буфера в байтах (2 на элемент). */
    public long sizeBytes() {
        return (long) data.length * 2L;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Fp16Tensor that)) {
            return false;
        }
        return Arrays.equals(shape, that.shape) && Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(shape) + Arrays.hashCode(data);
    }

    @Override
    public String toString() {
        return "Fp16Tensor{shape=" + Arrays.toString(shape) + ", elements=" + data.length + "}";
    }
}
