package com.veles.llm.jgpt.core;

import java.util.Arrays;
import java.util.Objects;

/**
 * Общие утилиты для тензоров: валидация формы, вычисление размера и шагов.
 * Используется всеми классами тензоров ({@link Tensor}, {@link Fp16Tensor},
 * {@link QuantizedTensor}, {@code GpuTensor}) для устранения дублирования.
 */
public final class TensorUtils {

    private TensorUtils() {
    }

    /**
     * Проверяет и клонирует массив shape.
     * @throws IllegalArgumentException если shape null/empty или содержит неположительные размеры
     */
    public static int[] validateAndCloneShape(int[] shape) {
        Objects.requireNonNull(shape, "shape cannot be null");
        if (shape.length == 0) {
            throw new IllegalArgumentException("shape must have at least one dimension");
        }
        int[] cloned = shape.clone();
        for (int i = 0; i < cloned.length; i++) {
            if (cloned[i] <= 0) {
                throw new IllegalArgumentException(
                        String.format("dimension %d must be positive, got %d", i, cloned[i]));
            }
        }
        return cloned;
    }

    /**
     * Вычисляет произведение размеров с проверкой переполнения.
     * @throws IllegalArgumentException если произведение превышает Integer.MAX_VALUE
     */
    public static int computeSizeOrThrow(int[] shape) {
        long sz = 1;
        for (int dim : shape) {
            sz *= dim;
            if (sz > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(String.format(
                        "Shape product overflow: %s → %d elements (max Integer.MAX_VALUE=%d)",
                        Arrays.toString(shape), sz, Integer.MAX_VALUE));
            }
        }
        return (int) sz;
    }

    /**
     * Вычисляет row-major strides.
     * Для rank-0 возвращает пустой массив.
     */
    public static int[] calculateStrides(int[] shape) {
        int rank = shape.length;
        int[] st = new int[rank];
        if (rank == 0) {
            return st;
        }
        st[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; i--) {
            st[i] = st[i + 1] * shape[i + 1];
        }
        return st;
    }

    /**
     * Вычисляет плоский индекс из многомерных индексов.
     * @param indices многомерные индексы
     * @param shape форма тензора
     * @param strides шаги
     * @param checkBounds проверять ли границы
     * @throws IllegalArgumentException если число измерений не совпадает
     * @throws IndexOutOfBoundsException если индекс вне границ (при checkBounds)
     */
    public static int flatIndex(int[] indices, int[] shape, int[] strides, boolean checkBounds) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                    String.format("Expected %d indices, got %d", shape.length, indices.length));
        }
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            if (checkBounds && (indices[i] < 0 || indices[i] >= shape[i])) {
                throw new IndexOutOfBoundsException(String.format(
                        "Index %d out of bounds for dimension %d (size %d)",
                        indices[i], i, shape[i]));
            }
            index += indices[i] * strides[i];
        }
        return index;
    }
}
