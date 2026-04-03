package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Fp16Tensor;
import com.veles.llm.jgpt.core.Tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Хранение одной активации для backward: либо ссылка на FP32 {@link Tensor} (как при обычном
 * forward), либо сжатие в {@link Fp16Tensor} для экономии памяти между forward и backward (gradient
 * checkpointing по слою).
 *
 * <p>При {@code useFp16 == true} значения копируются в half; {@link #getTensor()} при необходимости
 * восстанавливает FP32 (лениво один раз в {@code fp32Cache}). Градиенты по этим буферам в backward
 * идут в отдельные тензоры — потеря точности только в сохранённых активациях.
 *
 * <p><b>Потокобезопасность:</b> класс не потокобезопасен; вызывать из одного потока обучения.
 *
 * <p><b>Ожидаемый цикл:</b> {@link #store} после forward или {@link #ensureTensor} → запись → {@link
 * #finalizeAfterWrite}; {@link #getTensor} в backward; {@link #clear} по завершении.
 */
public final class ActivationTensorSlot {

    /** Как сейчас представлены данные в слоте (без учёта производного FP32-кэша после FP16). */
    public enum StorageMode {
        EMPTY,
        FP32,
        FP16
    }

    private Tensor fp32;
    private Fp16Tensor fp16;
    /** Ленивый FP32 после {@link #fp16}; сбрасывается при смене содержимого слота. */
    private Tensor fp32Cache;

    private int cachedShapeHash;

    public StorageMode mode() {
        if (fp32 != null) {
            return StorageMode.FP32;
        }
        if (fp16 != null) {
            return StorageMode.FP16;
        }
        return StorageMode.EMPTY;
    }

    public void store(Tensor activation, boolean useFp16) {
        Objects.requireNonNull(activation, "activation");
        clear();
        if (useFp16) {
            try {
                fp16 = Fp16Tensor.fromTensor(activation);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                        String.format(
                                "Failed to compress activation [shape=%s] to FP16: %s",
                                Arrays.toString(activation.getShape()), e.getMessage()),
                        e);
            }
        } else {
            fp32 = activation;
            cachedShapeHash = Arrays.hashCode(activation.getShape());
        }
    }

    /**
     * Возвращает или создаёт FP32-буфер нужной формы для записи (например D2H с GPU); после заполнения
     * вызвать {@link #finalizeAfterWrite(boolean)}.
     *
     * <p><b>Внимание:</b> возвращённый тензор — внутренний буфер слота; не использовать после {@link
     * #finalizeAfterWrite(boolean)}. Для изолированной копии — {@link #getTensor()} и клонирование
     * данных отдельно.
     */
    public Tensor ensureTensor(int[] shape) {
        Objects.requireNonNull(shape, "shape");
        if (fp32 != null) {
            int[] cur = fp32.getShape();
            if (shape.length == cur.length) {
                int h = Arrays.hashCode(shape);
                if (h == cachedShapeHash && Arrays.equals(cur, shape)) {
                    fp16 = null;
                    fp32Cache = null;
                    return fp32;
                }
            }
        }
        fp32 = new Tensor(shape);
        fp16 = null;
        fp32Cache = null;
        cachedShapeHash = Arrays.hashCode(shape);
        return fp32;
    }

    /**
     * После записи в буфер из {@link #ensureTensor}: при {@code compressToFp16} копирует в FP16 и
     * освобождает FP32 в слоте.
     */
    public void finalizeAfterWrite(boolean compressToFp16) {
        if (fp32 == null) {
            return;
        }
        if (compressToFp16) {
            Fp16Tensor compressed = Fp16Tensor.fromTensor(fp32);
            fp32 = null;
            fp16 = compressed;
            fp32Cache = null;
            cachedShapeHash = 0;
        }
    }

    /** FP32 для backward; при хранении только FP16 — одна ленивая конвертация с кэшированием. */
    public Tensor getTensor() {
        if (fp32 != null) {
            return fp32;
        }
        if (fp16 != null) {
            if (fp32Cache == null) {
                fp32Cache = fp16.toTensor();
            }
            return fp32Cache;
        }
        return null;
    }

    public boolean isEmpty() {
        return fp32 == null && fp16 == null;
    }

    public void clear() {
        fp32 = null;
        fp16 = null;
        fp32Cache = null;
        cachedShapeHash = 0;
    }

    @Override
    public String toString() {
        return switch (mode()) {
            case FP32 -> "ActivationTensorSlot{mode=FP32, shape=" + Arrays.toString(fp32.getShape()) + "}";
            case FP16 -> "ActivationTensorSlot{mode=FP16, shape=" + Arrays.toString(fp16.getShape()) + "}";
            case EMPTY -> "ActivationTensorSlot{mode=EMPTY}";
        };
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ActivationTensorSlot that)) {
            return false;
        }
        return Objects.equals(fp32, that.fp32) && Objects.equals(fp16, that.fp16);
    }

    @Override
    public int hashCode() {
        return Objects.hash(fp32, fp16);
    }
}
