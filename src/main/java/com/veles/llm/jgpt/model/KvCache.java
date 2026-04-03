package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;

import java.util.Arrays;

/**
 * Кэш K/V после RoPE по слоям: форма на слой {@code [batch=1, num_heads, max_seq_len, d_head]}.
 * Длина {@link #length()} — число заполненных позиций по оси последовательности.
 * <p>
 * <b>Thread-safety:</b> не потокобезопасен; один экземпляр — из одного потока инференса.
 */
public final class KvCache {

    private final Tensor[] k;
    private final Tensor[] v;
    private final int maxSeqLen;
    private int length;

    public KvCache(int numLayers, int numHeads, int dHead, int maxSeqLen) {
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive, got " + numLayers);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive, got " + numHeads);
        }
        if (dHead <= 0) {
            throw new IllegalArgumentException("dHead must be positive, got " + dHead);
        }
        if (maxSeqLen <= 0) {
            throw new IllegalArgumentException("maxSeqLen must be positive, got " + maxSeqLen);
        }
        this.maxSeqLen = maxSeqLen;
        this.k = new Tensor[numLayers];
        this.v = new Tensor[numLayers];
        int[] shape = {1, numHeads, maxSeqLen, dHead};
        for (int i = 0; i < numLayers; i++) {
            k[i] = new Tensor(shape);
            v[i] = new Tensor(shape);
        }
    }

    public Tensor getK(int layer) {
        return k[checkLayer(layer)];
    }

    public Tensor getV(int layer) {
        return v[checkLayer(layer)];
    }

    private int checkLayer(int layer) {
        if (layer < 0 || layer >= k.length) {
            throw new IndexOutOfBoundsException(
                    "layer " + layer + " out of range [0, " + k.length + ")");
        }
        return layer;
    }

    public int length() {
        return length;
    }

    /**
     * Устанавливает число заполненных позиций по seq; {@code 0 … maxSeqLen}.
     */
    public void setLength(int len) {
        if (len < 0 || len > maxSeqLen) {
            throw new IllegalArgumentException(
                    "len must be in [0, " + maxSeqLen + "], got " + len);
        }
        this.length = len;
    }

    /** Максимальная длина последовательности (размер оси кэша). */
    public int maxSeqLen() {
        return maxSeqLen;
    }

    /** Сколько ещё токенов можно добавить до заполнения кэша. */
    public int remainingCapacity() {
        return maxSeqLen - length;
    }

    /** Достигнут ли предел по длине контекста. */
    public boolean isFull() {
        return length >= maxSeqLen;
    }

    /**
     * Сбрасывает длину и обнуляет буферы K/V (чтобы не оставлять старые активации в памяти при новой сессии).
     */
    public void clear() {
        length = 0;
        for (Tensor t : k) {
            Arrays.fill(t.internalBuffer(), 0f);
        }
        for (Tensor t : v) {
            Arrays.fill(t.internalBuffer(), 0f);
        }
    }

    public int numLayers() {
        return k.length;
    }
}
