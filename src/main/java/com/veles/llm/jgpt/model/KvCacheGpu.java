package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;

/**
 * KV-кэш после RoPE на VRAM: на слой — плоский {@link GpuFloatBuffer} длины {@code num_heads * max_seq_len *
 * d_head} (тот же линейный порядок, что у {@link KvCache}).
 *
 * <p><b>Thread-safety:</b> как у {@link KvCache} — один поток инференса.
 */
public final class KvCacheGpu implements AutoCloseable {

    private final GpuFloatBuffer[] k;
    private final GpuFloatBuffer[] v;
    private final int maxSeqLen;
    private int length;

    public KvCacheGpu(int numLayers, int numHeads, int dHead, int maxSeqLen) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available for KvCacheGpu");
        }
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
        long perLayer = (long) numHeads * maxSeqLen * dHead;
        this.k = new GpuFloatBuffer[numLayers];
        this.v = new GpuFloatBuffer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            k[i] = GpuFloatBuffer.allocate(perLayer);
            v[i] = GpuFloatBuffer.allocate(perLayer);
            k[i].clear();
            v[i].clear();
        }
    }

    public GpuFloatBuffer getK(int layer) {
        return k[checkLayer(layer)];
    }

    public GpuFloatBuffer getV(int layer) {
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

    public void setLength(int len) {
        if (len < 0 || len > maxSeqLen) {
            throw new IllegalArgumentException("len must be in [0, " + maxSeqLen + "], got " + len);
        }
        this.length = len;
    }

    public int maxSeqLen() {
        return maxSeqLen;
    }

    public int remainingCapacity() {
        return maxSeqLen - length;
    }

    public boolean isFull() {
        return length >= maxSeqLen;
    }

    public void clear() {
        length = 0;
        for (GpuFloatBuffer b : k) {
            b.clear();
        }
        for (GpuFloatBuffer b : v) {
            b.clear();
        }
    }

    public int numLayers() {
        return k.length;
    }

    @Override
    public String toString() {
        boolean closed = k.length > 0 && k[0] != null && k[0].isClosed();
        long perLayer = k.length > 0 && k[0] != null ? k[0].numFloats() : 0L;
        return String.format(
                "KvCacheGpu{layers=%d, maxSeqLen=%d, len=%d, floatsPerLayer=%d, closed=%b}",
                k.length, maxSeqLen, length, perLayer, closed);
    }

    @Override
    public void close() {
        for (GpuFloatBuffer b : k) {
            b.close();
        }
        for (GpuFloatBuffer b : v) {
            b.close();
        }
    }
}
