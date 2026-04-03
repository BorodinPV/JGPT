package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;

/**
 * Thread-local scratch для {@link TensorOps#tryMultiHeadAttentionWithRoPEGpuResident}: активации на device,
 * веса — из {@link TensorOps.GpuAttnResidentBuffers}.
 *
 * <p><b>Потокобезопасность:</b> не потокобезопасен; один экземпляр на поток, доступ только из этого потока и под
 * {@link #exclusiveUseLock()} вместе с вызывающими ядрами.
 *
 * <p><b>Жизненный цикл:</b> {@link #ensure} перед forward-участком → использование буферов → {@link
 * #releaseThreadLocal()} при завершении работы потока.
 *
 * <p>Device-буферы при переиспользовании не очищаются (см. {@link GpuBufferUtils}).
 */
final class GpuAttentionResidentWorkspace {

    private static final ThreadLocal<GpuAttentionResidentWorkspace> LOCAL =
            ThreadLocal.withInitial(GpuAttentionResidentWorkspace::new);

    private final Object exclusiveUse = new Object();

    private int cachedRows = -1;
    private int cachedDModel = -1;

    private GpuFloatBuffer xIn;
    private GpuFloatBuffer xNorm;
    private GpuFloatBuffer q;
    private GpuFloatBuffer k;
    private GpuFloatBuffer v;
    private GpuFloatBuffer concatFlat;
    private GpuFloatBuffer attnOut;
    /** Квадрат маски [seqLen×seqLen] на device для resident SDPA без захвата хостового массива в graph. */
    private GpuFloatBuffer maskDev;

    private int maskSeqLen = -1;

    private float[] hostOut;

    GpuAttentionResidentWorkspace() {}

    Object exclusiveUseLock() {
        return exclusiveUse;
    }

    static GpuAttentionResidentWorkspace local() {
        return LOCAL.get();
    }

    static void releaseThreadLocal() {
        GpuAttentionResidentWorkspace w = LOCAL.get();
        if (w != null) {
            synchronized (w.exclusiveUse) {
                w.closeAll();
                w.cachedRows = -1;
                w.cachedDModel = -1;
                w.hostOut = null;
            }
        }
        LOCAL.remove();
    }

    void ensure(int rows, int dModel) {
        if (rows <= 0 || dModel <= 0) {
            throw new IllegalArgumentException("rows and dModel must be positive");
        }
        long plane = GpuBufferUtils.mulExact("rows*dModel", (long) rows, (long) dModel);
        int hostLen = GpuBufferUtils.intSize("hostOut", plane);
        if (rows == cachedRows && dModel == cachedDModel) {
            hostOut = GpuBufferUtils.ensureHost(hostOut, hostLen);
            return;
        }
        xIn = GpuBufferUtils.ensure(xIn, plane);
        xNorm = GpuBufferUtils.ensure(xNorm, plane);
        q = GpuBufferUtils.ensure(q, plane);
        k = GpuBufferUtils.ensure(k, plane);
        v = GpuBufferUtils.ensure(v, plane);
        concatFlat = GpuBufferUtils.ensure(concatFlat, plane);
        attnOut = GpuBufferUtils.ensure(attnOut, plane);
        cachedRows = rows;
        cachedDModel = dModel;
        hostOut = GpuBufferUtils.ensureHost(hostOut, hostLen);
    }

    /**
     * Буфер под attention mask (плотный float [seqLen×seqLen]); перевыделяется при смене {@code seqLen}.
     */
    void ensureMask(int seqLen) {
        if (seqLen <= 0) {
            throw new IllegalArgumentException("seqLen must be positive");
        }
        if (seqLen == maskSeqLen) {
            return;
        }
        long n = GpuBufferUtils.mulExact("seqLen^2", (long) seqLen, (long) seqLen);
        maskDev = GpuBufferUtils.ensure(maskDev, n);
        maskSeqLen = seqLen;
    }

    GpuFloatBuffer getMaskDev() {
        return maskDev;
    }

    private void closeAll() {
        xIn = GpuBufferUtils.closeAndNull(xIn);
        xNorm = GpuBufferUtils.closeAndNull(xNorm);
        q = GpuBufferUtils.closeAndNull(q);
        k = GpuBufferUtils.closeAndNull(k);
        v = GpuBufferUtils.closeAndNull(v);
        concatFlat = GpuBufferUtils.closeAndNull(concatFlat);
        attnOut = GpuBufferUtils.closeAndNull(attnOut);
        maskDev = GpuBufferUtils.closeAndNull(maskDev);
        maskSeqLen = -1;
    }

    @Override
    public String toString() {
        return String.format(
                "GpuAttentionResidentWorkspace{rows=%d, dModel=%d, cached=%b}",
                cachedRows, cachedDModel, cachedRows >= 0);
    }

    float[] getHostOut() {
        return hostOut;
    }

    GpuFloatBuffer getXIn() {
        return xIn;
    }

    GpuFloatBuffer getXNorm() {
        return xNorm;
    }

    GpuFloatBuffer getQ() {
        return q;
    }

    GpuFloatBuffer getK() {
        return k;
    }

    GpuFloatBuffer getV() {
        return v;
    }

    GpuFloatBuffer getConcatFlat() {
        return concatFlat;
    }

    GpuFloatBuffer getAttnOut() {
        return attnOut;
    }
}
