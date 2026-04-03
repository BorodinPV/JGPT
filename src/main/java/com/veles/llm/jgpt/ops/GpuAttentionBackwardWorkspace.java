package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;

/**
 * Thread-local scratch для fused attention backward на GPU.
 *
 * <p><b>Память:</b> при смене {@code (batch, numHeads, seqLen, dModel)} все device-буферы закрываются
 * одним проходом, затем выделяются заново — меньше чередования alloc/free, чем по одному буферу.
 *
 * <p><b>Потоки:</b> один экземпляр на поток. {@link #ensure} и последующие копирования/ядра — только под
 * внешним {@link #exclusiveUseLock()} (см. {@link TransformerBackward#tryFusedAttentionBackwardGpu});
 * вложенный synchronized на том же мониторе не используется.
 *
 * <p><b>Жизненный цикл:</b> {@link #ensure} перед backward → использование полей-геттеров → {@link
 * #releaseThreadLocal()} при остановке потока (см. {@link GpuWorkspaceCleanup#releaseAttentionBackwardThreadLocal()}).
 *
 * <p>Буферы после {@link #ensure} перезаполняются ядрами; при переиспользовании того же workspace с теми же
 * размерами вызывающий код не должен полагаться на предыдущее содержимое (см. {@link GpuBufferUtils}).
 *
 * <p><b>Остановка:</b> {@link GpuWorkspaceCleanup#releaseAttentionBackwardThreadLocal()} или
 * {@link com.veles.llm.jgpt.cuda.GpuPendingGradients#cleanupThreadLocal()}.
 */
final class GpuAttentionBackwardWorkspace {

    private static final ThreadLocal<GpuAttentionBackwardWorkspace> LOCAL =
            ThreadLocal.withInitial(GpuAttentionBackwardWorkspace::new);

    private final Object exclusiveUse = new Object();

    private int cachedBatch = -1;
    private int cachedNumHeads;
    private int cachedSeqLen;
    private int cachedDModel;

    private GpuFloatBuffer xFlat;
    private GpuFloatBuffer gradOutFlat;
    private GpuFloatBuffer wq;
    private GpuFloatBuffer wk;
    private GpuFloatBuffer wv;
    private GpuFloatBuffer wo;
    private GpuFloatBuffer qHeads;
    private GpuFloatBuffer kHeads;
    private GpuFloatBuffer vHeads;
    private GpuFloatBuffer probs;
    private GpuFloatBuffer concat;
    private GpuFloatBuffer gradConcat;
    private GpuFloatBuffer dHeads;
    private GpuFloatBuffer gradQh;
    private GpuFloatBuffer gradKh;
    private GpuFloatBuffer gradVh;
    private GpuFloatBuffer dQ;
    private GpuFloatBuffer dK;
    private GpuFloatBuffer dV;
    private GpuFloatBuffer gradX;
    private GpuFloatBuffer gradTmp;
    private GpuFloatBuffer gradWq;
    private GpuFloatBuffer gradWk;
    private GpuFloatBuffer gradWv;
    private GpuFloatBuffer gradWo;

    GpuAttentionBackwardWorkspace() {}

    /** Монитор для эксклюзивного использования workspace вместе с GPU-ядрами на этом потоке. */
    Object exclusiveUseLock() {
        return exclusiveUse;
    }

    static GpuAttentionBackwardWorkspace local() {
        return LOCAL.get();
    }

    /**
     * Освобождает VRAM буферов текущего потока и удаляет запись из {@link ThreadLocal}. Вызывать при
     * завершении обучения или перед уходом потока в пул.
     */
    static void releaseThreadLocal() {
        GpuAttentionBackwardWorkspace w = LOCAL.get();
        if (w != null) {
            synchronized (w.exclusiveUse) {
                w.closeAllGpuBuffers();
                w.cachedBatch = -1;
            }
        }
        LOCAL.remove();
    }

    void ensure(int batch, int numHeads, int seqLen, int dModel) {
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive, got " + numHeads);
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException(
                    "dModel must be divisible by numHeads: dModel=" + dModel + ", numHeads=" + numHeads);
        }
        if (batch == cachedBatch
                && numHeads == cachedNumHeads
                && seqLen == cachedSeqLen
                && dModel == cachedDModel) {
            return;
        }

        closeAllGpuBuffers();

        int dHead = dModel / numHeads;
        long rows = GpuBufferUtils.mulExact("batch*seqLen", (long) batch, (long) seqLen);
        long rowPlane = GpuBufferUtils.mulExact("rows*dModel", rows, (long) dModel);
        long batchHeads = GpuBufferUtils.mulExact("batch*numHeads", (long) batch, numHeads);
        long bhSeq = GpuBufferUtils.mulExact("batchHeads*seqLen", batchHeads, (long) seqLen);
        long headRows = GpuBufferUtils.mulExact("headFlat", bhSeq, (long) dHead);
        long probsSize = GpuBufferUtils.mulExact("probs", bhSeq, (long) seqLen);
        long weights = GpuBufferUtils.mulExact("dModel*dModel", (long) dModel, dModel);

        xFlat = GpuBufferUtils.ensure(xFlat, rowPlane);
        gradOutFlat = GpuBufferUtils.ensure(gradOutFlat, rowPlane);
        wq = GpuBufferUtils.ensure(wq, weights);
        wk = GpuBufferUtils.ensure(wk, weights);
        wv = GpuBufferUtils.ensure(wv, weights);
        wo = GpuBufferUtils.ensure(wo, weights);
        qHeads = GpuBufferUtils.ensure(qHeads, headRows);
        kHeads = GpuBufferUtils.ensure(kHeads, headRows);
        vHeads = GpuBufferUtils.ensure(vHeads, headRows);
        probs = GpuBufferUtils.ensure(probs, probsSize);
        concat = GpuBufferUtils.ensure(concat, rowPlane);
        gradConcat = GpuBufferUtils.ensure(gradConcat, rowPlane);
        dHeads = GpuBufferUtils.ensure(dHeads, headRows);
        gradQh = GpuBufferUtils.ensure(gradQh, headRows);
        gradKh = GpuBufferUtils.ensure(gradKh, headRows);
        gradVh = GpuBufferUtils.ensure(gradVh, headRows);
        dQ = GpuBufferUtils.ensure(dQ, rowPlane);
        dK = GpuBufferUtils.ensure(dK, rowPlane);
        dV = GpuBufferUtils.ensure(dV, rowPlane);
        gradX = GpuBufferUtils.ensure(gradX, rowPlane);
        gradTmp = GpuBufferUtils.ensure(gradTmp, rowPlane);
        gradWq = GpuBufferUtils.ensure(gradWq, weights);
        gradWk = GpuBufferUtils.ensure(gradWk, weights);
        gradWv = GpuBufferUtils.ensure(gradWv, weights);
        gradWo = GpuBufferUtils.ensure(gradWo, weights);

        cachedBatch = batch;
        cachedNumHeads = numHeads;
        cachedSeqLen = seqLen;
        cachedDModel = dModel;
    }

    private void closeAllGpuBuffers() {
        GpuFloatBuffer[] buffers = {
            xFlat,
            gradOutFlat,
            wq,
            wk,
            wv,
            wo,
            qHeads,
            kHeads,
            vHeads,
            probs,
            concat,
            gradConcat,
            dHeads,
            gradQh,
            gradKh,
            gradVh,
            dQ,
            dK,
            dV,
            gradX,
            gradTmp,
            gradWq,
            gradWk,
            gradWv,
            gradWo
        };
        for (int i = 0; i < buffers.length; i++) {
            buffers[i] = GpuBufferUtils.closeAndNull(buffers[i]);
        }
        xFlat = buffers[0];
        gradOutFlat = buffers[1];
        wq = buffers[2];
        wk = buffers[3];
        wv = buffers[4];
        wo = buffers[5];
        qHeads = buffers[6];
        kHeads = buffers[7];
        vHeads = buffers[8];
        probs = buffers[9];
        concat = buffers[10];
        gradConcat = buffers[11];
        dHeads = buffers[12];
        gradQh = buffers[13];
        gradKh = buffers[14];
        gradVh = buffers[15];
        dQ = buffers[16];
        dK = buffers[17];
        dV = buffers[18];
        gradX = buffers[19];
        gradTmp = buffers[20];
        gradWq = buffers[21];
        gradWk = buffers[22];
        gradWv = buffers[23];
        gradWo = buffers[24];
    }

    @Override
    public String toString() {
        return String.format(
                "GpuAttentionBackwardWorkspace{batch=%d, heads=%d, seq=%d, dModel=%d, allocated=%b}",
                cachedBatch, cachedNumHeads, cachedSeqLen, cachedDModel, cachedBatch >= 0);
    }

    GpuFloatBuffer getXFlat() {
        return xFlat;
    }

    GpuFloatBuffer getGradOutFlat() {
        return gradOutFlat;
    }

    GpuFloatBuffer getWq() {
        return wq;
    }

    GpuFloatBuffer getWk() {
        return wk;
    }

    GpuFloatBuffer getWv() {
        return wv;
    }

    GpuFloatBuffer getWo() {
        return wo;
    }

    GpuFloatBuffer getQHeads() {
        return qHeads;
    }

    GpuFloatBuffer getKHeads() {
        return kHeads;
    }

    GpuFloatBuffer getVHeads() {
        return vHeads;
    }

    GpuFloatBuffer getProbs() {
        return probs;
    }

    GpuFloatBuffer getConcat() {
        return concat;
    }

    GpuFloatBuffer getGradConcat() {
        return gradConcat;
    }

    GpuFloatBuffer getDHeads() {
        return dHeads;
    }

    GpuFloatBuffer getGradQh() {
        return gradQh;
    }

    GpuFloatBuffer getGradKh() {
        return gradKh;
    }

    GpuFloatBuffer getGradVh() {
        return gradVh;
    }

    GpuFloatBuffer getDQ() {
        return dQ;
    }

    GpuFloatBuffer getDK() {
        return dK;
    }

    GpuFloatBuffer getDV() {
        return dV;
    }

    GpuFloatBuffer getGradX() {
        return gradX;
    }

    GpuFloatBuffer getGradTmp() {
        return gradTmp;
    }

    GpuFloatBuffer getGradWq() {
        return gradWq;
    }

    GpuFloatBuffer getGradWk() {
        return gradWk;
    }

    GpuFloatBuffer getGradWv() {
        return gradWv;
    }

    GpuFloatBuffer getGradWo() {
        return gradWo;
    }
}
