package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;

/**
 * Thread-local scratch для внешнего контура {@link
 * TransformerBackward#transformerBlockBackwardGpuDevice}: пинг-понг буферы между FFN-backward, attention-backward и
 * первым RMSNorm, без alloc/free на каждый слой.
 *
 * <p><b>Память:</b> при смене {@code (rows, dModel)} (где {@code rows = batch * seqLen}) буферы перевыделяются через
 * {@link GpuBufferUtils#ensure}; при уменьшении shape буфер может оставаться большим (grow-only), логическая длина —
 * {@code rows * dModel}.
 *
 * <p><b>Потокобезопасность:</b> один экземпляр на поток; не передавать между потоками. Использовать только из потока,
 * который выполняет GPU backward decoder (как остальные workspace в этом пакете).
 *
 * <p><b>Контент:</b> после {@link #ensure} содержимое не гарантировано нулём; вызывающий код обязан перезаписать или
 * {@link GpuFloatBuffer#clear()} там, где ядра используют {@code +=} (см. {@link GpuBufferUtils}).
 *
 * <p><b>Жизненный цикл:</b> {@link GpuWorkspaceCleanup#releaseAllGpuWorkspacesThreadLocal()} или
 * {@link #releaseThreadLocal()} после {@link com.veles.llm.jgpt.TensorOpsGPU#synchronizeStream()} при остановке
 * обучения на этом потоке.
 */
final class GpuTransformerOuterBackwardWorkspace {

    private static final ThreadLocal<GpuTransformerOuterBackwardWorkspace> LOCAL =
            ThreadLocal.withInitial(GpuTransformerOuterBackwardWorkspace::new);

    private int cachedRows = -1;
    private int cachedDModel;

    private GpuFloatBuffer dGradXRes1;
    private GpuFloatBuffer dGradNorm1Path;
    private GpuFloatBuffer xInScratch;
    private GpuFloatBuffer dNorm1;
    /** Только если нет resident gamma в attention; иначе не используется. */
    private GpuFloatBuffer norm1GammaStaging;

    GpuTransformerOuterBackwardWorkspace() {}

    static GpuTransformerOuterBackwardWorkspace local() {
        return LOCAL.get();
    }

    static void releaseThreadLocal() {
        GpuTransformerOuterBackwardWorkspace w = LOCAL.get();
        if (w != null) {
            w.closeAllGpuBuffers();
            w.cachedRows = -1;
        }
        LOCAL.remove();
    }

    /**
     * Готовит буферы размера {@code rows * dModel} и {@code dNorm1} длины {@code dModel}.
     */
    void ensure(int rows, int dModel) {
        if (rows <= 0 || dModel <= 0) {
            throw new IllegalArgumentException("rows and dModel must be positive: rows=" + rows + ", dModel=" + dModel);
        }
        if (rows == cachedRows && dModel == cachedDModel) {
            return;
        }

        closeAllGpuBuffers();

        long flat = GpuBufferUtils.mulExact("rows*dModel", (long) rows, (long) dModel);

        dGradXRes1 = GpuBufferUtils.ensure(dGradXRes1, flat);
        dGradNorm1Path = GpuBufferUtils.ensure(dGradNorm1Path, flat);
        xInScratch = GpuBufferUtils.ensure(xInScratch, flat);
        dNorm1 = GpuBufferUtils.ensure(dNorm1, dModel);

        cachedRows = rows;
        cachedDModel = dModel;
    }

    /**
     * Стадия для копирования gamma первого RMSNorm с хоста, когда нет {@link TensorOps.GpuAttnResidentBuffers#normGamma()}.
     */
    GpuFloatBuffer norm1GammaStaging() {
        int dm = cachedDModel;
        if (cachedRows < 0 || dm <= 0) {
            throw new IllegalStateException("norm1GammaStaging: call ensure(rows, dModel) first");
        }
        norm1GammaStaging = GpuBufferUtils.ensure(norm1GammaStaging, dm);
        return norm1GammaStaging;
    }

    GpuFloatBuffer getDGradXRes1() {
        return dGradXRes1;
    }

    GpuFloatBuffer getDGradNorm1Path() {
        return dGradNorm1Path;
    }

    GpuFloatBuffer getXInScratch() {
        return xInScratch;
    }

    GpuFloatBuffer getDNorm1() {
        return dNorm1;
    }

    private void closeAllGpuBuffers() {
        dGradXRes1 = GpuBufferUtils.closeAndNull(dGradXRes1);
        dGradNorm1Path = GpuBufferUtils.closeAndNull(dGradNorm1Path);
        xInScratch = GpuBufferUtils.closeAndNull(xInScratch);
        dNorm1 = GpuBufferUtils.closeAndNull(dNorm1);
        norm1GammaStaging = GpuBufferUtils.closeAndNull(norm1GammaStaging);
    }
}
