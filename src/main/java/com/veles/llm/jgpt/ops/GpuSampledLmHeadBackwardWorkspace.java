package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;

/**
 * Thread-local scratch для {@link TransformerBackward#backwardSampledLmHeadDevice}: три буфера
 * ({@code dHidden}, {@code dGradBeforeNorm}, {@code dGGamma}) без {@code cudaMalloc} на каждый шаг.
 *
 * <p><b>Память:</b> grow-only через {@link GpuBufferUtils#ensure}; при первом вызове аллоцируются,
 * при смене {@code (rows, dModel)} перевыделяются.
 *
 * <p><b>Потокобезопасность:</b> один экземпляр на поток; не передавать между потоками.
 *
 * <p><b>Жизненный цикл:</b> {@link GpuWorkspaceCleanup#releaseAllGpuWorkspacesThreadLocal()} или
 * {@link #releaseThreadLocal()} после {@link com.veles.llm.jgpt.TensorOpsGPU#synchronizeStream()}.
 */
final class GpuSampledLmHeadBackwardWorkspace {

    private static final ThreadLocal<GpuSampledLmHeadBackwardWorkspace> LOCAL =
            ThreadLocal.withInitial(GpuSampledLmHeadBackwardWorkspace::new);

    private int cachedRows = -1;
    private int cachedDModel;

    /** ∂L/∂normedHidden из LM head: [rows * dModel]. */
    private GpuFloatBuffer dHidden;
    /** ∂L/∂x перед финальным RMSNorm: [rows * dModel]. */
    private GpuFloatBuffer dGradBeforeNorm;
    /** ∂L/∂gamma финального RMSNorm: [dModel]. */
    private GpuFloatBuffer dGGamma;

    GpuSampledLmHeadBackwardWorkspace() {}

    static GpuSampledLmHeadBackwardWorkspace local() {
        return LOCAL.get();
    }

    static void releaseThreadLocal() {
        GpuSampledLmHeadBackwardWorkspace w = LOCAL.get();
        if (w != null) {
            w.closeAllGpuBuffers();
            w.cachedRows = -1;
        }
        LOCAL.remove();
    }

    /**
     * Убеждается, что буферы готовы для {@code rows} строк и {@code dModel} скрытого измерения.
     * Grow-only: не пересоздаёт при уменьшении.
     */
    void ensure(int rows, int dModel) {
        if (rows <= 0 || dModel <= 0) {
            throw new IllegalArgumentException(
                    "rows and dModel must be positive: rows=" + rows + ", dModel=" + dModel);
        }
        long flat = GpuBufferUtils.mulExact("rows*dModel", (long) rows, (long) dModel);

        dHidden = GpuBufferUtils.ensure(dHidden, flat);
        dGradBeforeNorm = GpuBufferUtils.ensure(dGradBeforeNorm, flat);
        dGGamma = GpuBufferUtils.ensure(dGGamma, dModel);

        cachedRows = rows;
        cachedDModel = dModel;
    }

    GpuFloatBuffer getDHidden() {
        return dHidden;
    }

    GpuFloatBuffer getDGradBeforeNorm() {
        return dGradBeforeNorm;
    }

    GpuFloatBuffer getDGGamma() {
        return dGGamma;
    }

    private void closeAllGpuBuffers() {
        dHidden = GpuBufferUtils.closeAndNull(dHidden);
        dGradBeforeNorm = GpuBufferUtils.closeAndNull(dGradBeforeNorm);
        dGGamma = GpuBufferUtils.closeAndNull(dGGamma);
        cachedRows = -1;
    }
}
