package com.veles.llm.jgpt.ops;

/**
 * Освобождение thread-local GPU workspace в текущем потоке. Вызывать при завершении обучения/инференса на потоке
 * или перед возвратом потока в пул, чтобы не держать VRAM в {@link ThreadLocal}.
 *
 * <p>Каждый {@code release*} снимает свой {@link ThreadLocal}; {@link #releaseAllGpuWorkspacesThreadLocal()}
 * закрывает все workspace этого пакета в одном потоке.
 */
public final class GpuWorkspaceCleanup {

    private GpuWorkspaceCleanup() {}

    public static void releaseAttentionBackwardThreadLocal() {
        GpuAttentionBackwardWorkspace.releaseThreadLocal();
    }

    public static void releaseBlockWorkspaceThreadLocal() {
        GpuBlockWorkspace.releaseThreadLocal();
    }

    public static void releaseForwardBlockWorkspaceThreadLocal() {
        GpuForwardBlockWorkspace.releaseThreadLocal();
    }

    /** Освобождает все thread-local GPU workspace в текущем потоке. */
    public static void releaseAllGpuWorkspacesThreadLocal() {
        GpuAttentionBackwardWorkspace.releaseThreadLocal();
        GpuAttentionResidentWorkspace.releaseThreadLocal();
        GpuBlockWorkspace.releaseThreadLocal();
        GpuForwardBlockWorkspace.releaseThreadLocal();
        GpuSampledLmHeadBackwardWorkspace.releaseThreadLocal();
        GpuTransformerOuterBackwardWorkspace.releaseThreadLocal();
    }
}
