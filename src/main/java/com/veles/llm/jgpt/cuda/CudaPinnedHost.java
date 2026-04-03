package com.veles.llm.jgpt.cuda;

import java.nio.ByteBuffer;

/**
 * Page-locked host memory ({@code cudaHostAlloc}) для быстрых H2D копий. Требует загруженной
 * {@link TensorCudaLibrary}; освобождение — {@link #free(long)} (часто через {@code Cleaner} у {@link
 * com.veles.llm.jgpt.core.Tensor#allocatePinnedHost(int[])}).
 */
public final class CudaPinnedHost {

    private CudaPinnedHost() {}

    /** @return host pointer или {@code 0} при ошибке */
    public static native long allocBytes(long numBytes);

    public static native void free(long ptr);

    /** Обёртка указателя в direct {@link ByteBuffer} (не владеет памятью — держите ptr до {@link #free}). */
    public static native ByteBuffer directBuffer(long ptr, long numBytes);
}
