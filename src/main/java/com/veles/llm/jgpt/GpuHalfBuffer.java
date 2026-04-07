package com.veles.llm.jgpt;

import java.lang.ref.Cleaner;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Буфер FP16 (half) на GPU: {@code numHalfs} элементов по 2 байта. Используется для кэша активаций при
 * {@code JGPT_ACTIVATION_CACHE_FP16=1}.
 */
public final class GpuHalfBuffer implements AutoCloseable {

    private static final Cleaner CLEANER = Cleaner.create();

    private static final class FreeAction implements Runnable {
        private final long ptr;
        private final AtomicBoolean closedNormally;

        FreeAction(long ptr, AtomicBoolean closedNormally) {
            this.ptr = ptr;
            this.closedNormally = closedNormally;
        }

        @Override
        public void run() {
            if (ptr != 0L) {
                if (!closedNormally.get()) {
                    System.err.println(
                            "[GpuHalfBuffer] ПРЕДУПРЕЖДЕНИЕ: освобождение VRAM незакрытого буфера @0x"
                                    + Long.toHexString(ptr));
                }
                nativeFree(ptr);
            }
        }
    }

    static {
        TensorOpsGPU.isGpuAvailable();
    }

    private final AtomicBoolean closedNormally = new AtomicBoolean(false);
    private final Cleaner.Cleanable cleanable;
    private long devicePtr;
    private final long numHalfs;

    private GpuHalfBuffer(long devicePtr, long numHalfs) {
        this.devicePtr = devicePtr;
        this.numHalfs = numHalfs;
        this.cleanable = CLEANER.register(this, new FreeAction(devicePtr, closedNormally));
    }

    public static GpuHalfBuffer allocate(long numHalfs) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        if (numHalfs <= 0) {
            throw new IllegalArgumentException("numHalfs must be positive");
        }
        long p = nativeAlloc(numHalfs);
        if (p == 0L) {
            double mib = (numHalfs * 2.0) / (1024.0 * 1024.0);
            throw new OutOfMemoryError(
                    String.format(
                            Locale.ROOT,
                            "cudaMalloc(Async) failed for GpuHalfBuffer: numHalfs=%d (~%.1f MiB); "
                                    + "см. stderr: cudaMemGetInfo и строку GpuHalfBuffer.nativeAlloc",
                            numHalfs,
                            mib));
        }
        return new GpuHalfBuffer(p, numHalfs);
    }

    public long numHalfs() {
        return numHalfs;
    }

    public long devicePointer() {
        return devicePtr;
    }

    public boolean isClosed() {
        return devicePtr == 0L;
    }

    @Override
    public void close() {
        if (devicePtr != 0L) {
            closedNormally.set(true);
            cleanable.clean();
            devicePtr = 0L;
        }
    }

    void checkOpen() {
        if (devicePtr == 0L) {
            throw new IllegalStateException("GpuHalfBuffer is closed");
        }
    }

    private static native long nativeAlloc(long numHalfs);

    private static native void nativeFree(long ptr);
}
