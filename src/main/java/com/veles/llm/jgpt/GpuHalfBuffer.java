package com.veles.llm.jgpt;

import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Буфер FP16 (half) на GPU: {@code numHalfs} элементов по 2 байта. Используется для кэша активаций при
 * {@code JGPT_ACTIVATION_CACHE_FP16=1}.
 *
 * <p>Отложенное освобождение без {@link #close()} — см. {@link #drainLeaked()} (после GC). Предупреждение в stderr,
 * если {@link #close()} не вызывали.
 */
public final class GpuHalfBuffer implements AutoCloseable {

    private static final ReferenceQueue<GpuHalfBuffer> REF_QUEUE = new ReferenceQueue<>();

    private static final class HalfBufferPhantom extends PhantomReference<GpuHalfBuffer> {
        final long ptr;
        final AtomicBoolean nativeFreed;
        final AtomicBoolean closedExplicitly;

        HalfBufferPhantom(
                GpuHalfBuffer referent,
                ReferenceQueue<GpuHalfBuffer> q,
                long ptr,
                AtomicBoolean nativeFreed,
                AtomicBoolean closedExplicitly) {
            super(referent, q);
            this.ptr = ptr;
            this.nativeFreed = nativeFreed;
            this.closedExplicitly = closedExplicitly;
        }
    }

    static {
        TensorOpsGPU.isGpuAvailable();
    }

    private final AtomicBoolean closedExplicitly = new AtomicBoolean(false);
    private final AtomicBoolean nativeFreed = new AtomicBoolean(false);
    private long devicePtr;
    private final long numHalfs;

    private GpuHalfBuffer(long devicePtr, long numHalfs) {
        this.devicePtr = devicePtr;
        this.numHalfs = numHalfs;
        new HalfBufferPhantom(this, REF_QUEUE, devicePtr, nativeFreed, closedExplicitly);
    }

    /**
     * Обрабатывает очередь отложенных освобождений (после GC без {@link #close()}). Безопасно вызывать часто.
     */
    public static void drainLeaked() {
        Reference<?> r;
        while ((r = REF_QUEUE.poll()) != null) {
            if (r instanceof HalfBufferPhantom ph) {
                releasePhantom(ph);
            }
        }
    }

    private static void releasePhantom(HalfBufferPhantom ph) {
        try {
            if (!ph.nativeFreed.compareAndSet(false, true)) {
                return;
            }
            if (!ph.closedExplicitly.get()) {
                System.err.println(
                        "[GpuHalfBuffer] ПРЕДУПРЕЖДЕНИЕ: освобождение VRAM незакрытого буфера @0x"
                                + Long.toHexString(ph.ptr));
            }
            if (TensorOpsGPU.isGpuAvailable() && ph.ptr != 0L) {
                nativeFree(ph.ptr);
            }
        } finally {
            ph.clear();
        }
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
        if (devicePtr == 0L) {
            return;
        }
        closedExplicitly.set(true);
        if (nativeFreed.compareAndSet(false, true)) {
            nativeFree(devicePtr);
        }
        devicePtr = 0L;
    }

    void checkOpen() {
        if (devicePtr == 0L) {
            throw new IllegalStateException("GpuHalfBuffer is closed");
        }
    }

    private static native long nativeAlloc(long numHalfs);

    private static native void nativeFree(long ptr);
}
