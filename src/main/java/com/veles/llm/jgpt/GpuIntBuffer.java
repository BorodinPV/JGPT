package com.veles.llm.jgpt;

import java.lang.ref.Cleaner;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Буфер int32 на GPU (для CE targets и т.п.). Освобождать через {@link #close()} или try-with-resources.
 *
 * <p>При утечке ссылки без {@code close()} освобождение выполнит {@link Cleaner} (с предупреждением в stderr).
 * Класс не рассчитан на параллельные вызовы с одного объекта без внешней синхронизации.
 *
 * <p>Загрузка нативной библиотеки — при инициализации {@link TensorOpsGPU}.
 */
public final class GpuIntBuffer implements AutoCloseable {

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
                            "[GpuIntBuffer] ПРЕДУПРЕЖДЕНИЕ: освобождение VRAM незакрытого буфера @0x"
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
    private final long numInts;

    private GpuIntBuffer(long devicePtr, long numInts) {
        this.devicePtr = devicePtr;
        this.numInts = numInts;
        this.cleanable = CLEANER.register(this, new FreeAction(devicePtr, closedNormally));
    }

    public static GpuIntBuffer allocate(long numInts) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        if (numInts <= 0) {
            throw new IllegalArgumentException("numInts must be positive");
        }
        long p = nativeAlloc(numInts);
        if (p == 0L) {
            throw new OutOfMemoryError("cudaMalloc failed for GpuIntBuffer");
        }
        return new GpuIntBuffer(p, numInts);
    }

    public long numInts() {
        return numInts;
    }

    public long devicePointer() {
        return devicePtr;
    }

    public boolean isClosed() {
        return devicePtr == 0L;
    }

    public void copyFrom(int[] src, int srcOff, int len) {
        checkOpen();
        Objects.requireNonNull(src, "src");
        if (len < 0 || srcOff < 0 || (long) srcOff + len > src.length || len > numInts) {
            throw new IllegalArgumentException(
                    "copyFrom range invalid: src.length=" + src.length + ", srcOff=" + srcOff + ", len=" + len
                            + ", buffer.numInts=" + numInts);
        }
        nativeCopyHtoD(devicePtr, src, srcOff, len);
    }

    /** Копирование device → host (первые {@code len} int буфера). */
    public void copyTo(int[] dst, int dstOff, int len) {
        checkOpen();
        Objects.requireNonNull(dst, "dst");
        if (len < 0 || dstOff < 0 || (long) dstOff + len > dst.length || len > numInts) {
            throw new IllegalArgumentException(
                    "copyTo range invalid: dst.length=" + dst.length + ", dstOff=" + dstOff + ", len=" + len
                            + ", buffer.numInts=" + numInts);
        }
        nativeCopyDtoH(devicePtr, dst, dstOff, len);
    }

    /**
     * H2D из прямого {@link ByteBuffer}; {@code numBytes} обычно кратен 4 (размер int).
     * Смещения — в байтах от начала прямого буфера.
     */
    public void copyFromDirect(ByteBuffer src, long byteOffset, long numBytes) {
        checkOpen();
        Objects.requireNonNull(src, "src");
        if (!src.isDirect()) {
            throw new IllegalArgumentException("ByteBuffer must be direct");
        }
        if (byteOffset < 0 || numBytes < 0 || byteOffset + numBytes > src.capacity()) {
            throw new IllegalArgumentException("direct buffer range invalid");
        }
        if ((numBytes & 3L) != 0L) {
            throw new IllegalArgumentException("numBytes must be a multiple of 4 (sizeof int)");
        }
        if (numBytes > numInts * 4L) {
            throw new IllegalArgumentException("copy exceeds GpuIntBuffer size");
        }
        nativeCopyHtoDDirect(devicePtr, src, byteOffset, numBytes);
    }

    /** D2H в прямой {@link ByteBuffer}. */
    public void copyToDirect(ByteBuffer dst, long byteOffset, long numBytes) {
        checkOpen();
        Objects.requireNonNull(dst, "dst");
        if (!dst.isDirect()) {
            throw new IllegalArgumentException("ByteBuffer must be direct");
        }
        if (byteOffset < 0 || numBytes < 0 || byteOffset + numBytes > dst.capacity()) {
            throw new IllegalArgumentException("direct buffer range invalid");
        }
        if ((numBytes & 3L) != 0L) {
            throw new IllegalArgumentException("numBytes must be a multiple of 4 (sizeof int)");
        }
        if (numBytes > numInts * 4L) {
            throw new IllegalArgumentException("copy exceeds GpuIntBuffer size");
        }
        nativeCopyDtoHDirect(devicePtr, dst, byteOffset, numBytes);
    }

    public void clear() {
        checkOpen();
        nativeClear(devicePtr, numInts);
    }

    @Override
    public void close() {
        if (devicePtr != 0L) {
            closedNormally.set(true);
            cleanable.clean();
            devicePtr = 0L;
        }
    }

    @Override
    public String toString() {
        return "GpuIntBuffer{ptr=0x" + Long.toHexString(devicePtr) + ", numInts=" + numInts + ", closed="
                + (devicePtr == 0L) + "}";
    }

    private void checkOpen() {
        if (devicePtr == 0L) {
            throw new IllegalStateException("GpuIntBuffer is closed");
        }
    }

    private static native long nativeAlloc(long numInts);

    private static native void nativeFree(long ptr);

    private static native void nativeCopyHtoD(long devicePtr, int[] src, int offset, int length);

    private static native void nativeCopyDtoH(long devicePtr, int[] dst, int offset, int length);

    private static native void nativeCopyHtoDDirect(long devicePtr, ByteBuffer directBuf, long byteOffset, long numBytes);

    private static native void nativeCopyDtoHDirect(long devicePtr, ByteBuffer directBuf, long byteOffset, long numBytes);

    private static native void nativeClear(long devicePtr, long numInts);
}
