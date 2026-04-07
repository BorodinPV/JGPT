package com.veles.llm.jgpt;

import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Буфер float32 на GPU (opaque device pointer). Освобождать через {@link #close()} или try-with-resources.
 *
 * <p>При утечке ссылки без {@code close()} освобождение выполнит {@link Cleaner} (предупреждение в stderr только
 * если {@code close()} не вызывали).
 *
 * <p>Передачи с хоста: JNI при возможности копирует через thread-local <b>pinned</b> staging
 * ({@code cudaHostAlloc}), затем {@code cudaMemcpyAsync} на поток TensorOpsGPU — быстрее, чем прямой
 * {@code cudaMemcpy} из обычной (pageable) памяти {@code float[]}. Для долгоживущих данных предпочтительны
 * {@link #copyFromDirect(ByteBuffer, long, long)} / {@link #copyFromHost(FloatBuffer, long, long)} /
 * {@link MemorySegment} (нативный адрес, без кучи Java).
 *
 * <p><b>Потокобезопасность:</b> экземпляр не потокобезопасен; {@code copy*}, {@code clear} и {@link #close()}
 * — с одного потока или под внешней синхронизацией. После публикации ссылки чтение {@link #devicePointer()}
 * допустимо из других потоков, пока буфер не закрыт и не передаётся в нативный код конкурирующим образом.
 *
 * <p>Пример:
 *
 * <pre>{@code
 * try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(1024)) {
 *     buf.copyFrom(hostArray, 0, 1024);
 *     // TensorOpsGPU.someKernel(buf.devicePointer(), ...);
 * }
 * }</pre>
 *
 * <p>Загрузка нативной библиотеки — при инициализации {@link TensorOpsGPU}.
 */
public final class GpuFloatBuffer implements AutoCloseable {

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
                            "[GpuFloatBuffer] ПРЕДУПРЕЖДЕНИЕ: освобождение VRAM незакрытого буфера @0x"
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
    private final long numFloats;

    private GpuFloatBuffer(long devicePtr, long numFloats) {
        this.devicePtr = devicePtr;
        this.numFloats = numFloats;
        this.cleanable = CLEANER.register(this, new FreeAction(devicePtr, closedNormally));
    }

    /**
     * Выделяет {@code numFloats} float на устройстве. При сборке нативной библиотеки с CUDA 11.2+ используется
     * {@code cudaMallocAsync} на потоке TensorOpsGPU (освобождение — {@code cudaFreeAsync} на том же потоке).
     */
    public static GpuFloatBuffer allocate(long numFloats) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        if (numFloats <= 0) {
            throw new IllegalArgumentException("numFloats must be positive");
        }
        TensorOpsGPU.logVramBeforeDeviceFloatAlloc(numFloats, "GpuFloatBuffer.allocate");
        long p = nativeAlloc(numFloats);
        if (p == 0L) {
            throw new OutOfMemoryError("cudaMalloc failed for GpuFloatBuffer");
        }
        return new GpuFloatBuffer(p, numFloats);
    }

    public long numFloats() {
        return numFloats;
    }

    /** Указатель на device (для нативных вызовов). */
    public long devicePointer() {
        return devicePtr;
    }

    public boolean isClosed() {
        return devicePtr == 0L;
    }

    /** Копирует {@code len} float из {@code src[srcOff]} на GPU с начала буфера. */
    public void copyFrom(float[] src, int srcOff, int len) {
        checkOpen();
        Objects.requireNonNull(src, "src");
        if (len < 0 || srcOff < 0 || (long) srcOff + len > src.length || len > numFloats) {
            throw new IllegalArgumentException(
                    "copyFrom range invalid: src.length=" + src.length + ", srcOff=" + srcOff + ", len=" + len
                            + ", buffer.numFloats=" + numFloats);
        }
        nativeCopyHtoD(devicePtr, src, srcOff, len);
    }

    /**
     * H2D с смещением на device: пишет в {@code deviceFloatOffset … deviceFloatOffset + len)}.
     */
    public void copyFrom(float[] src, int srcOff, int len, int deviceFloatOffset) {
        checkOpen();
        Objects.requireNonNull(src, "src");
        if (len < 0
                || srcOff < 0
                || (long) srcOff + len > src.length
                || deviceFloatOffset < 0
                || (long) deviceFloatOffset + len > numFloats) {
            throw new IllegalArgumentException(
                    "copyFrom range invalid: src.length=" + src.length + ", srcOff=" + srcOff + ", len=" + len
                            + ", deviceFloatOffset=" + deviceFloatOffset + ", buffer.numFloats=" + numFloats);
        }
        nativeCopyHtoDOffset(devicePtr, deviceFloatOffset, src, srcOff, len);
    }

    /** Копирует с GPU в {@code dst[dstOff]} ({@code len} float). */
    public void copyTo(float[] dst, int dstOff, int len) {
        checkOpen();
        Objects.requireNonNull(dst, "dst");
        if (len < 0 || dstOff < 0 || (long) dstOff + len > dst.length || len > numFloats) {
            throw new IllegalArgumentException(
                    "copyTo range invalid: dst.length=" + dst.length + ", dstOff=" + dstOff + ", len=" + len
                            + ", buffer.numFloats=" + numFloats);
        }
        nativeCopyDtoH(devicePtr, dst, dstOff, len);
    }

    /**
     * D2H с смещением на device: читает с {@code deviceFloatOffset … + len)}.
     */
    public void copyTo(float[] dst, int dstOff, int len, int deviceFloatOffset) {
        checkOpen();
        Objects.requireNonNull(dst, "dst");
        if (len < 0
                || dstOff < 0
                || (long) dstOff + len > dst.length
                || deviceFloatOffset < 0
                || (long) deviceFloatOffset + len > numFloats) {
            throw new IllegalArgumentException(
                    "copyTo range invalid: dst.length=" + dst.length + ", dstOff=" + dstOff + ", len=" + len
                            + ", deviceFloatOffset=" + deviceFloatOffset + ", buffer.numFloats=" + numFloats);
        }
        nativeCopyDtoHOffset(devicePtr, deviceFloatOffset, dst, dstOff, len);
    }

    /** Обнуляет весь device-буфер. */
    public void clear() {
        checkOpen();
        nativeClear(devicePtr, numFloats);
    }

    /** Копирует {@code len} float с другого device-буфера. */
    public void copyFromDevice(GpuFloatBuffer src, int len) {
        checkOpen();
        Objects.requireNonNull(src, "src");
        src.checkOpen();
        if (len < 0 || len > numFloats || len > src.numFloats()) {
            throw new IllegalArgumentException(
                    "copyFromDevice range invalid: len=" + len + ", dst.numFloats=" + numFloats
                            + ", src.numFloats=" + src.numFloats());
        }
        nativeCopyDtoD(src.devicePtr, devicePtr, len);
    }

    /**
     * Прямой ByteBuffer (heap buffers не поддерживаются): копирует {@code numBytes} байт
     * в начало GPU-буфера (смещение {@code byteOffset} в буфере).
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
        if (numBytes > numFloats * 4L) {
            throw new IllegalArgumentException("copy exceeds GpuFloatBuffer size");
        }
        nativeCopyHtoDDirect(devicePtr, src, byteOffset, numBytes);
    }

    public void copyToDirect(ByteBuffer dst, long byteOffset, long numBytes) {
        checkOpen();
        Objects.requireNonNull(dst, "dst");
        if (!dst.isDirect()) {
            throw new IllegalArgumentException("ByteBuffer must be direct");
        }
        if (byteOffset < 0 || numBytes < 0 || byteOffset + numBytes > dst.capacity()) {
            throw new IllegalArgumentException("direct buffer range invalid");
        }
        if (numBytes > numFloats * 4L) {
            throw new IllegalArgumentException("copy exceeds GpuFloatBuffer size");
        }
        nativeCopyDtoHDirect(devicePtr, dst, byteOffset, numBytes);
    }

    /**
     * Direct {@link FloatBuffer}: копирует {@code copyFloats} float с индекса {@code srcFloatOffset}
     * в начало GPU-буфера (тот же контракт, что у {@link #copyFrom(float[], int, int)}).
     */
    public void copyFromHost(FloatBuffer src, long srcFloatOffset, long copyFloats) {
        checkOpen();
        Objects.requireNonNull(src, "src");
        if (!src.isDirect()) {
            throw new IllegalArgumentException("FloatBuffer must be direct (e.g. ByteBuffer.allocateDirect(...).asFloatBuffer())");
        }
        if (srcFloatOffset < 0 || copyFloats < 0) {
            throw new IllegalArgumentException("offsets and length must be non-negative");
        }
        if (srcFloatOffset + copyFloats > src.capacity()) {
            throw new IllegalArgumentException("FloatBuffer range invalid");
        }
        if (copyFloats > numFloats) {
            throw new IllegalArgumentException("copy exceeds GpuFloatBuffer size");
        }
        nativeCopyHtoDFloatBuffer(devicePtr, src, srcFloatOffset, copyFloats);
    }

    /**
     * Direct {@link FloatBuffer}: копирует {@code copyFloats} float с начала GPU-буфера в {@code dst}
     * с индекса {@code dstFloatOffset}.
     */
    public void copyToHost(FloatBuffer dst, long dstFloatOffset, long copyFloats) {
        checkOpen();
        Objects.requireNonNull(dst, "dst");
        if (!dst.isDirect()) {
            throw new IllegalArgumentException("FloatBuffer must be direct (e.g. ByteBuffer.allocateDirect(...).asFloatBuffer())");
        }
        if (dstFloatOffset < 0 || copyFloats < 0) {
            throw new IllegalArgumentException("offsets and length must be non-negative");
        }
        if (dstFloatOffset + copyFloats > dst.capacity()) {
            throw new IllegalArgumentException("FloatBuffer range invalid");
        }
        if (copyFloats > numFloats) {
            throw new IllegalArgumentException("copy exceeds GpuFloatBuffer size");
        }
        nativeCopyDtoHFloatBuffer(devicePtr, dst, dstFloatOffset, copyFloats);
    }

    /** Копирует весь буфер ({@link #numFloats()} float) в {@code dst} с индекса 0. */
    public void copyToHost(FloatBuffer dst) {
        copyToHost(dst, 0, numFloats);
    }

    /**
     * Копирует байты из нативного {@link MemorySegment} (например {@code Arena.ofAuto()} / off-heap) на GPU.
     * Сегмент должен быть {@link MemorySegment#isNative() native}.
     */
    public void copyFromMemorySegment(MemorySegment src, long byteOffset, long numBytes) {
        checkOpen();
        Objects.requireNonNull(src, "src");
        if (!src.isNative()) {
            throw new IllegalArgumentException("MemorySegment must be native (off-heap)");
        }
        if (byteOffset < 0 || numBytes < 0 || byteOffset + numBytes > src.byteSize()) {
            throw new IllegalArgumentException("segment range invalid");
        }
        if (numBytes > numFloats * 4L) {
            throw new IllegalArgumentException("copy exceeds GpuFloatBuffer size");
        }
        nativeCopyHtoDAddress(devicePtr, src.address() + byteOffset, numBytes);
    }

    /** Копирует с GPU в нативный {@link MemorySegment}. */
    public void copyToMemorySegment(MemorySegment dst, long byteOffset, long numBytes) {
        checkOpen();
        Objects.requireNonNull(dst, "dst");
        if (!dst.isNative()) {
            throw new IllegalArgumentException("MemorySegment must be native (off-heap)");
        }
        if (byteOffset < 0 || numBytes < 0 || byteOffset + numBytes > dst.byteSize()) {
            throw new IllegalArgumentException("segment range invalid");
        }
        if (numBytes > numFloats * 4L) {
            throw new IllegalArgumentException("copy exceeds GpuFloatBuffer size");
        }
        nativeCopyDtoHAddress(devicePtr, dst.address() + byteOffset, numBytes);
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
        return "GpuFloatBuffer{ptr=0x" + Long.toHexString(devicePtr) + ", numFloats=" + numFloats + ", closed="
                + (devicePtr == 0L) + "}";
    }

    private void checkOpen() {
        if (devicePtr == 0L) {
            throw new IllegalStateException("GpuFloatBuffer is closed");
        }
    }

    private static native long nativeAlloc(long numFloats);

    private static native void nativeFree(long ptr);

    private static native void nativeCopyHtoD(long devicePtr, float[] src, int offset, int length);

    private static native void nativeCopyHtoDOffset(
            long devicePtr, int deviceFloatOffset, float[] src, int srcOff, int len);

    private static native void nativeCopyDtoH(long devicePtr, float[] dst, int offset, int length);

    private static native void nativeCopyDtoHOffset(
            long devicePtr, int deviceFloatOffset, float[] dst, int dstOff, int len);

    private static native void nativeCopyHtoDDirect(long devicePtr, ByteBuffer buf, long byteOffset, long numBytes);

    private static native void nativeCopyDtoHDirect(long devicePtr, ByteBuffer buf, long byteOffset, long numBytes);

    private static native void nativeCopyHtoDFloatBuffer(long devicePtr, FloatBuffer buf, long floatOffset, long numFloats);

    private static native void nativeCopyDtoHFloatBuffer(long devicePtr, FloatBuffer buf, long floatOffset, long numFloats);

    private static native void nativeCopyHtoDAddress(long devicePtr, long hostAddress, long numBytes);

    private static native void nativeCopyDtoHAddress(long devicePtr, long hostAddress, long numBytes);

    private static native void nativeClear(long devicePtr, long numFloats);

    private static native void nativeCopyDtoD(long srcDevicePtr, long dstDevicePtr, int length);
}
