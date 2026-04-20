package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.core.TensorUtils;

import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Тензор, данные которого живут на GPU в одном {@link GpuFloatBuffer} (долгоживущий device-буфер).
 * Цель — шаги обучения без H2D/D2H на каждый GEMM: ядра и cuBLAS принимают {@link #devicePointer()}; на хост
 * копировать только для чекпоинта/отладки ({@link #downloadTo}, {@link #toHostTensor()}).
 *
 * <p>Ожидаемый цикл: {@link #allocate(int[])} / {@link #fromHostTensor(Tensor)} → при необходимости
 * {@link #zeroGrad()} перед backward → ядра пишут в {@link #gradDevicePointer()} → обновление весов на device
 * (например AdamW) → {@link #close()} или try-with-resources. Если не закрыть явно, отложенное освобождение
 * через {@link #drainLeaked()} после GC (в stderr — предупреждение). В пулах потоков всё равно вызывайте
 * {@code close()} по завершении задачи.
 *
 * <p>Градиент (опционально): ленивый второй буфер того же размера — {@link #zeroGrad()}, {@link #gradDevicePointer()}.
 *
 * <p><b>Не thread-safe.</b>
 */
public final class GpuTensor implements AutoCloseable {

    private static final ReferenceQueue<GpuTensor> REF_QUEUE = new ReferenceQueue<>();
    private static final Set<TensorPhantom> PHANTOMS = ConcurrentHashMap.newKeySet();

    /** Поля обновляются конструктором и {@link #zeroGrad()}; phantom читает актуальные ссылки. */
    private static final class BufferPair {
        GpuFloatBuffer data;
        GpuFloatBuffer grad;
    }

    private static final class TensorPhantom extends PhantomReference<GpuTensor> {
        final BufferPair pair;
        final AtomicBoolean closedExplicitly;
        final AtomicBoolean released;

        TensorPhantom(
                GpuTensor referent,
                ReferenceQueue<GpuTensor> q,
                BufferPair pair,
                AtomicBoolean closedExplicitly,
                AtomicBoolean released) {
            super(referent, q);
            this.pair = pair;
            this.closedExplicitly = closedExplicitly;
            this.released = released;
        }
    }

    private final int[] shape;
    private final int size;
    private final int[] strides;
    private final BufferPair buf;
    private final AtomicBoolean closedExplicitly = new AtomicBoolean(false);
    /** Один раз освободили вложенные буферы (явно или из phantom). */
    private final AtomicBoolean released = new AtomicBoolean(false);

    private GpuTensor(int[] shape, int size, int[] strides, GpuFloatBuffer data) {
        this.shape = shape.clone();
        this.size = size;
        this.strides = strides.clone();
        this.buf = new BufferPair();
        this.buf.data = Objects.requireNonNull(data, "data");
        TensorPhantom phantom = new TensorPhantom(this, REF_QUEUE, buf, closedExplicitly, released);
        PHANTOMS.add(phantom);
    }

    /**
     * Обрабатывает очередь отложенных освобождений (после GC без {@link #close()}). Безопасно вызывать часто.
     */
    public static void drainLeaked() {
        Reference<?> r;
        while ((r = REF_QUEUE.poll()) != null) {
            if (r instanceof TensorPhantom ph) {
                releasePhantom(ph);
            }
        }
    }

    private static void releasePhantom(TensorPhantom ph) {
        try {
            if (!ph.released.compareAndSet(false, true)) {
                return;
            }
            PHANTOMS.remove(ph);
            BufferPair pair = ph.pair;
            GpuFloatBuffer d = pair.data;
            GpuFloatBuffer g = pair.grad;
            if (d == null && g == null) {
                return;
            }
            if (!ph.closedExplicitly.get()) {
                System.err.println(
                        "[GpuTensor] ПРЕДУПРЕЖДЕНИЕ: освобождение VRAM незакрытого тензора (data ptr=0x"
                                + Long.toHexString(d != null ? d.devicePointer() : 0L)
                                + ")");
            }
            if (d != null) {
                d.close();
            }
            if (g != null) {
                g.close();
            }
            pair.data = null;
            pair.grad = null;
        } finally {
            ph.clear();
        }
    }

    /**
     * Выделяет нули на GPU ({@code cudaMemset}) для заданной формы.
     */
    public static GpuTensor allocate(int[] shape) {
        Objects.requireNonNull(shape, "shape");
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        int[] sh = TensorUtils.validateAndCloneShape(shape);
        int sz = TensorUtils.computeSizeOrThrow(sh);
        GpuFloatBuffer b = GpuFloatBuffer.allocate(sz);
        b.clear();
        int[] st = TensorUtils.calculateStrides(sh);
        return new GpuTensor(sh, sz, st, b);
    }

    /**
     * Однократная загрузка с хоста (например инициализация весов из {@link Tensor}).
     * Использует {@link Tensor#internalBuffer()} для heap-тензоров, чтобы избежать лишней копии.
     */
    public static GpuTensor fromHostTensor(Tensor host) {
        Objects.requireNonNull(host, "host");
        float[] buf = host.internalBuffer();
        GpuTensor g = allocate(host.getShape());
        g.uploadFrom(buf, 0, buf.length);
        return g;
    }

    public int[] getShape() {
        return shape.clone();
    }

    /**
     * Форма без копии — только для горячих путей; не изменять возвращённый массив.
     */
    public int[] shapeInternal() {
        return shape;
    }

    public int[] strides() {
        return strides.clone();
    }

    /**
     * Шаги без копии — только для горячих путей; не изменять возвращённый массив.
     */
    public int[] stridesInternal() {
        return strides;
    }

    public int size() {
        return size;
    }

    public int rank() {
        return shape.length;
    }

    /** Указатель на данные на device (row-major). */
    public long devicePointer() {
        checkOpen();
        return buf.data.devicePointer();
    }

    /** Тот же буфер, что {@link #devicePointer()}, для явной передачи в нативные device-операции. */
    public GpuFloatBuffer dataBuffer() {
        checkOpen();
        return buf.data;
    }

    /**
     * Лениво выделяет буфер градиента и обнуляет его. Вызывать перед backward, если ∂ накапливаются на GPU.
     */
    public void zeroGrad() {
        checkOpen();
        if (buf.grad == null) {
            buf.grad = GpuFloatBuffer.allocate(size);
        }
        buf.grad.clear();
    }

    public boolean hasGradBuffer() {
        return buf.grad != null;
    }

    public long gradDevicePointer() {
        checkOpen();
        if (buf.grad == null) {
            throw gradNotAllocated();
        }
        return buf.grad.devicePointer();
    }

    public GpuFloatBuffer gradBuffer() {
        checkOpen();
        if (buf.grad == null) {
            throw gradNotAllocated();
        }
        return buf.grad;
    }

    private IllegalStateException gradNotAllocated() {
        return new IllegalStateException(
                "grad not allocated for GpuTensor[shape="
                        + Arrays.toString(shape)
                        + ", size="
                        + size
                        + "]. Call zeroGrad() first.");
    }

    public void uploadFrom(float[] src) {
        Objects.requireNonNull(src, "src");
        checkOpen();
        if (src.length != size) {
            throw new IllegalArgumentException("src.length must equal tensor size: " + src.length + " != " + size);
        }
        buf.data.copyFrom(src, 0, size);
    }

    public void uploadFrom(float[] src, int srcOff, int len) {
        checkOpen();
        if (len < 0 || srcOff < 0 || (long) srcOff + len > src.length || len > size) {
            throw new IllegalArgumentException("upload range invalid");
        }
        buf.data.copyFrom(src, srcOff, len);
    }

    public void downloadTo(float[] dst, int dstOff, int len) {
        checkOpen();
        if (len < 0 || dstOff < 0 || (long) dstOff + len > dst.length || len > size) {
            throw new IllegalArgumentException("download range invalid");
        }
        buf.data.copyTo(dst, dstOff, len);
    }

    /**
     * Полная копия на хост: новый {@link Tensor} со своим {@code float[]} (изменения хост-массива не затрагивают
     * {@code wrap} от промежуточного буфера).
     */
    public Tensor toHostTensor() {
        checkOpen();
        float[] f = new float[size];
        buf.data.copyTo(f, 0, size);
        return Tensor.fromArray(f, getShape());
    }

    public boolean isClosed() {
        return buf.data == null;
    }

    @Override
    public void close() {
        if (buf.data == null && buf.grad == null) {
            return;
        }
        closedExplicitly.set(true);
        if (released.compareAndSet(false, true)) {
            GpuFloatBuffer d = buf.data;
            GpuFloatBuffer g = buf.grad;
            if (d != null) {
                d.close();
            }
            if (g != null) {
                g.close();
            }
            buf.data = null;
            buf.grad = null;
        }
    }

    @Override
    public String toString() {
        boolean closed = buf.data == null;
        long dptr = closed ? 0L : buf.data.devicePointer();
        String gradState = buf.grad != null ? "allocated" : "null";
        return "GpuTensor{shape="
                + Arrays.toString(shape)
                + ", size="
                + size
                + ", dataPtr=0x"
                + Long.toHexString(dptr)
                + ", grad="
                + gradState
                + ", closed="
                + closed
                + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof GpuTensor that)) {
            return false;
        }
        if (isClosed() || that.isClosed()) {
            return false;
        }
        return size == that.size
                && Arrays.equals(shape, that.shape)
                && buf.data.devicePointer() == that.buf.data.devicePointer();
    }

    @Override
    public int hashCode() {
        if (buf.data == null) {
            return System.identityHashCode(this);
        }
        int h = Objects.hash(size, buf.data.devicePointer());
        return 31 * h + Arrays.hashCode(shape);
    }

    private void checkOpen() {
        if (buf.data == null) {
            throw new IllegalStateException("GpuTensor is closed");
        }
    }
}
