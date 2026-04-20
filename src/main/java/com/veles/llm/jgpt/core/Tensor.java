package com.veles.llm.jgpt.core;

import com.veles.llm.jgpt.cuda.CudaPinnedHost;
import com.veles.llm.jgpt.cuda.TensorCudaLibrary;
import java.lang.ref.Cleaner;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable-shape mutable-data tensor with lazy gradient allocation.
 * <p>
 * <b>Thread-safety:</b> не потокобезопасен. Запись ({@code set*}, {@code fillData}, прямая запись в буферы)
 * требует внешней синхронизации. Чтение из нескольких потоков допустимо только при отсутствии гонок по записи.
 * Ленивое выделение {@link #grad} и сброс {@link #heapMirror} при модификации direct-данных не атомарны —
 * перед многопоточным использованием вызывайте {@link #zeroGrad()} и зафиксируйте контракт доступа.
 * <p>
 * <b>Память / JNI:</b> для heap-тензора {@link #internalBuffer()} — ссылка на {@code float[]} без копии.
 * Для {@linkplain Tensor.StorageMode#DIRECT direct}-хранилища {@link #internalBuffer()} синхронизирует снимок в
 * heap-массив (полная копия O(n)); это удобно для CPU-кода, но дорого при частых вызовах. Для настоящего
 * zero-copy с off-heap используйте {@link #directBufferForJNI()} или {@link #directByteBuffer()}.
 * <p>
 * {@link #getDataCopy()} всегда возвращает новую копию. {@link #gradBuffer()} — прямой heap-массив ∂L/∂data
 * после {@link #zeroGrad()}.
 * <p>
 * {@link #equals(Object)} и {@link #hashCode()} — O(n) по числу элементов; для direct-тензоров не поддерживают
 * постоянный heap-зеркальный кэш, но сравнение/хэш идут потоково по буферу (без лишнего удержания
 * {@code float[size]} между вызовами, если зеркало не используется).
 *
 * @implNote Shape and strides are immutable after construction.
 *           Data and gradient buffers are mutable.
 */
public final class Tensor {

    private static final Cleaner PINNED_CUDA_CLEANER = Cleaner.create();

    /** Режим хранения элементов данных. */
    public enum StorageMode {
        HEAP,
        DIRECT
    }

    private final float[] data;
    /** Direct storage: {@code data == null}, градиенты — только в {@link #heapMirror} / heap {@code grad}. */
    private final ByteBuffer directBytes;
    private final FloatBuffer directFloats;
    /** Копия direct-буфера для {@link #internalBuffer()} и CPU-путей. */
    private float[] heapMirror;

    /** {@code true} если данные в page-locked памяти ({@link #allocatePinnedHost(int[])}). */
    private final boolean pinnedCudaHost;

    /** ∂L/∂data; lazy allocation via {@link #zeroGrad()}. */
    private float[] grad;
    private final int[] shape;
    private final int[] strides;
    private final int size; // cached product of shape

    /**
     * Creates a new tensor with the given shape, initialized to zeros.
     * @param shape dimensions (must be positive, non-empty)
     * @throws IllegalArgumentException if shape is invalid or product overflows
     */
    public Tensor(int[] shape) {
        this.shape = validateAndCloneShape(shape);
        this.strides = calculateStrides(this.shape);
        this.size = computeSizeOrThrow(this.shape);
        this.data = new float[size];
        this.directBytes = null;
        this.directFloats = null;
        this.pinnedCudaHost = false;
        this.grad = null;
    }

    /**
     * Тензор на off-heap (direct) float-буфере; для JNI {@code cudaMemcpy} с pinned/direct без лишней копии в
     * {@code float[]}. Градиенты — в обычном heap-массиве.
     */
    public static Tensor allocateDirect(int[] shape) {
        int[] validatedShape = validateAndCloneShape(shape);
        int sz = computeSizeOrThrow(validatedShape);
        int[] strides = calculateStrides(validatedShape);
        ByteBuffer bb = ByteBuffer.allocateDirect(sz * Float.BYTES);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.limit(sz);
        return new Tensor(validatedShape, strides, sz, null, bb, fb, null, false);
    }

    /**
     * Тензор на <b>page-locked</b> host-памяти ({@code cudaHostAlloc}): быстрее H2D в CUDA, чем обычный
     * {@link #allocateDirect(int[])}. При ошибке выделения или без драйвера CUDA падает обратно на
     * {@link #allocateDirect(int[])}. Освобождение pinned-памяти — при сборке мусора тензора ({@link Cleaner}).
     */
    public static Tensor allocatePinnedHost(int[] shape) {
        TensorCudaLibrary.load();
        int[] validatedShape = validateAndCloneShape(shape);
        int sz = computeSizeOrThrow(validatedShape);
        int[] strides = calculateStrides(validatedShape);
        long bytesLong = (long) sz * (long) Float.BYTES;
        if (bytesLong <= 0L || bytesLong > Integer.MAX_VALUE) {
            return allocateDirect(validatedShape);
        }
        int bytes = (int) bytesLong;
        long ptr = CudaPinnedHost.allocBytes(bytes);
        if (ptr == 0L) {
            return allocateDirect(validatedShape);
        }
        try {
            ByteBuffer bb = CudaPinnedHost.directBuffer(ptr, bytes);
            if (bb == null) {
                return allocateDirect(validatedShape);
            }
            bb.order(ByteOrder.nativeOrder());
            FloatBuffer fb = bb.asFloatBuffer();
            fb.position(0);
            fb.limit(sz);
            Tensor t = new Tensor(validatedShape, strides, sz, null, bb, fb, null, true);
            PINNED_CUDA_CLEANER.register(t, () -> CudaPinnedHost.free(ptr));
            return t;
        } catch (Throwable e) {
            CudaPinnedHost.free(ptr);
            return allocateDirect(validatedShape);
        }
    }

    /** {@code true} для {@link #allocatePinnedHost(int[])} (данные всё ещё direct для JNI). */
    public boolean isPinnedCudaHostStorage() {
        return pinnedCudaHost;
    }

    /**
     * Обновляет ленивый heap-зеркальный массив из direct-буфера ({@link FloatBuffer#get(float[], int, int)} —
     * обычно с нативным ускорением).
     */
    private void syncHeapMirrorFromDirect() {
        if (directFloats == null) {
            return;
        }
        if (heapMirror == null) {
            heapMirror = new float[size];
        }
        directFloats.position(0);
        directFloats.limit(size);
        directFloats.get(heapMirror, 0, size);
    }

    public StorageMode storageMode() {
        return directBytes != null ? StorageMode.DIRECT : StorageMode.HEAP;
    }

    /** {@code true}, если данные в direct-буфере ({@link #directByteBuffer()} не {@code null}). */
    public boolean isDirectStorage() {
        return directBytes != null;
    }

    /**
     * Базовый direct {@link ByteBuffer} данных (native order); {@code null} для heap-тензора.
     */
    public ByteBuffer directByteBuffer() {
        return directBytes;
    }

    /**
     * Общий {@link FloatBuffer} данных (та же память, что у тензора). Меняет {@code position}/{@code limit}
     * у этого экземпляра. Для изолированной позиции предпочтительнее {@link #directBufferForJNI()}.
     */
    public FloatBuffer directFloatBuffer() {
        if (directFloats == null) {
            return null;
        }
        directFloats.position(0);
        directFloats.limit(size);
        return directFloats;
    }

    /**
     * Копия представления direct-памяти с {@code position=0}, {@code limit=size()} — безопасно для JNI/чтения,
     * не трогая позицию «канонического» буфера тензора.
     * <p>
     * Изменения float через возвращённый буфер отражаются в этом тензоре; не использовать из нескольких потоков без
     * синхронизации.
     *
     * @throws IllegalStateException если тензор на heap ({@link Tensor.StorageMode#HEAP})
     */
    public FloatBuffer directBufferForJNI() {
        if (directFloats == null) {
            throw new IllegalStateException("Tensor uses heap storage; use internalBuffer() for float[] access");
        }
        FloatBuffer dup = directFloats.duplicate();
        dup.position(0);
        dup.limit(size);
        return dup;
    }

    /**
     * Wraps an existing float array without copying (zero-copy view).
     * <p>
     * <b>Warning:</b> The provided {@code data} array MUST NOT be modified
     * externally after wrapping, as it becomes the tensor's internal buffer.
     *
     * @param data underlying buffer (length must equal product of shape)
     * @param shape tensor dimensions
     * @return new Tensor instance sharing the data buffer
     * @throws IllegalArgumentException if shape invalid or length mismatch
     */
    public static Tensor wrap(float[] data, int[] shape) {
        Objects.requireNonNull(data, "data cannot be null");
        int[] validatedShape = validateAndCloneShape(shape);
        int expectedSize = computeSizeOrThrow(validatedShape);
        if (data.length != expectedSize) {
            throw new IllegalArgumentException(String.format(
                    "data length %d != shape product %d for shape %s",
                    data.length, expectedSize, Arrays.toString(validatedShape)));
        }
        return new Tensor(validatedShape, data, calculateStrides(validatedShape), expectedSize, null, null, null);
    }

    /**
     * Creates a tensor by copying values from an array.
     */
    public static Tensor fromArray(float[] values, int[] shape) {
        Tensor t = new Tensor(shape); // allocates zeroed data
        if (values.length != t.size) {
            throw new IllegalArgumentException(String.format(
                    "values length %d != shape product %d", values.length, t.size));
        }
        System.arraycopy(values, 0, t.data, 0, values.length);
        return t;
    }

    private Tensor(int[] shape, float[] data, int[] strides, int size) {
        this(shape, strides, size, data, null, null, null, false);
    }

    private Tensor(int[] shape, float[] data, int[] strides, int size, float[] grad) {
        this(shape, strides, size, data, null, null, grad, false);
    }

    private Tensor(
            int[] shape,
            int[] strides,
            int size,
            float[] data,
            ByteBuffer directBytes,
            FloatBuffer directFloats,
            float[] grad,
            boolean pinnedCudaHost) {
        this.shape = shape;
        this.strides = strides;
        this.size = size;
        this.data = data;
        this.directBytes = directBytes;
        this.directFloats = directFloats;
        this.pinnedCudaHost = pinnedCudaHost;
        this.grad = grad;
        if ((data == null) == (directBytes == null)) {
            throw new IllegalArgumentException("exactly one of heap data or direct buffer must be set");
        }
    }

    private Tensor(int[] shape, float[] data, int[] strides, int size, ByteBuffer bb, FloatBuffer fb, float[] grad) {
        this(shape, strides, size, data, bb, fb, grad, false);
    }

    // ========== Validation helpers ==========

    private static int[] validateAndCloneShape(int[] shape) {
        return TensorUtils.validateAndCloneShape(shape);
    }

    private static int computeSizeOrThrow(int[] shape) {
        return TensorUtils.computeSizeOrThrow(shape);
    }

    private static int[] calculateStrides(int[] shape) {
        return TensorUtils.calculateStrides(shape);
    }

    // ========== Accessors ==========

    /**
     * Gets element at multi-dimensional indices.
     */
    public float get(int... indices) {
        return getLinear(flatIndex(indices));
    }

    /**
     * Gets element by row-major linear index ({@code 0 … size()-1}); один проход границ без varargs.
     */
    public float getLinear(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("linear index " + index + " for size " + size);
        }
        if (directFloats != null) {
            return directFloats.get(index);
        }
        return data[index];
    }

    /**
     * Sets element at multi-dimensional indices.
     */
    public void set(float value, int... indices) {
        setLinear(flatIndex(indices), value);
    }

    /**
     * Sets element by row-major linear index ({@code 0 … size()-1}).
     */
    public void setLinear(int index, float value) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("linear index " + index + " for size " + size);
        }
        if (directFloats != null) {
            heapMirror = null;
            directFloats.put(index, value);
            return;
        }
        data[index] = value;
    }

    /**
     * Computes flat index from multi-dimensional indices with bounds checking.
     */
    private int flatIndex(int[] indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(String.format(
                    "Expected %d indices, got %d", shape.length, indices.length));
        }
        return flatIndexUnchecked(indices);
    }

    /**
     * Без проверки числа измерений — вызывать только после проверки длины {@code indices}.
     */
    private int flatIndexUnchecked(int[] indices) {
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (idx < 0 || idx >= shape[i]) {
                throw new IndexOutOfBoundsException(String.format(
                        "Index %d out of bounds for dimension %d (size %d)",
                        idx, i, shape[i]));
            }
            index += idx * strides[i];
        }
        return index;
    }

    /**
     * Заполняет буфер данных одним значением (не трогает градиент).
     */
    public void fillData(float value) {
        if (directFloats != null) {
            heapMirror = null;
            FloatBuffer fb = directFloats.duplicate();
            fb.clear();
            int chunk = Math.min(size, 8192);
            float[] temp = new float[chunk];
            Arrays.fill(temp, value);
            int written = 0;
            while (written + chunk <= size) {
                fb.put(temp);
                written += chunk;
            }
            if (written < size) {
                fb.put(temp, 0, size - written);
            }
            return;
        }
        Arrays.fill(data, value);
    }

    /**
     * Копирует данные в {@code dest}, начиная с {@code offset}; нужно {@code dest.length >= offset + size()}.
     */
    public void copyDataTo(float[] dest, int offset) {
        if (offset < 0 || offset + size > dest.length) {
            throw new IllegalArgumentException(
                    "need dest.length >= " + (offset + size) + ", got offset=" + offset + ", dest.length=" + dest.length);
        }
        if (directFloats != null) {
            FloatBuffer src = directFloats.duplicate();
            src.position(0);
            src.limit(size);
            src.get(dest, offset, size);
            return;
        }
        System.arraycopy(data, 0, dest, offset, size);
    }

    /**
     * Новый вид того же буфера данных (и того же буфера градиента, если он есть): та же число элементов,
     * другая форма (row-major). Используется как reshape без копирования данных.
     */
    public Tensor viewReshape(int[] newShape) {
        int[] s = validateAndCloneShape(newShape);
        if (computeSizeOrThrow(s) != size) {
            throw new IllegalArgumentException(
                    "element count mismatch: size " + size + " vs new shape " + Arrays.toString(s));
        }
        int[] newStrides = calculateStrides(s);
        if (directBytes != null) {
            return new Tensor(s, newStrides, size, null, directBytes, directFloats, grad, pinnedCudaHost);
        }
        return new Tensor(s, data, newStrides, size, grad);
    }

    // ========== Gradient management ==========

    /**
     * Initializes or zeroes the gradient buffer.
     * <p>
     * При первом вызове выделяется {@code new float[size]} (уже нули). При повторных — тот же массив
     * обнуляется через {@link Arrays#fill}, т.к. градиенты нужно сбрасывать перед новым backward.
     */
    public void zeroGrad() {
        if (grad == null) {
            grad = new float[size];
        } else {
            Arrays.fill(grad, 0f);
        }
    }

    /**
     * Returns gradient buffer (after {@link #zeroGrad()}).
     * <p>
     * Прямой доступ без копии (аналог {@link #internalBuffer()} для градиента).
     * @throws IllegalStateException if grad not initialized
     */
    public float[] gradBuffer() {
        if (grad == null) {
            throw new IllegalStateException("Gradient not initialized — call zeroGrad() first");
        }
        return grad;
    }

    /**
     * Checks if gradient buffer is allocated.
     */
    public boolean hasGrad() {
        return grad != null;
    }

    // ========== Data access ==========

    /**
     * Доступ к данным для CPU-путей: heap — тот же {@code float[]}; direct — lazy heap-зеркало с полной копией
     * из off-heap (см. {@link #directBufferForJNI()} для zero-copy).
     * <p>
     * <b>Внимание:</b> мутации возвращённого массива для direct остаются в зеркале до следующей инвалидации
     * ({@code set*}, {@code fillData}); синхронизация обратно в direct не выполняется автоматически.
     */
    public float[] internalBuffer() {
        if (directFloats != null) {
            syncHeapMirrorFromDirect();
            return heapMirror;
        }
        return data;
    }

    /**
     * Returns a copy of the data buffer.
     * <p>
     * Prefer {@link #internalBuffer()} for performance if mutation is intended.
     */
    public float[] getDataCopy() {
        if (directFloats != null) {
            float[] c = new float[size];
            FloatBuffer src = directFloats.duplicate();
            src.position(0);
            src.limit(size);
            src.get(c, 0, size);
            return c;
        }
        return data.clone();
    }

    // ========== Metadata ==========

    /**
     * Returns a copy of the shape array.
     */
    public int[] getShape() {
        return shape.clone();
    }

    /**
     * Returns a copy of the strides array.
     */
    public int[] getStrides() {
        return strides.clone();
    }

    /**
     * Внутренний массив шагов без копии (для горячих путей в {@code com.veles.llm.jgpt.ops}).
     * <p>
     * Не изменять возвращённый массив. Внешнему коду без необходимости предпочтительнее {@link #getStrides()}.
     */
    public int[] stridesInternal() {
        return strides;
    }

    /**
     * Returns total number of elements.
     */
    public int size() {
        return size;
    }

    /**
     * Returns tensor rank (number of dimensions).
     */
    public int rank() {
        return shape.length;
    }

    // ========== Conversion ==========

    /**
     * Converts to FP16 representation (new buffer).
     */
    public Fp16Tensor toFp16() {
        return Fp16Tensor.fromTensor(this);
    }

    // ========== Object overrides ==========

    /**
     * Семантика по значению: форма и все элементы. O(n); для direct без обязательного построения постоянного
     * {@link #heapMirror}.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Tensor other)) {
            return false;
        }
        if (size != other.size || !Arrays.equals(shape, other.shape)) {
            return false;
        }
        return dataRegionEquals(this, other);
    }

    /**
     * Согласован с {@link #equals(Object)} (алгоритм как у {@link Arrays#hashCode(float[])} для данных).
     */
    @Override
    public int hashCode() {
        int result = Objects.hash(size);
        result = 31 * result + Arrays.hashCode(shape);
        if (directFloats != null) {
            result = 31 * result + chunkedFloatBufferHash(directDuplicateForRead(), size);
        } else {
            result = 31 * result + Arrays.hashCode(data);
        }
        return result;
    }

    @Override
    public String toString() {
        return String.format(
                "Tensor{shape=%s, size=%d, storage=%s, grad=%s}",
                Arrays.toString(shape),
                size,
                directBytes != null ? "DIRECT" : "HEAP",
                grad != null ? "allocated" : "null");
    }

    /** Duplicate с {@code position=0}, {@code limit=size} для чтения без изменения состояния поля {@link #directFloats}. */
    private FloatBuffer directDuplicateForRead() {
        FloatBuffer d = directFloats.duplicate();
        d.position(0);
        d.limit(size);
        return d;
    }

    private static boolean dataRegionEquals(Tensor a, Tensor b) {
        if (a.directFloats == null && b.directFloats == null) {
            return Arrays.equals(a.data, b.data);
        }
        if (a.directFloats != null && b.directFloats != null) {
            return directBuffersEqual(a.directDuplicateForRead(), a.size, b.directDuplicateForRead(), b.size);
        }
        if (a.directFloats != null) {
            return directBufferEqualsHeap(b.data, a.directDuplicateForRead(), a.size);
        }
        return directBufferEqualsHeap(a.data, b.directDuplicateForRead(), b.size);
    }

    private static boolean directBuffersEqual(FloatBuffer a, int n, FloatBuffer b, int m) {
        if (n != m) {
            return false;
        }
        int chunk = Math.min(n, 4096);
        float[] ca = new float[chunk];
        float[] cb = new float[chunk];
        int i = 0;
        while (i < n) {
            int len = Math.min(chunk, n - i);
            a.get(ca, 0, len);
            b.get(cb, 0, len);
            for (int j = 0; j < len; j++) {
                if (Float.floatToIntBits(ca[j]) != Float.floatToIntBits(cb[j])) {
                    return false;
                }
            }
            i += len;
        }
        return true;
    }

    private static boolean directBufferEqualsHeap(float[] heap, FloatBuffer fb, int n) {
        int chunk = Math.min(n, 4096);
        float[] temp = new float[chunk];
        int i = 0;
        while (i < n) {
            int len = Math.min(chunk, n - i);
            fb.get(temp, 0, len);
            for (int j = 0; j < len; j++) {
                if (Float.floatToIntBits(temp[j]) != Float.floatToIntBits(heap[i + j])) {
                    return false;
                }
            }
            i += len;
        }
        return true;
    }

    /** Тот же контракт, что {@link Arrays#hashCode(float[])} для первых {@code n} элементов буфера. */
    private static int chunkedFloatBufferHash(FloatBuffer fb, int n) {
        int result = 1;
        int chunk = Math.min(n, 4096);
        float[] temp = new float[chunk];
        int i = 0;
        while (i < n) {
            int len = Math.min(chunk, n - i);
            fb.get(temp, 0, len);
            for (int j = 0; j < len; j++) {
                result = 31 * result + Float.floatToIntBits(temp[j]);
            }
            i += len;
        }
        return result;
    }
}
