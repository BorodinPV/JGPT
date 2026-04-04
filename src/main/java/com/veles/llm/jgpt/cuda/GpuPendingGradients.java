package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.GpuWorkspaceCleanup;
import com.veles.llm.jgpt.training.LLMTrainer;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;

import jdk.incubator.vector.FloatVector;

/**
 * Thread-local accumulator for parameter gradients that are produced on GPU.
 * <p>
 * Ожидаемый цикл в обучении (один поток на {@link ThreadLocal}):
 * <ol>
 *   <li>{@link #accumulate(Tensor, GpuFloatBuffer, int)} — после backward на GPU</li>
 *   <li>{@link #flushMergeToGpuGrads(java.util.Map)} — перед шагом оптимизатора, если градиенты параметров на
 *       VRAM; или {@link #flushAllToHost()} — если накопление в host-{@code grad}</li>
 *   <li>при необходимости {@link #scaleAll(float)}</li>
 *   <li>{@link #cleanupThreadLocal()} при завершении работы потока с GPU (иначе
 *       {@link GpuFloatBuffer} останутся привязаны к потоку из пула → утечка VRAM)</li>
 * </ol>
 * <p>
 * В пуле потоков ({@code ExecutorService} и т.п.) после задачи вызывайте {@link #cleanupThreadLocal()} в
 * {@code finally} или используйте {@link Scope} (try-with-resources).
 * <p>
 * <b>Concurrency:</b> карта принадлежит одному потоку; {@link #accumulate} и flush — с этого же потока (как в
 * {@link LLMTrainer}). Поля {@link Entry} не {@code volatile}: reordering между потоками здесь не требуется.
 * <p>
 * <b>Ключи:</b> {@link IdentityHashMap} — ссылка на тензор параметра, не {@code equals}.
 * <p>
 * {@link TensorOpsGPU#accumulateAddGpuDevice} синхронизирует поток CUDA к возврату в Java.
 */
public final class GpuPendingGradients {

    /** Если буфер больше нужного размера более чем вдвое — пересоздаём, чтобы не держать лишнюю VRAM. */
    private static final int SHRINK_FACTOR = 2;

    private static final class ThreadLocalState {
        final Map<Tensor, Entry> map = new IdentityHashMap<>();
        GpuFloatBuffer[] anyNonFiniteScratchBufs;
        int[] anyNonFiniteScratchLens;
    }

    private static final ThreadLocal<ThreadLocalState> LOCAL = ThreadLocal.withInitial(ThreadLocalState::new);

    private static ThreadLocalState tls() {
        return LOCAL.get();
    }

    private GpuPendingGradients() {}

    /**
     * Обертка для try-with-resources: по закрытии вызывает {@link #cleanupThreadLocal()} для текущего потока.
     *
     * <pre>{@code
     * try (GpuPendingGradients.Scope scope = GpuPendingGradients.acquire()) {
     *     GpuPendingGradients.accumulate(...);
     * }
     * }</pre>
     */
    public static final class Scope implements AutoCloseable {
        private Scope() {}

        @Override
        public void close() {
            cleanupThreadLocal();
        }
    }

    /** Снимок guard для {@link Scope} (один поток). */
    public static Scope acquire() {
        return new Scope();
    }

    /**
     * Накопить дельту градиента с GPU в буфер по данному параметру.
     *
     * @param target тензор параметра (тот же объект, что в модели); не null
     * @param deviceDelta буфер на GPU; не null
     * @param length число float для сложения; должно совпадать с {@link Tensor#size()} у {@code target}
     */
    public static void accumulate(Tensor target, GpuFloatBuffer deviceDelta, int length) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(deviceDelta, "deviceDelta");
        if (!TensorOpsGPU.isGpuAvailable() || length <= 0) {
            return;
        }
        int expected = target.size();
        if (length != expected) {
            throw new IllegalArgumentException(
                    "length " + length + " != target.size() " + expected + " (GpuPendingGradients expects full parameter grad)");
        }
        Map<Tensor, Entry> map = tls().map;
        Entry e = map.get(target);
        if (e != null && e.expectedSize != expected) {
            throw new IllegalStateException(
                    "cached size " + e.expectedSize + " != target.size() " + expected + " for same Tensor key");
        }
        long need = length;
        long cap = e == null ? 0L : e.buffer.numFloats();
        boolean tooSmall = e == null || cap < need;
        boolean tooBig = e != null && cap > need * (long) SHRINK_FACTOR;
        if (tooSmall || tooBig) {
            if (e != null) {
                e.buffer.close();
            }
            e = new Entry(GpuFloatBuffer.allocate(need), expected);
            map.put(target, e);
        }
        if (!e.dirty) {
            e.buffer.clear();
            e.dirty = true;
            e.usedLength = 0;
        }
        TensorOpsGPU.accumulateAddGpuDevice(e.buffer, deviceDelta, length);
        e.usedLength = length;
    }

    /**
     * Слить накопленные GPU-дельты в {@link GpuTensor#gradBuffer()} (device-to-device, без D2H).
     *
     * @param paramToGpu маппинг CPU-тензора параметра → его {@link GpuTensor} на VRAM
     * @throws IllegalStateException если для какого-либо dirty параметра нет записи в {@code paramToGpu}
     */
    public static void flushMergeToGpuGrads(java.util.Map<com.veles.llm.jgpt.core.Tensor, GpuTensor> paramToGpu) {
        Map<Tensor, Entry> map = tls().map;
        for (Map.Entry<Tensor, Entry> it : map.entrySet()) {
            Tensor target = it.getKey();
            Entry e = it.getValue();
            if (!e.dirty || e.usedLength <= 0) {
                continue;
            }
            GpuTensor gt = paramToGpu.get(target);
            if (gt == null) {
                throw new IllegalStateException(
                        "flushMergeToGpuGrads: no GpuTensor for parameter [shape="
                                + Arrays.toString(target.getShape())
                                + ", size="
                                + target.size()
                                + "]. Ensure paramToGpu maps every training parameter (IdentityHashMap uses "
                                + "reference equality).");
            }
            if (!gt.hasGradBuffer()) {
                gt.zeroGrad();
            }
            TensorOpsGPU.accumulateAddGpuDevice(gt.gradBuffer(), e.buffer, e.usedLength);
            e.dirty = false;
            e.usedLength = 0;
        }
    }

    /**
     * Сбросить все «грязные» pending-буферы без слияния в {@link GpuTensor#gradBuffer()}. Нужно при overflow
     * (нечисловой loss или нечисловые pending-дельты), чтобы не выполнять {@link #flushMergeToGpuGrads} с NaN/Inf.
     */
    public static void discardDirtyPending() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        Map<Tensor, Entry> map = tls().map;
        for (Entry e : map.values()) {
            if (!e.dirty || e.usedLength <= 0) {
                continue;
            }
            e.buffer.clear();
            e.dirty = false;
            e.usedLength = 0;
        }
    }

    /**
     * Обнулить все thread-local pending-буферы (включая «не грязные»). После eval / {@code generateText}, до
     * следующего train-step — сильнее чем {@link #discardDirtyPending()}.
     */
    public static void clearAllPendingGpuBuffers() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        Map<Tensor, Entry> map = tls().map;
        for (Entry e : map.values()) {
            e.buffer.clear();
            e.dirty = false;
            e.usedLength = 0;
        }
    }

    /**
     * Проверка NaN/Inf в несброшенных накопленных дельтах (до {@link #flushAllToHost} /
     * {@link #flushMergeToGpuGrads}).
     *
     * <p>Использует фьюзированный GPU-вызов: один JNI и одна синхронизация стрима для всех dirty-
     * буферов, вместо отдельного round-trip на параметр.
     */
    public static boolean anyNonFinitePending() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return false;
        }
        ThreadLocalState st = tls();
        Map<Tensor, Entry> map = st.map;

        // Собираем dirty-буферы без аллокации на каждый вызов (scratch в thread-local, grow-only).
        int capacity = map.size();
        if (capacity == 0) {
            return false;
        }
        if (st.anyNonFiniteScratchBufs == null || st.anyNonFiniteScratchBufs.length < capacity) {
            st.anyNonFiniteScratchBufs = new GpuFloatBuffer[capacity];
            st.anyNonFiniteScratchLens = new int[capacity];
        }
        GpuFloatBuffer[] bufs = st.anyNonFiniteScratchBufs;
        int[] lens = st.anyNonFiniteScratchLens;
        int count = 0;
        for (Entry e : map.values()) {
            if (e.dirty && e.usedLength > 0) {
                bufs[count] = e.buffer;
                lens[count] = e.usedLength;
                count++;
            }
        }
        if (count == 0) {
            return false;
        }
        return TensorOpsGPU.anyNonFiniteGpuDeviceMulti(bufs, lens, count);
    }

    /**
     * Краткая метка первого dirty-параметра с нечисловыми значениями в pending (для логов отладки).
     */
    public static String firstNonFinitePendingDebugInfo() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return "";
        }
        Map<Tensor, Entry> map = tls().map;
        for (Map.Entry<Tensor, Entry> it : map.entrySet()) {
            Entry e = it.getValue();
            if (!e.dirty || e.usedLength <= 0) {
                continue;
            }
            if (TensorOpsGPU.anyNonFiniteGpuDevice(e.buffer, e.usedLength)) {
                Tensor t = it.getKey();
                return "id="
                        + System.identityHashCode(t)
                        + ",used="
                        + e.usedLength
                        + ",paramSize="
                        + t.size()
                        + ",shape="
                        + Arrays.toString(t.getShape());
            }
        }
        return "";
    }

    /**
     * {@code true}, если каждый параметр с несброшенной pending-дельтой есть в {@code paramToGpu} (для безопасного
     * {@link #flushMergeToGpuGrads}).
     */
    public static boolean allDirtyTargetsHaveGpuTensor(Map<Tensor, GpuTensor> paramToGpu) {
        Objects.requireNonNull(paramToGpu, "paramToGpu");
        Map<Tensor, Entry> map = tls().map;
        for (Map.Entry<Tensor, Entry> it : map.entrySet()) {
            Entry e = it.getValue();
            if (!e.dirty || e.usedLength <= 0) {
                continue;
            }
            if (paramToGpu.get(it.getKey()) == null) {
                return false;
            }
        }
        return true;
    }

    /**
     * Скопировать накопленные на GPU дельты в host-градиенты параметров (+=). После вызова буферы помечены
     * как «чистые» до следующего {@link #accumulate}.
     */
    public static void flushAllToHost() {
        Map<Tensor, Entry> map = tls().map;
        for (Map.Entry<Tensor, Entry> kv : map.entrySet()) {
            Tensor target = kv.getKey();
            Entry e = kv.getValue();
            if (!e.dirty || e.usedLength <= 0) {
                continue;
            }
            int n = e.usedLength;
            if (e.hostScratch == null || e.hostScratch.length < n) {
                e.hostScratch = new float[n];
            }
            e.buffer.copyTo(e.hostScratch, 0, n);
            if (!target.hasGrad()) {
                target.zeroGrad();
            }
            float[] g = target.gradBuffer();
            addHostScratchToGrad(g, e.hostScratch, n);
            e.dirty = false;
            e.usedLength = 0;
        }
    }

    /**
     * Масштабирует все накопленные на GPU dirty-градиенты в текущем потоке.
     */
    public static void scaleAll(float scale) {
        if (!TensorOpsGPU.isGpuAvailable() || scale == 1f) {
            return;
        }
        Map<Tensor, Entry> map = tls().map;
        for (Entry e : map.values()) {
            if (!e.dirty || e.usedLength <= 0) {
                continue;
            }
            TensorOpsGPU.scaleInPlaceGpuDevice(e.buffer, e.usedLength, scale);
        }
    }

    /** Есть ли несброшенные накопленные дельты для данного параметра в текущем потоке. */
    public static boolean isDirty(Tensor target) {
        Objects.requireNonNull(target, "target");
        Entry e = tls().map.get(target);
        return e != null && e.dirty && e.usedLength > 0;
    }

    /**
     * Краткая сводка по текущему потоку (для логов): число «грязных» записей и суммарный {@code usedLength}.
     */
    public static String currentThreadDebugSummary() {
        Map<Tensor, Entry> map = tls().map;
        int dirtyEntries = 0;
        long pendingFloats = 0L;
        for (Entry e : map.values()) {
            if (e.dirty && e.usedLength > 0) {
                dirtyEntries++;
                pendingFloats += e.usedLength;
            }
        }
        return "GpuPendingGradients{threadLocal, dirtyEntries="
                + dirtyEntries
                + ", pendingFloats="
                + pendingFloats
                + ", mapSize="
                + map.size()
                + "}";
    }

    /**
     * Закрывает все {@link GpuFloatBuffer} в текущем потоке и очищает {@link ThreadLocal} (освобождение VRAM).
     * Имеет смысл вызывать при завершении обучения в этом потоке или перед долгим простоем.
     */
    public static void cleanupThreadLocal() {
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
        ThreadLocalState st = LOCAL.get();
        for (Entry e : st.map.values()) {
            e.buffer.close();
        }
        st.map.clear();
        st.anyNonFiniteScratchBufs = null;
        st.anyNonFiniteScratchLens = null;
        LOCAL.remove();
    }

    private static void addHostScratchToGrad(float[] grad, float[] scratch, int n) {
        var species = FloatVector.SPECIES_PREFERRED;
        int i = 0;
        int bound = species.loopBound(n);
        for (; i < bound; i += species.length()) {
            FloatVector vg = FloatVector.fromArray(species, grad, i);
            FloatVector vs = FloatVector.fromArray(species, scratch, i);
            vg.add(vs).intoArray(grad, i);
        }
        for (; i < n; i++) {
            grad[i] += scratch[i];
        }
    }

    private static final class Entry {
        private final GpuFloatBuffer buffer;
        private final int expectedSize;
        private float[] hostScratch;
        private boolean dirty;
        private int usedLength;

        private Entry(GpuFloatBuffer buffer, int expectedSize) {
            this.buffer = buffer;
            this.expectedSize = expectedSize;
        }
    }
}
