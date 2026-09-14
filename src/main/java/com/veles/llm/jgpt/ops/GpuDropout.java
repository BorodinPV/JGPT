package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Dropout для VRAM-резидентного тренировочного пути (D2D forward/backward без хранения масок).
 *
 * <p>Маска в CUDA-ядре — детерминированная функция {@code (seed, index)}, поэтому backward воспроизводит её,
 * вызывая то же ядро с тем же seed на градиенте ветки. Seed складывается из:
 * <ul>
 *   <li>{@code stepSeed} — меняется на каждый тренировочный forward ({@link #beginTrainingForward()}), одинаков для
 *       forward и backward одного микробатча;</li>
 *   <li>индекса слоя ({@link #setCurrentLayer(int)}; forward выставляет его в {@code runDecoderLayerResidentEager},
 *       backward — в цикле по слоям);</li>
 *   <li>сайта внутри блока ({@link #SITE_ATTN}, {@link #SITE_FFN}, {@link #SITE_EMBED}).</li>
 * </ul>
 *
 * <p>Точки применения: выход Wo и выход W2 <b>до</b> residual-add (residual-поток не маскируется), эмбеддинги после
 * token+pos. В backward маскируется только градиент ветки (перед Wo/W2), residual проходит без маски.
 *
 * <p>Инференс/eval не трогают этот класс: forward применяет dropout только при {@code devCache != null}
 * (training), backward выполняется только в обучении. Пока dropout активен, decoder CUDA graph в training
 * отключается (seed — аргумент ядра и «запёкся» бы в графе).
 */
public final class GpuDropout {

    public static final int SITE_ATTN = 0;
    public static final int SITE_FFN = 1;
    public static final int SITE_EMBED = 2;

    private static volatile float residualProb = 0f;
    private static volatile float embeddingProb = 0f;
    private static volatile long stepSeed = 0L;
    private static volatile int currentLayer = 0;
    private static final AtomicLong forwardCounter = new AtomicLong();
    private static final long BASE_SEED = 0x2545F4914F6CDD1DL;

    private GpuDropout() {}

    /** Задаёт вероятности; {@code p<=0} выключает соответствующий dropout. Вызывать до обучения. */
    public static void configure(float residualDropout, float embeddingDropout) {
        residualProb = clamp(residualDropout);
        embeddingProb = clamp(embeddingDropout);
    }

    private static float clamp(float p) {
        if (!(p > 0f)) {
            return 0f;
        }
        return Math.min(p, 0.95f);
    }

    public static float residualProb() {
        return residualProb;
    }

    public static float embeddingProb() {
        return embeddingProb;
    }

    /** {@code true}, если хотя бы один dropout включён (training forward/backward должны его применять). */
    public static boolean isActive() {
        return residualProb > 0f || embeddingProb > 0f;
    }

    /** Новый seed на очередной тренировочный forward; действует до следующего вызова (т.е. на весь backward). */
    public static void beginTrainingForward() {
        if (stepSeedFrozen) {
            return;
        }
        stepSeed = mix64(BASE_SEED + forwardCounter.incrementAndGet() * 0x9E3779B97F4A7C15L);
    }

    private static volatile boolean stepSeedFrozen;

    /** Только для тестов (gradient check): фиксирует маску между несколькими forward. */
    public static void freezeStepSeedForTests(long seed) {
        stepSeed = seed;
        stepSeedFrozen = true;
    }

    public static void unfreezeStepSeedForTests() {
        stepSeedFrozen = false;
    }

    public static void setCurrentLayer(int layer) {
        currentLayer = layer;
    }

    static long seedFor(int layer, int site) {
        long h = stepSeed;
        h ^= (layer + 1L) * 0xC2B2AE3D27D4EB4FL;
        h ^= (site + 1L) * 0x165667B19E3779F9L;
        return mix64(h);
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /**
     * Residual dropout (in-place) на ветке текущего слоя; одинаково для forward (активация) и backward
     * (градиент ветки). No-op при {@code residualProb == 0}.
     */
    public static void applyResidualBranchInPlace(GpuFloatBuffer buf, int n, int site) {
        float p = residualProb;
        if (p <= 0f) {
            return;
        }
        TensorOpsGPU.dropoutGpuDevice(buf, buf, n, p, seedFor(currentLayer, site));
    }

    /** Как {@link #applyResidualBranchInPlace}, но {@code src → dst} (src не изменяется). */
    public static void applyResidualBranch(GpuFloatBuffer src, GpuFloatBuffer dst, int n, int site) {
        float p = residualProb;
        if (p <= 0f) {
            dst.copyFromDevice(src, n);
            return;
        }
        TensorOpsGPU.dropoutGpuDevice(src, dst, n, p, seedFor(currentLayer, site));
    }

    /** Embedding dropout (in-place): forward — на сумме token+pos, backward — на градиенте входа стека. */
    public static void applyEmbeddingInPlace(GpuFloatBuffer buf, int n) {
        float p = embeddingProb;
        if (p <= 0f) {
            return;
        }
        TensorOpsGPU.dropoutGpuDevice(buf, buf, n, p, seedFor(-1, SITE_EMBED));
    }
}
