package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.junit.TensorTestProbeResults;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Ручной «зонд» VRAM: ищет максимальные практичные {@code batchSize} и {@code accumulationSteps} для текущего
 * GPU и выбранного {@link LLMConfig}.
 *
 * <p>Пиковая память при последовательных микробатчах в основном задаётся одним микробатчем (активации не
 * копятся по глубине накопления). Увеличение {@code accumulationSteps} при том же {@code batchSize} обычно
 * почти не меняет пик VRAM, зато растёт эффективный размер батча для градиента.
 *
 * <p>Запуск (из корня проекта):
 *
 * <pre>
 * JGPT_BATCH_PROBE=1 mvn test -Dtest=EffectiveBatchProbeTest#probeMaxBatchAndAccumulation
 * </pre>
 *
 * <p>По умолчанию берётся та же база, что и в {@link com.veles.llm.jgpt.app.AllBooksTrain}:
 * {@link LLMConfig#small16M()} + {@link LLMConfig#applyBatchSizeOverrideFromEnv} (переменная {@code JGPT_BATCH_SIZE}),
 * а шаг оптимизатора строится из {@link LLMConfig#toTrainingConfig(String, int)} (LR, warmup, weight decay, клип,
 * косинусный LR и т.д.), с подставляемыми при зонде {@code batchSize} / {@code accumulationSteps}.
 *
 * <p>Опционально: {@code JGPT_PROBE_MODEL=nano|mini|small|small16m|smart50m} переопределяет пресет модели;
 * Для совпадения с {@code jgpt-smart.sh} задайте {@code JGPT_PROBE_TRAINING_ALIGNED=1} и те же env, что в
 * {@code env/*.env} ({@code JGPT_BATCH_SIZE}, {@code JGPT_MAX_SEQ_LEN}, {@code JGPT_PRESET_NUM_LAYERS},
 * {@code JGPT_ACCUMULATION_STEPS}).
 * {@code JGPT_PROBE_MAX_BATCH}, {@code JGPT_BATCH_PROBE_CPU=1}.
 *
 * <p>FP16 GEMM (Tensor Cores): статический блок ниже выставляет {@code jgpt.fp16.matmul=true} до первой
 * загрузки {@link com.veles.llm.jgpt.TensorOpsGPU} (как {@code JGPT_FP16_MATMUL=1} / {@code ./scripts/jgpt-smart.sh}).
 * Запускайте зонд отдельно: {@code mvn test -Dtest=EffectiveBatchProbeTest ...} — иначе другие тесты могут
 * загрузить TensorOpsGPU раньше, и в сводке будет FP16=выкл. Отключить для зонда: {@code JGPT_BATCH_PROBE_FP16=0}.
 */
@Tag("gpu")
final class EffectiveBatchProbeTest {

    static {
        if (!"0".equals(System.getenv("JGPT_BATCH_PROBE_FP16"))
                && !"false".equalsIgnoreCase(System.getenv("JGPT_BATCH_PROBE_FP16"))) {
            System.setProperty("jgpt.fp16.matmul", "true");
        }
    }

    private static final int SPEED_WARMUP = 1;
    private static final int SPEED_TIMED_RUNS = 3;

    private static final int[] ACCUM_CANDIDATES = {1, 2, 4, 8, 16, 32, 64, 128};

    private static final String SYNTHETIC_CORPUS =
            String.join(
                    " ",
                    repeat(
                            "the cat sat on the mat. the dog ran in the park. "
                                    + "the bird flew over the tree. the fish swam in the lake. "
                                    + "the cat and the dog are friends. ",
                            120));

    private static String[] repeat(String s, int n) {
        String[] a = new String[n];
        Arrays.fill(a, s);
        return a;
    }

    private static LLMConfig resolveProbeModel() {
        String raw = System.getenv("JGPT_PROBE_MODEL");
        LLMConfig base =
                raw == null || raw.isBlank()
                        ? LLMConfig.small16M()
                        : switch (raw.trim().toLowerCase(Locale.ROOT)) {
                            case "nano" -> LLMConfig.nano();
                            case "mini" -> LLMConfig.mini();
                            case "small" -> LLMConfig.small();
                            case "small16m", "small16", "16m" -> LLMConfig.small16M();
                            case "smart50m", "smart50", "50m" -> LLMConfig.smart50M();
                            default -> LLMConfig.small16M();
                        };
        return LLMConfig.applyBatchSizeOverrideFromEnv(base);
    }

    /**
     * Как {@link com.veles.llm.jgpt.app.AllBooksTrain}: те же env-оверрайды batch / seq / слои / accumulation.
     * Нужен для бенчмарка «как stable + smart50M» при {@code source env/02-stable.env}.
     */
    private static LLMConfig resolveProbeModelTrainingAligned() {
        String raw = System.getenv("JGPT_PROBE_MODEL");
        LLMConfig base =
                raw == null || raw.isBlank()
                        ? LLMConfig.small16M()
                        : switch (raw.trim().toLowerCase(Locale.ROOT)) {
                            case "nano" -> LLMConfig.nano();
                            case "mini" -> LLMConfig.mini();
                            case "small" -> LLMConfig.small();
                            case "small16m", "small16", "16m" -> LLMConfig.small16M();
                            case "smart50m", "smart50", "50m" -> LLMConfig.smart50M();
                            default -> LLMConfig.small16M();
                        };
        return LLMConfig.applyAccumulationStepsOverrideFromEnv(
                LLMConfig.applyPresetNumLayersOverrideFromEnv(
                        LLMConfig.applySeqLenOverrideFromEnv(
                                LLMConfig.applyBatchSizeOverrideFromEnv(base))));
    }

    /** Строка для сводки: эталонные batch/accum/LR/регуляризация как у полного обучения. */
    private static String formatReferenceTraining(LLMConfig lc, int modelVocabSize) {
        TrainingConfig t = lc.toTrainingConfig("checkpoints", modelVocabSize);
        return String.format(
                Locale.ROOT,
                "Эталон (LLMConfig + toTrainingConfig, как в AllBooksTrain): batch=%d, accum=%d, lr=%.6f, "
                        + "warmup=%.2f, weightDecay=%.5f, maxGradNorm=%.2f, lrSchedule=%s, minLrRatio=%.2f; FP16 GEMM=%s",
                lc.batchSize,
                lc.accumulationSteps,
                lc.learningRate,
                t.warmupRatio,
                t.weightDecay,
                t.maxGradNorm,
                t.lrSchedule,
                t.minLrRatio,
                TensorOpsGPU.useFp16Matmul() ? "вкл" : "выкл");
    }

    private static int probeMaxBatchCeiling() {
        String raw = System.getenv("JGPT_PROBE_MAX_BATCH");
        if (raw == null || raw.isBlank()) {
            return 256;
        }
        try {
            int v = Integer.parseInt(raw.trim());
            return v > 0 ? Math.min(v, 4096) : 256;
        } catch (NumberFormatException e) {
            return 256;
        }
    }

    private static boolean forceCpuProbe() {
        String e = System.getenv("JGPT_BATCH_PROBE_CPU");
        return "1".equals(e) || "true".equalsIgnoreCase(e);
    }

    private static TrainingConfig probeTrainingConfig(
            LLMConfig lc, int modelVocabSize, int batchSize, int accum, String checkpointDir) {
        TrainingConfig t = lc.toTrainingConfig(checkpointDir, modelVocabSize);
        return new TrainingConfig(
                t.vocabSize,
                t.maxSeqLen,
                t.dModel,
                t.numHeads,
                t.numLayers,
                t.dIntermediate,
                batchSize,
                accum,
                1,
                t.learningRate,
                t.warmupRatio,
                t.weightDecay,
                t.maxGradNorm,
                Integer.MAX_VALUE,
                Integer.MAX_VALUE,
                t.lrSchedule,
                t.minLrRatio,
                checkpointDir,
                Integer.MAX_VALUE,
                0,
                0,
                false,
                0f,
                0,
                false);
    }

    /**
     * Для BPE нужен достаточно длинный текст, иначе итоговый vocab может быть сильно меньше {@code targetVocab}
     * (модель всё равно согласована: используется {@link BPETokenizer#getVocabSize()}).
     */
    private static BPETokenizer createTokenizer(int targetVocab) {
        StringBuilder bpe = new StringBuilder(SYNTHETIC_CORPUS.length() * 50);
        for (int i = 0; i < 50; i++) {
            bpe.append(SYNTHETIC_CORPUS);
        }
        for (int i = 0; i < 2000; i++) {
            bpe.append(" tok_").append(i).append(" uniq_").append(i).append(".");
        }
        return BPETokenizer.train(List.of(bpe.toString()), targetVocab);
    }

    /**
     * Сколько непересекающихся окон длины {@code maxSeqLen+1} получится из уже закодированных токенов
     * (та же логика среза, что в {@link DataLoader#loadText}).
     */
    private static int countSlidingWindows(int[] tokens, int maxSeqLen) {
        int count = 0;
        for (int i = 0; i + maxSeqLen < tokens.length; i += maxSeqLen) {
            count++;
        }
        return count;
    }

    /**
     * Повторяет синтетический текст, пока из него не получится хотя бы {@code minSequences} окон — иначе рост
     * batch упирается в длину корпуса, а не в VRAM (как при «need 186, got 185»).
     */
    private static String ensureLoaderText(BPETokenizer tokenizer, int maxSeqLen, int minSequences) {
        StringBuilder sb = new StringBuilder(SYNTHETIC_CORPUS.length() * 8);
        for (int round = 0; round < 50_000; round++) {
            sb.append(SYNTHETIC_CORPUS);
            int[] tokens = tokenizer.encode(sb.toString(), true);
            if (countSlidingWindows(tokens, maxSeqLen) >= minSequences) {
                return sb.toString();
            }
        }
        throw new IllegalStateException(
                "Не удалось собрать текст для " + minSequences + " окон (maxSeqLen=" + maxSeqLen + ")");
    }

    /**
     * Один шаг оптимизатора (полная группа накопления), как в реальном обучении: {@link LLMTrainer#train()}.
     * Ровно {@code batchSize * accumulationSteps} окон — одна эпоха даёт ровно один optimizer step.
     */
    static void runSingleOptimizerStep(
            LLMConfig lc, BPETokenizer tokenizer, Path checkpointDir, int batchSize, int accum, String loaderText)
            throws IOException {
        int modelVocab = tokenizer.getVocabSize();
        TrainingConfig tc = probeTrainingConfig(lc, modelVocab, batchSize, accum, checkpointDir.toString());

        GPTModel model =
                new GPTModel(
                        tc.vocabSize,
                        tc.maxSeqLen,
                        tc.dModel,
                        tc.numHeads,
                        tc.numLayers,
                        tc.dIntermediate);

        DataLoader loader = new DataLoader(tokenizer, lc.maxSeqLen, batchSize);
        loader.setMaxSequences(batchSize * accum);
        loader.loadText(loaderText);
        if (loader.numSequences() < batchSize * accum) {
            throw new IllegalStateException(
                    "Not enough sequences: need " + (batchSize * accum) + ", got " + loader.numSequences());
        }

        LLMTrainer trainer = new LLMTrainer(model, tc, loader);
        trainer.train();
    }

    /**
     * Среднее время одного шага оптимизатора (нс), с разогревом; отдельные каталоги на каждый вызов.
     */
    private static double meanOptimizerStepNanos(
            LLMConfig lc,
            BPETokenizer tokenizer,
            Path parentDir,
            String label,
            int batchSize,
            int accum,
            int warmup,
            int runs,
            String loaderText)
            throws IOException {
        for (int i = 0; i < warmup; i++) {
            Path runDir = Files.createDirectories(parentDir.resolve(label + "_warm_" + i));
            runSingleOptimizerStep(lc, tokenizer, runDir, batchSize, accum, loaderText);
        }
        long sum = 0L;
        for (int i = 0; i < runs; i++) {
            Path runDir = Files.createDirectories(parentDir.resolve(label + "_timed_" + i));
            long t0 = System.nanoTime();
            runSingleOptimizerStep(lc, tokenizer, runDir, batchSize, accum, loaderText);
            sum += System.nanoTime() - t0;
        }
        return sum / (double) runs;
    }

    private static double tokensPerSecond(long tokensPerOptimizerStep, double meanNanos) {
        if (meanNanos <= 0d || tokensPerOptimizerStep <= 0L) {
            return Double.NaN;
        }
        return tokensPerOptimizerStep * 1e9 / meanNanos;
    }

    @Test
    @DisplayName("opt-in: печатает макс. batchSize и accumulationSteps (JGPT_BATCH_PROBE=1)")
    @EnabledIfEnvironmentVariable(named = "JGPT_BATCH_PROBE", matches = "1")
    void probeMaxBatchAndAccumulation(@TempDir Path checkpointDir) throws IOException {
        if (!forceCpuProbe()) {
            Assumptions.assumeTrue(
                    TensorOpsGPU.isGpuAvailable(), "Задайте JGPT_BATCH_PROBE_CPU=1 для прогона без GPU");
        }

        boolean trainingAligned =
                "1".equals(System.getenv("JGPT_PROBE_TRAINING_ALIGNED"))
                        || "true".equalsIgnoreCase(System.getenv("JGPT_PROBE_TRAINING_ALIGNED"));
        LLMConfig lc = trainingAligned ? resolveProbeModelTrainingAligned() : resolveProbeModel();
        BPETokenizer tokenizer = createTokenizer(lc.vocabSize);

        int ceiling = probeMaxBatchCeiling();
        int maxAccumCandidate = 0;
        for (int a : ACCUM_CANDIDATES) {
            maxAccumCandidate = Math.max(maxAccumCandidate, a);
        }
        int minSequencesForLoader = Math.max(ceiling, maxAccumCandidate) + 64;
        String loaderText = ensureLoaderText(tokenizer, lc.maxSeqLen, minSequencesForLoader);

        int maxBatch = 0;
        Throwable lastFail = null;

        for (int bs = 1; bs <= ceiling; bs++) {
            Path runDir = Files.createDirectories(checkpointDir.resolve("batch_" + bs));
            try {
                runSingleOptimizerStep(lc, tokenizer, runDir, bs, 1, loaderText);
                maxBatch = bs;
            } catch (Throwable t) {
                lastFail = t;
                break;
            }
        }

        // Микробатч = 1: проверяем, какой максимум шагов накопления проходит (пик VRAM обычно как у одного микробатча).
        final int microBatchForAccumProbe = 1;
        int maxAccum = 0;
        for (int acc : ACCUM_CANDIDATES) {
            Path runDir = Files.createDirectories(checkpointDir.resolve("accum_" + acc));
            try {
                runSingleOptimizerStep(lc, tokenizer, runDir, microBatchForAccumProbe, acc, loaderText);
                maxAccum = acc;
            } catch (Throwable t) {
                break;
            }
        }

        Path speedDir = Files.createDirectories(checkpointDir.resolve("speed"));
        double tpsMaxBatch = Double.NaN;
        if (maxBatch >= 1) {
            double ns =
                    meanOptimizerStepNanos(
                            lc,
                            tokenizer,
                            speedDir,
                            "bmax",
                            maxBatch,
                            1,
                            SPEED_WARMUP,
                            SPEED_TIMED_RUNS,
                            loaderText);
            long tok = (long) maxBatch * lc.maxSeqLen;
            tpsMaxBatch = tokensPerSecond(tok, ns);
        }
        double tpsMaxAccum = Double.NaN;
        if (maxAccum >= 1) {
            double ns =
                    meanOptimizerStepNanos(
                            lc,
                            tokenizer,
                            speedDir,
                            "amax",
                            microBatchForAccumProbe,
                            maxAccum,
                            SPEED_WARMUP,
                            SPEED_TIMED_RUNS,
                            loaderText);
            long tok = (long) maxAccum * lc.maxSeqLen;
            tpsMaxAccum = tokensPerSecond(tok, ns);
        }

        String failLine = null;
        if (lastFail != null) {
            String msg = lastFail.getMessage() != null ? lastFail.getMessage() : "";
            if (msg.startsWith("Not enough sequences")) {
                failLine =
                        "первая ошибка при росте batch (шаг "
                                + (maxBatch + 1)
                                + "): не хватило окон в корпусе после ensureLoaderText (увеличьте повторы или maxSeqLen) — "
                                + msg;
            } else {
                failLine =
                        "первая ошибка при росте batch (шаг "
                                + (maxBatch + 1)
                                + "): "
                                + lastFail.getClass().getSimpleName()
                                + " — "
                                + msg;
            }
        }

        TensorTestProbeResults.record(
                lc.name,
                lc.maxSeqLen,
                tokenizer.getVocabSize(),
                maxBatch,
                maxAccum,
                failLine,
                tpsMaxBatch,
                tpsMaxAccum,
                formatReferenceTraining(lc, tokenizer.getVocabSize()));

        assertTrue(maxBatch >= 1, "даже batchSize=1 должен проходить на выбранной конфигурации");
    }
}
