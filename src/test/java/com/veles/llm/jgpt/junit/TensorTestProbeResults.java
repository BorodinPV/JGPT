package com.veles.llm.jgpt.junit;

import com.veles.llm.jgpt.TensorOpsGPU;

import java.util.Locale;

/**
 * Результаты opt-in прогона {@link com.veles.llm.jgpt.training.EffectiveBatchProbeTest} для вывода в конце
 * всего тест-плана ({@link TensorTestExecutionSummaryListener}).
 */
public final class TensorTestProbeResults {

    private static final Object LOCK = new Object();

    private static boolean present;
    private static String modelName;
    private static int maxSeqLen;
    private static int vocabSize;
    private static int maxBatch;
    private static int maxAccum;
    private static String firstBatchFailureLine;
    /** Токены/с на один шаг оптимизатора при (maxBatch, 1); NaN если не измеряли. */
    private static double tokensPerSecMaxBatch;
    /** Токены/с на один шаг при (1, maxAccum); NaN если не измеряли. */
    private static double tokensPerSecMaxAccum;
    /** Эталонные настройки (как в {@link com.veles.llm.jgpt.training.LLMConfig#toTrainingConfig}). */
    private static String referenceTrainingLine;

    private TensorTestProbeResults() {
    }

    public static void reset() {
        synchronized (LOCK) {
            present = false;
            modelName = "";
            maxSeqLen = 0;
            vocabSize = 0;
            maxBatch = 0;
            maxAccum = 0;
            firstBatchFailureLine = null;
            tokensPerSecMaxBatch = Double.NaN;
            tokensPerSecMaxAccum = Double.NaN;
            referenceTrainingLine = null;
        }
    }

    /**
     * @param firstBatchFailure кратко о первой ошибке при переборе batch (или {@code null})
     */
    public static void record(
            String modelName,
            int maxSeqLen,
            int vocabSize,
            int maxBatch,
            int maxAccum,
            String firstBatchFailure,
            double tokensPerSecAtMaxBatchAccum1,
            double tokensPerSecAtBatch1MaxAccum,
            String referenceTrainingLine) {
        synchronized (LOCK) {
            present = true;
            TensorTestProbeResults.modelName = modelName != null ? modelName : "";
            TensorTestProbeResults.maxSeqLen = maxSeqLen;
            TensorTestProbeResults.vocabSize = vocabSize;
            TensorTestProbeResults.maxBatch = maxBatch;
            TensorTestProbeResults.maxAccum = maxAccum;
            TensorTestProbeResults.firstBatchFailureLine = firstBatchFailure;
            TensorTestProbeResults.tokensPerSecMaxBatch = tokensPerSecAtMaxBatchAccum1;
            TensorTestProbeResults.tokensPerSecMaxAccum = tokensPerSecAtBatch1MaxAccum;
            TensorTestProbeResults.referenceTrainingLine = referenceTrainingLine;
        }
    }

    public static void appendTo(StringBuilder out) {
        synchronized (LOCK) {
            if (!present) {
                return;
            }
            out.append("========== Подбор batch / accumulation (прогон) ==========\n");
            out.append("Модель: ")
                    .append(modelName)
                    .append(", maxSeqLen=")
                    .append(maxSeqLen)
                    .append(", vocab=")
                    .append(vocabSize)
                    .append('\n');
            if (referenceTrainingLine != null && !referenceTrainingLine.isBlank()) {
                out.append(referenceTrainingLine).append('\n');
            }
            out.append("max batchSize (accumulationSteps=1): ").append(maxBatch).append('\n');
            out.append("max accumulationSteps (batchSize=1): ").append(maxAccum).append('\n');
            if (firstBatchFailureLine != null && !firstBatchFailureLine.isBlank()) {
                out.append(firstBatchFailureLine).append('\n');
            }
            if (TensorOpsGPU.isGpuAvailable()) {
                out.append("GPU: ")
                        .append(TensorOpsGPU.getGpuName())
                        .append(", ")
                        .append(TensorOpsGPU.getGpuMemory())
                        .append(" MB\n");
            } else {
                out.append("GPU: нет (CPU)\n");
            }
            long eff1 = (long) maxBatch * maxSeqLen;
            long eff2 = (long) maxAccum * maxSeqLen;
            out.append("Эффективно токенов за шаг оптимизатора: ")
                    .append(maxBatch)
                    .append("×1×seq = ")
                    .append(eff1)
                    .append(" или 1×")
                    .append(maxAccum)
                    .append("×seq = ")
                    .append(eff2)
                    .append('\n');
            out.append("Скорость обучения (токенов/с, один шаг оптимизатора, после разогрева):\n");
            if (!Double.isNaN(tokensPerSecMaxBatch)) {
                out.append("  batch=")
                        .append(maxBatch)
                        .append(", accum=1  →  ")
                        .append(String.format(Locale.ROOT, "%.0f", tokensPerSecMaxBatch))
                        .append(" tok/s\n");
            } else {
                out.append("  batch=max, accum=1  →  (не измерено)\n");
            }
            if (!Double.isNaN(tokensPerSecMaxAccum)) {
                out.append("  batch=1, accum=")
                        .append(maxAccum)
                        .append("  →  ")
                        .append(String.format(Locale.ROOT, "%.0f", tokensPerSecMaxAccum))
                        .append(" tok/s\n");
            } else {
                out.append("  batch=1, accum=max  →  (не измерено)\n");
            }
            out.append("===========================================================\n");
        }
    }
}
