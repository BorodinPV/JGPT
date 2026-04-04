package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.model.GPTModel;

/**
 * Конфигурации моделей разного размера (совместимо с {@link GPTModel} / {@link TrainingConfig}).
 */
public final class LLMConfig {

    public final String name;
    public final int vocabSize;
    public final int maxSeqLen;
    public final int dModel;
    public final int numHeads;
    public final int numLayers;
    public final int dIntermediate;
    public final int batchSize;
    /** Шагов накопления градиента на один optimizer step (см. {@link TrainingConfig#accumulationSteps}). */
    public final int accumulationSteps;
    public final float learningRate;
    public final int epochs;

    public LLMConfig(
            String name,
            int vocabSize,
            int maxSeqLen,
            int dModel,
            int numHeads,
            int numLayers,
            int dIntermediate,
            int batchSize,
            int accumulationSteps,
            float learningRate,
            int epochs) {
        this.name = name;
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.dIntermediate = dIntermediate;
        this.batchSize = batchSize;
        this.accumulationSteps = Math.max(1, accumulationSteps);
        this.learningRate = learningRate;
        this.epochs = epochs;
    }

    /** Мини-модель для быстрого теста. */
    public static LLMConfig nano() {
        return new LLMConfig("Nano", 500, 64, 64, 4, 2, 256, 8, 1, 0.01f, 5);
    }

    /** Маленькая модель для видимых результатов. */
    public static LLMConfig mini() {
        return new LLMConfig("Mini", 4000, 128, 128, 8, 4, 512, 16, 1, 0.001f, 10);
    }

    /**
     * Средняя модель (~12.5M параметров по {@link #estimateParameters()}, vocab 8000), тяжелее {@link #mini()}.
     */
    public static LLMConfig small() {
        return new LLMConfig(
                "Small",
                8000,
                256,
                256,
                16,
                8,
                1024,
                6,
                1,
                0.001f,
                20);
    }

    /**
     * Multi-book по умолчанию: vocab 8000, контекст 512, 11 трансформер-слоёв — около **15.8M** параметров по
     * {@link #estimateParameters()}; длиннее контекст, чем у {@link #small()}, меньше слоёв, чем у 16-слойного варианта.
     */
    public static LLMConfig small16M() {
        return new LLMConfig(
                "Small~16M",
                8000,
                512,
                256,
                16,
                11,
                1024,
                6,
                1,
                0.001f,
                20);
    }

    /**
     * Крупный пресет: vocab 8000, контекст <b>2048</b>, 20 слоёв, ~55M параметров по {@link #estimateParameters()}.
     * <p>
     * Внимание по VRAM: attention и кэши растут примерно как {@code O(seq²)} на слой; {@code batchSize=1} и
     * {@code sampled CE} обязательны на картах ~10 ГиБ. При {@code cudaMalloc}/OOM: уменьшить {@code maxSeqLen},
     * {@code numLayers} или {@code JGPT_SAMPLED_CE_CANDIDATES}; не поднимать {@code JGPT_BATCH_SIZE}.
     */
    public static LLMConfig smart50M() {
        return new LLMConfig(
                "JGPT-50M-Smart",
                8000,
                2048,
                384,
                24,
                20,
                1536,
                1,
                6,
                0.0005f,
                20);
    }

    /**
     * @deprecated Заменён на {@link #small16M()} (≈16M, контекст 512).
     */
    @Deprecated
    public static LLMConfig small18M() {
        return small16M();
    }

    /**
     * Runtime override from env {@code JGPT_BATCH_SIZE}.
     */
    public static LLMConfig applyBatchSizeOverrideFromEnv(LLMConfig base) {
        int overridden = readPositiveEnvInt("JGPT_BATCH_SIZE", base.batchSize);
        if (overridden == base.batchSize) {
            return base;
        }
        return new LLMConfig(
                base.name,
                base.vocabSize,
                base.maxSeqLen,
                base.dModel,
                base.numHeads,
                base.numLayers,
                base.dIntermediate,
                overridden,
                base.accumulationSteps,
                base.learningRate,
                base.epochs);
    }

    /**
     * Runtime override from env {@code JGPT_MAX_SEQ_LEN}.
     * <p>
     * Позволяет уменьшить контекст без перекомпиляции. Актуально при OOM: на RTX 3080 10 ГиБ
     * с 20 слоями и 24 головами кэш attention для backward = {@code heads × seq² × 2 bytes × layers}.
     * При seq=2048 → ~4 ГиБ; при seq=1024 → ~1 ГиБ.
     * <p>Пример: {@code JGPT_MAX_SEQ_LEN=1024 ./scripts/train-e2e-gpu.sh}
     */
    public static LLMConfig applySeqLenOverrideFromEnv(LLMConfig base) {
        int overridden = readPositiveEnvInt("JGPT_MAX_SEQ_LEN", base.maxSeqLen);
        if (overridden == base.maxSeqLen) {
            return base;
        }
        return new LLMConfig(
                base.name,
                base.vocabSize,
                overridden,
                base.dModel,
                base.numHeads,
                base.numLayers,
                base.dIntermediate,
                base.batchSize,
                base.accumulationSteps,
                base.learningRate,
                base.epochs);
    }

    /**
     * Переопределяет число эпох через переменную окружения {@code JGPT_EPOCHS}.
     *
     * <p>Пример: {@code JGPT_EPOCHS=40 ./scripts/train-e2e-gpu.sh allbooks}
     */
    public static LLMConfig applyEpochsOverrideFromEnv(LLMConfig base) {
        int overridden = readPositiveEnvInt("JGPT_EPOCHS", base.epochs);
        if (overridden == base.epochs) {
            return base;
        }
        return new LLMConfig(
                base.name,
                base.vocabSize,
                base.maxSeqLen,
                base.dModel,
                base.numHeads,
                base.numLayers,
                base.dIntermediate,
                base.batchSize,
                base.accumulationSteps,
                base.learningRate,
                overridden);
    }

    /**
     * Явный запрос из env {@code JGPT_TRAIN_GPU_RESIDENT=1} / {@code true} (пустое значение — {@code false}).
     */
    public static boolean gpuResidentTrainingExplicitlyOn() {
        String e = System.getenv("JGPT_TRAIN_GPU_RESIDENT");
        if (e == null || e.isBlank()) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    /**
     * Резидентные веса при обучении: при доступной CUDA и пустом / не заданном env — <b>вкл.</b> по умолчанию;
     * {@code JGPT_TRAIN_GPU_RESIDENT=0}/{@code false} — принудительно выкл.; явный {@code 1}/{@code true} — вкл.
     * при наличии CUDA.
     */
    public static boolean effectiveGpuResidentTraining() {
        String e = System.getenv("JGPT_TRAIN_GPU_RESIDENT");
        if (e != null && !e.isBlank()) {
            String t = e.trim();
            if ("0".equals(t) || "false".equalsIgnoreCase(t)) {
                return false;
            }
            if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                return TensorOpsGPU.isGpuAvailable();
            }
            return false;
        }
        return TensorOpsGPU.isGpuAvailable();
    }

    /** Env {@code JGPT_FULL_GPU_TRAIN} / prop {@code jgpt.fullGpuTrain}. */
    public static boolean fullGpuTrainStepFromEnv() {
        return readBoolEnvOrProp("JGPT_FULL_GPU_TRAIN", "jgpt.fullGpuTrain");
    }

    /** Эффективный: env-флаг + реально доступная CUDA. */
    public static boolean effectiveFullGpuTrainStepFromEnv() {
        return fullGpuTrainStepFromEnv() && TensorOpsGPU.isGpuAvailable();
    }

    /** Env / prop; при незаданных — как при {@link #decoderGpuPipelineFromEnvOrProp()}: вкл., если CUDA есть. */
    public static boolean deviceLogitsTrainStepFromEnv() {
        return readBoolEnvOrPropDefaultGpuWhenUnset(
                "JGPT_DEVICE_LOGITS_TRAIN", "jgpt.deviceLogitsTrain");
    }

    /** Env / prop; при незаданных — вкл., если CUDA есть. */
    public static boolean deviceDecoderBackwardFromEnv() {
        return readBoolEnvOrPropDefaultGpuWhenUnset(
                "JGPT_DEVICE_DECODER_BWD", "jgpt.deviceDecoderBackward");
    }

    /** Train-only loss mode: env {@code JGPT_TRAIN_LOSS_MODE} / prop {@code jgpt.trainLossMode}. */
    public static TrainLossMode trainLossModeFromEnvOrProp() {
        return readTrainLossMode(
                firstNonBlank(System.getenv("JGPT_TRAIN_LOSS_MODE"), System.getProperty("jgpt.trainLossMode")));
    }

    /** Env {@code JGPT_SAMPLED_CE_CANDIDATES} / prop {@code jgpt.sampledCe.candidates}. */
    public static int sampledCeCandidatesFromEnv() {
        return Math.max(2, readPositiveEnvOrPropInt("JGPT_SAMPLED_CE_CANDIDATES", "jgpt.sampledCe.candidates", 128));
    }

    /**
     * Env {@code JGPT_INTERACTIVE_EVERY}: через сколько шагов оптимизатора генерировать текст во время
     * обучения. {@code 0} или {@code -1} — отключить. По умолчанию 200.
     * <p>Внимание: генерация (инференс) после каждого eval резко снижает FP16 loss scale (÷64),
     * что вызывает overflow на следующем обучающем шаге. При нестабильном FP16 лучше выставить {@code 0}.
     */
    public static int interactiveEveryFromEnv(int defaultValue) {
        String env = System.getenv("JGPT_INTERACTIVE_EVERY");
        if (env != null && !env.isBlank()) {
            try {
                int v = Integer.parseInt(env.trim());
                return Math.max(0, v);
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        String prop = System.getProperty("jgpt.interactiveEvery");
        if (prop != null && !prop.isBlank()) {
            try {
                int v = Integer.parseInt(prop.trim());
                return Math.max(0, v);
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    /** Env {@code JGPT_SAMPLED_CE_NEGATIVE_MODE} / prop {@code jgpt.sampledCe.negativeMode}. */
    public static SampledNegativeMode sampledCeNegativeModeFromEnvOrProp() {
        String raw =
                firstNonBlank(
                        System.getenv("JGPT_SAMPLED_CE_NEGATIVE_MODE"),
                        System.getProperty("jgpt.sampledCe.negativeMode"));
        if (raw == null) {
            return SampledNegativeMode.BATCH_SHARED_UNIFORM;
        }
        String normalized = raw.trim().replace('-', '_').toUpperCase();
        if ("BATCH_SHARED_UNIFORM".equals(normalized)) {
            return SampledNegativeMode.BATCH_SHARED_UNIFORM;
        }
        throw new IllegalArgumentException(
                "Unsupported sampled CE negative mode: "
                        + raw
                        + " (expected batch_shared_uniform)");
    }

    /**
     * Пресет end-to-end GPU: env {@code JGPT_GPU_E2E_TRAIN=1} / prop {@code jgpt.gpu.e2eTrain}. Включает
     * {@link TrainingConfig#useGpuResident}, {@link TrainingConfig#fullGpuTrainStep}, device logits и device
     * decoder backward согласованно (см. {@link #toTrainingConfig(String, int)}). Требует
     * {@link #effectiveGpuResidentTraining()} и доступную CUDA; иначе {@link #toTrainingConfig} бросает
     * {@link IllegalStateException}. Нужен {@code JGPT_DECODER_GPU_PIPELINE=1} — без него
     * {@link #toTrainingConfig(String, int)} бросает {@link IllegalStateException}. Часть вспомогательной работы
     * (данные, I/O чекпоинта) по-прежнему на хосте.
     */
    public static boolean gpuE2eTrainFromEnv() {
        return readBoolEnvOrProp("JGPT_GPU_E2E_TRAIN", "jgpt.gpu.e2eTrain");
    }

    /**
     * Разрешение decoder GPU pipeline: env {@code JGPT_DECODER_GPU_PIPELINE=1} / prop
     * {@code jgpt.decoder.gpu.pipeline} — совпадает с {@link GPTModel} (см. {@link #toTrainingConfig}).
     */
    public static boolean decoderGpuPipelineFromEnvOrProp() {
        return readBoolEnvOrPropDefaultGpuWhenUnset(
                "JGPT_DECODER_GPU_PIPELINE", "jgpt.decoder.gpu.pipeline");
    }

    /**
     * Один CUDA graph на полный декодер-слой (MHA+FFN на {@code kTensorCudaStream}): env {@code
     * JGPT_DECODER_LAYER_CUDA_GRAPH=1} / prop {@code jgpt.decoder.layer.cudaGraph}. Включайте явно; при сбое
     * захвата выполнение откатывается на обычную цепочку launch’ов.
     */
    public static boolean decoderLayerCudaGraphFromEnvOrProp() {
        return readBoolEnvOrProp("JGPT_DECODER_LAYER_CUDA_GRAPH", "jgpt.decoder.layer.cudaGraph");
    }

    /**
     * Один JNI для второго RMSNorm + проекций SwiGLU W1/W3: env {@code JGPT_FUSED_FFN_RMS_W1W3=1} / prop {@code
     * jgpt.fused.ffn.rms.w1w3}.
     */
    public static boolean fusedFfnRmsW1W3FromEnvOrProp() {
        return readBoolEnvOrProp("JGPT_FUSED_FFN_RMS_W1W3", "jgpt.fused.ffn.rms.w1w3");
    }

    /**
     * Полный GPU-шаг из env ({@link #gpuE2eTrainFromEnv()} или {@link #effectiveFullGpuTrainStepFromEnv()}) требует
     * pipeline, иначе {@link GPTModel#canFullGpuTrain()} ложен.
     */
    private static void ensureDecoderGpuPipelineForFullGpuTrainRequest() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        if (!decoderGpuPipelineFromEnvOrProp()) {
            throw new IllegalStateException(
                    "Full GPU training requires JGPT_DECODER_GPU_PIPELINE=1 "
                            + "(or -Djgpt.decoder.gpu.pipeline=true) so that GPTModel.canFullGpuTrain() is true.");
        }
    }

    private static boolean readBoolEnvOrProp(String envKey, String propKey) {
        String e = System.getenv(envKey);
        if (e != null && !e.isBlank()) {
            String t = e.trim();
            return "1".equals(t) || "true".equalsIgnoreCase(t);
        }
        String p = System.getProperty(propKey);
        if (p != null && !p.isBlank()) {
            String t = p.trim();
            return "1".equals(t) || "true".equalsIgnoreCase(t);
        }
        return false;
    }

    /**
     * Явное {@code 0}/{@code false} — выкл.; {@code 1}/{@code true} — вкл.; не задано ни env, ни property —
     * {@link TensorOpsGPU#isGpuAvailable()} (предпочитать GPU-реализацию, если она есть).
     */
    private static boolean readBoolEnvOrPropDefaultGpuWhenUnset(String envKey, String propKey) {
        String e = System.getenv(envKey);
        if (e != null && !e.isBlank()) {
            String t = e.trim();
            if ("0".equals(t) || "false".equalsIgnoreCase(t)) {
                return false;
            }
            if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                return true;
            }
            return false;
        }
        String p = System.getProperty(propKey);
        if (p != null && !p.isBlank()) {
            String t = p.trim();
            if ("0".equals(t) || "false".equalsIgnoreCase(t)) {
                return false;
            }
            if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                return true;
            }
            return false;
        }
        return TensorOpsGPU.isGpuAvailable();
    }

    private static int readPositiveEnvInt(String key, int defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        try {
            int parsed = Integer.parseInt(raw.trim());
            return parsed > 0 ? parsed : defaultValue;
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private static int readPositiveEnvOrPropInt(String envKey, String propKey, int defaultValue) {
        String env = System.getenv(envKey);
        if (env != null && !env.isBlank()) {
            try {
                int parsed = Integer.parseInt(env.trim());
                return parsed > 0 ? parsed : defaultValue;
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        String prop = System.getProperty(propKey);
        if (prop != null && !prop.isBlank()) {
            try {
                int parsed = Integer.parseInt(prop.trim());
                return parsed > 0 ? parsed : defaultValue;
            } catch (NumberFormatException ignored) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    private static String firstNonBlank(String first, String second) {
        if (first != null && !first.isBlank()) {
            return first;
        }
        if (second != null && !second.isBlank()) {
            return second;
        }
        return null;
    }

    private static TrainLossMode readTrainLossMode(String raw) {
        if (raw == null) {
            return TrainLossMode.FULL;
        }
        String normalized = raw.trim().replace('-', '_').toUpperCase();
        if ("FULL".equals(normalized)) {
            return TrainLossMode.FULL;
        }
        if ("SAMPLED".equals(normalized)) {
            return TrainLossMode.SAMPLED;
        }
        throw new IllegalArgumentException(
                "Unsupported train loss mode: " + raw + " (expected full or sampled)");
    }

    /**
     * {@link TrainingConfig} для {@link LLMTrainer} с разумными значениями по умолчанию.
     */
    public TrainingConfig toTrainingConfig() {
        return toTrainingConfig("checkpoints", vocabSize);
    }

    public TrainingConfig toTrainingConfig(String checkpointDir) {
        return toTrainingConfig(checkpointDir, vocabSize);
    }

    /**
     * @param modelVocabSize фактический размер словаря (например {@link BPETokenizer#getVocabSize()} после train)
     *     <p>Если эффективен {@link #effectiveFullGpuTrainStepFromEnv()}, запросы device logits / decoder
     *     приводятся к {@code true}, чтобы не оставался частичный GPU-путь.
     */
    public TrainingConfig toTrainingConfig(String checkpointDir, int modelVocabSize) {
        TensorOpsGPU.requireCuda("LLMConfig.toTrainingConfig");
        if (gpuE2eTrainFromEnv()) {
            if (!TensorOpsGPU.isGpuAvailable()) {
                throw new IllegalStateException(
                        "JGPT_GPU_E2E_TRAIN requires CUDA (GPU not available).");
            }
            if (!effectiveGpuResidentTraining()) {
                throw new IllegalStateException(
                        "JGPT_GPU_E2E_TRAIN requires GPU resident training (CUDA and JGPT_TRAIN_GPU_RESIDENT not 0/false)");
            }
            ensureDecoderGpuPipelineForFullGpuTrainRequest();
            return new TrainingConfig(
                    modelVocabSize,
                    maxSeqLen,
                    dModel,
                    numHeads,
                    numLayers,
                    dIntermediate,
                    batchSize,
                    accumulationSteps,
                    epochs,
                    learningRate,
                    0.1f,
                    0.01f,
                    1.0f,
                    500,
                    100,
                    LearningRateSchedule.COSINE,
                    0f,
                    checkpointDir,
                    50,
                    interactiveEveryFromEnv(200),
                    3,
                    true,
                    1e-8f,
                    8,
                    true,
                    true,
                    true,
                    true,
                    false,
                    trainLossModeFromEnvOrProp(),
                    sampledCeCandidatesFromEnv(),
                    sampledCeNegativeModeFromEnvOrProp());
        }
        boolean useGpu = effectiveGpuResidentTraining();
        boolean fullStep = effectiveFullGpuTrainStepFromEnv();
        boolean deviceLogits = useGpu && deviceLogitsTrainStepFromEnv();
        boolean deviceDec = useGpu && deviceDecoderBackwardFromEnv();
        if (useGpu && TensorOpsGPU.isGpuAvailable()) {
            // Единый device-путь обучения: без «полупутей» (host decoder backward + H2D кэша).
            fullStep = true;
            deviceLogits = true;
            deviceDec = true;
            ensureDecoderGpuPipelineForFullGpuTrainRequest();
        } else if (fullStep) {
            deviceLogits = true;
            deviceDec = true;
            ensureDecoderGpuPipelineForFullGpuTrainRequest();
        }
        return new TrainingConfig(
                modelVocabSize,
                maxSeqLen,
                dModel,
                numHeads,
                numLayers,
                dIntermediate,
                batchSize,
                accumulationSteps,
                epochs,
                learningRate,
                0.1f,
                0.01f,
                1.0f,
                500,
                100,
                LearningRateSchedule.COSINE,
                0f,
                checkpointDir,
                50,
                interactiveEveryFromEnv(200),
                3,
                true,
                1e-8f,
                8,
                useGpu,
                fullStep,
                deviceLogits,
                deviceDec,
                false,
                trainLossModeFromEnvOrProp(),
                sampledCeCandidatesFromEnv(),
                sampledCeNegativeModeFromEnvOrProp());
    }

    /** Должно совпадать с {@link GPTModel#countParameters()} для тех же гиперпараметров. */
    public long estimateParameters() {
        long params = 0;
        params += (long) vocabSize * dModel;
        params += (long) maxSeqLen * dModel;
        long perLayer =
                4L * dModel * dModel
                        + 2L * dModel * dIntermediate
                        + (long) dIntermediate * dModel
                        + 2L * dModel;
        params += numLayers * perLayer;
        params += (long) dModel;
        params += (long) dModel * vocabSize;
        return params;
    }

    @Override
    public String toString() {
        return String.format(
                "%s Model: ~%,d params, seq=%d, layers=%d, heads=%d",
                name, estimateParameters(), maxSeqLen, numLayers, numHeads);
    }
}
