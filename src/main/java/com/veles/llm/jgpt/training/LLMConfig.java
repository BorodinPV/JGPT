package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.model.GPTModel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Конфигурации моделей разного размера (совместимо с {@link GPTModel} / {@link TrainingConfig}).
 */
public final class LLMConfig {

    private static final Logger log = LoggerFactory.getLogger(LLMConfig.class);

    private static final String ENV_FALSE = "false";

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
     * Каноническая геометрия AllBooks / InferChat (~34.9M по {@link #estimateParameters()}):
     * vocab 8000, seq 1024, d_model 384, 24 головы (d_head=16), 12 слоёв, SwiGLU d_intermediate=1536.
     * Пресеты {@code env/*.env} могут сменить batch/seq/layers/width ({@code JGPT_D_MODEL}/{@code JGPT_NUM_HEADS}/{@code JGPT_D_INTERMEDIATE}) через {@code JGPT_*}; без override это и есть train.
     */
    public static LLMConfig canonical() {
        return new LLMConfig(
                "JGPT-35M",
                8000,
                1024,
                384,
                24,
                12,
                1536,
                1,
                6,
                0.0005f,
                20);
    }

    /**
     * Историческое имя; то же, что {@link #canonical()}.
     *
     * @deprecated используйте {@link #canonical()}
     */
    @Deprecated(since = "1.0", forRemoval = false)
    public static LLMConfig smart50M() {
        return canonical();
    }

    /**
     * @deprecated Заменён на {@link #small16M()} (≈16M, контекст 512).
     */
    @Deprecated(since = "1.0", forRemoval = false)
    public static LLMConfig small18M() {
        return small16M();
    }

    /**
     * Переопределение {@link #learningRate}: {@code JGPT_LEARNING_RATE} или краткий псевдоним {@code JGPT_LR}
     * (положительное конечное число; десятичный разделитель «.» или «,»). Удобно для дообучения на плато.
     *
     * <p>Пример: {@code JGPT_LEARNING_RATE=1e-4 ./scripts/linux/jgpt-smart.sh}
     */
    public static LLMConfig applyLearningRateOverrideFromEnv(LLMConfig base) {
        float lr = readLearningRateFromEnvOrDefault(base.learningRate);
        if (lr == base.learningRate) {
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
                lr,
                base.epochs);
    }

    private static float readLearningRateFromEnvOrDefault(float defaultValue) {
        String raw = firstNonBlank(System.getenv("JGPT_LEARNING_RATE"), System.getenv("JGPT_LR"));
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        try {
            float v = Float.parseFloat(raw.trim().replace(',', '.'));
            if (v > 0f && Float.isFinite(v)) {
                return v;
            }
        } catch (NumberFormatException _) {
            // ignore invalid LR override
        }
        return defaultValue;
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
     * Позволяет уменьшить контекст без перекомпиляции. Актуально при OOM:
     * attention backward ~ {@code heads × seq² × 2 bytes × layers}.
     * При seq=1024 и 12 слоях это около 0.6 ГиБ; при seq=512 — вчетверо меньше.
     * <p>Пример: {@code JGPT_MAX_SEQ_LEN=1024 ./scripts/linux/jgpt-smart.sh}
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
     * Переопределяет {@link #vocabSize} через {@code JGPT_VOCAB_SIZE} (целевой размер BPE при обучении
     * токенизатора). Уже существующий файл токенизатора задаёт фактический vocab модели.
     *
     * <p>Пример: {@code JGPT_VOCAB_SIZE=16000 ./scripts/linux/jgpt-train-37L-sft.sh}
     */
    public static LLMConfig applyVocabSizeOverrideFromEnv(LLMConfig base) {
        int overridden = readPositiveEnvInt("JGPT_VOCAB_SIZE", base.vocabSize);
        if (overridden == base.vocabSize) {
            return base;
        }
        return new LLMConfig(
                base.name,
                overridden,
                base.maxSeqLen,
                base.dModel,
                base.numHeads,
                base.numLayers,
                base.dIntermediate,
                base.batchSize,
                base.accumulationSteps,
                base.learningRate,
                base.epochs);
    }

    public static LLMConfig applyWidthOverrideFromEnv(LLMConfig base) {
        int d = readPositiveEnvInt("JGPT_D_MODEL", base.dModel);
        int h = readPositiveEnvInt("JGPT_NUM_HEADS", base.numHeads);
        int ff = readPositiveEnvInt("JGPT_D_INTERMEDIATE", base.dIntermediate);
        if (d == base.dModel && h == base.numHeads && ff == base.dIntermediate) {
            return base;
        }
        if (h < 1 || d % h != 0) {
            throw new IllegalArgumentException(
                    "JGPT_D_MODEL=" + d + " must be divisible by JGPT_NUM_HEADS=" + h);
        }
        return new LLMConfig(
                base.name,
                base.vocabSize,
                base.maxSeqLen,
                d,
                h,
                base.numLayers,
                ff,
                base.batchSize,
                base.accumulationSteps,
                base.learningRate,
                base.epochs);
    }

    /**
     * Переопределяет число эпох через переменную окружения {@code JGPT_EPOCHS}.
     *
     * <p>Пример: {@code JGPT_EPOCHS=40 ./scripts/linux/jgpt-smart.sh}
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
     * Переопределяет {@link #accumulationSteps} через переменную окружения {@code JGPT_ACCUMULATION_STEPS}.
     *
     * <p>Микробатчей градиента на один шаг оптимизатора (см. {@link TrainingConfig#accumulationSteps}):
     * CE — global token-mean по валидным токенам окна. Пример: {@code JGPT_ACCUMULATION_STEPS=4 ./scripts/linux/jgpt-smart.sh}
     */
    public static LLMConfig applyAccumulationStepsOverrideFromEnv(LLMConfig base) {
        int overridden = readPositiveEnvInt("JGPT_ACCUMULATION_STEPS", base.accumulationSteps);
        if (overridden == base.accumulationSteps) {
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
                overridden,
                base.learningRate,
                base.epochs);
    }

    /**
     * Переопределяет {@link #numLayers} через {@code JGPT_PRESET_NUM_LAYERS}.
     * Используется в {@link com.veles.llm.jgpt.app.AllBooksTrain#main(String[])} при запуске через
     * {@code jgpt-smart.sh} ({@code env/*.env} задаёт только переменные окружения процесса).
     */
    public static LLMConfig applyPresetNumLayersOverrideFromEnv(LLMConfig base) {
        int overridden = readPositiveEnvInt("JGPT_PRESET_NUM_LAYERS", base.numLayers);
        if (overridden == base.numLayers) {
            return base;
        }
        return new LLMConfig(
                base.name,
                base.vocabSize,
                base.maxSeqLen,
                base.dModel,
                base.numHeads,
                overridden,
                base.dIntermediate,
                base.batchSize,
                base.accumulationSteps,
                base.learningRate,
                base.epochs);
    }

    /**
     * Канонический GPU-train: CUDA доступна. Обучение всегда resident + decoder pipeline + device
     * logits/backward; отдельные {@code JGPT_TRAIN_GPU_RESIDENT} / {@code JGPT_FULL_GPU_TRAIN} /
     * {@code JGPT_GPU_E2E_TRAIN} / {@code JGPT_DEVICE_LOGITS_TRAIN} / {@code JGPT_DEVICE_DECODER_BWD} /
     * {@code JGPT_DECODER_GPU_PIPELINE} больше не выбирают путь.
     */
    public static boolean canonicalGpuTrain() {
        return TensorOpsGPU.isGpuAvailable();
    }

    /**
     * Устар.: сырой разбор {@code JGPT_TRAIN_GPU_RESIDENT=1}. На путь обучения не влияет.
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
     * GPU-резидентные веса: совпадает с {@link #canonicalGpuTrain()}. {@code JGPT_TRAIN_GPU_RESIDENT=0}
     * игнорируется (предупреждение в {@link #toTrainingConfig(String, int)}).
     */
    public static boolean effectiveGpuResidentTraining() {
        return canonicalGpuTrain();
    }

    /** Устар.: сырой {@code JGPT_FULL_GPU_TRAIN} / {@code jgpt.fullGpuTrain}. На {@link #toTrainingConfig} не влияет. */
    public static boolean fullGpuTrainStepFromEnv() {
        return readBoolEnvOrProp("JGPT_FULL_GPU_TRAIN", "jgpt.fullGpuTrain");
    }

    /** Устар.: сырой флаг + CUDA. Канонический путь — {@link #canonicalGpuTrain()}. */
    public static boolean effectiveFullGpuTrainStepFromEnv() {
        return canonicalGpuTrain();
    }

    /** Устар.: сырой env/prop. Канонический путь включает device logits при CUDA. */
    public static boolean deviceLogitsTrainStepFromEnv() {
        return readBoolEnvOrPropDefaultGpuWhenUnset(
                "JGPT_DEVICE_LOGITS_TRAIN", "jgpt.deviceLogitsTrain");
    }

    /** Устар.: сырой env/prop. Канонический путь включает device decoder backward при CUDA. */
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
     * <p>Env {@code JGPT_SAMPLE_PROMPT}: пользовательский промпт для промежуточной генерации.
     * Можно задать несколько промптов через {@code |}, тогда они чередуются по шагам:
     * {@code JGPT_SAMPLE_PROMPT="он вышел из дома|весна пришла|тихая ночь"}.
     * Если не задан — используются встроенные русские промпты.
     */
    /** Целое ≥ {@code min} из env; иначе {@code defaultValue}. */
    private static int positiveIntFromEnv(String name, int defaultValue, int min) {
        String env = System.getenv(name);
        if (env == null || env.isBlank()) {
            return defaultValue;
        }
        try {
            return Math.max(min, Integer.parseInt(env.trim()));
        } catch (NumberFormatException _) {
            return defaultValue;
        }
    }

    /** {@code JGPT_SAVE_EVERY_STEPS} — период {@code checkpoint_step_N} (шагов оптимизатора). */
    public static int saveEveryStepsFromEnv(int defaultValue) {
        return positiveIntFromEnv("JGPT_SAVE_EVERY_STEPS", defaultValue, 1);
    }

    /** {@code JGPT_EVAL_EVERY_STEPS} — период eval (и checkpoint_best при улучшении). */
    public static int evalEveryStepsFromEnv(int defaultValue) {
        return positiveIntFromEnv("JGPT_EVAL_EVERY_STEPS", defaultValue, 1);
    }

    /**
     * {@code JGPT_DROPOUT} — вероятность dropout (residual после attention/FFN и embedding) в {@code [0, 0.9)};
     * не задано — {@code defaultValue}. На GPU-пути реализуется ядром dropout (см. {@code TensorOpsGPU}).
     */
    public static float dropoutFromEnv(float defaultValue) {
        String env = System.getenv("JGPT_DROPOUT");
        if (env == null || env.isBlank()) {
            return defaultValue;
        }
        try {
            float v = Float.parseFloat(env.trim().replace(',', '.'));
            if (!(v >= 0f) || v >= 0.9f) {
                return defaultValue;
            }
            return v;
        } catch (NumberFormatException _) {
            return defaultValue;
        }
    }

    public static int interactiveEveryFromEnv(int defaultValue) {
        String env = System.getenv("JGPT_INTERACTIVE_EVERY");
        if (env != null && !env.isBlank()) {
            try {
                int v = Integer.parseInt(env.trim());
                return Math.max(0, v);
            } catch (NumberFormatException _) {
                return defaultValue;
            }
        }
        String prop = System.getProperty("jgpt.interactiveEvery");
        if (prop != null && !prop.isBlank()) {
            try {
                int v = Integer.parseInt(prop.trim());
                return Math.max(0, v);
            } catch (NumberFormatException _) {
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
     * Интервал логирования train loss: env {@code JGPT_LOG_EVERY_STEPS} / prop {@code jgpt.logEverySteps}.
     * {@code 1} — каждый шаг; {@code 0} или {@code -1} — отключить. По умолчанию 1.
     */
    public static int logEveryStepsFromEnv(int defaultValue) {
        String env = System.getenv("JGPT_LOG_EVERY_STEPS");
        if (env != null && !env.isBlank()) {
            try {
                int v = Integer.parseInt(env.trim());
                return v <= 0 ? Integer.MAX_VALUE : v;
            } catch (NumberFormatException _) {
                return defaultValue;
            }
        }
        String prop = System.getProperty("jgpt.logEverySteps");
        if (prop != null && !prop.isBlank()) {
            try {
                int v = Integer.parseInt(prop.trim());
                return v <= 0 ? Integer.MAX_VALUE : v;
            } catch (NumberFormatException _) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    /**
     * Устар.: сырой {@code JGPT_GPU_E2E_TRAIN} / {@code jgpt.gpu.e2eTrain}. Канонический GPU-train включается
     * сам при CUDA; флаг на {@link #toTrainingConfig} не влияет.
     */
    public static boolean gpuE2eTrainFromEnv() {
        return readBoolEnvOrProp("JGPT_GPU_E2E_TRAIN", "jgpt.gpu.e2eTrain");
    }

    /**
     * Устар.: сырой {@code JGPT_DECODER_GPU_PIPELINE} / {@code jgpt.decoder.gpu.pipeline}. У {@link GPTModel}
     * pipeline по умолчанию включён при {@code gpuResident}; обучение не читает этот флаг.
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
     * Подробные логи указателей decoder CUDA graph (перед capture/replay, сравнение с эталоном): env {@code
     * JGPT_DECODER_LAYER_CUDA_GRAPH_LOG=1} / prop {@code jgpt.decoder.layer.cudaGraph.log}.
     */
    public static boolean decoderLayerCudaGraphDebugLogFromEnvOrProp() {
        return readBoolEnvOrProp("JGPT_DECODER_LAYER_CUDA_GRAPH_LOG", "jgpt.decoder.layer.cudaGraph.log");
    }

    /**
     * Лог VRAM после первого успешного graph launch слоя (MiB): env {@code JGPT_DECODER_CUDA_GRAPH_MEM_LOG=1} / prop
     * {@code jgpt.decoder.cudaGraph.memLog}.
     */
    public static boolean decoderCudaGraphMemLogFromEnvOrProp() {
        return readBoolEnvOrProp("JGPT_DECODER_CUDA_GRAPH_MEM_LOG", "jgpt.decoder.cudaGraph.memLog");
    }

    /**
     * Минимальный cudaMemGetInfo free (MiB) перед graph-path слоя decoder: env {@code JGPT_DECODER_GRAPH_MIN_FREE_MIB} /
     * prop {@code jgpt.decoder.graph.minFreeMib}. {@code 0} — выкл. Если {@code total − used} меньше порога, graph до
     * конца текущего forward отключается (только eager). При OOM на {@code cudaGraphLaunch} при ~134 MiB free в логах
     * можно задать, например, {@code 192}.
     */
    public static int decoderGraphMinFreeMibFromEnvOrProp() {
        String e = System.getenv("JGPT_DECODER_GRAPH_MIN_FREE_MIB");
        if (e != null && !e.isBlank()) {
            try {
                return Math.max(0, Integer.parseInt(e.trim()));
            } catch (NumberFormatException _) {
                return 0;
            }
        }
        String p = System.getProperty("jgpt.decoder.graph.minFreeMib");
        if (p != null && !p.isBlank()) {
            try {
                return Math.max(0, Integer.parseInt(p.trim()));
            } catch (NumberFormatException _) {
                return 0;
            }
        }
        return 0;
    }

    /** Байтовый порог для {@link #decoderGraphMinFreeMibFromEnvOrProp()}; {@code 0} — проверка выключена. */
    public static long decoderGraphMinFreeBytesFromEnvOrProp() {
        int mib = decoderGraphMinFreeMibFromEnvOrProp();
        return mib <= 0 ? 0L : (long) mib * 1024L * 1024L;
    }

    /**
     * Снимок VRAM вокруг training decoder forward (SLF4J {@code [VRAM] decoderBefore/After}): env {@code JGPT_TRAIN_VRAM_STEP_PROBE=1} /
     * prop {@code jgpt.train.vramStepProbe}. Интервал по счётчику вызовов {@link com.veles.llm.jgpt.model.GPTModel#forwardGpuDecoder}:
     * env {@code JGPT_TRAIN_VRAM_STEP_PROBE_EVERY} / prop {@code jgpt.train.vramStepProbeEvery} (по умолчанию {@code 50}).
     *
     * <p>Интерпретация: рост поля {@code used} на последовательных {@code decoderBefore} — намёк на накопление между
     * шагами; большой скачок только между {@code decoderBefore} и {@code decoderAfter} на одном {@code seq} — пик внутри
     * forward, а не обязательно утечка.
     */
    public static boolean trainVramStepProbeFromEnvOrProp() {
        return readBoolEnvOrProp("JGPT_TRAIN_VRAM_STEP_PROBE", "jgpt.train.vramStepProbe");
    }

    public static int trainVramStepProbeEveryFromEnvOrProp() {
        return readPositiveEnvOrPropInt(
                "JGPT_TRAIN_VRAM_STEP_PROBE_EVERY", "jgpt.train.vramStepProbeEvery", 50);
    }

    /**
     * Один JNI для второго RMSNorm + проекций SwiGLU W1/W3: env {@code JGPT_FUSED_FFN_RMS_W1W3=1} / prop {@code
     * jgpt.fused.ffn.rms.w1w3}.
     */
    public static boolean fusedFfnRmsW1W3FromEnvOrProp() {
        return readBoolEnvOrProp("JGPT_FUSED_FFN_RMS_W1W3", "jgpt.fused.ffn.rms.w1w3");
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
     * Переопределяет терпение раннего останова через {@code JGPT_EARLY_STOP_EVAL_PATIENCE}.
     * {@code 0} — отключить останов по отсутствию улучшения eval loss.
     * По умолчанию {@code defaultValue}.
     */
    public static int earlyStopEvalPatienceFromEnv(int defaultValue) {
        return readNonNegativeEnvInt("JGPT_EARLY_STOP_EVAL_PATIENCE", defaultValue);
    }

    /**
     * Переопределяет проверку переобучения (train↓ + eval↑) через {@code JGPT_EARLY_STOP_OVERFIT}.
     * {@code 0}/{@code false} — отключить. По умолчанию {@code defaultValue}.
     */
    public static boolean earlyStopOverfitFromEnv(boolean defaultValue) {
        String e = System.getenv("JGPT_EARLY_STOP_OVERFIT");
        if (e == null || e.isBlank()) {
            return defaultValue;
        }
        String t = e.trim();
        return isTruthyToken(t) || (!isFalsyToken(t) && defaultValue);
    }

    private static int readNonNegativeEnvInt(String key, int defaultValue) {
        String e = System.getenv(key);
        if (e != null && !e.isBlank()) {
            try {
                int v = Integer.parseInt(e.trim());
                return Math.max(0, v);
            } catch (NumberFormatException _) {
                // ignore invalid env int
            }
        }
        return defaultValue;
    }

    /**
     * Явное {@code 0}/{@code false} — выкл.; {@code 1}/{@code true} — вкл.; не задано ни env, ни property —
     * {@link TensorOpsGPU#isGpuAvailable()} (предпочитать GPU-реализацию, если она есть).
     */
    private static boolean isFalsyToken(String t) {
        return "0".equals(t) || ENV_FALSE.equalsIgnoreCase(t);
    }

    private static boolean isTruthyToken(String t) {
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    private static boolean readBoolEnvOrPropDefaultGpuWhenUnset(String envKey, String propKey) {
        String e = System.getenv(envKey);
        if (e != null && !e.isBlank()) {
            return isTruthyToken(e.trim());
        }
        String p = System.getProperty(propKey);
        if (p != null && !p.isBlank()) {
            return isTruthyToken(p.trim());
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
        } catch (NumberFormatException _) {
            return defaultValue;
        }
    }

    private static int readPositiveEnvOrPropInt(String envKey, String propKey, int defaultValue) {
        String env = System.getenv(envKey);
        if (env != null && !env.isBlank()) {
            try {
                int parsed = Integer.parseInt(env.trim());
                return parsed > 0 ? parsed : defaultValue;
            } catch (NumberFormatException _) {
                return defaultValue;
            }
        }
        String prop = System.getProperty(propKey);
        if (prop != null && !prop.isBlank()) {
            try {
                int parsed = Integer.parseInt(prop.trim());
                return parsed > 0 ? parsed : defaultValue;
            } catch (NumberFormatException _) {
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
     *     <p>Канонический GPU-train: resident + полный шаг + device logits + device decoder. Наследие
     *     {@code JGPT_FULL_GPU_TRAIN} / {@code JGPT_GPU_E2E_TRAIN} / {@code JGPT_DEVICE_*} /
     *     {@code JGPT_DECODER_GPU_PIPELINE} не меняет путь.
     */
    public TrainingConfig toTrainingConfig(String checkpointDir, int modelVocabSize) {
        TensorOpsGPU.requireCuda("LLMConfig.toTrainingConfig");
        warnIfLegacyGpuTrainFlagsSet();
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
                0.1f,
                1.0f,
                dropoutFromEnv(0.1f),
                0f,
                dropoutFromEnv(0.1f),
                saveEveryStepsFromEnv(500),
                evalEveryStepsFromEnv(100),
                LearningRateSchedule.COSINE,
                0f,
                checkpointDir,
                logEveryStepsFromEnv(1),
                interactiveEveryFromEnv(200),
                earlyStopEvalPatienceFromEnv(3),
                earlyStopOverfitFromEnv(true),
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

    private static void warnIfLegacyGpuTrainFlagsSet() {
        warnLegacyGpuFlagIfOff("JGPT_TRAIN_GPU_RESIDENT", null);
        warnLegacyGpuFlagIfOff("JGPT_FULL_GPU_TRAIN", "jgpt.fullGpuTrain");
        warnLegacyGpuFlagIfOff("JGPT_GPU_E2E_TRAIN", "jgpt.gpu.e2eTrain");
        warnLegacyGpuFlagIfOff("JGPT_DEVICE_LOGITS_TRAIN", "jgpt.deviceLogitsTrain");
        warnLegacyGpuFlagIfOff("JGPT_DEVICE_DECODER_BWD", "jgpt.deviceDecoderBackward");
        warnLegacyGpuFlagIfOff("JGPT_DECODER_GPU_PIPELINE", "jgpt.decoder.gpu.pipeline");
    }

    private static void warnLegacyGpuFlagIfOff(String envKey, String propKey) {
        if (explicitlyOff(System.getenv(envKey)) || (propKey != null && explicitlyOff(System.getProperty(propKey)))) {
            log.warn(
                    "{}=0/false игнорируется: канонический GPU-train всегда полный путь при CUDA",
                    envKey);
        }
    }

    private static boolean explicitlyOff(String raw) {
        if (raw == null || raw.isBlank()) {
            return false;
        }
        String t = raw.trim();
        return "0".equals(t) || ENV_FALSE.equalsIgnoreCase(t);
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
