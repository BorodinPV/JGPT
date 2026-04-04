package com.veles.llm.jgpt.training;

/**
 * Конфигурация обучения LLM.
 */
public final class TrainingConfig {

    public final int vocabSize;
    public final int maxSeqLen;
    public final int dModel;
    public final int numHeads;
    public final int numLayers;
    public final int dIntermediate;

    public final int batchSize;
    /** Микробатчей на один шаг оптимизатора; градиенты CE масштабируются как 1/N. */
    public final int accumulationSteps;
    public final int epochs;
    public final float learningRate;
    public final float warmupRatio;
    public final float weightDecay;
    public final float maxGradNorm;

    public final int saveEverySteps;
    public final int evalEverySteps;
    /** Режим LR после warmup (см. {@link LearningRateScheduler}). */
    public final LearningRateSchedule lrSchedule;
    /**
     * Нижняя граница LR как доля от {@link #learningRate} в конце (косинус/линейный/inv-sqrt);
     * {@code 0} — для косинуса конец уходит в 0 (как раньше).
     */
    public final float minLrRatio;
    public final String checkpointDir;
    /** Лог train loss / lr не чаще чем раз в N шагов оптимизатора (минимум 1). */
    public final int logEverySteps;
    /**
     * Раз в N шагов — автоматическая генерация по короткому русскому промпту; {@code 0} — выключено.
     */
    public final int interactiveSampleEverySteps;

    /**
     * Ранний останов: {@code 0} — выкл. Иначе число подряд eval без нового best eval loss (perplexity не
     * улучшается).
     */
    public final int earlyStopEvalPatience;

    /**
     * Если {@code true}: останов при росте eval loss при одновременном снижении train loss относительно
     * предыдущего eval (признак переобучения).
     */
    public final boolean earlyStopTrainDownEvalUp;

    /**
     * Ранний останов при «затухании» градиентов: минимальная глобальная L2-норма после unscale; {@code 0f} —
     * не проверять.
     */
    public final float earlyStopMinGradNorm;

    /**
     * Число подряд успешных шагов оптимизатора с нормой &lt; {@link #earlyStopMinGradNorm} до останова.
     */
    public final int earlyStopGradNormPatience;

    /**
     * Запрос «GPU-резидентного» цикла: требуется CUDA и {@link com.veles.llm.jgpt.model.GPTModel} с
     * {@code gpuResident=true}; VRAM-копии весов синхронизируются после шага Adam. Полный forward/backward на
     * device — по мере готовности (см. {@code docs/ROADMAP_GPU_RESIDENT.md}).
     */
    public final boolean useGpuResident;

    /**
     * Полный GPU-шаг обучения: forward/backward/clip/Adam на VRAM без лишнего скачивания весов на хост.
     * Требует {@link #useGpuResident} и {@link com.veles.llm.jgpt.model.GPTModel#canFullGpuTrain()}.
     * В приложении при запросе full GPU из env см. {@link com.veles.llm.jgpt.training.LLMConfig#toTrainingConfig}:
     * там же включаются device logits и device decoder без «полупутей».
     */
    public final boolean fullGpuTrainStep;

    /**
     * CE + backward LM head на device (логиты и ∂logits остаются на VRAM).
     * Автоматически выключается если {@code !fullGpuTrainStep}.
     */
    public final boolean deviceLogitsTrainStep;

    /**
     * Backward декодерных блоков на VRAM (grad ping-pong, device activation cache).
     * Автоматически выключается если {@code !deviceLogitsTrainStep}.
     */
    public final boolean deviceDecoderBackward;

    /**
     * Только для тестов/режима сравнения: единый device logits + decoder backward при {@code fullGpuTrainStep=false}
     * и оптимизация через {@link LLMTrainer} по пути merge-first (D2H перед Adam на хосте).
     * Несовместимо с {@code fullGpuTrainStep=true}.
     */
    public final boolean mergeFirstGpuResidentTrain;

    /** Train-only loss: full-vocab CE или sampled candidate loss. Eval всегда считает full-vocab CE. */
    public final TrainLossMode trainLossMode;

    /** Число кандидатов на строку в sampled train loss (target + negatives), используется только при {@link #trainLossMode}. */
    public final int sampledCeCandidates;

    /** Режим выбора negative candidates для sampled train loss. */
    public final SampledNegativeMode sampledCeNegativeMode;

    /** Полная конфигурация с train-loss режимом, ранним остановом и GPU-флагами. */
    public TrainingConfig(
            int vocabSize,
            int maxSeqLen,
            int dModel,
            int numHeads,
            int numLayers,
            int dIntermediate,
            int batchSize,
            int accumulationSteps,
            int epochs,
            float learningRate,
            float warmupRatio,
            float weightDecay,
            float maxGradNorm,
            int saveEverySteps,
            int evalEverySteps,
            LearningRateSchedule lrSchedule,
            float minLrRatio,
            String checkpointDir,
            int logEverySteps,
            int interactiveSampleEverySteps,
            int earlyStopEvalPatience,
            boolean earlyStopTrainDownEvalUp,
            float earlyStopMinGradNorm,
            int earlyStopGradNormPatience,
            boolean useGpuResident,
            boolean fullGpuTrainStep,
            boolean deviceLogitsTrainStep,
            boolean deviceDecoderBackward,
            boolean mergeFirstGpuResidentTrain,
            TrainLossMode trainLossMode,
            int sampledCeCandidates,
            SampledNegativeMode sampledCeNegativeMode) {
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.dIntermediate = dIntermediate;
        this.batchSize = batchSize;
        this.accumulationSteps = Math.max(1, accumulationSteps);
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.warmupRatio = warmupRatio;
        this.weightDecay = weightDecay;
        this.maxGradNorm = maxGradNorm;
        this.saveEverySteps = saveEverySteps;
        this.evalEverySteps = evalEverySteps;
        this.lrSchedule = lrSchedule != null ? lrSchedule : LearningRateSchedule.COSINE;
        this.minLrRatio = clampRatio(minLrRatio);
        this.checkpointDir = checkpointDir;
        this.logEverySteps = Math.max(1, logEverySteps);
        this.interactiveSampleEverySteps = Math.max(0, interactiveSampleEverySteps);
        this.earlyStopEvalPatience = Math.max(0, earlyStopEvalPatience);
        this.earlyStopTrainDownEvalUp = earlyStopTrainDownEvalUp;
        this.earlyStopMinGradNorm = Math.max(0f, earlyStopMinGradNorm);
        this.earlyStopGradNormPatience = Math.max(0, earlyStopGradNormPatience);
        this.useGpuResident = useGpuResident;
        this.fullGpuTrainStep = fullGpuTrainStep;
        this.mergeFirstGpuResidentTrain = mergeFirstGpuResidentTrain;
        if (mergeFirstGpuResidentTrain) {
            if (fullGpuTrainStep) {
                throw new IllegalArgumentException(
                        "mergeFirstGpuResidentTrain requires fullGpuTrainStep=false");
            }
            this.deviceLogitsTrainStep = deviceLogitsTrainStep;
            this.deviceDecoderBackward = deviceDecoderBackward && this.deviceLogitsTrainStep;
        } else {
            this.deviceLogitsTrainStep = deviceLogitsTrainStep && this.fullGpuTrainStep;
            this.deviceDecoderBackward = deviceDecoderBackward && this.deviceLogitsTrainStep;
        }
        this.trainLossMode = trainLossMode != null ? trainLossMode : TrainLossMode.FULL;
        this.sampledCeCandidates = Math.max(2, sampledCeCandidates);
        this.sampledCeNegativeMode =
                sampledCeNegativeMode != null
                        ? sampledCeNegativeMode
                        : SampledNegativeMode.BATCH_SHARED_UNIFORM;
    }

    /** Полная конфигурация (ранний останов + GPU-флаги). */
    public TrainingConfig(
            int vocabSize,
            int maxSeqLen,
            int dModel,
            int numHeads,
            int numLayers,
            int dIntermediate,
            int batchSize,
            int accumulationSteps,
            int epochs,
            float learningRate,
            float warmupRatio,
            float weightDecay,
            float maxGradNorm,
            int saveEverySteps,
            int evalEverySteps,
            LearningRateSchedule lrSchedule,
            float minLrRatio,
            String checkpointDir,
            int logEverySteps,
            int interactiveSampleEverySteps,
            int earlyStopEvalPatience,
            boolean earlyStopTrainDownEvalUp,
            float earlyStopMinGradNorm,
            int earlyStopGradNormPatience,
            boolean useGpuResident,
            boolean fullGpuTrainStep,
            boolean deviceLogitsTrainStep,
            boolean deviceDecoderBackward,
            boolean mergeFirstGpuResidentTrain) {
        this(
                vocabSize,
                maxSeqLen,
                dModel,
                numHeads,
                numLayers,
                dIntermediate,
                batchSize,
                accumulationSteps,
                epochs,
                learningRate,
                warmupRatio,
                weightDecay,
                maxGradNorm,
                saveEverySteps,
                evalEverySteps,
                lrSchedule,
                minLrRatio,
                checkpointDir,
                logEverySteps,
                interactiveSampleEverySteps,
                earlyStopEvalPatience,
                earlyStopTrainDownEvalUp,
                earlyStopMinGradNorm,
                earlyStopGradNormPatience,
                useGpuResident,
                fullGpuTrainStep,
                deviceLogitsTrainStep,
                deviceDecoderBackward,
                mergeFirstGpuResidentTrain,
                TrainLossMode.FULL,
                128,
                SampledNegativeMode.BATCH_SHARED_UNIFORM);
    }

    /** Конструктор без GPU-флагов (обратная совместимость с ранним остановом). */
    public TrainingConfig(
            int vocabSize,
            int maxSeqLen,
            int dModel,
            int numHeads,
            int numLayers,
            int dIntermediate,
            int batchSize,
            int accumulationSteps,
            int epochs,
            float learningRate,
            float warmupRatio,
            float weightDecay,
            float maxGradNorm,
            int saveEverySteps,
            int evalEverySteps,
            LearningRateSchedule lrSchedule,
            float minLrRatio,
            String checkpointDir,
            int logEverySteps,
            int interactiveSampleEverySteps,
            int earlyStopEvalPatience,
            boolean earlyStopTrainDownEvalUp,
            float earlyStopMinGradNorm,
            int earlyStopGradNormPatience,
            boolean useGpuResident) {
        this(
                vocabSize,
                maxSeqLen,
                dModel,
                numHeads,
                numLayers,
                dIntermediate,
                batchSize,
                accumulationSteps,
                epochs,
                learningRate,
                warmupRatio,
                weightDecay,
                maxGradNorm,
                saveEverySteps,
                evalEverySteps,
                lrSchedule,
                minLrRatio,
                checkpointDir,
                logEverySteps,
                interactiveSampleEverySteps,
                earlyStopEvalPatience,
                earlyStopTrainDownEvalUp,
                earlyStopMinGradNorm,
                earlyStopGradNormPatience,
                useGpuResident,
                false,
                false,
                false,
                false,
                TrainLossMode.FULL,
                128,
                SampledNegativeMode.BATCH_SHARED_UNIFORM);
    }

    /** Без раннего останова и GPU-флагов (обратная совместимость). */
    public TrainingConfig(
            int vocabSize,
            int maxSeqLen,
            int dModel,
            int numHeads,
            int numLayers,
            int dIntermediate,
            int batchSize,
            int accumulationSteps,
            int epochs,
            float learningRate,
            float warmupRatio,
            float weightDecay,
            float maxGradNorm,
            int saveEverySteps,
            int evalEverySteps,
            LearningRateSchedule lrSchedule,
            float minLrRatio,
            String checkpointDir,
            int logEverySteps,
            int interactiveSampleEverySteps) {
        this(
                vocabSize,
                maxSeqLen,
                dModel,
                numHeads,
                numLayers,
                dIntermediate,
                batchSize,
                accumulationSteps,
                epochs,
                learningRate,
                warmupRatio,
                weightDecay,
                maxGradNorm,
                saveEverySteps,
                evalEverySteps,
                lrSchedule,
                minLrRatio,
                checkpointDir,
                logEverySteps,
                interactiveSampleEverySteps,
                0,
                false,
                0f,
                0,
                false,
                false,
                false,
                false,
                false);
    }

    private static float clampRatio(float r) {
        if (r < 0f) {
            return 0f;
        }
        if (r > 1f) {
            return 1f;
        }
        return r;
    }

    /** Конфигурация по умолчанию для мини-модели */
    public static TrainingConfig defaultMini() {
        return new TrainingConfig(
                1000,
                128,
                128,
                8,
                4,
                512,
                16,
                1,
                10,
                0.001f,
                0.1f,
                0.01f,
                1.0f,
                100,
                50,
                LearningRateSchedule.COSINE,
                0f,
                "checkpoints",
                50,
                0,
                0,
                false,
                0f,
                0,
                false,
                false,
                false,
                false,
                false);
    }

    public boolean usesSampledTrainLoss() {
        return trainLossMode.usesSampledCandidates();
    }

    @Override
    public String toString() {
        return String.format(
                "TrainingConfig{vocab=%d, seq=%d, d_model=%d, heads=%d, layers=%d, batch=%d, accum=%d, epochs=%d, lr=%.4f, lr_sched=%s, min_lr_r=%.4f, log_every=%d, interactive_every=%d, early_stop_eval_p=%d, early_stop_overfit=%s, early_stop_grad_min=%s, early_stop_grad_p=%d, gpu_resident=%s, full_gpu_step=%s, device_logits=%s, device_decoder_bwd=%s, merge_first_gpu=%s, train_loss=%s, sampled_candidates=%d, sampled_negative_mode=%s}",
                vocabSize,
                maxSeqLen,
                dModel,
                numHeads,
                numLayers,
                batchSize,
                accumulationSteps,
                epochs,
                learningRate,
                lrSchedule,
                minLrRatio,
                logEverySteps,
                interactiveSampleEverySteps,
                earlyStopEvalPatience,
                earlyStopTrainDownEvalUp,
                earlyStopMinGradNorm > 0f ? String.format("%.1e", earlyStopMinGradNorm) : "off",
                earlyStopGradNormPatience,
                useGpuResident,
                fullGpuTrainStep,
                deviceLogitsTrainStep,
                deviceDecoderBackward,
                mergeFirstGpuResidentTrain,
                trainLossMode,
                sampledCeCandidates,
                sampledCeNegativeMode);
    }
}
