package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.GpuIntBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.cuda.TensorCudaLibrary;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.BlockActivationCacheDevice;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.ops.GpuWorkspaceCleanup;
import com.veles.llm.jgpt.ops.TensorOps;
import com.veles.llm.jgpt.util.DebugGpuTrain;
import com.veles.llm.jgpt.util.LogFmt;

import java.util.Map;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Trainer: forward, CE loss, градиент по logits, backward через LM head, AdamW, clipping.
 * Поддерживается {@link TrainingConfig#accumulationSteps}: CE и backward с масштабом 1/N, шаг оптимизатора
 * после N микробатчей (в конце эпохи — неполная группа с поправкой N/r к градиентам параметров).
 *
 * <p><b>Скаляр CE:</b> все ветки loss+∂logits возвращают <em>среднее</em> по токенам. Fused GPU (хостовые
 * логиты) и device-CE получают это значение напрямую из JNI; отдельный D2H-буфер только для скаляра loss не
 * используется. Множитель для ∂logits в fused-путях: {@code microbatchGradScale * lossScale / numTokens},
 * {@code lossScale} — текущий {@link DynamicLossScaler} при FP16 matmul (см. {@link LlmTrainerCrossEntropy#ceFusedGradScaleOverTotal}).
 */
public final class LLMTrainer {

    private static final Logger log = LoggerFactory.getLogger(LLMTrainer.class);


    /**
     * После GPU-инференса вне шага обучения (промежуточная генерация, и т.п.): тот же сброс, что
     * после {@code generateGpuKv}, иначе первый следующий train-step иногда даёт нечисловые градиенты при том же
     * потоке CUDA и decoder graphs.
     */
    void synchronizeTrainingPipelineAfterGpuAuxiliaryInfer(String reason) {
        if (!model.isGpuResident() || !TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        model.prepareForTrainingAfterInteractiveGeneration();
        sampledTrainCandidatesPerRow = 0;
        /* Инвариант перед train: не тащить ∂ параметров и pending с инференса (KV/CE gradScale=0 всё равно могут
         * оставить мусор на путях fusion). Затем барьер stream — см. Коммент к overflow после eval. */
        model.zeroGradParameters();
        model.zeroGpuTrainableParameterGrads();
        gpuTrainableParamGradsKnownClean = true;
        GpuPendingGradients.clearAllPendingGpuBuffers();
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
        TensorOpsGPU.synchronizeStream();
        TensorOpsGPU.drainDeferredGpuBuffers();
        /* После eval/sample infer: trim async memory pools — иначе фрагментация и задержка возврата блоков после
         * cudaFreeAsync часто дают ложный OOM на следующем cudaMallocAsync/cudaMalloc. */
        TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
        if (fp16Matmul && dynamicLossScaler != null) {
            dynamicLossScaler.resetConsecutiveNonOverflowAfterAuxiliaryGpuWork();
            if (LlmTrainerGpuUtils.fp16AuxSoftenScaleAfterInfer()) {
                float s0 = dynamicLossScaler.getScale();
                dynamicLossScaler.softenScaleAfterAuxiliaryGpuWork(reason);
                float s1 = dynamicLossScaler.getScale();
                if (DebugGpuTrain.isEnabled()) {
                    LlmTrainerDebugLog.b39372(
                            "H_aux_soften",
                            "LLMTrainer.synchronizeTrainingPipelineAfterGpuAuxiliaryInfer",
                            "post_fp16_soften",
                            "{\"globalStep\":"
                                    + globalStep
                                    + ",\"reason\":\""
                                    + LlmTrainerDebugLog.jsonEsc(reason)
                                    + "\",\"scaleBefore\":"
                                    + s0
                                    + ",\"scaleAfter\":"
                                    + s1
                                    + "}");
                }
            }
        }
        if (DebugGpuTrain.isEnabled()) {
            LlmTrainerDebugLog.b39372(
                    "H_aux_fence",
                    "LLMTrainer",
                    "post_aux_infer_reset",
                    "{\"globalStep\":"
                            + globalStep
                            + ",\"reason\":\""
                            + LlmTrainerDebugLog.jsonEsc(reason)
                            + "\"}");
        }
    }

    /**
     * После завершения сегмента обучения (например между книгами в цепочке): освобождает VRAM тренера и модели до выхода из
     * области видимости, чтобы не полагаться на {@code finalize} у {@link GpuTensor}/{@link GpuFloatBuffer}.
     */
    public void releaseGpuResourcesAfterBook() {
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
        }
        GpuPendingGradients.cleanupThreadLocal();
            GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
        optimizer.releaseGpuMomentBuffers();
        if (ceTargetsDevice != null) {
            ceTargetsDevice.close();
            ceTargetsDevice = null;
            ceTargetsCapRows = 0;
        }
        if (sampledCandidateIdsDevice != null) {
            sampledCandidateIdsDevice.close();
            sampledCandidateIdsDevice = null;
            sampledCandidateIdsCapElems = 0;
        }
        if (sampledCandidateLogitsDevice != null) {
            sampledCandidateLogitsDevice.close();
            sampledCandidateLogitsDevice = null;
        }
        if (sampledCandidateGradDevice != null) {
            sampledCandidateGradDevice.close();
            sampledCandidateGradDevice = null;
            sampledCandidateFloatCapElems = 0;
        }
        sampledTrainCandidatesPerRow = 0;
        model.clearSampledTrainLossGrad();
        model.closeGpuResidentWeights();
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
            TensorOpsGPU.drainDeferredGpuBuffers();
            TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
            TensorOpsGPU.cleanupCudaThreadResources();
        }
        if (DebugGpuTrain.isEnabled()) {
            LlmTrainerDebugLog.b39372(
                    "H_book_handoff",
                    "LLMTrainer.releaseGpuResourcesAfterBook",
                    "after_book_cleanup",
                    "{}");
        }
    }

    final GPTModel model;
    final TrainingConfig config;
    final DataLoader dataLoader;
    /**
     * Отдельный hold-out для; {@code null} — как раньше, eval идёт по батчам train-loader
     * (с сохранением/восстановлением индекса).
     */
    final DataLoader evalDataLoader;
    final AdamOptimizer optimizer;
    int globalStep;
    float bestLoss;
    /**
     * Линия отсчёта для внешнего graceful shutdown ({@code AllBooksTrain}): после загрузки чекпоинта совпадает с
     * {@link #globalStep}; после {@link #restartOptimizerScheduleForNewPlan} и {@link #resetGlobalStep} — с текущим
     * {@code globalStep} (обычно 0). Иначе сравнение «прогресс после restart_schedule» с устаревшим шагом из файла
     * даёт ложное «нет прогресса» и не обновляет {@code checkpoint_final.bin}.
     */
    int shutdownProgressBaselineStep = 0;
    /**
     * После {@link #loadCheckpoint}: с какого индекса эпохи (0-based) начинать внешний цикл в {@link #train()}.
     * В чекпоинт пишется {@link #pendingCheckpointEpochIndex} — номер эпохи для следующего запуска после
     * полного завершения эпохи {@code epoch} это {@code epoch + 1}; внутри незавершённой эпохи — {@code epoch}.
     */
    int loadedResumeEpochIndex = 0;

    /** Значение, сериализуемое в следующий {@link #saveCheckpoint(String)} (см. {@link #loadedResumeEpochIndex}). */
    int pendingCheckpointEpochIndex = 0;

    /**
     * Индекс первой последовательности текущего батча в {@link DataLoader} ({@link DataLoader#getCurrentIndex()}),
     * сериализуется в v4-чекпоинт.
     */
    int pendingCheckpointDataLoaderIndex = 0;

    /** После {@link #loadCheckpoint} (v3/v4): смещение внутри эпохи для первого прохода {@link #train()}. */
    int loadedResumeDataLoaderIndex = 0;

    /**
     * После загрузки v3/v4: на первой итерации внешнего цикла повторить {@link DataLoader#shuffle()} столько раз,
     * как при полном прогоне от начала до старта сохранённой эпохи (детерминированный {@code Random(42)}).
     */
    boolean resumeReplayCheckpointShuffles = false;

    final List<Tensor> parameters;
    /** Всего шагов оптимизатора за всё обучение: {@code epochs × ceil(numBatches / accumulationSteps)}. */
    final int totalTrainingSteps;
    /** Линейный warmup: шаги {@code 1 … warmupSteps}, доля от {@link TrainingConfig#warmupRatio}. */
    final int warmupSteps;

    /**
     * При FP16 matmul не {@code null}: динамический loss scale (overflow → уменьшить, серия успешных шагов →
     * увеличить). При обучении без FP16 matmul — {@code null}.
     */
    final DynamicLossScaler dynamicLossScaler;
    /** Ранний выход после N шагов оптимизатора (env {@code JGPT_EXIT_AFTER_STEP}). */
    final int exitAfterOptimizerSteps;
    /**
     * Периодический вызов {@link TensorOpsGPU#cudaTrimDeviceMemoryPoolsBestEffort()} каждые N успешных шагов
     * оптимизатора (см. цикл в {@link #train()}). Env {@code JGPT_CUDA_TRIM_EVERY_STEPS}; {@code 0} — выкл.;
     * по умолчанию 500 — смягчает рост allocated/фрагментации при длинных прогонах.
     */
    final int cudaTrimEveryOptimizerSteps;

    /**
     * Кэш {@link TensorOpsGPU#useFp16Matmul()} на время жизни тренера (не меняется без перезапуска JVM).
     */
    final boolean fp16Matmul;

    /**
     * {@code JGPT_CE_ASYNC}: CE в поток ставит kernel и D2H скаляра; вызывающий делает барьер перед
     * {@link GPTModel#backward} (готовность ∂logits) и после backward (готовность параметрических ∂ перед clip/step),
     * затем читает loss из pinned host (см. {@code train()}).
     */
    final boolean ceAsyncDevice;
    /** {@code JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH}: сброс loss scale в начале каждой эпохи. */
    final boolean fp16DynamicResetEachEpoch;

    /** Плановый номер шага ({@code globalStep + 1}) для агрегирования повторных overflow до успешного шага. */
    int overflowLogPlannedStepKey = -1;
    int overflowSkipRepeatCount;

    /** Переиспользуемый device-буфер int32 для CE (device logits path). */
    GpuIntBuffer ceTargetsDevice;
    int ceTargetsCapRows;
    int[] ceHostTargetScratch;
    GpuIntBuffer sampledCandidateIdsDevice;
    int sampledCandidateIdsCapElems;
    GpuFloatBuffer sampledCandidateLogitsDevice;
    GpuFloatBuffer sampledCandidateGradDevice;
    int sampledCandidateFloatCapElems;
    int[] sampledCandidateIdsHostScratch;
    int[] sampledSharedNegativeScratch;
    /** Число кандидатов на строку после (для CE без повторной подготовки). */
    int sampledTrainCandidatesPerRow;

    /** Скрач ∂logits для fused CE в (градиент не используется, {@code gradScale=0}). */
    float[] evalCeGradScratch;

    /**
     * После: ∂ обучаемых параметров на VRAM гарантированно нули.
     * Сбрасывается в (хостовый Adam без zeroGpuGrads) и при первом запуске.
     */
    boolean gpuTrainableParamGradsKnownClean;

    /** Переиспользуемые массивы для (без аллокации на шаг). */
    GpuFloatBuffer[] nonFiniteParamBufsScratch = new GpuFloatBuffer[0];
    int[] nonFiniteParamLensScratch = new int[0];

    /** Переиспользуемые массивы для sumSquares по GPU grad-буферам (без аллокации на шаг). */
    long[] sumSqPtrsScratch = new long[0];
    int[] sumSqLensScratch = new int[0];

    /** Scratch-список тензоров для FP16 unscale logits (размер <= 1, alloc пропускается если ещё нет). */
    final List<Tensor> logitsOnlyScratch = new ArrayList<>(1);

    /**
     * Если задано: после синхронного v2-чекпоинта запись model_*.bin и токенизатора уходит в фон
     * (снимок весов клонируется на heap до возврата из {@link #saveCheckpoint}).
     */
    final boolean checkpointAsyncIo;
    /** Очередь на один поток; цепочка {@link #checkpointIoTail} сохраняет порядок записей. */
    final ExecutorService checkpointIoExecutor;
    volatile CompletableFuture<Void> checkpointIoTail = CompletableFuture.completedFuture(null);

    /** Глобальная L2-норма градиентов после unscale, до/с клипом (последний успешный шаг). */
    float lastGlobalGradNorm;

    final TrainingEventCallback trainingEventCallback;
    /** {@code state/stats.json} для {@code dashboard.html}; {@code null} при {@code JGPT_STATS_JSON=0}. */
    final TrainingStatsWriter trainingStatsWriter;
    /** Запрос на выход из {@link #train()} для смены пресета супервизором. */
    volatile boolean supervisedStopRequested;
    /** {@code true}, если последний {@link #train()} вышел из‑за {@link #requestSupervisedStop()}. */
    volatile boolean exitedDueToSupervisorRequest;

    public LLMTrainer(GPTModel model, TrainingConfig config, DataLoader dataLoader) {
        this(model, config, dataLoader, null, null, null);
    }

    /**
     * @param evalDataLoader hold-out для метрик eval; {@code null} — eval на train-потоке (прежнее поведение).
     */
    public LLMTrainer(GPTModel model, TrainingConfig config, DataLoader dataLoader, DataLoader evalDataLoader) {
        this(model, config, dataLoader, evalDataLoader, null, null);
    }

    /**
     * @param trainingEventCallback колбэк метрик (может быть {@code null} — эквивалентно {@link
     *     TrainingEventCallback#NOOP})
     * @param lossScalerOverride если не {@code null}, используется вместо {@link
     *     DynamicLossScaler#fromEnvironmentIfFp16()}
     */
    public LLMTrainer(
            GPTModel model,
            TrainingConfig config,
            DataLoader dataLoader,
            TrainingEventCallback trainingEventCallback,
            DynamicLossScaler lossScalerOverride) {
        this(model, config, dataLoader, null, trainingEventCallback, lossScalerOverride);
    }

    /**
     * Полный конструктор: опциональный {@code evalDataLoader} для hold-out validation.
     */
    public LLMTrainer(
            GPTModel model,
            TrainingConfig config,
            DataLoader dataLoader,
            DataLoader evalDataLoader,
            TrainingEventCallback trainingEventCallback,
            DynamicLossScaler lossScalerOverride) {
        TensorOpsGPU.requireCuda("LLMTrainer");
        this.exitedDueToSupervisorRequest = false;
        this.model = model;
        this.config = config;
        this.dataLoader = dataLoader;
        this.evalDataLoader = evalDataLoader;
        this.trainingEventCallback =
                trainingEventCallback != null ? trainingEventCallback : TrainingEventCallback.NOOP;
        this.supervisedStopRequested = false;
        this.optimizer = AdamOptimizer.fromConfig(config);
        this.globalStep = 0;
        this.bestLoss = Float.MAX_VALUE;
        this.parameters = new ArrayList<>(model.getParameters());
        int batchesPerEpoch = Math.max(1, dataLoader.numBatches());
        int acc = config.accumulationSteps;
        int optimizerStepsPerEpoch = (batchesPerEpoch + acc - 1) / acc;
        this.totalTrainingSteps = Math.max(1, config.epochs * optimizerStepsPerEpoch);
        this.trainingStatsWriter =
                LlmTrainerEnvUtils.readBooleanEnv("JGPT_STATS_JSON", true)
                        ? new TrainingStatsWriter(Path.of("state"))
                        : null;
        if (trainingStatsWriter != null) {
            trainingStatsWriter.setConfig(
                    config,
                    totalTrainingSteps,
                    LlmTrainerEnvUtils.envTrimOrDefault("JGPT_STATS_PRESET", "-"),
                    LlmTrainerEnvUtils.envTrimOrDefault("JGPT_STATS_PRESET_IDX", "-"));
            trainingStatsWriter.setEvalDataSource(evalDataLoader != null ? "validation" : "train");
        }
        float wr = config.warmupRatio;
        if (wr > 0f) {
            this.warmupSteps = Math.max(1, (int) (totalTrainingSteps * wr));
        } else {
            this.warmupSteps = 0;
        }
        this.fp16Matmul = TensorOpsGPU.useFp16Matmul();
        this.dynamicLossScaler =
                lossScalerOverride != null ? lossScalerOverride : DynamicLossScaler.fromEnvironmentIfFp16();
        this.fp16DynamicResetEachEpoch =
                LlmTrainerEnvUtils.readBooleanEnv("JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH", false);
        boolean ceAsyncEnv = LlmTrainerEnvUtils.readBooleanEnv("JGPT_CE_ASYNC", false);
        this.ceAsyncDevice = ceAsyncEnv;
        if (ceAsyncDevice) {
            log.info(
                    "JGPT_CE_ASYNC: enqueue CE + D2H loss; synchronizeStream до backward (∂logits ready) "
                            + "и после backward (градиенты ready перед optimizer step); loss читается из pinned host.");
        }
        if (ceAsyncDevice && fp16Matmul) {
            log.info(
                    "JGPT_CE_ASYNC + FP16 matmul: pinned CE staging thread-local в native; один обучающий поток на GPU stream.");
        }
        if (fp16Matmul && dynamicLossScaler != null) {
            log.info(
                    "Динамический FP16 loss scale: старт {}×, рост каждые {} успешных шагов (макс {}×)",
                    String.format(Locale.ROOT, "%.0f", dynamicLossScaler.getScale()),
                    dynamicLossScaler.getGrowthInterval(),
                    String.format(Locale.ROOT, "%.0f", dynamicLossScaler.getMaxScale()));
            if (dynamicLossScaler.getRecoveryAfterMinStreak() > 0) {
                log.info(
                        "JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK={}: после стольких overflow подряд на min — сброс к {}×",
                        dynamicLossScaler.getRecoveryAfterMinStreak(),
                        String.format(Locale.ROOT, "%.4g", dynamicLossScaler.getBaselineScale()));
            }
            if (fp16DynamicResetEachEpoch) {
                log.info(
                        "JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH=1: в начале каждой эпохи loss scale сбрасывается к стартовому.");
            }
        }
        this.exitAfterOptimizerSteps = LlmTrainerEnvUtils.readPositiveEnvInt("JGPT_EXIT_AFTER_STEP", 0);
        this.cudaTrimEveryOptimizerSteps = LlmTrainerEnvUtils.readCudaTrimEveryOptimizerStepsFromEnv();
        if (cudaTrimEveryOptimizerSteps > 0) {
            log.info(
                    "{} JGPT_CUDA_TRIM_EVERY_STEPS={} — периодический trim пулов CUDA после успешного шага",
                    LogFmt.badge("CFG"),
                    cudaTrimEveryOptimizerSteps);
        }
        this.lastGlobalGradNorm = 0f;
        this.checkpointAsyncIo =
                LlmTrainerEnvUtils.readBooleanEnv("JGPT_CHECKPOINT_ASYNC", false)
                        || Boolean.getBoolean("jgpt.checkpoint.async");
        this.checkpointIoExecutor =
                checkpointAsyncIo
                        ? Executors.newSingleThreadExecutor(
                                r -> {
                                    Thread t = new Thread(r, "jgpt-checkpoint-io");
                                    t.setDaemon(true);
                                    return t;
                                })
                        : null;
        if (config.useGpuResident) {
            if (!TensorOpsGPU.isGpuAvailable()) {
                throw new IllegalArgumentException(
                        "TrainingConfig.useGpuResident requires CUDA (GPU not available)");
            }
            if (!model.isGpuResident()) {
                throw new IllegalArgumentException(
                        "TrainingConfig.useGpuResident requires GPTModel constructed with gpuResident=true");
            }
            if (config.mergeFirstGpuResidentTrain) {
                if (!config.deviceLogitsTrainStep || !config.deviceDecoderBackward) {
                    throw new IllegalArgumentException(
                            "mergeFirstGpuResidentTrain requires deviceLogitsTrainStep and deviceDecoderBackward");
                }
            } else if (!config.fullGpuTrainStep || !config.deviceDecoderBackward) {
                throw new IllegalArgumentException(
                        "TrainingConfig.useGpuResident with CUDA requires fullGpuTrainStep and deviceDecoderBackward "
                                + "(unified VRAM training path); use LLMConfig.toTrainingConfig() or the full TrainingConfig constructor.");
            }
        }
        if (config.fullGpuTrainStep) {
            if (!config.useGpuResident) {
                throw new IllegalArgumentException("fullGpuTrainStep requires useGpuResident");
            }
            if (!model.canFullGpuTrain()) {
                throw new IllegalArgumentException(
                        "fullGpuTrainStep requires canFullGpuTrain (decoder pipeline)");
            }
        }
        if (config.mergeFirstGpuResidentTrain && !model.canFullGpuTrain()) {
            throw new IllegalArgumentException(
                    "mergeFirstGpuResidentTrain requires canFullGpuTrain (decoder pipeline)");
        }
        if (config.usesSampledTrainLoss()) {
            if (!config.fullGpuTrainStep || !config.deviceLogitsTrainStep || !config.deviceDecoderBackward) {
                throw new IllegalArgumentException(
                        "sampled train loss requires unified full GPU path "
                                + "(fullGpuTrainStep + deviceLogitsTrainStep + deviceDecoderBackward)");
            }
            if (config.sampledCeNegativeMode != SampledNegativeMode.BATCH_SHARED_UNIFORM) {
                throw new IllegalArgumentException(
                        "sampled train loss currently supports only BATCH_SHARED_UNIFORM negatives");
            }
            if (ceAsyncDevice) {
                throw new IllegalArgumentException(
                        "sampled train loss does not support JGPT_CE_ASYNC yet; disable async CE for sampled mode");
            }
            log.info(
                    "{} train-only sampled CE: candidates={} mode={} (eval остаётся full-vocab)",
                    LogFmt.badge("LOSS"),
                    config.sampledCeCandidates,
                    config.sampledCeNegativeMode);
            log.info(
                    "{} sampled PERF accounting: candidate-id prep + candidate LM-head входят в \"прямой\"; "
                            + "\"лосс+∂CE\" покрывает только sampled CE",
                    LogFmt.badge("PERF"));
        }
        if (config.deviceLogitsTrainStep) {
            model.setDeviceLogitsEnabled(true);
        }
        if (config.deviceDecoderBackward) {
            model.setDeviceDecoderBackward(true);
        }
    }

    /** Сигнал {@link #train()} завершить цикл и выйти (сохранение — на усмотрение вызывающего). */
    public void requestSupervisedStop() {
        this.supervisedStopRequested = true;
    }

    public boolean exitedDueToSupervisorRequest() {
        return exitedDueToSupervisorRequest;
    }

    /** Масштаб для CE и градиента по логитам (текущий шаг оптимизатора / накопление). */
    float lossScaleForForward() {
        if (!fp16Matmul) {
            return 1f;
        }
        return dynamicLossScaler.getScale();
    }

    /**
     * Эффективный LR: линейный warmup, затем косинусный спад к нулю к концу обучения.
     *
     * @param stepForLr номер шага 1…totalTrainingSteps (совпадает с {@code globalStep + 1} до инкремента).
     */
    float learningRateForStep(int stepForLr) {
        return LearningRateScheduler.learningRateAtStep(
                stepForLr,
                totalTrainingSteps,
                warmupSteps,
                config.learningRate,
                config.lrSchedule,
                config.minLrRatio);
    }

    /**
     * Сводка env и эффективных флагов для сверки с {@code scripts/jgpt-smart.sh}; ключи зондов
     * (JGPT_BATCH_PROBE*, JGPT_PROBE_*) в {@link #train()} не читаются.
     */
    private void logTensorTrainingEnvSnapshot() {
        log.info(
                "Сводка JGPT_* (обучение): батч direct={}, pinned={}; prefetch={}; async чекпоинт={}; "
                        + "выход после шага={}; trainPerf={}; profile={}; timings={}; generateGpuKv={}",
                dataLoader.usesDirectBatchBuffers(),
                dataLoader.usesPinnedHostBatchBuffers(),
                LlmTrainerEnvUtils.batchPrefetchEnabled(),
                checkpointAsyncIo,
                exitAfterOptimizerSteps > 0 ? Integer.toString(exitAfterOptimizerSteps) : "0 (выкл.)",
                LlmTrainerEnvUtils.envRawOrDash("JGPT_TRAIN_PERF"),
                LlmTrainerEnvUtils.envRawOrDash("JGPT_PROFILE"),
                LlmTrainerEnvUtils.envRawOrDash("JGPT_TIMINGS"),
                LlmTrainerEnvUtils.envRawOrDash("JGPT_GENERATE_GPU_KV"));
        log.info(
                "  пресет/env: E2E={} резидент (эфф.)={} резидент (env)={} pipeline декодера={} полный GPU шаг={} "
                        + "логиты GPU={} decoder bwd GPU={} train loss={} sampled candidates={} sampled negatives={} "
                        + "размер батча (ovr)={} кэш FP16={} FP16 dyn старт={} FP16 dyn интервал={} FP16 dyn макс={} "
                        + "FP16 aux soften scale={} CUDA_LIB={}",
                LLMConfig.gpuE2eTrainFromEnv(),
                LLMConfig.effectiveGpuResidentTraining(),
                LlmTrainerEnvUtils.envRawOrDash("JGPT_TRAIN_GPU_RESIDENT"),
                LLMConfig.decoderGpuPipelineFromEnvOrProp(),
                LLMConfig.fullGpuTrainStepFromEnv(),
                LLMConfig.deviceLogitsTrainStepFromEnv(),
                LLMConfig.deviceDecoderBackwardFromEnv(),
                config.trainLossMode,
                config.usesSampledTrainLoss() ? Integer.toString(config.sampledCeCandidates) : "-",
                config.usesSampledTrainLoss() ? config.sampledCeNegativeMode : "-",
                LlmTrainerEnvUtils.envRawOrDash("JGPT_BATCH_SIZE"),
                LlmTrainerEnvUtils.envRawOrDash("JGPT_ACTIVATION_CACHE_FP16"),
                LlmTrainerEnvUtils.envRawOrDash("JGPT_FP16_DYNAMIC_INITIAL"),
                LlmTrainerEnvUtils.envRawOrDash("JGPT_FP16_DYNAMIC_GROWTH_INTERVAL"),
                LlmTrainerEnvUtils.envRawOrDash("JGPT_FP16_DYNAMIC_MAX"),
                LlmTrainerGpuUtils.fp16AuxSoftenScaleAfterInfer() ? "1" : "0",
                LlmTrainerEnvUtils.envCudaLibSummary());
        if (TensorOpsGPU.isGpuAvailable()) {
            log.info(
                    "  TensorOpsGPU (при загрузке класса): FP16_MATMUL={} CE_мин_элементов={} FlashAttention={}",
                    TensorOpsGPU.useFp16Matmul(),
                    TensorOpsGPU.ceGpuMinElements(),
                    TensorOpsGPU.FLASH_ATTENTION);
        }
        {
            String tensorJavaMem = System.getenv("JGPT_JAVA_MEM");
            log.info(
                    "  JVM: maxHeap≈{} МиБ; env JGPT_JAVA_MEM={}",
                    Runtime.getRuntime().maxMemory() / (1024L * 1024L),
                    tensorJavaMem == null || tensorJavaMem.isBlank()
                            ? "—"
                            : tensorJavaMem.strip());
        }
        String legacyDynFlag = System.getenv("JGPT_FP16_DYNAMIC_LOSS_SCALE");
        if (legacyDynFlag != null && !legacyDynFlag.isBlank()) {
            log.warn(
                    "JGPT_FP16_DYNAMIC_LOSS_SCALE={} в env не используется: при FP16 matmul всегда динамический loss scale (JGPT_FP16_DYNAMIC_INITIAL / _GROWTH_INTERVAL / _MAX)",
                    legacyDynFlag.strip());
        }
        log.info(
                "  в train() не используются: JGPT_BATCH_PROBE* / JGPT_PROBE_* (зонды в отдельных тестах)");
    }

    /**
     * Сводка для сопоставимости экспериментов: эффективный batch в токенах, шаги оптимизатора на эпоху, интервал eval
     * в «эпохах», источник eval (train / hold-out).
     */
    private void logExperimentScheduleSummary() {
        int batches = Math.max(1, dataLoader.numBatches());
        int acc = Math.max(1, config.accumulationSteps);
        int optPerEpoch = (batches + acc - 1) / acc;
        long tokensPerOpt = (long) config.batchSize * (long) acc * (long) config.maxSeqLen;
        double evalEveryEpochs =
                optPerEpoch > 0 ? (double) config.evalEverySteps / (double) optPerEpoch : 0d;
        log.info(
                "{} эффективный batch: {} токенов на шаг оптимизатора (batch×accum×seq)",
                LogFmt.badge("EXP"),
                tokensPerOpt);
        log.info(
                "{} ~{} шагов оптимизатора на эпоху; eval каждые {} шагов ≈ каждые {} эпох; базовый LR={}",
                LogFmt.badge("EXP"),
                optPerEpoch,
                config.evalEverySteps,
                String.format(Locale.ROOT, "%.3f", evalEveryEpochs),
                String.format(Locale.ROOT, "%.4g", config.learningRate));
        if (evalDataLoader != null) {
            log.info(
                    "{} eval: отдельная валидация — {} окон, до {} полных батчей за проход (train не пересекается)",
                    LogFmt.badge("EXP"),
                    evalDataLoader.numSequences(),
                    evalDataLoader.numBatches());
        } else {
            log.info(
                    "{} eval: по train-потоку (индекс батча сохраняется). Hold-out: задайте JGPT_VAL_FRACTION в AllBooksTrain",
                    LogFmt.badge("EXP"));
        }
    }

    public void train() throws IOException {
        exitedDueToSupervisorRequest = false;
        supervisedStopRequested = false;
        log.info("{} старт обучения", LogFmt.badge("TRAIN"));
        log.info("{} конфигурация: {}", LogFmt.badge("CFG"), config);
        log.info(
                "Расписание LR: {} (min_lr_ratio={}), разогрев: шаги {} из {} (доля warmup {})",
                config.lrSchedule,
                config.minLrRatio,
                warmupSteps,
                totalTrainingSteps,
                config.warmupRatio);
        log.info(
                "Накопление градиента: {} микробатч(а/ей) на один шаг оптимизатора",
                config.accumulationSteps);
        log.info(
                "Логирование: каждые {} шаг(ов); автогенерация текста: {}",
                config.logEverySteps,
                config.interactiveSampleEverySteps > 0
                        ? ("каждые " + config.interactiveSampleEverySteps + " шагов (русские промпты)")
                        : "выкл.");
        log.info(
                "Данные: {} последовательностей, {} батчей/эпоха, всего ~{} шагов оптимизатора",
                dataLoader.numSequences(),
                dataLoader.numBatches(),
                totalTrainingSteps);
        logExperimentScheduleSummary();
        if (config.useGpuResident) {
            log.info(
                    config.fullGpuTrainStep
                            ? "GPU-резидент: веса и шаг оптимизатора на VRAM; синхронизация с хостом — ленивая при чекпоинте."
                            : "GPU-резидент: LM head / финальный RMSNorm на VRAM; после Adam — синхронизация с хостом.");
        }
        log.info(
                "GPU-путь (факт): полный шаг={}, логиты на GPU={}, backward декодера на GPU={}, "
                        + "допускается полный GPU-цикл={}, pipeline декодера={}",
                config.fullGpuTrainStep,
                config.deviceLogitsTrainStep,
                config.deviceDecoderBackward,
                model.canFullGpuTrain(),
                model.isDecoderGpuPipeline());
        if (config.earlyStopEvalPatience > 0
                || config.earlyStopTrainDownEvalUp
                || (config.earlyStopMinGradNorm > 0f && config.earlyStopGradNormPatience > 0)) {
            log.info(
                    "Ранний останов: терпение_eval={}, train↓_eval↑={}, мин_норма_grad={}, терпение_grad={}",
                    config.earlyStopEvalPatience,
                    config.earlyStopTrainDownEvalUp,
                    config.earlyStopMinGradNorm > 0f
                            ? String.format(Locale.ROOT, "%.1e", config.earlyStopMinGradNorm)
                            : "выкл.",
                    config.earlyStopGradNormPatience);
        }

        int trainCeFloatElems =
                config.usesSampledTrainLoss()
                        ? config.batchSize
                                * config.maxSeqLen
                                * LlmTrainerCrossEntropy.effectiveSampledCandidateCount(this, config.vocabSize)
                        : config.batchSize * config.maxSeqLen * config.vocabSize;
        int evalFullLogitElems = config.batchSize * config.maxSeqLen * config.vocabSize;
        log.info("Ускорение (см. также логи TensorOpsGPU при старте JVM):");
        log.info(
                "  • FP16 GEMM (matmul / внимание / matmul+ReLU): {}",
                fp16Matmul ? "вкл." : "выкл.");
        if (dynamicLossScaler != null) {
            log.info(
                    "  • Динамический loss scale: текущий {}×, рост каждые {} успешных шагов (max {}×); снимается перед клипом",
                    String.format("%.0f", dynamicLossScaler.getScale()),
                    dynamicLossScaler.getGrowthInterval(),
                    String.format("%.0f", dynamicLossScaler.getMaxScale()));
            log.info(
                    "  • При overflow шаг пропускается, scale снижается; после стабильных шагов — осторожный рост (интервал {})",
                    dynamicLossScaler.getGrowthInterval());
        } else {
            log.info("  • Масштаб loss FP16: не используется (FP16 matmul выкл.)");
        }
        if (config.usesSampledTrainLoss()) {
            log.info(
                    "  • CE train (sampled) на GPU: {} — буфер кандидатных логитов ~{} float (batch×seq×candidates); "
                            + "eval по-прежнему full-vocab ~{} float; порог CE ≥ {} float (устар., перем. {})",
                    TensorOpsGPU.shouldUseGpuCrossEntropy(trainCeFloatElems) ? "вкл." : "выкл.",
                    trainCeFloatElems,
                    evalFullLogitElems,
                    TensorOpsGPU.ceGpuMinElements(),
                    "JGPT_CE_GPU_MIN_ELEMENTS");
        } else {
            log.info(
                    "  • CE и градиент по логитам на GPU: {} (логиты: {} float; порог CE ≥ {} float, перем. {})",
                    TensorOpsGPU.shouldUseGpuCrossEntropy(trainCeFloatElems) ? "вкл." : "выкл.",
                    trainCeFloatElems,
                    TensorOpsGPU.ceGpuMinElements(),
                    "JGPT_CE_GPU_MIN_ELEMENTS");
        }

        logTensorTrainingEnvSnapshot();

        Files.createDirectories(Path.of(config.checkpointDir));

        TrainingProfiler profiler = TrainingProfiler.fromEnv();
        TrainingTimings timings = TrainingTimings.fromEnv();
        if (TensorTrainingPerfEnv.enabled()) {
            log.info(
                    "JGPT_TRAIN_PERF=1: как JGPT_PROFILE=1 + JGPT_TIMINGS=1 — подробный PERF на первые {} шаг(ов) после старта/reset/eval; у train_loss — ток/с и фазы каждые {} шаг(ов). "
                            + "После forward выполняется cuda stream sync — «прямой» включает GPU-forward, «лосс+∂CE» в основном CE (артефакт только этого режима).",
                    profiler != null ? profiler.maxDetailSteps() : 0,
                    config.logEverySteps);
        } else {
            if (profiler != null) {
                profiler.printBanner();
            }
            if (timings != null) {
                log.info(
                        "JGPT_TIMINGS=1: в строке train_loss — кратко ток/с и Σ фаз; развёрнуто — на следующей строке (только там же, где train_loss, каждые {} шаг(ов) оптимизатора).",
                        config.logEverySteps);
            }
        }
        if (profiler != null) {
            profiler.armDetailWindow("start");
        }
        final boolean profile = profiler != null || timings != null;

        if (globalStep >= totalTrainingSteps) {
            StepBeyondPlanPolicy beyond = StepBeyondPlanPolicy.fromEnvironment();
            if (beyond == StepBeyondPlanPolicy.RESTART_SCHEDULE) {
                int prev = globalStep;
                log.warn(
                        "globalStep={} >= totalTrainingSteps={} — JGPT_IF_STEP_BEYOND_PLAN=restart_schedule: "
                                + "сброс счётчика шагов и LR-расписания; веса, Adam и лучший eval сохранены. "
                                + "Иначе: JGPT_EPOCHS (расширить план), JGPT_FINETUNE=1 или другой чекпоинт.",
                        prev,
                        totalTrainingSteps);
                restartOptimizerScheduleForNewPlan();
            } else if (beyond == StepBeyondPlanPolicy.FAIL) {
                throw new TrainingPlanExhaustedException(
                        String.format(
                                Locale.ROOT,
                                "globalStep=%d >= totalTrainingSteps=%d — JGPT_IF_STEP_BEYOND_PLAN=fail. "
                                        + "Увеличьте JGPT_EPOCHS, задайте restart_schedule/skip или смените чекпоинт.",
                                globalStep,
                                totalTrainingSteps));
            } else {
                log.info(
                        "План обучения уже выполнен: шаг {}/{} — дальнейшие шаги пропускаются. "
                                + "(JGPT_IF_STEP_BEYOND_PLAN=skip; для продолжения: restart_schedule)",
                        globalStep,
                        totalTrainingSteps);
                if (profiler != null) {
                    profiler.printSummary();
                }
                return;
            }
        }

        ExecutorService prefetchExecutor =
                LlmTrainerEnvUtils.batchPrefetchEnabled()
                        ? Executors.newSingleThreadExecutor(
                                r -> {
                                    Thread t = new Thread(r, "jgpt-batch-prefetch");
                                    t.setDaemon(true);
                                    return t;
                                })
                        : null;
        try {
        if (prefetchExecutor != null) {
            log.info(
                    "Фоновая подготовка следующего батча на CPU ({}=1; выключить: 0 или -Djgpt.batch.prefetch=false).",
                    "JGPT_BATCH_PREFETCH");
        }
        int evalsWithoutImprovement = 0;
        float lastEvalLossSnapshot = Float.NaN;
        float lastTrainAtEval = Float.NaN;
        int gradNormSmallStreak = 0;
        boolean trainingStoppedEarly = false;
        boolean forceScalerResetNextEpoch = false;

        int startEpoch = Math.max(0, Math.min(loadedResumeEpochIndex, config.epochs));
        int resumeSeqCopy = loadedResumeDataLoaderIndex;
        boolean replayCkpt = resumeReplayCheckpointShuffles;
        loadedResumeEpochIndex = 0;
        loadedResumeDataLoaderIndex = 0;
        resumeReplayCheckpointShuffles = false;

        if (startEpoch > 0 || resumeSeqCopy > 0) {
            log.info(
                    "{} продолжение: эпоха {}/{}, индекс последовательности в эпохе {} (повтор shuffle={})",
                    LogFmt.badge("CKPT"),
                    startEpoch + 1,
                    config.epochs,
                    resumeSeqCopy,
                    replayCkpt);
        }
        if (trainingStatsWriter != null) {
            trainingStatsWriter.syncProgressFromResume(globalStep, startEpoch + 1, config.epochs);
            trainingStatsWriter.syncBestLossFromResume(bestLoss);
        }

        outer:
        for (int epoch = startEpoch; epoch < config.epochs; epoch++) {
            pendingCheckpointEpochIndex = epoch;
            long epochStartNs = System.nanoTime();
            log.info(
                    "{} эпоха {}/{}: батчей в эпохе {}",
                    LogFmt.badge("EPOCH"),
                    epoch + 1,
                    config.epochs,
                    dataLoader.numBatches());
            if (replayCkpt && epoch == startEpoch) {
                /* Повтор shuffle имитирует прошлые эпохи для согласованности RNG/порядка; лог — один раз. */
                for (int k = 0; k < epoch + 1; k++) {
                    dataLoader.shuffle(k == epoch);
                }
                int clamped = clampResumeSequenceIndex(resumeSeqCopy);
                dataLoader.setCurrentIndex(clamped);
                if (resumeSeqCopy != clamped) {
                    log.warn(
                            "{} индекс последовательности из чекпоинта {} приведён к {} (границы эпохи/батча)",
                            LogFmt.badge("CKPT"),
                            resumeSeqCopy,
                            clamped);
                }
            } else {
                dataLoader.shuffle();
            }
            if (fp16Matmul
                    && dynamicLossScaler != null
                    && (fp16DynamicResetEachEpoch || forceScalerResetNextEpoch)) {
                dynamicLossScaler.resetToInitial();
                log.info(
                        "{} reset к {}× в начале эпохи ({})",
                        LogFmt.badge("FP16"),
                        String.format(Locale.ROOT, "%.4g", dynamicLossScaler.getScale()),
                        fp16DynamicResetEachEpoch
                                ? "JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH"
                                : "auto-recovery после неуспешной эпохи");
                if (profiler != null) {
                    profiler.armDetailWindow("fp16-reset");
                }
                forceScalerResetNextEpoch = false;
            }

            float epochLoss = 0f;
            int epochOptimizerAttempts = 0;
            int epochSuccessfulOptimizerSteps = 0;
            int microInAccum = 0;
            float accumLoss = 0f;
            long accFwdNs = 0;
            long accLossCeNs = 0;
            long accBwdNs = 0;
            long accTokens = 0;

            CompletableFuture<DataLoader.Batch> prefetchFut = null;

            if (!dataLoader.hasMore()) {
                log.warn(
                        "Эпоха {}: нет полных батчей (данных меньше, чем размер_батча × длина_последовательности).",
                        epoch + 1);
            }
            while (dataLoader.hasMore()) {
                if (supervisedStopRequested) {
                    log.warn(
                            "{} обучение прервано по запросу супервизора (смена пресета)",
                            LogFmt.badge("SMART"));
                    trainingStoppedEarly = true;
                    exitedDueToSupervisorRequest = true;
                    break outer;
                }
                DataLoader.Batch batch;
                if (prefetchExecutor != null && prefetchFut != null) {
                    batch = LlmTrainerBatchPrefetch.takeNextBatchPrefetched(dataLoader, prefetchFut);
                } else {
                    batch = dataLoader.nextBatch();
                }
                prefetchFut = LlmTrainerBatchPrefetch.scheduleBatchPrefetch(dataLoader, prefetchExecutor);

                boolean lastBatchOfEpoch = !dataLoader.hasMore();
                microInAccum++;
                accTokens += (long) batch.input.getShape()[0] * config.maxSeqLen;

                long t0 = profile ? System.nanoTime() : 0L;
                Tensor logits;
                if (LlmTrainerCrossEntropy.canDeviceSampledTrainForward(this)) {
                    int batchSz = batch.input.getShape()[0];
                    int seqLen = batch.input.getShape()[1];
                    int vocabSize = config.vocabSize;
                    int rows = batchSz * seqLen;
                    int candCount = LlmTrainerCrossEntropy.effectiveSampledCandidateCount(this, vocabSize);
                    LlmTrainerCrossEntropy.prepareSampledCandidateIds(this, batch.target, rows, vocabSize, candCount);
                    logits =
                            model.forwardTrainingDeviceSampled(
                                    batch.input, sampledCandidateIdsDevice, sampledCandidateLogitsDevice, candCount);
                } else {
                    logits = model.forward(batch.input, true, config.useGpuResident);
                }
                if (TensorTrainingPerfEnv.enabled() && TensorOpsGPU.isGpuAvailable()) {
                    /* Честное разбиение PERF: весь GPU-forward до барьера входит в «прямой»; на скорость шага не влияет. */
                    TensorOpsGPU.synchronizeStream();
                }
                long t1 = profile ? System.nanoTime() : 0L;
                float ceScale = 1f / (float) config.accumulationSteps;
                float loss;
                if (ceAsyncDevice && !config.usesSampledTrainLoss() && config.deviceLogitsTrainStep && model.hasDeviceLogitsBuffers()) {
                    LlmTrainerCrossEntropy.applyCrossEntropyLossAndGradDeviceAsync(this, logits, batch.target, ceScale);
                    /* Async CE: kernel пишет ∂logits на device и D2H скаляра loss — до backward нужен барьер. */
                    TensorOpsGPU.synchronizeStream();
                    long t3 = profile ? System.nanoTime() : 0L;
                    model.backward(
                            logits,
                            microInAccum == 1,
                            config.fullGpuTrainStep
                                    && model.isGpuResident()
                                    && gpuTrainableParamGradsKnownClean
                                    && microInAccum == 1);
                    /* Backward ставит GPU kernels; перед clip/overflow-check нужен барьер готовности ∂. */
                    TensorOpsGPU.synchronizeStream();
                    loss = TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost();
                    long t4 = profile ? System.nanoTime() : 0L;
                    if (profile) {
                        accFwdNs += t1 - t0;
                        accLossCeNs += t3 - t1;
                        accBwdNs += t4 - t3;
                    }
                } else {
                    loss = LlmTrainerCrossEntropy.applyTrainLossAndGrad(this, logits, batch.target, ceScale);
                    long t3 = profile ? System.nanoTime() : 0L;
                    model.backward(
                            logits,
                            microInAccum == 1,
                            config.fullGpuTrainStep
                                    && model.isGpuResident()
                                    && gpuTrainableParamGradsKnownClean
                                    && microInAccum == 1);
                    long t4 = profile ? System.nanoTime() : 0L;
                    if (profile) {
                        accFwdNs += t1 - t0;
                        accLossCeNs += t3 - t1;
                        accBwdNs += t4 - t3;
                    }
                }
                accumLoss += loss;

                boolean shouldStep =
                        microInAccum >= config.accumulationSteps || lastBatchOfEpoch;
                if (!shouldStep) {
                    continue;
                }

                float partialScale = 1f;
                if (lastBatchOfEpoch && microInAccum < config.accumulationSteps) {
                    partialScale = (float) config.accumulationSteps / (float) microInAccum;
                }
                if (partialScale != 1f) {
                    LlmTrainerOptimizerStep.scaleGradients(parameters, partialScale);
                    if (config.fullGpuTrainStep) {
                        LlmTrainerOptimizerStep.scaleGpuGradients(model.gpuTensorByTrainableParameter(), partialScale);
                    }
                }

                float avgMicroLoss = accumLoss / (float) microInAccum;
                long t5 = profile ? System.nanoTime() : 0L;
                boolean stepped =
                        config.fullGpuTrainStep
                                ? LlmTrainerOptimizerStep.clipAndOptimizerStepFullGpu(this, logits, avgMicroLoss)
                                : LlmTrainerOptimizerStep.clipAndOptimizerStep(this, logits, avgMicroLoss);
                long t6 = profile ? System.nanoTime() : 0L;

                epochOptimizerAttempts++;
                epochLoss += avgMicroLoss;

                microInAccum = 0;
                accumLoss = 0f;

                if (!stepped) {
                    if (config.accumulationSteps > 1) {
                        LlmTrainerOptimizerStep.zeroGradients(this, logits);
                        LlmTrainerOptimizerStep.clearGpuParamGradsAfterOverflowSkip(this);
                        if (config.fullGpuTrainStep && model.isGpuResident()) {
                            LlmTrainerOptimizerStep.zeroGpuGradsMarkingParamGradsClean(this, model.gpuTensorByTrainableParameter());
                        }
                        LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip();
                    }
                    accFwdNs = 0;
                    accLossCeNs = 0;
                    accBwdNs = 0;
                    accTokens = 0;
                    continue;
                }

                overflowLogPlannedStepKey = -1;
                overflowSkipRepeatCount = 0;

                if (config.earlyStopMinGradNorm > 0f && config.earlyStopGradNormPatience > 0) {
                    if (lastGlobalGradNorm < config.earlyStopMinGradNorm) {
                        gradNormSmallStreak++;
                        if (gradNormSmallStreak >= config.earlyStopGradNormPatience) {
                            log.warn(
                                    "Ранний останов: глобальная норма градиента {} ниже порога {} — {} шагов подряд",
                                    String.format(Locale.ROOT, "%.2e", lastGlobalGradNorm),
                                    String.format(Locale.ROOT, "%.2e", config.earlyStopMinGradNorm),
                                    config.earlyStopGradNormPatience);
                            trainingStoppedEarly = true;
                            break outer;
                        }
                    } else {
                        gradNormSmallStreak = 0;
                    }
                }

                long snapFwdNs = accFwdNs;
                long snapLossCeNs = accLossCeNs;
                long snapBwdNs = accBwdNs;
                long snapAccTokens = accTokens;
                long snapOptNs = t6 - t5;
                if (profiler != null) {
                    profiler.recordStep(
                            accFwdNs,
                            accLossCeNs,
                            accBwdNs,
                            t6 - t5,
                            accTokens,
                            globalStep + 1);
                }
                if (timings != null) {
                    timings.recordOptimizerStep(accFwdNs, accLossCeNs, accBwdNs, t6 - t5, accTokens);
                }
                accFwdNs = 0;
                accLossCeNs = 0;
                accBwdNs = 0;
                accTokens = 0;

                epochSuccessfulOptimizerSteps++;
                globalStep++;
                trainingEventCallback.onOptimizerStepCompleted(globalStep, epoch + 1);
                if (trainingStatsWriter != null) {
                    long totalNs = snapFwdNs + snapLossCeNs + snapBwdNs + snapOptNs;
                    int tps =
                            totalNs > 0L
                                    ? (int)
                                            Math.min(
                                                    (long) Integer.MAX_VALUE,
                                                    snapAccTokens * 1_000_000_000L / totalNs)
                                    : 0;
                    trainingStatsWriter.onStep(
                            globalStep,
                            epoch + 1,
                            config.epochs,
                            avgMicroLoss,
                            learningRateForStep(globalStep),
                            tps);
                }

                if (globalStep == 1 || globalStep % config.logEverySteps == 0) {
                    float lr = learningRateForStep(globalStep);
                    int ep = epoch + 1;
                    if (timings != null) {
                        TrainingTimings.WindowLog timing = timings.formatForLogAndReset();
                        log.info(
                                "{} эпоха {}/{}  шаг {}  {}={}  lr={}  (прогресс шагов {}/{}){}",
                                LogFmt.badge("STEP"),
                                ep,
                                config.epochs,
                                globalStep,
                                config.usesSampledTrainLoss() ? "sampled_train_loss" : "train_loss",
                                String.format("%.4f", avgMicroLoss),
                                String.format("%.2e", lr),
                                globalStep,
                                totalTrainingSteps,
                                timing.inline().isEmpty() ? "" : ("  " + timing.inline()));
                        if (!timing.detailMultiline().isEmpty()) {
                            log.info("{}", timing.detailMultiline());
                        }
                    } else {
                        log.info(
                                "{} эпоха {}/{}  шаг {}  {}={}  lr={}  (прогресс шагов {}/{})",
                                LogFmt.badge("STEP"),
                                ep,
                                config.epochs,
                                globalStep,
                                config.usesSampledTrainLoss() ? "sampled_train_loss" : "train_loss",
                                String.format("%.4f", avgMicroLoss),
                                String.format("%.2e", lr),
                                globalStep,
                                totalTrainingSteps);
                    }
                }

                if (globalStep % config.evalEverySteps == 0) {
                    float evalLoss = LlmTrainerEvalAndSample.evaluate(this);
                    /* До saveCheckpoint / следующего train-batch: сброс кэша/graph и стрим, иначе иногда нечисловые ∂ на
                     * первом шаге после eval (пересечение infer-цепочки с обучением на том же stream). */
                    synchronizeTrainingPipelineAfterGpuAuxiliaryInfer("eval");
                    if (profiler != null) {
                        profiler.armDetailWindow("eval");
                    }
                    if (Float.isFinite(evalLoss)) {
                        boolean improvedBest = evalLoss < bestLoss;
                        if (improvedBest) {
                            bestLoss = evalLoss;
                            saveCheckpoint("best");
                            evalsWithoutImprovement = 0;
                        } else if (config.earlyStopEvalPatience > 0) {
                            evalsWithoutImprovement++;
                            if (evalsWithoutImprovement >= config.earlyStopEvalPatience) {
                                log.warn(
                                        "Ранний останов: оценочный loss не улучшал лучший {} раз подряд (перплексия не падает)",
                                        config.earlyStopEvalPatience);
                                trainingStoppedEarly = true;
                                break outer;
                            }
                        }
                        if (config.earlyStopTrainDownEvalUp
                                && !Float.isNaN(lastEvalLossSnapshot)
                                && evalLoss > lastEvalLossSnapshot
                                && avgMicroLoss < lastTrainAtEval) {
                            log.warn(
                                     "Ранний останов: train loss снизился относительно прошлого eval, eval loss вырос — признак переобучения");
                            trainingStoppedEarly = true;
                            break outer;
                        }
                        lastEvalLossSnapshot = evalLoss;
                        lastTrainAtEval = avgMicroLoss;
                        if (evalDataLoader != null) {
                            log.info(
                                    "{} эпоха {}/{}: val_loss={} (лучший по val={})",
                                    LogFmt.badge("EVAL"),
                                    epoch + 1,
                                    config.epochs,
                                    String.format(Locale.ROOT, "%.4f", evalLoss),
                                    LlmTrainerTrainingFormat.formatEvalBestLossForLog(bestLoss));
                        } else {
                            log.info(
                                    "{} эпоха {}/{}: eval_loss={} (лучший сохранённый={})",
                                    LogFmt.badge("EVAL"),
                                    epoch + 1,
                                    config.epochs,
                                    String.format(Locale.ROOT, "%.4f", evalLoss),
                                    LlmTrainerTrainingFormat.formatEvalBestLossForLog(bestLoss));
                        }
                        trainingEventCallback.onEvalCompleted(
                                epoch + 1, evalLoss, bestLoss, improvedBest);
                        if (trainingStatsWriter != null) {
                            trainingStatsWriter.onEval(
                                    globalStep,
                                    evalLoss,
                                    (float) Math.exp((double) evalLoss),
                                    bestLoss);
                        }
                    }
                    if (TensorOpsGPU.isGpuAvailable()) {
                        TensorOpsGPU.synchronizeStream();
                    }
                }

                if (cudaTrimEveryOptimizerSteps > 0
                        && globalStep % cudaTrimEveryOptimizerSteps == 0
                        && model.isGpuResident()
                        && TensorOpsGPU.isGpuAvailable()) {
                    TensorOpsGPU.synchronizeStream();
                    TensorOpsGPU.drainDeferredGpuBuffers();
                    TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
                }

                /* Периодическая очистка активационных кэшей модели каждые N шагов для предотвращения
                 * утечки VRAM при GROW_ONLY=1. Буферы в blockCachesDevice[] растут до максимума
                 * и не освобождаются автоматически в течение эпохи.
                 * Настраивается через JGPT_VRAM_CLEANUP_EVERY_STEPS (по умолчанию 1000, 0 - отключить). */
                int vramCleanupEverySteps = LlmTrainerEnvUtils.readVramCleanupEveryStepsFromEnv();
                if (vramCleanupEverySteps > 0 && globalStep % vramCleanupEverySteps == 0 && model.isGpuResident()) {
                    if (log.isDebugEnabled()) {
                        log.debug("{} периодическая очистка VRAM на шаге {}", LogFmt.badge("VRAM"), globalStep);
                    }
                    /* Очистка ThreadLocal пула и активационных кэшей модели */
                    BlockActivationCacheDevice.purgeThreadLocalPool();
                    model.prepareForTrainingAfterInteractiveGeneration();
                    GpuPendingGradients.cleanupThreadLocal();
                    GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
                    TensorOpsGPU.synchronizeStream();
                    TensorOpsGPU.drainDeferredGpuBuffers();
                    TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
                }

                if (globalStep % config.saveEverySteps == 0) {
                    saveCheckpoint("step_" + globalStep);
                }

                LlmTrainerEvalAndSample.maybeAutoSample(this, epoch + 1);
                if (model.isGpuResident() && TensorOpsGPU.isGpuAvailable()) {
                    /* saveCheckpoint/syncWeights — async D2H на том же stream; барьер до следующего train-forward. */
                    TensorOpsGPU.synchronizeStream();
                }
                if (globalStep >= totalTrainingSteps) {
                    log.info(
                            "Достигнут лимит плана обучения: шаг {}/{} — завершение.",
                            globalStep,
                            totalTrainingSteps);
                    if (profiler != null) {
                        profiler.printSummary();
                    }
                    return;
                }
                if (exitAfterOptimizerSteps > 0 && globalStep >= exitAfterOptimizerSteps) {
                    log.info(
                            "Ранний выход: переменная JGPT_EXIT_AFTER_STEP={}",
                            exitAfterOptimizerSteps);
                    if (profiler != null) {
                        profiler.printSummary();
                    }
                    return;
                }
            }

            float avgLoss = epochLoss / Math.max(1, epochOptimizerAttempts);
            long epochElapsedNs = System.nanoTime() - epochStartNs;
            String durationStr = LlmTrainerTrainingFormat.formatEpochDuration(epochElapsedNs);
            String lossStr = String.format("%.4f", avgLoss);
            if (epochSuccessfulOptimizerSteps == 0 && epochOptimizerAttempts > 0) {
                log.warn(
                        "Эпоха {}: успешных шагов оптимизатора 0 из {}; средний train_loss — по forward (веса не обновлялись).",
                        epoch + 1,
                        epochOptimizerAttempts);
                if (fp16Matmul && dynamicLossScaler != null && !fp16DynamicResetEachEpoch) {
                    forceScalerResetNextEpoch = true;
                    log.warn(
                            "Эпоха {}: авто-восстановление FP16 scaler — в начале следующей эпохи будет reset к {}× "
                                    + "(включите {}=1, чтобы всегда сбрасывать между эпохами).",
                            epoch + 1,
                            String.format(Locale.ROOT, "%.4g", dynamicLossScaler.getBaselineScale()),
                            "JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH");
                }
            }
            String epochTail =
                    epochSuccessfulOptimizerSteps < epochOptimizerAttempts
                            ? String.format(
                                    Locale.ROOT,
                                    "  (шаги оптимизатора: %d/%d)",
                                    epochSuccessfulOptimizerSteps,
                                    epochOptimizerAttempts)
                            : "";
            log.info(
                    "{} эпоха {} завершена: средний train_loss={}, длительность={}{}",
                    LogFmt.badge("EPOCH"),
                    epoch + 1,
                    lossStr,
                    LogFmt.success(durationStr),
                    epochTail);

            pendingCheckpointEpochIndex = epoch + 1;
            pendingCheckpointDataLoaderIndex = 0;
            saveCheckpoint("epoch_" + (epoch + 1));
            dataLoader.reset();
            /* После каждой эпохи сбрасываем async memory pool — иначе при большом batch (batch>1) пул
            /* После каждой эпохи сбрасываем async memory pool — иначе при большом batch (batch>1) пул
             * накапливает фрагментацию между eval-вызовами: eval+trim происходит 1 раз в ~batch/1 эпох,
             * а временные буферы forward/backward занимают пул между trimами. */
            if (model.isGpuResident() && TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.drainDeferredGpuBuffers();
                TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
            }

            /* Очистка ThreadLocal пула BlockActivationCacheDevice для предотвращения утечки VRAM.
             * Пул накапливает буферы при POOL=1, которые не освобождаются при смерти потоков. */
            BlockActivationCacheDevice.purgeThreadLocalPool();
            if (log.isInfoEnabled()) {
                log.info("{} очистка VRAM после эпохи: пул BlockActivationCacheDevice очищен", LogFmt.badge("VRAM"));
            }

            /* Очистка активационных кэшей модели для предотвращения утечки VRAM при GROW_ONLY=1.
             * Буферы в blockCachesDevice[] растут до максимального размера и не освобождаются между эпохами. */
            if (model.isGpuResident()) {
                model.prepareForTrainingAfterInteractiveGeneration();
                TensorOpsGPU.synchronizeStream();
                TensorOpsGPU.drainDeferredGpuBuffers();
                TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
                if (log.isInfoEnabled()) {
                    log.info("{} очистка VRAM после эпохи: активационные кэши модели очищены", LogFmt.badge("VRAM"));
                }
            }

            /* Очистка ThreadLocal GpuPendingGradients для предотвращения утечки VRAM. */
            GpuPendingGradients.cleanupThreadLocal();
            GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
        }
        if (trainingStoppedEarly) {
            log.info("{} обучение остановлено по раннему критерию", LogFmt.badge("STOP"));
        } else {
            log.info("{} обучение по плану завершено", LogFmt.badge("TRAIN"));
        }
        if (fp16Matmul) {
            Fp16Metrics.global().logStats(log);
        }
        if (bestLoss == Float.MAX_VALUE) {
            log.info(
                    "Лучший оценочный loss: нет данных (не было ни одной оценки: шагов меньше evalEverySteps или интервал слишком большой).");
        } else {
            log.info("Лучший оценочный loss (сохранённый): {}", LlmTrainerTrainingFormat.formatEvalBestLossForLog(bestLoss));
        }
        if (profiler != null) {
            profiler.printSummary();
        }
        } finally {
            if (prefetchExecutor != null) {
                prefetchExecutor.shutdown();
                try {
                    if (!prefetchExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                        prefetchExecutor.shutdownNow();
                    }
                } catch (InterruptedException ie) {
                    prefetchExecutor.shutdownNow();
                    Thread.currentThread().interrupt();
                }
            }
            GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
            GpuPendingGradients.cleanupThreadLocal();
            GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
            if (TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.synchronizeStream();
                TensorOpsGPU.drainDeferredGpuBuffers();
                TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
                TensorOpsGPU.cleanupCudaThreadResources();
            }
        }
    }

    public void saveCheckpoint(String name) throws IOException {
        LlmTrainerCheckpointIo.saveCheckpoint(this, name);
    }

    public void saveModelWeights(String name) throws IOException {
        LlmTrainerCheckpointIo.saveModelWeights(this, name);
    }

    public void awaitPendingCheckpointWrites() {
        LlmTrainerCheckpointIo.awaitPendingCheckpointWrites(this);
    }

    public void loadCheckpoint(String path) throws IOException, ClassNotFoundException {
        LlmTrainerCheckpointIo.loadCheckpoint(this, path);
    }

    public void loadModelWeights(String name) throws IOException, ClassNotFoundException {
        String modelPath = Path.of(config.checkpointDir).resolve("model_" + name + ".bin").toString();
        model.loadWeights(modelPath);
    }

    public int getGlobalStep() {
        return globalStep;
    }

    /**
     * См. {@link #shutdownProgressBaselineStep}: {@code getGlobalStep() > getShutdownProgressBaselineStep()} означает,
     * что после загрузки чекпоина и возможного сброса плана был хотя бы один новый шаг оптимизатора.
     */
    public int getShutdownProgressBaselineStep() {
        return shutdownProgressBaselineStep;
    }

    void syncShutdownProgressBaselineFromGlobalStep() {
        shutdownProgressBaselineStep = globalStep;
    }

    /** Плановое число шагов оптимизатора за весь прогон ({@code epochs × stepsPerEpoch}). */
    public int getTotalTrainingSteps() {
        return totalTrainingSteps;
    }

    public float getBestLoss() {
        return bestLoss;
    }

    /**
     * Сброс позиции в плане после смены длины эпохи/батча: шаг 0 и начало эпохи, без сброса {@link #bestLoss}
     * и без очистки весов/Adam (в отличие от {@link #resetGlobalStep()}).
     */
    public void restartOptimizerScheduleForNewPlan() {
        globalStep = 0;
        optimizer.setStep(0);
        loadedResumeEpochIndex = 0;
        loadedResumeDataLoaderIndex = 0;
        pendingCheckpointEpochIndex = 0;
        pendingCheckpointDataLoaderIndex = 0;
        resumeReplayCheckpointShuffles = false;
        syncShutdownProgressBaselineFromGlobalStep();
    }

    /**
     * Сбрасывает {@code globalStep} в 0, обновляет шаг оптимайзера и сбрасывает лучший eval loss.
     * Используется при дообучении ({@code JGPT_FINETUNE=1}): веса и Adam-буферы
     * сохраняются, но косинусный LR-расписание и счётчик раннего останова перезапускаются.
     */
    public void resetGlobalStep() {
        globalStep = 0;
        optimizer.setStep(0);
        bestLoss = Float.MAX_VALUE;
        loadedResumeEpochIndex = 0;
        loadedResumeDataLoaderIndex = 0;
        pendingCheckpointEpochIndex = 0;
        pendingCheckpointDataLoaderIndex = 0;
        resumeReplayCheckpointShuffles = false;
        syncShutdownProgressBaselineFromGlobalStep();
    }

    /**
     * Приводит сырой индекс из чекпоинта к допустимому старту полного батча (кратно {@code batchSize}, не больше
     * последнего старта с {@link DataLoader#hasMore()}).
     */
    private int clampResumeSequenceIndex(int raw) {
        int n = dataLoader.numSequences();
        if (n <= 0) {
            return 0;
        }
        int bs = config.batchSize;
        int maxStart = n >= bs ? n - bs : 0;
        int idx = raw;
        if (idx < 0) {
            idx = 0;
        }
        if (idx > n) {
            idx = n;
        }
        if (bs > 1) {
            idx = (idx / bs) * bs;
        }
        return Math.min(idx, maxStart);
    }

    // ========== Test harness / full-GPU training API ==========

    /** Result of a single micro-batch forward + CE + backward. */
    public static final class TestMicrobatchResult {
        public final Tensor logits;
        public final float ceLoss;

        public TestMicrobatchResult(Tensor logits, float ceLoss) {
            this.logits = logits;
            this.ceLoss = ceLoss;
        }
    }

    /**
     * One micro-batch: forward → CE → backward. Used by tests to exercise the full pipeline.
     *
     * @param batch input + target
     * @param zeroGrads zero parameter gradients before backward
     * @return logits and CE loss
     */
    public TestMicrobatchResult testHarnessForwardCeBackward(DataLoader.Batch batch, boolean zeroGrads) {
        Tensor logits;
        if (LlmTrainerCrossEntropy.canDeviceSampledTrainForward(this)) {
            int batchSz = batch.input.getShape()[0];
            int seqLen = batch.input.getShape()[1];
            int vocabSize = config.vocabSize;
            int rows = batchSz * seqLen;
            int candCount = LlmTrainerCrossEntropy.effectiveSampledCandidateCount(this, vocabSize);
            LlmTrainerCrossEntropy.prepareSampledCandidateIds(this, batch.target, rows, vocabSize, candCount);
            logits =
                    model.forwardTrainingDeviceSampled(
                            batch.input, sampledCandidateIdsDevice, sampledCandidateLogitsDevice, candCount);
        } else {
            logits = model.forward(batch.input, true, config.useGpuResident);
        }
        float ceScale = 1f / (float) config.accumulationSteps;
        float loss;
        if (ceAsyncDevice && !config.usesSampledTrainLoss() && config.deviceLogitsTrainStep && model.hasDeviceLogitsBuffers()) {
            LlmTrainerCrossEntropy.applyCrossEntropyLossAndGradDeviceAsync(this, logits, batch.target, ceScale);
            TensorOpsGPU.synchronizeStream();
            model.backward(logits, zeroGrads);
            TensorOpsGPU.synchronizeStream();
            loss = TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost();
        } else {
            loss = LlmTrainerCrossEntropy.applyTrainLossAndGrad(this, logits, batch.target, ceScale);
            model.backward(logits, zeroGrads);
        }
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
        }
        return new TestMicrobatchResult(logits, loss);
    }

    /**
     * Clip + optimizer step. Routes to full-GPU path when {@code config.fullGpuTrainStep}.
     *
     * @param logits last micro-batch logits (for overflow check on host path)
     * @param loss average loss across accumulated micro-batches
     * @param partialScale correction factor for incomplete accumulation group ({@code accumulationSteps / actual})
     * @return true if weights updated
     */
    public boolean testHarnessClipAndOptimizerStep(Tensor logits, float loss, float partialScale) {
        boolean stepped;
        if (config.fullGpuTrainStep) {
            if (partialScale != 1f) {
                LlmTrainerOptimizerStep.scaleGradients(parameters, partialScale);
                LlmTrainerOptimizerStep.scaleGpuGradients(model.gpuTensorByTrainableParameter(), partialScale);
            }
            stepped = LlmTrainerOptimizerStep.clipAndOptimizerStepFullGpu(this, logits, loss);
        } else {
            if (partialScale != 1f) {
                LlmTrainerOptimizerStep.scaleGradients(parameters, partialScale);
            }
            stepped = LlmTrainerOptimizerStep.clipAndOptimizerStep(this, logits, loss);
        }
        if (stepped) {
            overflowLogPlannedStepKey = -1;
            overflowSkipRepeatCount = 0;
        }
        return stepped;
    }

}
