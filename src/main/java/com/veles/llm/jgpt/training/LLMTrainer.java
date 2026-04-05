package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.GpuIntBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.app.LlmTextGeneration;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.cuda.TensorCudaLibrary;
import com.veles.llm.jgpt.data.DataLoader;
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
import java.util.concurrent.ExecutionException;
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
 * {@code lossScale} — текущий {@link DynamicLossScaler} при FP16 matmul (см. {@link #ceFusedGradScaleOverTotal}).
 */
public final class LLMTrainer {

    private static final Logger log = LoggerFactory.getLogger(LLMTrainer.class);

    private static void agentLogB39372(String hypothesisId, String location, String message, String dataJson) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\""
                        + hypothesisId
                        + "\",\"location\":\""
                        + location
                        + "\",\"message\":\""
                        + message
                        + "\",\"data\":"
                        + dataJson
                        + ",\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }

    private static String jsonEsc(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    /**
     * После overflow: все {@code zeroGrad} / clear на GPU ставят async-работу в очередь. Без барьера следующий
     * forward/backward может пересечься с ними (ложные NaN / нечисловые градиенты на всех следующих шагах). Реальная
     * ошибка CUDA на stream при этом всплывёт в {@link TensorOpsGPU#synchronizeStream()} (исключение).
     */
    private static void synchronizeGpuAfterOverflowSkip() {
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
        }
    }

    /**
     * Доп. деление FP16 loss scale после eval/генерации ({@link DynamicLossScaler#softenScaleAfterAuxiliaryGpuWork}).
     * Отключить, если достаточно сброса градиентов/pending: env {@code JGPT_FP16_AUX_SOFTEN=0} или
     * {@code -Djgpt.fp16.aux.soften=false}.
     */
    private static boolean fp16AuxSoftenScaleAfterInfer() {
        try {
            String e = System.getenv("JGPT_FP16_AUX_SOFTEN");
            if (e != null && !e.isBlank()) {
                String s = e.trim();
                if ("0".equals(s) || "false".equalsIgnoreCase(s)) {
                    return false;
                }
                return true;
            }
        } catch (Exception ignored) {
        }
        try {
            String p = System.getProperty("jgpt.fp16.aux.soften");
            if (p != null && !p.isBlank()) {
                String s = p.trim();
                if ("0".equals(s) || "false".equalsIgnoreCase(s)) {
                    return false;
                }
            }
        } catch (Exception ignored) {
        }
        return true;
    }

    /**
     * После GPU-инференса вне шага обучения (промежуточная генерация, {@link #evaluate}, и т.п.): тот же сброс, что
     * после {@code generateGpuKv}, иначе первый следующий train-step иногда даёт нечисловые градиенты при том же
     * потоке CUDA и decoder graphs.
     */
    private void synchronizeTrainingPipelineAfterGpuAuxiliaryInfer(String reason) {
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
        if (fp16Matmul && dynamicLossScaler != null) {
            dynamicLossScaler.resetConsecutiveNonOverflowAfterAuxiliaryGpuWork();
            if (fp16AuxSoftenScaleAfterInfer()) {
                float s0 = dynamicLossScaler.getScale();
                dynamicLossScaler.softenScaleAfterAuxiliaryGpuWork(reason);
                float s1 = dynamicLossScaler.getScale();
                if (DebugGpuTrain.isEnabled()) {
                    agentLogB39372(
                            "H_aux_soften",
                            "LLMTrainer.synchronizeTrainingPipelineAfterGpuAuxiliaryInfer",
                            "post_fp16_soften",
                            "{\"globalStep\":"
                                    + globalStep
                                    + ",\"reason\":\""
                                    + jsonEsc(reason)
                                    + "\",\"scaleBefore\":"
                                    + s0
                                    + ",\"scaleAfter\":"
                                    + s1
                                    + "}");
                }
            }
        }
        if (DebugGpuTrain.isEnabled()) {
            agentLogB39372(
                    "H_aux_fence",
                    "LLMTrainer",
                    "post_aux_infer_reset",
                    "{\"globalStep\":"
                            + globalStep
                            + ",\"reason\":\""
                            + jsonEsc(reason)
                            + "\"}");
        }
    }

    /**
     * После завершения книги в цепочке ({@code MultiBookTrain}): освобождает VRAM тренера и модели до выхода из
     * области видимости, чтобы не полагаться на {@code finalize} у {@link GpuTensor}/{@link GpuFloatBuffer}.
     */
    public void releaseGpuResourcesAfterBook() {
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
        }
        GpuPendingGradients.cleanupThreadLocal();
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
        }
        if (DebugGpuTrain.isEnabled()) {
            agentLogB39372(
                    "H_book_handoff",
                    "LLMTrainer.releaseGpuResourcesAfterBook",
                    "after_book_cleanup",
                    "{}");
        }
    }

    private final GPTModel model;
    private final TrainingConfig config;
    private final DataLoader dataLoader;
    private final AdamOptimizer optimizer;
    private int globalStep;
    private float bestLoss;
    private final List<Tensor> parameters;
    /** Всего шагов оптимизатора за всё обучение: {@code epochs × ceil(numBatches / accumulationSteps)}. */
    private final int totalTrainingSteps;
    /** Линейный warmup: шаги {@code 1 … warmupSteps}, доля от {@link TrainingConfig#warmupRatio}. */
    private final int warmupSteps;

    /**
     * При FP16 matmul не {@code null}: динамический loss scale (overflow → уменьшить, серия успешных шагов →
     * увеличить). При обучении без FP16 matmul — {@code null}.
     */
    private final DynamicLossScaler dynamicLossScaler;
    /** Ранний выход после N шагов оптимизатора (env {@code JGPT_EXIT_AFTER_STEP}). */
    private final int exitAfterOptimizerSteps;

    /**
     * Кэш {@link TensorOpsGPU#useFp16Matmul()} на время жизни тренера (не меняется без перезапуска JVM).
     */
    private final boolean fp16Matmul;

    /**
     * {@code JGPT_CE_ASYNC}: CE в поток ставит kernel и D2H скаляра; вызывающий делает барьер перед
     * {@link GPTModel#backward} (готовность ∂logits) и после backward (готовность параметрических ∂ перед clip/step),
     * затем читает loss из pinned host (см. {@code train()}).
     */
    private final boolean ceAsyncDevice;
    /** {@code JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH}: сброс loss scale в начале каждой эпохи. */
    private final boolean fp16DynamicResetEachEpoch;

    /** Плановый номер шага ({@code globalStep + 1}) для агрегирования повторных overflow до успешного шага. */
    private int overflowLogPlannedStepKey = -1;
    private int overflowSkipRepeatCount;

    /** Переиспользуемый device-буфер int32 для CE (device logits path). */
    private GpuIntBuffer ceTargetsDevice;
    private int ceTargetsCapRows;
    private int[] ceHostTargetScratch;
    private GpuIntBuffer sampledCandidateIdsDevice;
    private int sampledCandidateIdsCapElems;
    private GpuFloatBuffer sampledCandidateLogitsDevice;
    private GpuFloatBuffer sampledCandidateGradDevice;
    private int sampledCandidateFloatCapElems;
    private int[] sampledCandidateIdsHostScratch;
    private int[] sampledSharedNegativeScratch;
    /** Число кандидатов на строку после {@link #prepareSampledCandidateIds} (для CE без повторной подготовки). */
    private int sampledTrainCandidatesPerRow;

    /** Скрач ∂logits для fused CE в {@link #evaluate()} (градиент не используется, {@code gradScale=0}). */
    private float[] evalCeGradScratch;

    /**
     * После {@link #zeroGpuGradsMarkingParamGradsClean}: ∂ обучаемых параметров на VRAM гарантированно нули.
     * Сбрасывается в {@link #clipAndOptimizerStep} (хостовый Adam без zeroGpuGrads) и при первом запуске.
     */
    private boolean gpuTrainableParamGradsKnownClean;

    /** Переиспользуемые массивы для {@link #checkGpuParamGradsNonFiniteFused} (без аллокации на шаг). */
    private GpuFloatBuffer[] nonFiniteParamBufsScratch = new GpuFloatBuffer[0];
    private int[] nonFiniteParamLensScratch = new int[0];

    /** Переиспользуемые массивы для sumSquares по GPU grad-буферам (без аллокации на шаг). */
    private long[] sumSqPtrsScratch = new long[0];
    private int[] sumSqLensScratch = new int[0];

    /** Scratch-список тензоров для FP16 unscale logits (размер <= 1, alloc пропускается если ещё нет). */
    private final List<Tensor> logitsOnlyScratch = new ArrayList<>(1);

    /**
     * Если задано: после синхронного v2-чекпоинта запись model_*.bin и токенизатора уходит в фон
     * (снимок весов клонируется на heap до возврата из {@link #saveCheckpoint}).
     */
    private final boolean checkpointAsyncIo;
    /** Очередь на один поток; цепочка {@link #checkpointIoTail} сохраняет порядок записей. */
    private final ExecutorService checkpointIoExecutor;
    private volatile CompletableFuture<Void> checkpointIoTail = CompletableFuture.completedFuture(null);

    /** Глобальная L2-норма градиентов после unscale, до/с клипом (последний успешный шаг). */
    private float lastGlobalGradNorm;

    public LLMTrainer(GPTModel model, TrainingConfig config, DataLoader dataLoader) {
        TensorOpsGPU.requireCuda("LLMTrainer");
        this.model = model;
        this.config = config;
        this.dataLoader = dataLoader;
        this.optimizer = AdamOptimizer.fromConfig(config);
        this.globalStep = 0;
        this.bestLoss = Float.MAX_VALUE;
        this.parameters = new ArrayList<>(model.getParameters());
        int batchesPerEpoch = Math.max(1, dataLoader.numBatches());
        int acc = config.accumulationSteps;
        int optimizerStepsPerEpoch = (batchesPerEpoch + acc - 1) / acc;
        this.totalTrainingSteps = Math.max(1, config.epochs * optimizerStepsPerEpoch);
        float wr = config.warmupRatio;
        if (wr > 0f) {
            this.warmupSteps = Math.max(1, (int) (totalTrainingSteps * wr));
        } else {
            this.warmupSteps = 0;
        }
        this.fp16Matmul = TensorOpsGPU.useFp16Matmul();
        this.dynamicLossScaler = DynamicLossScaler.fromEnvironmentIfFp16();
        this.fp16DynamicResetEachEpoch = readBooleanEnv("JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH", false);
        boolean ceAsyncEnv = readBooleanEnv("JGPT_CE_ASYNC", false);
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
        this.exitAfterOptimizerSteps = readPositiveEnvInt("JGPT_EXIT_AFTER_STEP", 0);
        this.lastGlobalGradNorm = 0f;
        this.checkpointAsyncIo =
                readBooleanEnv("JGPT_CHECKPOINT_ASYNC", false)
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

    private static boolean readBooleanEnv(String key, boolean defaultValue) {
        try {
            String e = System.getenv(key);
            if (e != null) {
                String t = e.trim();
                if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                    return true;
                }
                if ("0".equals(t) || "false".equalsIgnoreCase(t)) {
                    return false;
                }
            }
        } catch (Exception ignored) {
        }
        return defaultValue;
    }

    /** Масштаб для CE и градиента по логитам (текущий шаг оптимизатора / накопление). */
    private float lossScaleForForward() {
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
     * Сводка env и эффективных флагов для сверки с {@code scripts/run-training-gpu.sh}; ключи зондов
     * (JGPT_BATCH_PROBE*, JGPT_PROBE_*) в {@link #train()} не читаются.
     */
    private void logTensorTrainingEnvSnapshot() {
        log.info(
                "Сводка JGPT_* (обучение): батч direct={}, pinned={}; prefetch={}; async чекпоинт={}; "
                        + "выход после шага={}; trainPerf={}; profile={}; timings={}; generateGpuKv={}",
                dataLoader.usesDirectBatchBuffers(),
                dataLoader.usesPinnedHostBatchBuffers(),
                batchPrefetchEnabled(),
                checkpointAsyncIo,
                exitAfterOptimizerSteps > 0 ? Integer.toString(exitAfterOptimizerSteps) : "0 (выкл.)",
                envRawOrDash("JGPT_TRAIN_PERF"),
                envRawOrDash("JGPT_PROFILE"),
                envRawOrDash("JGPT_TIMINGS"),
                envRawOrDash("JGPT_GENERATE_GPU_KV"));
        log.info(
                "  пресет/env: E2E={} резидент (эфф.)={} резидент (env)={} pipeline декодера={} полный GPU шаг={} "
                        + "логиты GPU={} decoder bwd GPU={} train loss={} sampled candidates={} sampled negatives={} "
                        + "размер батча (ovr)={} кэш FP16={} FP16 dyn старт={} FP16 dyn интервал={} FP16 dyn макс={} "
                        + "FP16 aux soften scale={} CUDA_LIB={}",
                LLMConfig.gpuE2eTrainFromEnv(),
                LLMConfig.effectiveGpuResidentTraining(),
                envRawOrDash("JGPT_TRAIN_GPU_RESIDENT"),
                LLMConfig.decoderGpuPipelineFromEnvOrProp(),
                LLMConfig.fullGpuTrainStepFromEnv(),
                LLMConfig.deviceLogitsTrainStepFromEnv(),
                LLMConfig.deviceDecoderBackwardFromEnv(),
                config.trainLossMode,
                config.usesSampledTrainLoss() ? Integer.toString(config.sampledCeCandidates) : "-",
                config.usesSampledTrainLoss() ? config.sampledCeNegativeMode : "-",
                envRawOrDash("JGPT_BATCH_SIZE"),
                envRawOrDash("JGPT_ACTIVATION_CACHE_FP16"),
                envRawOrDash("JGPT_FP16_DYNAMIC_INITIAL"),
                envRawOrDash("JGPT_FP16_DYNAMIC_GROWTH_INTERVAL"),
                envRawOrDash("JGPT_FP16_DYNAMIC_MAX"),
                fp16AuxSoftenScaleAfterInfer() ? "1" : "0",
                envCudaLibSummary());
        if (TensorOpsGPU.isGpuAvailable()) {
            log.info(
                    "  TensorOpsGPU (при загрузке класса): FP16_MATMUL={} CE_мин_элементов={}",
                    TensorOpsGPU.useFp16Matmul(),
                    TensorOpsGPU.ceGpuMinElements());
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

    private static String envRawOrDash(String key) {
        String v = System.getenv(key);
        if (v == null) {
            return "—";
        }
        String t = v.trim();
        return t.isEmpty() ? "—" : t;
    }

    private static String envCudaLibSummary() {
        String v = System.getenv("JGPT_CUDA_LIB");
        if (v == null || v.isBlank()) {
            return "—";
        }
        return "(задан)";
    }

    public void train() throws IOException {
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
                        ? config.batchSize * config.maxSeqLen * effectiveSampledCandidateCount(config.vocabSize)
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
            log.info(
                    "План обучения уже выполнен: шаг {}/{} — дальнейшие шаги пропускаются.",
                    globalStep,
                    totalTrainingSteps);
            if (profiler != null) {
                profiler.printSummary();
            }
            return;
        }

        ExecutorService prefetchExecutor =
                batchPrefetchEnabled()
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

        outer:
        for (int epoch = 0; epoch < config.epochs; epoch++) {
            long epochStartNs = System.nanoTime();
            log.info(
                    "{} эпоха {}/{}: батчей в эпохе {}",
                    LogFmt.badge("EPOCH"),
                    epoch + 1,
                    config.epochs,
                    dataLoader.numBatches());
            dataLoader.shuffle();
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
                DataLoader.Batch batch;
                if (prefetchExecutor != null && prefetchFut != null) {
                    batch = takeNextBatchPrefetched(dataLoader, prefetchFut);
                } else {
                    batch = dataLoader.nextBatch();
                }
                prefetchFut = scheduleBatchPrefetch(dataLoader, prefetchExecutor);

                boolean lastBatchOfEpoch = !dataLoader.hasMore();
                microInAccum++;
                accTokens += (long) batch.input.getShape()[0] * config.maxSeqLen;

                long t0 = profile ? System.nanoTime() : 0L;
                Tensor logits;
                if (canDeviceSampledTrainForward()) {
                    int batchSz = batch.input.getShape()[0];
                    int seqLen = batch.input.getShape()[1];
                    int vocabSize = config.vocabSize;
                    int rows = batchSz * seqLen;
                    int candCount = effectiveSampledCandidateCount(vocabSize);
                    prepareSampledCandidateIds(batch.target, rows, vocabSize, candCount);
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
                    applyCrossEntropyLossAndGradDeviceAsync(logits, batch.target, ceScale);
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
                    loss = applyTrainLossAndGrad(logits, batch.target, ceScale);
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
                    scaleGradients(parameters, partialScale);
                    if (config.fullGpuTrainStep) {
                        scaleGpuGradients(model.gpuTensorByTrainableParameter(), partialScale);
                    }
                }

                float avgMicroLoss = accumLoss / (float) microInAccum;
                long t5 = profile ? System.nanoTime() : 0L;
                boolean stepped =
                        config.fullGpuTrainStep
                                ? clipAndOptimizerStepFullGpu(logits, avgMicroLoss)
                                : clipAndOptimizerStep(logits, avgMicroLoss);
                long t6 = profile ? System.nanoTime() : 0L;

                epochOptimizerAttempts++;
                epochLoss += avgMicroLoss;

                microInAccum = 0;
                accumLoss = 0f;

                if (!stepped) {
                    if (config.accumulationSteps > 1) {
                        zeroGradients(logits);
                        clearGpuParamGradsAfterOverflowSkip();
                        if (config.fullGpuTrainStep && model.isGpuResident()) {
                            zeroGpuGradsMarkingParamGradsClean(model.gpuTensorByTrainableParameter());
                        }
                        synchronizeGpuAfterOverflowSkip();
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
                    float evalLoss = evaluate();
                    /* До saveCheckpoint / следующего train-batch: сброс кэша/graph и стрим, иначе иногда нечисловые ∂ на
                     * первом шаге после eval (пересечение infer-цепочки с обучением на том же stream). */
                    synchronizeTrainingPipelineAfterGpuAuxiliaryInfer("eval");
                    if (profiler != null) {
                        profiler.armDetailWindow("eval");
                    }
                    if (Float.isFinite(evalLoss)) {
                        if (evalLoss < bestLoss) {
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
                        log.info(
                                "{} эпоха {}/{}: loss={} (лучший сохранённый={})",
                                LogFmt.badge("EVAL"),
                                epoch + 1,
                                config.epochs,
                                String.format(Locale.ROOT, "%.4f", evalLoss),
                                formatEvalBestLossForLog(bestLoss));
                    }
                    if (TensorOpsGPU.isGpuAvailable()) {
                        TensorOpsGPU.synchronizeStream();
                    }
                }

                if (globalStep % config.saveEverySteps == 0) {
                    saveCheckpoint("step_" + globalStep);
                }

                maybeAutoSample(epoch + 1);
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
            String durationStr = formatEpochDuration(epochElapsedNs);
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

            saveCheckpoint("epoch_" + (epoch + 1));
            dataLoader.reset();
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
            log.info("Лучший оценочный loss (сохранённый): {}", formatEvalBestLossForLog(bestLoss));
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
            if (TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.synchronizeStream();
            }
            GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    private static boolean batchPrefetchEnabled() {
        if (Boolean.getBoolean("jgpt.batch.prefetch")) {
            return true;
        }
        String p = System.getProperty("jgpt.batch.prefetch");
        if (p != null && ("0".equals(p.trim()) || "false".equalsIgnoreCase(p.trim()))) {
            return false;
        }
        String v = System.getenv("JGPT_BATCH_PREFETCH");
        if (v != null) {
            String t = v.trim();
            if ("0".equals(t) || "false".equalsIgnoreCase(t) || "no".equalsIgnoreCase(t)) {
                return false;
            }
        }
        return true;
    }

    private static CompletableFuture<DataLoader.Batch> scheduleBatchPrefetch(
            DataLoader loader, ExecutorService prefetchExecutor) {
        if (prefetchExecutor == null || !loader.hasMore()) {
            return null;
        }
        return CompletableFuture.supplyAsync(loader::buildBatchNoAdvance, prefetchExecutor);
    }

    private static DataLoader.Batch takeNextBatchPrefetched(
            DataLoader loader, CompletableFuture<DataLoader.Batch> prefetchFut) {
        try {
            DataLoader.Batch b = prefetchFut.get();
            loader.advanceAfterPreparedBatch();
            return b;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (ExecutionException e) {
            Throwable c = e.getCause();
            if (c instanceof Error) {
                throw (Error) c;
            }
            if (c instanceof RuntimeException) {
                throw (RuntimeException) c;
            }
            throw new RuntimeException(c);
        }
    }

    /**
     * После backward: при GPU-resident — merge ∂ в VRAM, finite-check на GPU, unscale/clip на device, затем D2H в
     * host-градиенты и Adam; иначе {@link GpuPendingGradients#flushAllToHost()} и проверка на CPU.
     *
     * @param avgMicroLoss средний loss по микробатчам в группе накопления (NaN/Inf → overflow)
     * @return {@code true}, если веса обновлены; {@code false} при overflow / нечисловым градиентам
     */
    private boolean clipAndOptimizerStep(Tensor logits, float avgMicroLoss) {
        if (model.isGpuResident()
                && TensorOpsGPU.isGpuAvailable()
                && GpuPendingGradients.allDirtyTargetsHaveGpuTensor(model.gpuTensorByTrainableParameter())) {
            return clipAndOptimizerStepGpuResidentMergeFirst(logits, avgMicroLoss);
        }
        GpuPendingGradients.flushAllToHost();
        if (fp16Matmul) {
            Fp16Metrics.global().recordStep();
        }
        boolean hasOverflow = checkGradientOverflow(logits, avgMicroLoss);
        if (fp16Matmul && hasOverflow) {
            Fp16Metrics.global().recordOverflow();
        }

        if (fp16Matmul) {
            if (!dynamicLossScaler.step(hasOverflow)) {
                zeroGradients(logits);
                clearGpuParamGradsAfterOverflowSkip();
                markGpuTrainableParamGradsMaybeDirtyAfterHostOptimizerPath();
                logGradientOverflowSkipped(dynamicLossScaler.getScale());
                synchronizeGpuAfterOverflowSkip();
                return false;
            }
        } else if (hasOverflow) {
            zeroGradients(logits);
            clearGpuParamGradsAfterOverflowSkip();
            markGpuTrainableParamGradsMaybeDirtyAfterHostOptimizerPath();
            synchronizeGpuAfterOverflowSkip();
            return false;
        }

        List<Tensor> gradsToUnscale = collectGradTensorsWithLossScale(logits);
        if (fp16Matmul) {
            dynamicLossScaler.unscaleGradients(gradsToUnscale);
        }

        float gradNorm = 0f;
        if (!gradsToUnscale.isEmpty()) {
            gradNorm = AdamOptimizer.clipGradientsGlobal(gradsToUnscale, config.maxGradNorm);
        }
        lastGlobalGradNorm = gradNorm;

        optimizerStep();
        model.onParametersUpdated();

        zeroGradients(logits);
        markGpuTrainableParamGradsMaybeDirtyAfterHostOptimizerPath();
        return true;
    }

    /**
     * Merge-first resident clip/optimizer: как {@link #clipAndOptimizerStepFullGpu} до Adam — overflow/unscale/clip по
     * ∂ на VRAM (и по logits на хосте); затем D2H ∂ в {@link Tensor#gradBuffer()} и {@link #optimizerStep()} с
     * хостовым Adam ({@link AdamOptimizer#stepAllWithParamGrad}). В отличие от full GPU здесь нет
     * {@link AdamOptimizer#stepAllGpuDevice}. Конфиг {@link TrainingConfig#mergeFirstGpuResidentTrain} задаётся вместе
     * с {@link TrainingConfig#deviceLogitsTrainStep} и {@link TrainingConfig#deviceDecoderBackward} — тот же
     * device-backward, что и при full GPU: накопление ∂ обучаемых параметров только в
     * {@link GpuTensor#gradBuffer()} / пуле {@link GpuPendingGradients}, без {@code accumulateAddGpuFromHost} после
     * {@link GpuPendingGradients#flushMergeToGpuGrads}.
     */
    private boolean clipAndOptimizerStepGpuResidentMergeFirst(Tensor logits, float avgMicroLoss) {
        if (fp16Matmul) {
            Fp16Metrics.global().recordStep();
        }
        Map<Tensor, GpuTensor> paramMap = model.gpuTensorByTrainableParameter();
        TensorOpsGPU.synchronizeStream();
        boolean lossNonFinite = !Float.isFinite(avgMicroLoss);
        boolean pendingNonFinite = GpuPendingGradients.anyNonFinitePending();
        boolean hasOverflow = lossNonFinite || pendingNonFinite;
        if (lossNonFinite || pendingNonFinite) {
            GpuPendingGradients.discardDirtyPending();
        } else {
            GpuPendingGradients.flushMergeToGpuGrads(paramMap);
        }
        boolean deviceGradNonFinite = false;
        if (!hasOverflow) {
            String info = checkGpuParamGradsNonFiniteFused(paramMap);
            if (info != null) {
                hasOverflow = true;
                deviceGradNonFinite = true;
            }
        }
        boolean logitsGradNonFinite =
                !hasOverflow && logits.hasGrad() && !floatArrayIsFinite(logits.gradBuffer());
        if (logitsGradNonFinite) {
            hasOverflow = true;
        }

        if (fp16Matmul && hasOverflow) {
            Fp16Metrics.global().recordOverflow();
        }
        if (fp16Matmul) {
            if (!dynamicLossScaler.step(hasOverflow)) {
                zeroGradients(logits);
                clearGpuParamGradsAfterOverflowSkip();
                zeroGpuGradsMarkingParamGradsClean(paramMap);
                logGradientOverflowSkipped(dynamicLossScaler.getScale());
                synchronizeGpuAfterOverflowSkip();
                return false;
            }
        } else if (hasOverflow) {
            zeroGradients(logits);
            clearGpuParamGradsAfterOverflowSkip();
            zeroGpuGradsMarkingParamGradsClean(paramMap);
            synchronizeGpuAfterOverflowSkip();
            return false;
        }

        float lossScaleForUnscale = fp16Matmul ? dynamicLossScaler.getScale() : 1f;
        if (lossScaleForUnscale > 1f) {
            DynamicLossScaler.unscaleGpuDeviceGrads(paramMap, lossScaleForUnscale);
            if (logits.hasGrad()) {
                logitsOnlyScratch.clear();
                logitsOnlyScratch.add(logits);
                dynamicLossScaler.unscaleGradients(logitsOnlyScratch);
            }
        }

        double sumSq = sumSquaresGpuParamGrads(paramMap);
        if (logits.hasGrad()) {
            float[] lg = logits.gradBuffer();
            if (lg.length > 0) {
                sumSq += TensorOpsGPU.sumSquaresGPU(lg, lg.length);
            }
        }
        float totalNorm = (float) Math.sqrt(sumSq);
        lastGlobalGradNorm = totalNorm;
        if (totalNorm > config.maxGradNorm && config.maxGradNorm > 0f) {
            float clipCoeff = config.maxGradNorm / totalNorm;
            for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
                GpuTensor gt = e.getValue();
                if (gt.hasGradBuffer()) {
                    TensorOpsGPU.scaleInPlaceGpuDevice(gt.gradBuffer(), e.getKey().size(), clipCoeff);
                }
            }
            if (logits.hasGrad()) {
                float[] lg = logits.gradBuffer();
                if (lg.length > 0) {
                    TensorOpsGPU.scaleInPlaceGPU(lg, lg.length, clipCoeff);
                }
            }
        }

        for (Tensor p : parameters) {
            GpuTensor gt = paramMap.get(p);
            if (gt != null && p.hasGrad() && gt.hasGradBuffer()) {
                gt.gradBuffer().copyTo(p.gradBuffer(), 0, p.size());
            }
        }

        optimizerStep();
        model.onParametersUpdated();

        zeroGpuGradsMarkingParamGradsClean(paramMap);
        zeroGradients(logits);
        return true;
    }

    /**
     * При серии неудачных попыток одного и того же планового шага ({@code globalStep + 1} без роста {@code globalStep})
     * — одно предупреждение; повторы на уровне debug.
     */
    private void logGradientOverflowSkipped(float scaleAfterSkip) {
        int planned = globalStep + 1;
        if (planned != overflowLogPlannedStepKey) {
            overflowLogPlannedStepKey = planned;
            overflowSkipRepeatCount = 0;
        }
        overflowSkipRepeatCount++;
        String scaleStr = String.format(Locale.ROOT, "%.4g", scaleAfterSkip);
        if (overflowSkipRepeatCount == 1) {
            log.warn(
                    "Шаг {} пропущен: переполнение градиентов/loss, масштаб loss (после сброса) {}×",
                    planned,
                    scaleStr);
        } else {
            log.debug(
                    "Шаг {} пропущен снова ({} подряд; scale {})",
                    planned,
                    overflowSkipRepeatCount,
                    scaleStr);
        }
    }

    /**
     * NaN/Inf в хост-градиентах параметров/логитов, при необходимости в device-градиентах (если full GPU + resident),
     * или в среднем loss. На горячем пути {@link TrainingConfig#fullGpuTrainStep} вызывается
     * {@link #clipAndOptimizerStepFullGpu(Tensor, float)}; этот метод — после {@link GpuPendingGradients#flushAllToHost()}.
     */
    private static int firstNonFiniteIndex(float[] a) {
        if (a == null) {
            return -1;
        }
        for (int i = 0; i < a.length; i++) {
            if (!Float.isFinite(a[i])) {
                return i;
            }
        }
        return -1;
    }

    private boolean checkGradientOverflow(Tensor logits, float avgMicroLoss) {
        int plannedStep = globalStep + 1;
        float ls = fp16Matmul && dynamicLossScaler != null ? dynamicLossScaler.getScale() : 1f;
        if (!Float.isFinite(avgMicroLoss)) {
            return true;
        }
        for (int pi = 0; pi < parameters.size(); pi++) {
            Tensor p = parameters.get(pi);
            if (p.hasGrad() && !floatArrayIsFinite(p.gradBuffer())) {
                return true;
            }
        }
        if (logits.hasGrad() && !floatArrayIsFinite(logits.gradBuffer())) {
            return true;
        }
        if (config.fullGpuTrainStep && model.isGpuResident()) {
            Map<Tensor, GpuTensor> gpuParams = model.gpuTensorByTrainableParameter();
            if (checkGpuParamGradsNonFiniteFused(gpuParams) != null) {
                return true;
            }
        }
        return false;
    }

    private List<Tensor> collectGradTensorsWithLossScale(Tensor logits) {
        List<Tensor> list = new ArrayList<>(parameters.size() + 1);
        for (Tensor p : parameters) {
            if (p.hasGrad()) {
                list.add(p);
            }
        }
        if (logits.hasGrad()) {
            list.add(logits);
        }
        return list;
    }

    private static boolean floatArrayIsFinite(float[] a) {
        for (float v : a) {
            if (!Float.isFinite(v)) {
                return false;
            }
        }
        return true;
    }

    private static void scaleGradients(List<Tensor> tensors, float scale) {
        for (Tensor t : tensors) {
            scaleTensorGrad(t, scale);
        }
    }

    private static void scaleGpuGradients(Map<Tensor, GpuTensor> paramMap, float scale) {
        if (scale == 1f) {
            return;
        }
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt.hasGradBuffer()) {
                TensorOpsGPU.scaleInPlaceGpuDevice(gt.gradBuffer(), e.getKey().size(), scale);
            }
        }
        GpuPendingGradients.scaleAll(scale);
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

    private static void scaleTensorGrad(Tensor t, float scale) {
        if (!t.hasGrad()) {
            return;
        }
        float[] g = t.gradBuffer();
        if (TensorOpsGPU.shouldUseGpuOptimizer(g.length)) {
            TensorOpsGPU.scaleInPlaceGPU(g, g.length, scale);
            return;
        }
        for (int i = 0; i < g.length; i++) {
            g[i] *= scale;
        }
    }

    private void zeroGradients(Tensor logits) {
        /* При full GPU step хостовые ∂ параметров не участвуют в backward/Adam — fill избыточен. */
        if (!config.fullGpuTrainStep || !model.isGpuResident()) {
            for (Tensor p : parameters) {
                if (p.hasGrad()) {
                    p.zeroGrad();
                }
            }
        }
        model.clearSampledTrainLossGrad();
        if (logits.hasGrad()) {
            logits.zeroGrad();
        }
    }

    /**
     * При пропуске шага оптимизатора из‑за overflow: хост-градиенты уже обнуляются в {@link #zeroGradients};
     * для GPU-резидентных весов и накопления микробатчей дополнительно сбрасываем ∂ на VRAM, иначе следующий
     * {@code backward(..., false)} может сложить NaN в device-буферы.
     */
    private void clearGpuParamGradsAfterOverflowSkip() {
        if (config.accumulationSteps > 1 && model.isGpuResident()) {
            model.zeroGpuTrainableParameterGrads();
        }
    }

    /**
     * CE loss + ∂L/∂logits: full-vocab CE или sampled train-only loss.
     * Возвращаемое значение — средний loss по токенам внутри выбранного train path.
     */
    private float applyTrainLossAndGrad(Tensor logits, Tensor target, float gradScale) {
        if (config.usesSampledTrainLoss()) {
            return applySampledTrainLossAndGradDevice(logits, target, gradScale);
        }
        return applyCrossEntropyLossAndGrad(logits, target, gradScale);
    }

    /**
     * CE loss + ∂L/∂logits: fused softmax+CE на GPU (хостовые или device-логиты).
     * Возвращаемое значение — средний loss по токенам (единый контракт для логирования и overflow).
     */
    private float applyCrossEntropyLossAndGrad(Tensor logits, Tensor target, float gradScale) {
        model.clearSampledTrainLossGrad();
        if (config.deviceLogitsTrainStep && model.hasDeviceLogitsBuffers()) {
            return applyCrossEntropyLossAndGradDevice(logits, target, gradScale);
        }
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];
        int totalTokens = batch * seqLen;
        if (!logits.hasGrad()) {
            logits.zeroGrad();
        }
        float[] gradData = logits.gradBuffer();

        final float lossScale = lossScaleForForward();
        if (!TensorOpsGPU.shouldUseGpuCrossEntropy(logits.size())) {
            return 0f;
        }
        float gradScaleOverTotal = ceFusedGradScaleOverTotal(totalTokens, gradScale);
        if (logits.isDirectStorage() && target.isDirectStorage()) {
            return TensorOpsGPU.crossEntropySoftmaxGradLossGpuDirectEx(
                    logits.directByteBuffer(),
                    0L,
                    target.directByteBuffer(),
                    0L,
                    gradData,
                    batch,
                    seqLen,
                    vocabSize,
                    gradScaleOverTotal,
                    fp16Matmul);
        }
        float[] logitData = logits.internalBuffer();
        float[] targetData = target.internalBuffer();
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpu(
                logitData, targetData, gradData, batch, seqLen, vocabSize, gradScaleOverTotal);
    }

    /** Множитель для ∂logits в fused CE (GPU хост-буфер и device logits): включает FP16 loss scale. */
    private float ceFusedGradScaleOverTotal(int totalTokens, float microbatchGradScale) {
        return microbatchGradScale * lossScaleForForward() / (float) totalTokens;
    }

    /**
     * CE по логитам на VRAM: JNI возвращает скаляр loss на хост — после завершения CE-kernel на потоке TensorOpsGPU
     * (синхронизация внутри native). При {@link #ceAsyncDevice} используется
     * {@link #applyCrossEntropyLossAndGradDeviceAsync}: {@link TensorOpsGPU#synchronizeStream()} перед
     * {@link GPTModel#backward}; затем loss — {@link TensorOpsGPU#crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost()}.
     */
    /**
     * Индекс класса для CE на GPU: невалидные значения → {@code -1} (строка в CE-kernel полностью в ноль).
     */
    private static int ceTokenIdOrInvalid(float v, int vocabSize) {
        if (!Float.isFinite(v) || v < 0f || v >= (float) vocabSize) {
            return -1;
        }
        int t = (int) v;
        if ((float) t != v) {
            return -1;
        }
        return t;
    }

    private void fillCeTargetsHostSanitized(Tensor target, int nrows, int vocabSize) {
        if (ceHostTargetScratch == null || ceHostTargetScratch.length < nrows) {
            ceHostTargetScratch = new int[nrows];
        }
        int invalid = 0;
        if (target.isDirectStorage()) {
            ByteBuffer bb = target.directByteBuffer();
            FloatBuffer fb = bb.asFloatBuffer();
            fb.clear();
            int capFloats = fb.limit();
            if (nrows > capFloats) {
                throw new IllegalArgumentException(
                        "CE targets: need " + nrows + " float ids, direct buffer has " + capFloats);
            }
            for (int i = 0; i < nrows; i++) {
                int t = ceTokenIdOrInvalid(fb.get(), vocabSize);
                ceHostTargetScratch[i] = t;
                if (t < 0) {
                    invalid++;
                }
            }
        } else {
            float[] td = target.internalBuffer();
            if (td.length < nrows) {
                throw new IllegalArgumentException(
                        "CE targets: need " + nrows + " floats, host buffer has " + td.length);
            }
            for (int i = 0; i < nrows; i++) {
                int t = ceTokenIdOrInvalid(td[i], vocabSize);
                ceHostTargetScratch[i] = t;
                if (t < 0) {
                    invalid++;
                }
            }
        }
        if (invalid > 0 && DebugGpuTrain.isEnabled()) {
            agentLogB39372(
                    "H_targets",
                    "LLMTrainer.fillCeTargetsDeviceSanitized",
                    "invalid_ce_targets",
                    "{\"globalStep\":"
                            + globalStep
                            + ",\"invalid\":"
                            + invalid
                            + ",\"nrows\":"
                            + nrows
                            + ",\"vocabSize\":"
                            + vocabSize
                            + "}");
        }
    }

    private void fillCeTargetsDeviceSanitized(Tensor target, int nrows, int vocabSize) {
        fillCeTargetsHostSanitized(target, nrows, vocabSize);
        ceTargetsDevice.copyFrom(ceHostTargetScratch, 0, nrows);
    }

    private float applyCrossEntropyLossAndGradDevice(Tensor logits, Tensor target, float gradScale) {
        model.clearSampledTrainLossGrad();
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];
        int totalTokens = batch * seqLen;
        float gradScaleOverTotal = ceFusedGradScaleOverTotal(totalTokens, gradScale);
        GpuFloatBuffer logitsGpu = model.deviceLogitsBuffer();
        GpuFloatBuffer gradGpu = model.ensureDeviceLogitsGradBuffer(batch * seqLen * vocabSize);
        gradGpu.clear();
        int nrows = totalTokens;
        if (ceTargetsDevice == null || ceTargetsCapRows < nrows) {
            if (ceTargetsDevice != null) {
                ceTargetsDevice.close();
            }
            ceTargetsDevice = GpuIntBuffer.allocate(nrows);
            ceTargetsCapRows = nrows;
        }
        fillCeTargetsDeviceSanitized(target, nrows, vocabSize);
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceTargetsDevice(
                logitsGpu,
                ceTargetsDevice,
                gradGpu,
                batch,
                seqLen,
                vocabSize,
                gradScaleOverTotal,
                fp16Matmul);
    }

    /**
     * Подготовка targets + async CE (без sync внутри). Вызывающий обязан: один
     * {@link TensorOpsGPU#synchronizeStream()} до {@link GPTModel#backward} и второй — после backward до clip/step;
     * loss — {@link TensorOpsGPU#crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost()}.
     */
    private void applyCrossEntropyLossAndGradDeviceAsync(Tensor logits, Tensor target, float gradScale) {
        model.clearSampledTrainLossGrad();
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];
        int totalTokens = batch * seqLen;
        float gradScaleOverTotal = ceFusedGradScaleOverTotal(totalTokens, gradScale);
        GpuFloatBuffer logitsGpu = model.deviceLogitsBuffer();
        GpuFloatBuffer gradGpu = model.ensureDeviceLogitsGradBuffer(batch * seqLen * vocabSize);
        gradGpu.clear();
        int nrows = totalTokens;
        if (ceTargetsDevice == null || ceTargetsCapRows < nrows) {
            if (ceTargetsDevice != null) {
                ceTargetsDevice.close();
            }
            ceTargetsDevice = GpuIntBuffer.allocate(nrows);
            ceTargetsCapRows = nrows;
        }
        fillCeTargetsDeviceSanitized(target, nrows, vocabSize);
        TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceTargetsDeviceAsync(
                logitsGpu,
                ceTargetsDevice,
                gradGpu,
                batch,
                seqLen,
                vocabSize,
                gradScaleOverTotal,
                fp16Matmul);
    }

    private int effectiveSampledCandidateCount(int vocabSize) {
        return Math.max(2, Math.min(config.sampledCeCandidates, vocabSize));
    }

    /** Sampled train: декодер GPU + кандидатный LM-head без полных логитов на VRAM. */
    private boolean canDeviceSampledTrainForward() {
        return config.usesSampledTrainLoss()
                && config.deviceLogitsTrainStep
                && config.deviceDecoderBackward
                && model.canFullGpuTrain()
                && model.isDeviceLogitsEnabled();
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 33)) * 0xff51afd7ed558ccdL;
        z = (z ^ (z >>> 33)) * 0xc4ceb9fe1a85ec53L;
        return z ^ (z >>> 33);
    }

    private static int deterministicCandidateIndex(long key, int vocabSize) {
        return (int) Long.remainderUnsigned(mix64(key), vocabSize);
    }

    private static boolean containsCandidate(int[] ids, int base, int used, int candidate) {
        for (int i = 0; i < used; i++) {
            if (ids[base + i] == candidate) {
                return true;
            }
        }
        return false;
    }

    private static int nextDistinctCandidate(
            int vocabSize, int excludedTarget, int[] ids, int base, int used, long key) {
        if (vocabSize <= 1) {
            return -1;
        }
        int candidate = deterministicCandidateIndex(key, vocabSize);
        for (int tries = 0; tries < vocabSize; tries++) {
            if (candidate != excludedTarget && !containsCandidate(ids, base, used, candidate)) {
                return candidate;
            }
            candidate++;
            if (candidate >= vocabSize) {
                candidate = 0;
            }
        }
        return -1;
    }

    private void ensureSampledCandidateDeviceBuffers(int totalCandidateElems) {
        if (sampledCandidateIdsDevice == null || sampledCandidateIdsCapElems < totalCandidateElems) {
            if (sampledCandidateIdsDevice != null) {
                sampledCandidateIdsDevice.close();
            }
            sampledCandidateIdsDevice = GpuIntBuffer.allocate(totalCandidateElems);
            sampledCandidateIdsCapElems = totalCandidateElems;
        }
        if (sampledCandidateLogitsDevice == null || sampledCandidateFloatCapElems < totalCandidateElems) {
            if (sampledCandidateLogitsDevice != null) {
                sampledCandidateLogitsDevice.close();
            }
            if (sampledCandidateGradDevice != null) {
                sampledCandidateGradDevice.close();
            }
            sampledCandidateLogitsDevice = GpuFloatBuffer.allocate(totalCandidateElems);
            sampledCandidateGradDevice = GpuFloatBuffer.allocate(totalCandidateElems);
            sampledCandidateFloatCapElems = totalCandidateElems;
        }
    }

    private int prepareSampledCandidateIds(Tensor target, int rows, int vocabSize, int candidates) {
        fillCeTargetsHostSanitized(target, rows, vocabSize);
        int total = rows * candidates;
        if (sampledCandidateIdsHostScratch == null || sampledCandidateIdsHostScratch.length < total) {
            sampledCandidateIdsHostScratch = new int[total];
        }
        int negativeCount = candidates - 1;
        if (sampledSharedNegativeScratch == null || sampledSharedNegativeScratch.length < negativeCount) {
            sampledSharedNegativeScratch = new int[negativeCount];
        }
        long sharedSeed =
                mix64((((long) globalStep + 1L) << 32) ^ ((long) rows << 8) ^ ((long) vocabSize << 1) ^ candidates);
        for (int j = 0; j < negativeCount; j++) {
            sampledSharedNegativeScratch[j] =
                    nextDistinctCandidate(
                            vocabSize,
                            -1,
                            sampledSharedNegativeScratch,
                            0,
                            j,
                            sharedSeed ^ ((long) (j + 1) * 0x9E3779B97F4A7C15L));
        }
        for (int row = 0; row < rows; row++) {
            int base = row * candidates;
            int targetId = ceHostTargetScratch[row];
            sampledCandidateIdsHostScratch[base] = targetId;
            if (targetId < 0) {
                for (int j = 1; j < candidates; j++) {
                    sampledCandidateIdsHostScratch[base + j] = -1;
                }
                continue;
            }
            int used = 1;
            for (int j = 0; j < negativeCount; j++) {
                int neg = sampledSharedNegativeScratch[j];
                if (neg == targetId || containsCandidate(sampledCandidateIdsHostScratch, base, used, neg)) {
                    neg =
                            nextDistinctCandidate(
                                    vocabSize,
                                    targetId,
                                    sampledCandidateIdsHostScratch,
                                    base,
                                    used,
                                    sharedSeed
                                            ^ ((long) (row + 1) * 0xD1B54A32D192ED03L)
                                            ^ ((long) (j + 1) * 0x94D049BB133111EBL));
                }
                sampledCandidateIdsHostScratch[base + used] = neg;
                used++;
            }
        }
        ensureSampledCandidateDeviceBuffers(total);
        sampledCandidateIdsDevice.copyFrom(sampledCandidateIdsHostScratch, 0, total);
        sampledTrainCandidatesPerRow = candidates;
        return candidates;
    }

    private float applySampledTrainLossAndGradDevice(Tensor logits, Tensor target, float gradScale) {
        if (!config.deviceLogitsTrainStep || !model.hasDeviceSampledTrainLmHeadActivations()) {
            throw new IllegalStateException("sampled train loss requires sampled device forward (LM activations without full logits)");
        }
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int rows = batch * seqLen;
        int candidates = sampledTrainCandidatesPerRow;
        if (candidates <= 0) {
            throw new IllegalStateException("sampled train loss: candidates not prepared before forward");
        }
        Objects.requireNonNull(target, "target");
        if (sampledCandidateLogitsDevice == null
                || sampledCandidateGradDevice == null
                || sampledCandidateIdsDevice == null) {
            throw new IllegalStateException("sampled train buffers missing");
        }
        long need = (long) rows * candidates;
        if (sampledCandidateLogitsDevice.numFloats() < need
                || sampledCandidateGradDevice.numFloats() < need
                || sampledCandidateIdsDevice.numInts() < need) {
            throw new IllegalStateException("sampled train buffers too small for logits shape");
        }
        sampledCandidateGradDevice.clear();
        model.clearSampledTrainLossGrad();
        float loss =
                TensorOpsGPU.sampledCrossEntropyGradLossGpuDeviceFirstSlot(
                        sampledCandidateLogitsDevice,
                        sampledCandidateIdsDevice,
                        sampledCandidateGradDevice,
                        rows,
                        candidates,
                        ceFusedGradScaleOverTotal(rows, gradScale));
        model.setSampledTrainLossGrad(sampledCandidateIdsDevice, sampledCandidateGradDevice, candidates);
        return loss;
    }

    private float evaluateCrossEntropyLoss(Tensor logits, Tensor target) {
        int len = logits.size();
        if (!TensorOpsGPU.shouldUseGpuCrossEntropy(len)) {
            return 0f;
        }
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];
        float[] logitData = logits.internalBuffer();
        float[] targetData = target.internalBuffer();
        if (evalCeGradScratch == null || evalCeGradScratch.length < len) {
            evalCeGradScratch = new float[len];
        }
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpu(
                logitData, targetData, evalCeGradScratch, batch, seqLen, vocabSize, 0f);
    }

    /** CE на device: логиты уже в {@code logitsGpu}; {@code gradScale=0}, буфер ∂ переиспользуется с обнулением. */
    private float evaluateCrossEntropyLossDevice(
            Tensor target, GpuFloatBuffer logitsGpu, int batch, int seqLen, int vocabSize) {
        int len = batch * seqLen * vocabSize;
        if (!TensorOpsGPU.shouldUseGpuCrossEntropy(len)) {
            return 0f;
        }
        GpuFloatBuffer grad = model.ensureDeviceLogitsGradBuffer(len);
        grad.clear();
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpuDevice(
                logitsGpu,
                target.internalBuffer(),
                grad,
                batch,
                seqLen,
                vocabSize,
                0f,
                fp16Matmul);
    }

    private void optimizerStep() {
        int stepForLr = globalStep + 1;
        optimizer.setLearningRate(learningRateForStep(stepForLr));
        optimizer.beginStep();
        optimizer.stepAllWithParamGrad(parameters);
    }

    private static final int SAMPLE_MAX_NEW_TOKENS = 64;
    private static final float SAMPLE_TEMP = 0.9f;
    private static final int SAMPLE_TOP_K = 40;

    /** Ровно по пять слов — для сэмпла во время обучения. */
    private static final String[] AUTO_PROMPTS_RU = {
        "мороз и солнце день чудесный",
        "весна пришла в город сегодня",
        "он сказал ей одно слово",
        "книга лежала на столе тихо",
        "ветер гонит тучи прочь сейчас",
        "старинная улица встретила утро зарёю",
        "тишина стояла в комнате ночью",
        "дети бежали навстречу лету радостно"
    };

    /**
     * Читает промпт для промежуточной генерации.
     * Если задан {@code JGPT_SAMPLE_PROMPT} — использует его (можно задать несколько через {@code |},
     * тогда выбирается по шагу: {@code "промпт1|промпт2|промпт3"}).
     * Иначе — берёт из {@link #AUTO_PROMPTS_RU}.
     */
    private String pickSamplePrompt(int epochOneBased) {
        String env = System.getenv("JGPT_SAMPLE_PROMPT");
        if (env != null && !env.isBlank()) {
            String[] parts = env.split("\\|");
            return parts[(globalStep + epochOneBased) % parts.length].trim();
        }
        return AUTO_PROMPTS_RU[(globalStep + epochOneBased) % AUTO_PROMPTS_RU.length];
    }

    private void maybeAutoSample(int epochOneBased) {
        if (config.interactiveSampleEverySteps <= 0) {
            return;
        }
        if (globalStep % config.interactiveSampleEverySteps != 0) {
            return;
        }
        String prompt = pickSamplePrompt(epochOneBased);
        log.info(
                "{} промежуточная генерация: эпоха {}/{}, шаг {}",
                LogFmt.badge("SAMPLE"),
                epochOneBased,
                config.epochs,
                globalStep);
        log.info("{} промпт: {}", LogFmt.badge("SAMPLE"), prompt);
        try {
            model.zeroGradParameters();
            String out =
                    LlmTextGeneration.generateText(
                            model,
                            dataLoader.getTokenizer(),
                            prompt,
                            SAMPLE_MAX_NEW_TOKENS,
                            SAMPLE_TEMP,
                            SAMPLE_TOP_K);
            log.info("{} сгенерировано: {}", LogFmt.badge("SAMPLE"), out);
        } catch (Exception e) {
            log.warn("{} генерация не удалась: {}", LogFmt.badge("SAMPLE"), e.getMessage());
        } finally {
            synchronizeTrainingPipelineAfterGpuAuxiliaryInfer("sample");
        }
    }

    /**
     * Оценка loss на нескольких батчах (не горячий цикл обучения): fused CE на GPU; при пустых логитах — 0.
     * При полном VRAM infer-пайплайне и {@code useGpuResident} — forward без D2H логитов и CE по
     * {@link GPTModel#deviceLogitsBuffer()} (меньше PCIe на батч; граница стрима только в нативном CE при D2H
     * скаляра loss).
     */
    private float evaluate() {
        int saved = dataLoader.getCurrentIndex();
        float total = 0f;
        int n = 0;
        int maxBatches = Math.min(8, dataLoader.numBatches());
        boolean deviceLogitsEval = false;
        for (int i = 0; i < maxBatches && dataLoader.hasMore(); i++) {
            DataLoader.Batch batch = dataLoader.nextBatch();
            int[] inSh = batch.input.getShape();
            int batchSize = inSh[0];
            int seqLen = inSh[1];
            if (i == 0) {
                deviceLogitsEval =
                        config.useGpuResident && model.canInferLogitsOnDevice(batchSize, seqLen);
            }
            if (deviceLogitsEval) {
                model.forward(batch.input, false, true, true);
                GpuFloatBuffer logitsGpu = model.deviceLogitsBuffer();
                total +=
                        evaluateCrossEntropyLossDevice(
                                batch.target, logitsGpu, batchSize, seqLen, config.vocabSize);
            } else {
                Tensor logits = model.forward(batch.input, false, config.useGpuResident);
                total += evaluateCrossEntropyLoss(logits, batch.target);
            }
            n++;
        }
        dataLoader.setCurrentIndex(saved);
        if (deviceLogitsEval && n > 0 && TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
        }
        if (n == 0) {
            log.warn(
                    "{} ни одного eval-батча (hasMore={}) — не обновляем best/patience early-stop",
                    LogFmt.badge("EVAL"),
                    dataLoader.hasMore());
            return Float.NaN;
        }
        float loss = total / n;
        float perplexity = (float) Math.exp(loss);
        log.info("{} перплексия: {}", LogFmt.badge("EVAL"), String.format("%.2f", perplexity));
        return loss;
    }

    private static final String CHECKPOINT_FORMAT_V2 = "veles.ckpt.v2";

    public void saveCheckpoint(String name) throws IOException {
        Path dir = Path.of(config.checkpointDir);
        Files.createDirectories(dir);
        String path = config.checkpointDir + "/checkpoint_" + name + ".bin";

        // Обновляем state/last_step.txt для внешнего монитора (jgpt-monitor.sh)
        try {
            Path stateDir = Path.of("state");
            Files.createDirectories(stateDir);
            Files.writeString(stateDir.resolve("last_step.txt"), String.valueOf(globalStep));
        } catch (IOException ignored) {
            // Не критично — мониторинг опциональный
        }

        if (config.fullGpuTrainStep && model.isGpuResident()) {
            model.syncWeightsFromGpu(model.gpuTensorByTrainableParameter());
        }

        try (DataOutputStream out =
                new DataOutputStream(
                        new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeUTF(CHECKPOINT_FORMAT_V2);
            out.writeInt(globalStep);
            out.writeFloat(bestLoss);
            optimizer.setStep(globalStep);
            optimizer.writeMomentBuffers(out, parameters);
        }
        log.info("{} checkpoint(v2+Adam): {}", LogFmt.badge("CKPT"), path);

        if (checkpointAsyncIo && checkpointIoExecutor != null) {
            List<Tensor> params = model.getParameters();
            List<float[]> weightSnap = new ArrayList<>(params.size());
            for (Tensor p : params) {
                weightSnap.add(p.internalBuffer().clone());
            }
            checkpointIoTail =
                    checkpointIoTail.thenRunAsync(
                            () -> {
                                try {
                                    writeModelWeightsFromSnapshot(name, weightSnap);
                                } catch (IOException e) {
                                    log.error("Асинхронная запись весов чекпоинта не удалась: {}", name, e);
                                }
                            },
                            checkpointIoExecutor);
            log.info("{} веса checkpoint '{}' поставлены в очередь асинхронной записи", LogFmt.badge("CKPT"), name);
        } else {
            saveModelWeights(name);
        }
    }

    private static void writeFloatArrayBigEndian(DataOutputStream out, float[] buf) throws IOException {
        if (buf.length == 0) {
            return;
        }
        ByteBuffer bb = ByteBuffer.allocate(buf.length * 4).order(ByteOrder.BIG_ENDIAN);
        bb.asFloatBuffer().put(buf);
        out.write(bb.array());
    }

    /** Сохраняет веса модели в {@code <checkpointDir>/model_<name>.bin}. */
    public void saveModelWeights(String name) throws IOException {
        Path dir = Path.of(config.checkpointDir);
        Files.createDirectories(dir);
        String modelPath = dir.resolve("model_" + name + ".bin").toString();

        if (config.fullGpuTrainStep && model.isGpuResident()) {
            model.syncWeightsFromGpu(model.gpuTensorByTrainableParameter());
        }

        try (DataOutputStream out =
                new DataOutputStream(new BufferedOutputStream(new FileOutputStream(modelPath)))) {
            List<Tensor> params = model.getParameters();
            out.writeUTF(GPTModel.MODEL_WEIGHTS_FORMAT_V1);
            out.writeInt(params.size());
            for (Tensor param : params) {
                int[] shape = param.getShape();
                out.writeInt(shape.length);
                for (int d : shape) {
                    out.writeInt(d);
                }
                writeFloatArrayBigEndian(out, param.internalBuffer());
            }
        }
        log.info("{} веса модели записаны: {}", LogFmt.badge("CKPT"), modelPath);

        String tokPath = dir.resolve("tokenizer_" + name + ".bin").toString();
        dataLoader.getTokenizer().save(tokPath);
        log.info("{} токенизатор записан: {}", LogFmt.badge("CKPT"), tokPath);
    }

    /**
     * Запись model_/tokenizer_ без повторного {@link GPTModel#syncWeightsFromGpu}; веса — уже клоны буферов
     * с момента снимка.
     */
    private void writeModelWeightsFromSnapshot(String name, List<float[]> weightSnap) throws IOException {
        Path dir = Path.of(config.checkpointDir);
        Files.createDirectories(dir);
        String modelPath = dir.resolve("model_" + name + ".bin").toString();
        List<Tensor> params = model.getParameters();
        if (params.size() != weightSnap.size()) {
            throw new IllegalStateException("weight snapshot size mismatch");
        }
        try (DataOutputStream out =
                new DataOutputStream(new BufferedOutputStream(new FileOutputStream(modelPath)))) {
            out.writeUTF(GPTModel.MODEL_WEIGHTS_FORMAT_V1);
            out.writeInt(params.size());
            for (int i = 0; i < params.size(); i++) {
                int[] shape = params.get(i).getShape();
                out.writeInt(shape.length);
                for (int d : shape) {
                    out.writeInt(d);
                }
                writeFloatArrayBigEndian(out, weightSnap.get(i));
            }
        }
        log.info("{} веса модели записаны (асинхронный снимок): {}", LogFmt.badge("CKPT"), modelPath);
        String tokPath = dir.resolve("tokenizer_" + name + ".bin").toString();
        dataLoader.getTokenizer().save(tokPath);
        log.info("{} токенизатор записан: {}", LogFmt.badge("CKPT"), tokPath);
    }

    /** Дождаться завершения фоновых записей чекпоинта (для тестов / корректного выхода). */
    public void awaitPendingCheckpointWrites() {
        if (checkpointIoExecutor == null) {
            return;
        }
        try {
            checkpointIoTail.get();
        } catch (Exception e) {
            log.warn("Ожидание фоновой записи чекпоинта: {}", e.toString());
        }
    }

    public void loadCheckpoint(String path) throws IOException, ClassNotFoundException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(path))) {
            bis.mark(1 << 20);
            DataInputStream dis = new DataInputStream(bis);
            String tag;
            try {
                tag = dis.readUTF();
            } catch (IOException e) {
                bis.reset();
                loadLegacyCheckpoint(path);
                return;
            }
            if (CHECKPOINT_FORMAT_V2.equals(tag)) {
                globalStep = dis.readInt();
                bestLoss = dis.readFloat();
                if (bestLoss == 0f) {
                    log.warn(
                            "Чекпоинт: в файле bestLoss=0 (часто артефакт eval без батчей в старых прогонах) — сброс к «ещё не зафиксирован»");
                    bestLoss = Float.MAX_VALUE;
                }
                optimizer.setStep(globalStep);
                optimizer.readMomentBuffers(dis, parameters);
                log.info(
                        "Чекпоинт загружен (v2 + Adam): {} (шаг {}, лучший оценочный loss {})",
                        path,
                        globalStep,
                        formatEvalBestLossForLog(bestLoss));
                return;
            }
            bis.reset();
        }
        loadLegacyCheckpoint(path);
    }

    private void loadLegacyCheckpoint(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            globalStep = in.readInt();
            bestLoss = in.readFloat();
            if (bestLoss == 0f) {
                log.warn(
                        "Чекпоинт (legacy): bestLoss=0 — сброс к «ещё не зафиксирован»");
                bestLoss = Float.MAX_VALUE;
            }
            optimizer.setStep(globalStep);
            log.info(
                    "Чекпоинт загружен (старый формат, без буферов Adam m/v): {} (шаг {}, лучший loss {})",
                    path,
                    globalStep,
                    formatEvalBestLossForLog(bestLoss));
        }
    }

    /** Загрузка весов из {@code <checkpointDir>/model_<name>.bin}. */
    public void loadModelWeights(String name) throws IOException, ClassNotFoundException {
        String modelPath = Path.of(config.checkpointDir).resolve("model_" + name + ".bin").toString();
        model.loadWeights(modelPath);
    }

    public int getGlobalStep() {
        return globalStep;
    }

    public float getBestLoss() {
        return bestLoss;
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
    }

    /**
     * В чекпоинте и при старте «лучший eval loss» часто равен {@link Float#MAX_VALUE} — признак того, что ещё не было
     * оценки или улучшения; без этого {@code %.4f} даёт огромное «мусорное» число в логе.
     */
    private static String formatEvalBestLossForLog(float v) {
        if (!Float.isFinite(v) || v == Float.MAX_VALUE) {
            return "нет (ещё не зафиксирован)";
        }
        return String.format(Locale.ROOT, "%.4f", v);
    }

    /** Человекочитаемая длительность (ru-RU); если эпоха короче минуты — с долями секунды. */
    private static String formatEpochDuration(long nanos) {
        if (nanos <= 0L) {
            return "0 с";
        }
        long totalSec = nanos / 1_000_000_000L;
        double secExact = nanos / 1_000_000_000.0;
        long h = totalSec / 3600L;
        long m = (totalSec % 3600L) / 60L;
        long s = totalSec % 60L;
        Locale ru = Locale.forLanguageTag("ru-RU");
        if (h > 0L) {
            return String.format(ru, "%d ч %02d мин %02d с", h, m, s);
        }
        if (m > 0L) {
            return String.format(ru, "%d мин %02d с", m, s);
        }
        return String.format(ru, "%.1f с", secExact);
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
        if (canDeviceSampledTrainForward()) {
            int batchSz = batch.input.getShape()[0];
            int seqLen = batch.input.getShape()[1];
            int vocabSize = config.vocabSize;
            int rows = batchSz * seqLen;
            int candCount = effectiveSampledCandidateCount(vocabSize);
            prepareSampledCandidateIds(batch.target, rows, vocabSize, candCount);
            logits =
                    model.forwardTrainingDeviceSampled(
                            batch.input, sampledCandidateIdsDevice, sampledCandidateLogitsDevice, candCount);
        } else {
            logits = model.forward(batch.input, true, config.useGpuResident);
        }
        float ceScale = 1f / (float) config.accumulationSteps;
        float loss;
        if (ceAsyncDevice && !config.usesSampledTrainLoss() && config.deviceLogitsTrainStep && model.hasDeviceLogitsBuffers()) {
            applyCrossEntropyLossAndGradDeviceAsync(logits, batch.target, ceScale);
            TensorOpsGPU.synchronizeStream();
            model.backward(logits, zeroGrads);
            TensorOpsGPU.synchronizeStream();
            loss = TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost();
        } else {
            loss = applyTrainLossAndGrad(logits, batch.target, ceScale);
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
                scaleGradients(parameters, partialScale);
                scaleGpuGradients(model.gpuTensorByTrainableParameter(), partialScale);
            }
            stepped = clipAndOptimizerStepFullGpu(logits, loss);
        } else {
            if (partialScale != 1f) {
                scaleGradients(parameters, partialScale);
            }
            stepped = clipAndOptimizerStep(logits, loss);
        }
        if (stepped) {
            overflowLogPlannedStepKey = -1;
            overflowSkipRepeatCount = 0;
        }
        return stepped;
    }

    /**
     * Full-GPU clip/optimizer step: overflow check, unscale, clip, Adam — все ∂ параметров на VRAM.
     * После {@link GpuPendingGradients#flushMergeToGpuGrads} градиенты уже полные на device (декодер через pending,
     * эмбеддинги/LM head — прямое накопление в {@link GpuTensor#gradBuffer()} в backward); хостовый {@link
     * Tensor#gradBuffer()} не мержится.
     */
    private boolean clipAndOptimizerStepFullGpu(Tensor logits, float avgMicroLoss) {
        Map<Tensor, GpuTensor> paramMap = model.gpuTensorByTrainableParameter();
        int stepForLr = globalStep + 1;

        if (fp16Matmul) {
            Fp16Metrics.global().recordStep();
        }
        TensorOpsGPU.synchronizeStream();
        boolean lossNonFinite = !Float.isFinite(avgMicroLoss);
        boolean pendingNonFinite = GpuPendingGradients.anyNonFinitePending();
        String pendingDebug = pendingNonFinite ? GpuPendingGradients.firstNonFinitePendingDebugInfo() : "";
        boolean hasOverflow = lossNonFinite || pendingNonFinite;
        if (lossNonFinite || pendingNonFinite) {
            GpuPendingGradients.discardDirtyPending();
        } else {
            GpuPendingGradients.flushMergeToGpuGrads(paramMap);
        }
        String firstNonFiniteGrad = null;
        if (!hasOverflow) {
            firstNonFiniteGrad = checkGpuParamGradsNonFiniteFused(paramMap);
            if (firstNonFiniteGrad != null) {
                hasOverflow = true;
            }
        }
        boolean logitsGradNonFinite = !hasOverflow && logits.hasGrad() && !floatArrayIsFinite(logits.gradBuffer());
        if (logitsGradNonFinite) {
            hasOverflow = true;
        }

        if (DebugGpuTrain.isEnabled() && (hasOverflow || globalStep >= 10)) {
            String primary;
            if (lossNonFinite) {
                primary = "avgMicroLoss";
            } else if (pendingNonFinite) {
                primary = "pendingNonFinite";
            } else if (firstNonFiniteGrad != null) {
                primary = "deviceParamGrad";
            } else if (logitsGradNonFinite) {
                primary = "logitsHostGrad";
            } else if (!hasOverflow) {
                primary = "preUnscaleOk";
            } else {
                primary = "unknown";
            }
            agentLogB39372(
                    "H_ovf",
                    "LLMTrainer.clipAndOptimizerStepFullGpu",
                    "state",
                    "{\"plannedStep\":"
                            + stepForLr
                            + ",\"globalStep\":"
                            + globalStep
                            + ",\"hasOverflow\":"
                            + hasOverflow
                            + ",\"primary\":\""
                            + jsonEsc(primary)
                            + "\",\"avgMicroLoss\":"
                            + avgMicroLoss
                            + ",\"lossScale\":"
                            + (fp16Matmul && dynamicLossScaler != null ? dynamicLossScaler.getScale() : 1f)
                            + ",\"pending\":"
                            + pendingNonFinite
                            + ",\"logitsHostBad\":"
                            + logitsGradNonFinite
                            + ",\"devGradHint\":\""
                            + jsonEsc(firstNonFiniteGrad)
                            + "\",\"pendingKey\":\""
                            + jsonEsc(pendingDebug)
                            + "\",\"cudaLib\":\""
                            + jsonEsc(
                                    TensorCudaLibrary.getLastLoadedPath() != null
                                            ? TensorCudaLibrary.getLastLoadedPath()
                                            : "")
                            + "\"}");
        }

        if (fp16Matmul && hasOverflow) {
            Fp16Metrics.global().recordOverflow();
        }
        if (fp16Matmul) {
            if (!dynamicLossScaler.step(hasOverflow)) {
                zeroGradients(logits);
                clearGpuParamGradsAfterOverflowSkip();
                zeroGpuGradsMarkingParamGradsClean(paramMap);
                logGradientOverflowSkipped(dynamicLossScaler.getScale());
                synchronizeGpuAfterOverflowSkip();
                return false;
            }
        } else if (hasOverflow) {
            zeroGradients(logits);
            clearGpuParamGradsAfterOverflowSkip();
            zeroGpuGradsMarkingParamGradsClean(paramMap);
            synchronizeGpuAfterOverflowSkip();
            return false;
        }

        float lossScaleForUnscale = fp16Matmul ? dynamicLossScaler.getScale() : 1f;
        if (lossScaleForUnscale > 1f) {
            DynamicLossScaler.unscaleGpuDeviceGrads(paramMap, lossScaleForUnscale);
            if (logits.hasGrad()) {
                logitsOnlyScratch.clear();
                logitsOnlyScratch.add(logits);
                dynamicLossScaler.unscaleGradients(logitsOnlyScratch);
            }
        }

        double sumSq = sumSquaresGpuParamGrads(paramMap);
        if (logits.hasGrad()) {
            float[] lg = logits.gradBuffer();
            if (lg.length > 0) {
                sumSq += TensorOpsGPU.sumSquaresGPU(lg, lg.length);
            }
        }
        float totalNorm = (float) Math.sqrt(sumSq);
        lastGlobalGradNorm = totalNorm;
        if (!Float.isFinite(totalNorm)) {
            if (DebugGpuTrain.isEnabled()) {
                agentLogB39372(
                        "H_norm",
                        "LLMTrainer.clipAndOptimizerStepFullGpu",
                        "non_finite_total_norm",
                        "{\"plannedStep\":"
                                + stepForLr
                                + ",\"globalStep\":"
                                + globalStep
                                + ",\"sumSq\":"
                                + sumSq
                                + ",\"totalNorm\":"
                                + totalNorm
                                + ",\"lossScale\":"
                                + (fp16Matmul && dynamicLossScaler != null ? dynamicLossScaler.getScale() : 1f)
                                + "}");
            }
            if (fp16Matmul) {
                Fp16Metrics.global().recordOverflow();
                dynamicLossScaler.step(true);
            }
            zeroGradients(logits);
            clearGpuParamGradsAfterOverflowSkip();
            zeroGpuGradsMarkingParamGradsClean(paramMap);
            logGradientOverflowSkipped(fp16Matmul && dynamicLossScaler != null ? dynamicLossScaler.getScale() : 1f);
            synchronizeGpuAfterOverflowSkip();
            return false;
        }
        float clipCoeff = 1f;
        if (totalNorm > config.maxGradNorm && config.maxGradNorm > 0f) {
            clipCoeff = config.maxGradNorm / totalNorm;
            for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
                GpuTensor gt = e.getValue();
                if (gt.hasGradBuffer()) {
                    TensorOpsGPU.scaleInPlaceGpuDevice(gt.gradBuffer(), e.getKey().size(), clipCoeff);
                }
            }
            if (logits.hasGrad()) {
                float[] lg = logits.gradBuffer();
                if (lg.length > 0) {
                    TensorOpsGPU.scaleInPlaceGPU(lg, lg.length, clipCoeff);
                }
            }
        }
        optimizer.setLearningRate(learningRateForStep(stepForLr));
        optimizer.beginStep();
        optimizer.stepAllGpuDevice(paramMap);
        model.onGpuParametersUpdated();

        zeroGpuGradsMarkingParamGradsClean(paramMap);
        zeroGradients(logits);
        return true;
    }

    private String firstNonFiniteGpuParamInfo(Map<Tensor, GpuTensor> paramMap, boolean grad) {
        for (int i = 0; i < parameters.size(); i++) {
            Tensor cpu = parameters.get(i);
            GpuTensor gt = paramMap.get(cpu);
            if (gt == null) {
                continue;
            }
            if (grad) {
                if (gt.hasGradBuffer() && TensorOpsGPU.anyNonFiniteGpuDevice(gt.gradBuffer(), cpu.size())) {
                    return "param#" + i + "/grad/size=" + cpu.size();
                }
            } else if (TensorOpsGPU.anyNonFiniteGpuDevice(gt.dataBuffer(), cpu.size())) {
                return "param#" + i + "/weight/size=" + cpu.size();
            }
        }
        return null;
    }

    /**
     * Фьюзированная проверка NaN/Inf во всех GPU-градиентах параметров: один JNI-вызов и одна
     * синхронизация стрима в нормальном случае.
     *
     * <p>Если overflow не обнаружен — возвращает {@code null} (один быстрый round-trip).
     * Если обнаружен — вызывает медленный per-param скан {@link #firstNonFiniteGpuParamInfo} только
     * для получения debug-строки; этот путь редкий (только при реальном overflow/NaN).
     *
     * @return {@code null} если все градиенты конечны; иначе строка-описание первого «плохого» параметра
     */
    private String checkGpuParamGradsNonFiniteFused(Map<Tensor, GpuTensor> paramMap) {
        int n = paramMap.size();
        if (n == 0) {
            return null;
        }
        if (nonFiniteParamBufsScratch.length < n) {
            nonFiniteParamBufsScratch = new GpuFloatBuffer[n];
            nonFiniteParamLensScratch = new int[n];
        }
        int count = 0;
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt != null && gt.hasGradBuffer()) {
                nonFiniteParamBufsScratch[count] = gt.gradBuffer();
                nonFiniteParamLensScratch[count] = e.getKey().size();
                count++;
            }
        }
        if (count == 0) {
            return null;
        }
        if (!TensorOpsGPU.anyNonFiniteGpuDeviceMulti(nonFiniteParamBufsScratch, nonFiniteParamLensScratch, count)) {
            return null;
        }
        // Overflow detected (редкий путь): per-param скан для debug-строки
        return firstNonFiniteGpuParamInfo(paramMap, true);
    }

    /** Хостовый clip/step без {@link #zeroGpuGrads} — ∂ на VRAM могут остаться ненулевыми после backward. */
    private void markGpuTrainableParamGradsMaybeDirtyAfterHostOptimizerPath() {
        if (model.isGpuResident()) {
            gpuTrainableParamGradsKnownClean = false;
        }
    }

    /**
     * Сумма квадратов GPU-градиентов всех параметров в {@code paramMap}: использует
     * grow-only scratch-массивы {@code sumSqPtrsScratch}/{@code sumSqLensScratch}, без аллокации на шаг.
     */
    private double sumSquaresGpuParamGrads(Map<Tensor, GpuTensor> paramMap) {
        int n = paramMap.size();
        if (n == 0) return 0.0;
        if (sumSqPtrsScratch.length < n) {
            sumSqPtrsScratch = new long[n];
            sumSqLensScratch = new int[n];
        }
        int count = 0;
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt != null && !gt.isClosed() && gt.hasGradBuffer()) {
                sumSqPtrsScratch[count] = gt.gradBuffer().devicePointer();
                sumSqLensScratch[count] = e.getKey().size();
                count++;
            }
        }
        if (count == 0) return 0.0;
        return TensorOpsGPU.sumSquaresGPUDeviceFused(sumSqPtrsScratch, sumSqLensScratch, count);
    }

    private void zeroGpuGradsMarkingParamGradsClean(Map<Tensor, GpuTensor> paramMap) {
        zeroGpuGrads(paramMap);
        if (model.isGpuResident()) {
            gpuTrainableParamGradsKnownClean = true;
        }
    }

    private static void zeroGpuGrads(Map<Tensor, GpuTensor> paramMap) {
        for (GpuTensor gt : paramMap.values()) {
            if (gt.hasGradBuffer()) {
                gt.zeroGrad();
            }
        }
    }

}
