package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.GpuIntBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.ops.TensorOps;
import com.veles.llm.jgpt.ops.TensorOpsBackward;
import com.veles.llm.jgpt.ops.TransformerBackward;
import com.veles.llm.jgpt.training.LLMConfig;
import com.veles.llm.jgpt.training.LLMTrainer;
import com.veles.llm.jgpt.util.DebugGpuTrain;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GPT-подобная модель: эмбеддинги токенов и позиций, стек трансформер-блоков, финальный RMSNorm и LM head.
 *
 * <p><b>Потокобезопасность:</b> не потокобезопасна; один экземпляр — один поток обучения/инференса.
 *
 * <p><b>Жизненный цикл:</b> конструктор → {@link #forward} / {@link #backward} / оптимизатор → при необходимости
 * {@link #closeGpuResidentWeights()} или освобождение ресурсов KV/GPU.
 */
public final class GPTModel {

    private static final Logger log = LoggerFactory.getLogger(GPTModel.class);

    private static void agentLogB39372LayerGrad(int layer, int flat) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H_layer_grad\",\"location\":\"GPTModel.backwardDecoderLayersDevice\",\"message\":\"nonfinite_grad_after_block\",\"data\":{\"layer\":"
                        + layer
                        + ",\"flat\":"
                        + flat
                        + "},\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }

    private static void agentLogB39372RmsBuf(
            String hypothesisId, String location, String message, float a, float b, float c, float d, long n) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\""
                        + hypothesisId
                        + "\",\"location\":\""
                        + location
                        + "\",\"message\":\""
                        + message
                        + "\",\"data\":{\"n\":"
                        + n
                        + ",\"s0\":"
                        + a
                        + ",\"s1\":"
                        + b
                        + ",\"s2\":"
                        + c
                        + ",\"s3\":"
                        + d
                        + "},\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }

    /**
     * mask: b0..b4 = bwdPing,bwdPong,chainPing,chainPong,lmFwdScratch; b5..b8 = lastHidden,xBeforeNorm,lastLogits,
     * lastLogitsGrad (1 if cleared).
     */
    private static void agentLogB39372ScratchHandoff(int mask) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H_scratch_handoff\",\"location\":\"GPTModel.clearGpuResidentDecoderScratchForTrainHandoff\",\"message\":\"cleared_resident_scratch\",\"data\":{\"mask\":"
                        + mask
                        + "},\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }

    private static final AtomicBoolean FUSED_LM_HEAD_FAILURE_LOGGED = new AtomicBoolean();

    /** Env {@code JGPT_FUSED_LM_HEAD}: один JNI RMSNorm + matmul на LM-head (см. {@link TensorOpsGPU#rmsNormMatmulLmHeadGpuDevice}). */
    private static boolean fusedLmHeadFromEnv() {
        String e = System.getenv("JGPT_FUSED_LM_HEAD");
        if (e == null) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    private static void applyLmHeadSplitOps(
            GpuFloatBuffer xBeforeNorm,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer normScratch,
            GpuFloatBuffer w,
            GpuFloatBuffer logitsOut,
            int rows,
            int dModel,
            int vocab) {
        TensorOpsGPU.rmsNormGpuDevice(xBeforeNorm, gamma, eps, normScratch, rows, dModel);
        TensorOpsGPU.matmulGpuDevice(normScratch, w, logitsOut, rows, dModel, vocab);
    }

    /**
     * Предпочитает fused JNI при {@link #fusedLmHeadFromEnv()}; при {@link Throwable} — один раз в лог и откат на
     * раздельный путь (ядро/линковка/JNI).
     */
    private void applyLmHeadFusedPreferredThenSplit(
            GpuFloatBuffer xBeforeNorm,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer normScratch,
            GpuFloatBuffer w,
            GpuFloatBuffer logitsOut,
            int rows,
            int dModel,
            int vocab) {
        if (!fusedLmHeadFromEnv()) {
            applyLmHeadSplitOps(xBeforeNorm, gamma, eps, normScratch, w, logitsOut, rows, dModel, vocab);
            return;
        }
        try {
            TensorOpsGPU.rmsNormMatmulLmHeadGpuDevice(
                    xBeforeNorm,
                    gamma,
                    eps,
                    normScratch,
                    w,
                    logitsOut,
                    rows,
                    dModel,
                    vocab,
                    TensorOpsGPU.useFp16Matmul());
        } catch (Throwable t) {
            if (FUSED_LM_HEAD_FAILURE_LOGGED.compareAndSet(false, true)) {
                log.warn(
                        "JGPT_FUSED_LM_HEAD: fused LM head недоступен или выбросил исключение; откат на RMSNorm+matmul ({})",
                        t.toString());
            }
            applyLmHeadSplitOps(xBeforeNorm, gamma, eps, normScratch, w, logitsOut, rows, dModel, vocab);
        }
    }

    /** Бинарный формат {@link com.veles.llm.jgpt.training.LLMTrainer#saveModelWeights(String)} (UTF-тег + raw float). */
    public static final String MODEL_WEIGHTS_FORMAT_V1 = "veles.weights.v1";

    private final int vocabSize;
    private final int maxSeqLen;
    private final int dModel;
    private final int numHeads;
    private final int numLayers;
    private final int dIntermediate;

    private final TokenEmbedding tokenEmbedding;
    private final PositionEmbedding positionEmbedding;
    private final DecoderBlock[] blocks;
    private final Tensor layerNormFinal;
    private final Tensor lmHead;

    /**
     * При {@code true} (и доступной CUDA) дубликаты {@link #layerNormFinal} и {@link #lmHead} лежат в
     * {@link GptGpuWeights} для путей вроде {@link #forwardGpuLmHead(Tensor)} без H2D весов на каждый шаг.
     */
    private final boolean gpuResident;

    private final GptGpuWeights gpuResidentHead;

    /** VRAM-копии весов декодер-слоя (pre-norm + Q/K/V/O + FFN); без H2D весов в fused/attention путях. */
    private final GptGpuDecoderLayerGpuWeights[] gpuDecoderLayer;

    /**
     * {@code true} если GPU-резидентно и decoder-pipeline разрешён (env {@code JGPT_DECODER_GPU_PIPELINE=1}
     * или prop {@code jgpt.decoder.gpu.pipeline=true}).
     */
    private final boolean decoderGpuPipeline;

    /**
     * Запрошен env {@code JGPT_DECODER_LAYER_CUDA_GRAPH}: при успешном захвате — один graph exec на слой (MHA+FFN).
     */
    private final boolean decoderLayerCudaGraphWanted;

    /** native {@code cudaGraphExec_t} на слой; {@code 0} — ещё не захвачен или сброшен. */
    private final long[] decoderLayerGraphExec;

    private int decoderLayerGraphCaptureKey = Integer.MIN_VALUE;

    private boolean decoderLayerGraphRuntimeDisabled;

    /** CE + LM head backward на device; включается через {@link #setDeviceLogitsEnabled(boolean)}. */
    private boolean deviceLogitsEnabled;

    /** Decoder backward на VRAM; включается через {@link #setDeviceDecoderBackward(boolean)}. */
    private boolean deviceDecoderBackward;

    /** Скрытое состояние перед LM head (после финального RMSNorm); для backward последнего слоя. */
    private Tensor lastHidden;

    /** Вход в финальный RMSNorm (до γ); нужен для backward. */
    private Tensor xBeforeFinalNorm;

    /** Кэш активаций по слоям при {@link #forward(Tensor, boolean)} с training=true. */
    private volatile BlockActivationCache[] blockCaches;

    /** Device-копии активаций по слоям для полного decoder backward без host round-trip. */
    private volatile BlockActivationCacheDevice[] blockCachesDevice;

    /**
     * Сохранять активации блока в FP16 в {@link BlockActivationCache} (env {@code
     * JGPT_ACTIVATION_CACHE_FP16=1} или {@code -Djgpt.activationCache.fp16=true}).
     */
    private final boolean useFp16ActivationCache;

    /** Последний вызов {@link #forward(Tensor, boolean)}: было ли {@code training=true}. */
    private boolean lastForwardWasTraining;

    private Tensor lastMask;
    private Tensor cachedCausalMask;
    private int cachedCausalMaskSeqLen = -1;
    private Tensor cachedLmHeadTranspose;
    private boolean cachedLmHeadTransposeValid;

    /** Кэш {@link #gpuTensorByTrainableParameter()}: пересоздаётся только при явном сбросе. */
    private Map<Tensor, GpuTensor> cachedGpuParamMap;
    private boolean cachedGpuParamMapValid;
    private Tensor lastInputTokens;
    private int lastSeqLen;
    private Tensor backwardGradHidden;
    private Tensor backwardGradBeforeNorm;
    private Tensor backwardGradPing;
    private Tensor backwardGradPong;

    /** GPU-копии последних активаций/логитов для device CE + logits backward. */
    private GpuFloatBuffer lastHiddenGpu;
    private GpuFloatBuffer xBeforeFinalNormGpu;
    private GpuFloatBuffer lastLogitsGpu;
    private GpuFloatBuffer lastLogitsGradGpu;
    private GpuIntBuffer lastSampledCandidateIdsGpu;
    private GpuFloatBuffer lastSampledCandidateGradGpu;
    private int lastSampledCandidateCount;

    /** Ping-pong ∂L/∂x между decoder-слоями на VRAM при {@link #deviceDecoderBackward}. */
    private GpuFloatBuffer decoderBwdGradPing;
    private GpuFloatBuffer decoderBwdGradPong;

    /**
     * [batch, seq, d_model] на VRAM: gather токенов + позиции без D2H между ними (см. {@link #ensureEmbeddingScratchGpu}).
     */
    private GpuTensor embeddingScratchGpu;

    private int embeddingScratchBatch = -1;
    private int embeddingScratchSeqLen = -1;

    /** Цепочка decoder: ping-pong на VRAM без D2H между слоями ({@link #forwardGpuDecoder}). */
    private GpuFloatBuffer decoderChainPing;

    private GpuFloatBuffer decoderChainPong;

    /**
     * Strided-batched QKV/FFN pack на VRAM при {@link #decoderLayerCudaGraphWanted}: фиксированные указатели для
     * {@link TensorOpsGPU#setStridedBatchedPackOverride} на время {@link #runDecoderStackLayers} (инвариант CUDA graph).
     */
    private GpuFloatBuffer decoderGraphStridedPackW;

    private GpuFloatBuffer decoderGraphStridedPackC;

    /** RMSNorm перед LM head на VRAM ({@link #forwardGpuLmHeadFromDevice}). */
    private GpuFloatBuffer lmHeadNormScratchGpu;

    /**
     * Заглушка формы логитов для API {@link Tensor} при обучении с логитами только на GPU (данные не читаются —
     * CE и backward идут по {@link #lastLogitsGpu}).
     */
    private float[] logitsShapeStub;

    /** Одна строка логитов для {@link #sampleNextToken} без порчи буфера logits. */
    private float[] sampleLogitsScratch;

    /** Маска «входит в top-k» при сэмплинге; совпадает с порогом стабильной сортировки по (logit↓, индекс↑). */
    private boolean[] sampleTopKMember;

    /** Переиспользование {@code [1,1]} в {@link #generate} (декодирование одного токена). */
    private Tensor reusableDecodeOneToken;

    /**
     * Слайс токенов {@code [1, sliceLen]} при скользящем KV-окне: один экземпляр на ту же длину {@code sliceLen},
     * чтобы не аллоцировать тензор на каждом срабатывании (длина может меняться между шагами).
     */
    private Tensor reusableSlidingPrefillInput;

    public GPTModel(
            int vocabSize,
            int maxSeqLen,
            int dModel,
            int numHeads,
            int numLayers,
            int dIntermediate) {
        this(vocabSize, maxSeqLen, dModel, numHeads, numLayers, dIntermediate, false);
    }

    /**
     * @param gpuResident при {@code true} и доступной CUDA — копии финального RMSNorm и LM head на VRAM
     *                    ({@link #forwardGpuLmHead(Tensor)}), веса слоёв декодера (аттеншн + FFN) на VRAM.
     */
    public GPTModel(
            int vocabSize,
            int maxSeqLen,
            int dModel,
            int numHeads,
            int numLayers,
            int dIntermediate,
            boolean gpuResident) {
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.dIntermediate = dIntermediate;

        float embedScale = 1.0f / (float) Math.sqrt(dModel);
        float projScale = 1.0f / (float) Math.sqrt(dModel);
        float ffnScale = 1.0f / (float) Math.sqrt(dIntermediate);

        boolean resident = gpuResident && TensorOpsGPU.isGpuAvailable();
        if (gpuResident && !resident) {
            log.warn("gpuResident=true, но GPU недоступен — VRAM-копии LM head не создаются");
        }
        this.gpuResident = resident;

        this.tokenEmbedding = new TokenEmbedding(vocabSize, dModel, embedScale, resident);
        this.positionEmbedding = new PositionEmbedding(maxSeqLen, dModel, embedScale, resident);

        this.blocks = new DecoderBlock[numLayers];
        for (int i = 0; i < numLayers; i++) {
            blocks[i] = new DecoderBlock(dModel, numHeads, dIntermediate, projScale, ffnScale);
        }

        this.layerNormFinal = TensorOps.onesTensor(new int[]{dModel});
        this.lmHead = TensorOps.randomTensor(new int[]{dModel, vocabSize}, projScale);
        this.useFp16ActivationCache = resolveUseFp16ActivationCache();

        if (this.gpuResident) {
            this.gpuResidentHead = GptGpuWeights.upload(lmHead, layerNormFinal);
            this.gpuDecoderLayer = new GptGpuDecoderLayerGpuWeights[numLayers];
            for (int i = 0; i < numLayers; i++) {
                gpuDecoderLayer[i] =
                        GptGpuDecoderLayerGpuWeights.upload(
                                blocks[i].getNorm1(),
                                blocks[i].getWq(),
                                blocks[i].getWk(),
                                blocks[i].getWv(),
                                blocks[i].getWo(),
                                blocks[i].getNorm2(),
                                blocks[i].getW1(),
                                blocks[i].getW2(),
                                blocks[i].getW3());
            }
            log.info(
                    "  GPU-резидентно: токен/позиция эмбеддинги + веса слоёв декодера (аттеншн+FFN) + финальный"
                            + " RMSNorm + LM head");
        } else {
            this.gpuResidentHead = null;
            this.gpuDecoderLayer = null;
        }

        this.decoderGpuPipeline = this.gpuResident && resolveDecoderGpuPipeline();
        if (this.decoderGpuPipeline && this.gpuDecoderLayer != null && LLMConfig.decoderLayerCudaGraphFromEnvOrProp()) {
            this.decoderLayerCudaGraphWanted = true;
            this.decoderLayerGraphExec = new long[numLayers];
        } else {
            this.decoderLayerCudaGraphWanted = false;
            this.decoderLayerGraphExec = null;
        }

        log.info(
                "Модель GPT инициализирована: vocab={}, seq_len={}, d_model={}, heads={}, слоёв={}",
                vocabSize,
                maxSeqLen,
                dModel,
                numHeads,
                numLayers);
        log.info("  оценка числа параметров: ~{}", String.format("%,d", countParameters()));
        if (useFp16ActivationCache) {
            log.info(
                    "  кэш активаций блоков: FP16 (хост при классическом backward; на VRAM — слоты half при device-кэше)");
        }
    }

    /** {@code true}, если активны VRAM-копии LM head / финального RMSNorm ({@link #forwardGpuLmHead(Tensor)}). */
    public boolean isGpuResident() {
        return gpuResident;
    }

    public boolean isDecoderGpuPipeline() {
        return decoderGpuPipeline;
    }

    /** Env/prop запросил CUDA graph на декодер-слой ({@code JGPT_DECODER_LAYER_CUDA_GRAPH}). */
    public boolean isDecoderLayerCudaGraphRequested() {
        return decoderLayerCudaGraphWanted;
    }

    /**
     * Graph-путь ещё не отключён из-за ошибки захвата; не гарантирует, что все {@link #decoderLayerGraphExec} уже
     * инстанцированы.
     */
    public boolean isDecoderLayerCudaGraphActive() {
        return decoderLayerCudaGraphWanted && !decoderLayerGraphRuntimeDisabled;
    }

    /** Все условия для полного GPU-обучения: GPU-резидентно, pipeline, CUDA доступна. */
    public boolean canFullGpuTrain() {
        return gpuResident && decoderGpuPipeline && TensorOpsGPU.isGpuAvailable();
    }

    /**
     * Инференс с логитами в {@link #deviceLogitsBuffer()} (без D2H логитов) — те же условия, что внутренний путь
     * {@code vramInferDecoderLm} при {@code useGpuResidentHead=true}.
     */
    public boolean canInferLogitsOnDevice(int batch, int seqLen) {
        if (batch <= 0 || seqLen <= 0 || seqLen > maxSeqLen) {
            return false;
        }
        return decoderGpuPipeline
                && gpuDecoderLayer != null
                && TensorOpsGPU.isGpuAvailable()
                && gpuResident
                && gpuResidentHead != null
                && tokenEmbedding.hasWeightsGpu()
                && positionEmbedding.hasWeightsGpu()
                && TensorOpsGPU.shouldUseGpuElementwise(batch * seqLen * dModel);
    }

    /**
     * Включает device-side CE + LM head backward; требует {@code decoderGpuPipeline}.
     */
    public void setDeviceLogitsEnabled(boolean v) {
        if (v && !decoderGpuPipeline) {
            throw new IllegalStateException(
                    "setDeviceLogitsEnabled(true) requires decoderGpuPipeline (gpuResident + env/prop)");
        }
        this.deviceLogitsEnabled = v;
    }

    public boolean isDeviceLogitsEnabled() {
        return deviceLogitsEnabled;
    }

    /**
     * @throws IllegalStateException если {@code deviceLogitsEnabled} не включён
     */
    public void setDeviceDecoderBackward(boolean v) {
        if (v && !deviceLogitsEnabled) {
            throw new IllegalStateException(
                    "setDeviceDecoderBackward(true) requires deviceLogitsEnabled");
        }
        this.deviceDecoderBackward = v;
    }

    public boolean isDeviceDecoderBackward() {
        return deviceDecoderBackward;
    }

    /**
     * Обнуляет ∂ на VRAM для всех обучаемых параметров — зеркало {@link #zeroGradParameters()} при
     * {@code zeroParamGrads=true}. Иначе после {@link com.veles.llm.jgpt.cuda.GpuPendingGradients#flushMergeToGpuGrads}
     * или снимка градиентов буферы {@link GpuTensor} остаются ненулевыми и следующий backward даёт неверные ∂.
     */
    public void zeroGpuTrainableParameterGrads() {
        if (!gpuResident) {
            return;
        }
        for (GpuTensor gt : gpuTensorByTrainableParameter().values()) {
            gt.zeroGrad();
        }
    }

    /**
     * Маппинг CPU-тензора параметра → его {@link GpuTensor} на VRAM. Нужен для
     * {@link com.veles.llm.jgpt.cuda.GpuPendingGradients#flushMergeToGpuGrads} и clip/optimizer на VRAM; при unified
     * device-backward все ∂ обучаемых весов накапливаются в {@link GpuTensor#gradBuffer()} / пуле pending, без
     * опоры на хостовый {@link Tensor#gradBuffer()} у параметров.
     *
     * @throws IllegalStateException если модель не GPU-резидентная
     */
    public Map<Tensor, GpuTensor> gpuTensorByTrainableParameter() {
        if (!gpuResident) {
            throw new IllegalStateException("gpuTensorByTrainableParameter requires gpuResident=true");
        }
        if (cachedGpuParamMapValid && cachedGpuParamMap != null) {
            return cachedGpuParamMap;
        }
        Map<Tensor, GpuTensor> map = new IdentityHashMap<>();
        if (tokenEmbedding.weightsGpu != null) {
            map.put(tokenEmbedding.weights, tokenEmbedding.weightsGpu);
        }
        if (positionEmbedding.weightsGpu != null) {
            map.put(positionEmbedding.weights, positionEmbedding.weightsGpu);
        }
        for (int i = 0; i < numLayers; i++) {
            gpuDecoderLayer[i].collectMapping(map);
        }
        gpuResidentHead.collectMapping(lmHead, layerNormFinal, map);
        cachedGpuParamMap = map;
        cachedGpuParamMapValid = true;
        return map;
    }

    /** Сбрасывает кэш {@link #gpuTensorByTrainableParameter()} — вызывать при добавлении/удалении GPU-тензоров. */
    public void invalidateGpuParamMapCache() {
        cachedGpuParamMapValid = false;
    }

    private static boolean resolveDecoderGpuPipeline() {
        return LLMConfig.decoderGpuPipelineFromEnvOrProp();
    }

    private void destroyDecoderLayerCudaGraphs() {
        if (decoderLayerGraphExec == null) {
            return;
        }
        for (int i = 0; i < decoderLayerGraphExec.length; i++) {
            if (decoderLayerGraphExec[i] != 0L) {
                TensorOpsGPU.cudaGraphExecDestroy(decoderLayerGraphExec[i]);
                decoderLayerGraphExec[i] = 0L;
            }
        }
    }

    private void disableDecoderLayerCudaGraph(String reason) {
        if (decoderLayerGraphRuntimeDisabled) {
            return;
        }
        decoderLayerGraphRuntimeDisabled = true;
        destroyDecoderLayerCudaGraphs();
        decoderLayerGraphCaptureKey = Integer.MIN_VALUE;
        if (decoderLayerCudaGraphWanted) {
            log.warn("JGPT_DECODER_LAYER_CUDA_GRAPH: отключён ({}).", reason);
        }
    }

    private int decoderLayerGraphKey(Tensor mask, boolean trainingStep, int batch, int seqLen) {
        int fp16Mm = TensorOpsGPU.useFp16Matmul() ? 1 : 0;
        int maskId = mask == null ? 0 : System.identityHashCode(mask);
        int cacheFp16 = 0;
        if (trainingStep && blockCachesDevice != null && blockCachesDevice.length > 0 && blockCachesDevice[0] != null) {
            cacheFp16 = blockCachesDevice[0].isFp16ActivationStorage() ? 1 : 0;
        }
        /*
         * Обязательно различать training vs infer в ключе: при cacheFp16==0 (нет FP16-хранения в кэше) и одинаковых
         * batch/seq/mask ключ совпадал бы с {@link #forwardGpuDecoderInfer} (layerCache=null), и следующий train-step
         * мог бы replay графа без записи активаций в {@link BlockActivationCacheDevice} → битый backward / non-finite.
         */
        int trainMode = trainingStep ? 1 : 0;
        return Objects.hash(batch, seqLen, dModel, numHeads, maskId, fp16Mm, cacheFp16, trainMode);
    }

    /**
     * Гарантирует pack-буферы под decoder CUDA graph; при росте — сбрасывает exec (новые указатели).
     */
    private void ensureDecoderGraphStridedPacks(long rows) {
        long[] need = TensorOpsGPU.stridedBatchedPackNeed(rows, dModel, dIntermediate);
        long wNeed = need[0];
        long cNeed = need[1];
        boolean needGrow =
                decoderGraphStridedPackW == null
                        || decoderGraphStridedPackW.isClosed()
                        || decoderGraphStridedPackW.numFloats() < wNeed
                        || decoderGraphStridedPackC == null
                        || decoderGraphStridedPackC.isClosed()
                        || decoderGraphStridedPackC.numFloats() < cNeed;
        if (!needGrow) {
            return;
        }
        destroyDecoderLayerCudaGraphs();
        /* Не сбрасывать decoderLayerGraphCaptureKey: он уже совпадает с текущим nk из runDecoderStackLayers;
         * иначе на следующем forward сработает ложное «ключ изменился» и графы пересоздадутся дважды подряд. */
        decoderGraphStridedPackW = closeGpuBuffer(decoderGraphStridedPackW);
        decoderGraphStridedPackC = closeGpuBuffer(decoderGraphStridedPackC);
        decoderGraphStridedPackW = GpuFloatBuffer.allocate(wNeed);
        decoderGraphStridedPackC = GpuFloatBuffer.allocate(cNeed);
    }

    private boolean runDecoderLayerResidentEager(
            int layer,
            GpuFloatBuffer cur,
            GpuFloatBuffer attnOut,
            GpuFloatBuffer blockOut,
            Tensor mask,
            int batch,
            int seqLen,
            BlockActivationCacheDevice devCache) {
        if (!TensorOps.multiHeadAttentionResidentDeviceToDevice(
                cur,
                attnOut,
                TensorOpsGPU.rmsNormEps(),
                gpuDecoderLayer[layer].attnBuffers(),
                batch,
                seqLen,
                dModel,
                numHeads,
                mask,
                true,
                devCache)) {
            return false;
        }
        return TensorOps.fusedNormResidualSwiGLUResidentDeviceToDevice(
                attnOut,
                blockOut,
                gpuDecoderLayer[layer].ffnBuffers(),
                batch,
                seqLen,
                dModel,
                devCache);
    }

    /**
     * @return активации после последнего блока (planar на GPU)
     */
    private GpuFloatBuffer runDecoderStackLayers(
            GpuFloatBuffer xDevice,
            Tensor mask,
            int batch,
            int seqLen,
            boolean trainingStep,
            BlockActivationCacheDevice[] cachesPerLayer) {
        long planeLong = (long) batch * seqLen * dModel;
        int plane = (int) planeLong;
        decoderChainPing = ensureGpuBuffer(decoderChainPing, plane);
        decoderChainPong = ensureGpuBuffer(decoderChainPong, plane);

        boolean wantGraph =
                decoderLayerCudaGraphWanted
                        && !decoderLayerGraphRuntimeDisabled
                        && decoderLayerGraphExec != null
                        && TensorOpsGPU.isGpuAvailable();
        if (wantGraph) {
            int nk = decoderLayerGraphKey(mask, trainingStep, batch, seqLen);
            if (decoderLayerGraphCaptureKey != nk) {
                destroyDecoderLayerCudaGraphs();
                decoderLayerGraphCaptureKey = nk;
            }
        }

        boolean stridedPackOverride = false;
        if (wantGraph) {
            ensureDecoderGraphStridedPacks((long) batch * seqLen);
            TensorOpsGPU.setStridedBatchedPackOverride(
                    decoderGraphStridedPackW.devicePointer(),
                    decoderGraphStridedPackC.devicePointer(),
                    decoderGraphStridedPackW.numFloats(),
                    decoderGraphStridedPackC.numFloats());
            stridedPackOverride = true;
        }
        try {
            GpuFloatBuffer cur = xDevice;
            if (wantGraph && mask != null) {
                /*
                 * Один H2D маски на весь decoder-stack: при replay граф не содержит upload; при capture skip H2D
                 * внутри графа полагается на уже заполненный maskDev (см. TensorOps).
                 */
                TensorOps.primeDecoderGraphAttentionMaskDevice(mask, seqLen);
            }
            for (int i = 0; i < numLayers; i++) {
                GpuFloatBuffer attnOut = decoderScratchOther(cur, decoderChainPing, decoderChainPong);
                GpuFloatBuffer blockOut = decoderScratchOther(attnOut, decoderChainPing, decoderChainPong);
                BlockActivationCacheDevice layerCache = cachesPerLayer != null ? cachesPerLayer[i] : null;

                boolean executed = false;
                if (wantGraph && !decoderLayerGraphRuntimeDisabled) {
                    long ex = decoderLayerGraphExec[i];
                    if (ex != 0L) {
                        if (TensorOpsGPU.cudaGraphExecLaunch(ex)) {
                            executed = true;
                        } else {
                            disableDecoderLayerCudaGraph("cudaGraphLaunch failed");
                            /*
                             * После ошибки запуска графа (часто illegal memory) дальнейшие kernel'ы в этом процессе
                             * ненадёжны; eager-fallback лишь множит ошибки (cuBLAS 14, FFN и т.д.). Требуется новый JVM.
                             */
                            throw new IllegalStateException(
                                    "cudaGraphLaunch не удался — контекст CUDA считается недействительным. "
                                            + "Перезапустите JVM. Чтобы не использовать decoder CUDA graph: не задавайте "
                                            + "JGPT_DECODER_LAYER_CUDA_GRAPH=1 (графы включаются только явно).");
                        }
                    } else {
                        TensorOpsGPU.ensureStridedBatchedPackScratch(
                                (long) batch * seqLen, dModel, dIntermediate);
                        TensorOps.primeDecoderGraphLayerWorkspaces(
                                batch, seqLen, dModel, numHeads, dIntermediate, mask, layerCache);
                        TensorOpsGPU.synchronizeStream();
                        if (!TensorOpsGPU.cudaStreamBeginCapture()) {
                            disableDecoderLayerCudaGraph("cudaStreamBeginCapture failed");
                        } else {
                            TensorOps.setDecoderGraphCaptureSkipAttentionMaskHostUpload(true);
                            boolean captureStillActive = true;
                            try {
                                boolean ok =
                                        runDecoderLayerResidentEager(
                                                i, cur, attnOut, blockOut, mask, batch, seqLen, layerCache);
                                long nexec = TensorOpsGPU.cudaStreamEndCaptureAndInstantiate();
                                captureStillActive = false;
                                if (!ok) {
                                    disableDecoderLayerCudaGraph("layer forward failed during capture");
                                    TensorOpsGPU.synchronizeStream();
                                    throw new IllegalStateException(
                                            "decoder layer " + i + " during CUDA graph capture");
                                }
                                if (nexec == 0L) {
                                    disableDecoderLayerCudaGraph("cudaStreamEndCapture/instantiate failed");
                                    TensorOpsGPU.synchronizeStream();
                                    /* граф не создан — захват не исполнял слой; нужен eager */
                                    executed = false;
                                } else {
                                    decoderLayerGraphExec[i] = nexec;
                                    /* Захват только записывает ядра, не исполняет их.
                                     * Немедленно запускаем только что захваченный граф,
                                     * чтобы blockOut был заполнен реальными данными. */
                                    if (TensorOpsGPU.cudaGraphExecLaunch(nexec)) {
                                        executed = true;
                                    } else {
                                        TensorOpsGPU.cudaGraphExecDestroy(nexec);
                                        decoderLayerGraphExec[i] = 0L;
                                        disableDecoderLayerCudaGraph("cudaGraphLaunch failed on capture step");
                                        executed = false;
                                    }
                                }
                            } finally {
                                TensorOps.setDecoderGraphCaptureSkipAttentionMaskHostUpload(false);
                                if (captureStillActive) {
                                    TensorOpsGPU.abortCudaStreamCaptureIfActive();
                                }
                            }
                        }
                    }
                }
                if (!executed) {
                    if (!runDecoderLayerResidentEager(i, cur, attnOut, blockOut, mask, batch, seqLen, layerCache)) {
                        throw new IllegalStateException("decoder layer " + i + " (eager path)");
                    }
                }
                cur = blockOut;
            }
            return cur;
        } finally {
            if (stridedPackOverride) {
                TensorOpsGPU.clearStridedBatchedPackOverride();
            }
        }
    }

    /**
     * Обновляет GpuTensor с CPU после {@link #loadWeights(String)} или ручной правки {@link #getLmHead()} /
     * {@link #getLayerNormFinal()}.
     */
    public void syncGpuResidentWeightsFromHost() {
        if (gpuResident) {
            tokenEmbedding.syncGpuWeightsFromHost();
            positionEmbedding.syncGpuWeightsFromHost();
        }
        if (gpuResidentHead != null) {
            gpuResidentHead.syncFromHost();
        }
        if (gpuDecoderLayer != null) {
            for (GptGpuDecoderLayerGpuWeights w : gpuDecoderLayer) {
                w.syncFromHost();
            }
        }
    }

    /** Освобождает {@link GpuTensor} финальных весов; после вызова {@link #isGpuResident()} остаётся {@code true}, но
     * {@link #forwardGpuLmHead(Tensor)} бросит (буферы закрыты). */
    public void closeGpuResidentWeights() {
        invalidateGpuParamMapCache();
        tokenEmbedding.closeGpuWeights();
        positionEmbedding.closeGpuWeights();
        closeEmbeddingScratchGpu();
        closeTrainingGpuBuffers();
        closeBlockCachesDevice();
        if (gpuDecoderLayer != null) {
            for (GptGpuDecoderLayerGpuWeights w : gpuDecoderLayer) {
                w.close();
            }
        }
        if (gpuResidentHead != null) {
            gpuResidentHead.close();
        }
    }

    private void closeEmbeddingScratchGpu() {
        if (embeddingScratchGpu != null) {
            embeddingScratchGpu.close();
            embeddingScratchGpu = null;
            embeddingScratchBatch = -1;
            embeddingScratchSeqLen = -1;
        }
    }

    private void closeTrainingGpuBuffers() {
        destroyDecoderLayerCudaGraphs();
        decoderLayerGraphCaptureKey = Integer.MIN_VALUE;
        TensorOpsGPU.clearStridedBatchedPackOverride();
        decoderGraphStridedPackW = closeGpuBuffer(decoderGraphStridedPackW);
        decoderGraphStridedPackC = closeGpuBuffer(decoderGraphStridedPackC);
        lastHiddenGpu = closeGpuBuffer(lastHiddenGpu);
        xBeforeFinalNormGpu = closeGpuBuffer(xBeforeFinalNormGpu);
        lastLogitsGpu = closeGpuBuffer(lastLogitsGpu);
        lastLogitsGradGpu = closeGpuBuffer(lastLogitsGradGpu);
        decoderBwdGradPing = closeGpuBuffer(decoderBwdGradPing);
        decoderBwdGradPong = closeGpuBuffer(decoderBwdGradPong);
        decoderChainPing = closeGpuBuffer(decoderChainPing);
        decoderChainPong = closeGpuBuffer(decoderChainPong);
        lmHeadNormScratchGpu = closeGpuBuffer(lmHeadNormScratchGpu);
        logitsShapeStub = null;
    }

    private void closeBlockCachesDevice() {
        if (blockCachesDevice == null) {
            return;
        }
        for (BlockActivationCacheDevice cache : blockCachesDevice) {
            if (cache != null) {
                cache.close();
            }
        }
        blockCachesDevice = null;
    }

    /**
     * После инференс-only путей, которые не вызывают {@link #forward(Tensor, boolean, boolean)} с
     * {@code training=true} — например {@link #generateGpuKv} или цепочка {@code forward(..., false)} в оценке loss:
     * сбрасывает VRAM activation cache обучения и уничтожает захваченные CUDA-graph слоёв декодера, чтобы следующий
     * training-forward не смешивал состояние с infer-путём на тех же resident-буферах и потоке.
     */
    public void prepareForTrainingAfterInteractiveGeneration() {
        closeBlockCachesDevice();
        closeEmbeddingScratchGpu();
        clearGpuResidentDecoderScratchForTrainHandoff();
        clearSampledTrainLossGrad();
        if (decoderLayerGraphExec != null) {
            destroyDecoderLayerCudaGraphs();
            decoderLayerGraphCaptureKey = Integer.MIN_VALUE;
        }
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
        }
    }

    /**
     * Infer и train делят {@link #decoderChainPing}/{@link #decoderChainPong} и ping-pong градиентов декодера без
     * смены указателей между эпохами: после eval/KV необходимо обнулить, иначе первый train-step иногда даёт
     * нечисловые ∂ (ядра с += / неполная перезапись).
     *
     * <p>Цепочка LM на VRAM ({@link #lastHiddenGpu}, {@link #xBeforeFinalNormGpu}, {@link #lastLogitsGpu},
     * {@link #lastLogitsGradGpu}) заполняется только обычным {@link #forward}; пути {@code forwardPrefill} /
     * {@code forwardDecode} с KV её не обновляют. После eval размер [B,S] часто другой — без {@link GpuFloatBuffer#clear()}
     * в швах с train возможны нечисловые ∂ в {@link #backwardFromDeviceLogits} / декодере (см. debug H_scratch_handoff).
     */
    private void clearGpuResidentDecoderScratchForTrainHandoff() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int m = 0;
        if (decoderBwdGradPing != null && !decoderBwdGradPing.isClosed()) {
            decoderBwdGradPing.clear();
            m |= 1;
        }
        if (decoderBwdGradPong != null && !decoderBwdGradPong.isClosed()) {
            decoderBwdGradPong.clear();
            m |= 2;
        }
        if (decoderChainPing != null && !decoderChainPing.isClosed()) {
            decoderChainPing.clear();
            m |= 4;
        }
        if (decoderChainPong != null && !decoderChainPong.isClosed()) {
            decoderChainPong.clear();
            m |= 8;
        }
        if (lmHeadNormScratchGpu != null && !lmHeadNormScratchGpu.isClosed()) {
            lmHeadNormScratchGpu.clear();
            m |= 16;
        }
        if (lastHiddenGpu != null && !lastHiddenGpu.isClosed()) {
            lastHiddenGpu.clear();
            m |= 32;
        }
        if (xBeforeFinalNormGpu != null && !xBeforeFinalNormGpu.isClosed()) {
            xBeforeFinalNormGpu.clear();
            m |= 64;
        }
        if (lastLogitsGpu != null && !lastLogitsGpu.isClosed()) {
            lastLogitsGpu.clear();
            m |= 128;
        }
        if (lastLogitsGradGpu != null && !lastLogitsGradGpu.isClosed()) {
            lastLogitsGradGpu.clear();
            m |= 256;
        }
        agentLogB39372ScratchHandoff(m);
    }

    private static GpuFloatBuffer closeGpuBuffer(GpuFloatBuffer buf) {
        if (buf != null && !buf.isClosed()) {
            buf.close();
        }
        return null;
    }

    private static GpuFloatBuffer ensureGpuBuffer(GpuFloatBuffer buf, long size) {
        if (buf != null && !buf.isClosed() && buf.numFloats() >= size) {
            return buf;
        }
        if (buf != null && !buf.isClosed()) {
            buf.close();
        }
        return GpuFloatBuffer.allocate(size);
    }

    private GpuTensor ensureEmbeddingScratchGpu(int batch, int seqLen) {
        if (embeddingScratchGpu != null
                && embeddingScratchBatch == batch
                && embeddingScratchSeqLen == seqLen) {
            return embeddingScratchGpu;
        }
        closeEmbeddingScratchGpu();
        embeddingScratchGpu = GpuTensor.allocate(new int[]{batch, seqLen, dModel});
        embeddingScratchBatch = batch;
        embeddingScratchSeqLen = seqLen;
        return embeddingScratchGpu;
    }

    public Tensor getLayerNormFinal() {
        return layerNormFinal;
    }

    /**
     * Финальный RMSNorm + LM head на GPU при {@link #isGpuResident()}; веса не копируются с хоста на каждый вызов
     * (активации — один H2D входа и один D2H логитов).
     *
     * @param xBeforeFinalNorm [batch, seq_len, d_model] — тот же этап, что перед RMSNorm в {@link #forward(Tensor, boolean)}
     */
    public Tensor forwardGpuLmHead(Tensor xBeforeFinalNorm) {
        return forwardGpuLmHead(xBeforeFinalNorm, false);
    }

    /**
     * @param fillLastHiddenForBackward если {@code true} — после RMSNorm на device копирует нормализованные
     *     активации в {@link #lastHidden} (нужно для {@link #backward(Tensor, boolean)} после forward с тем же
     *     шагом).
     */
    public Tensor forwardGpuLmHead(Tensor xBeforeFinalNorm, boolean fillLastHiddenForBackward) {
        if (!gpuResident || gpuResidentHead == null) {
            throw new IllegalStateException("forwardGpuLmHead требует gpuResident=true и доступную CUDA");
        }
        int[] sh = xBeforeFinalNorm.getShape();
        if (sh.length != 3 || sh[2] != dModel) {
            throw new IllegalArgumentException(
                    "ожидается [batch, seq, d_model], d_model=" + dModel + ", получено " + Arrays.toString(sh));
        }
        int batch = sh[0];
        int seqLen = sh[1];
        int rows = batch * seqLen;
        int flat = rows * dModel;
        int logitsFlatElems = rows * vocabSize;
        float eps = TensorOpsGPU.rmsNormEps();

        try (GpuFloatBuffer xFlat = GpuFloatBuffer.allocate(flat);
                GpuFloatBuffer normed = GpuFloatBuffer.allocate(flat);
                GpuFloatBuffer logitsFlat = GpuFloatBuffer.allocate(logitsFlatElems)) {
            xFlat.copyFrom(xBeforeFinalNorm.internalBuffer(), 0, flat);
            TensorOpsGPU.rmsNormGpuDevice(
                    xFlat,
                    gpuResidentHead.layerNormGammaGpu().dataBuffer(),
                    eps,
                    normed,
                    rows,
                    dModel);
            if (fillLastHiddenForBackward) {
                xBeforeFinalNormGpu = ensureGpuBuffer(xBeforeFinalNormGpu, flat);
                xBeforeFinalNormGpu.copyFromDevice(xFlat, flat);
                lastHiddenGpu = ensureGpuBuffer(lastHiddenGpu, flat);
                lastHiddenGpu.copyFromDevice(normed, flat);
                float[] normedHost = new float[flat];
                normed.copyTo(normedHost, 0, flat);
                lastHidden = Tensor.wrap(normedHost, new int[]{batch, seqLen, dModel});
            }
            TensorOpsGPU.matmulGpuDevice(
                    normed,
                    gpuResidentHead.lmHeadGpu().dataBuffer(),
                    logitsFlat,
                    rows,
                    dModel,
                    vocabSize);
            if (fillLastHiddenForBackward) {
                lastLogitsGpu = ensureGpuBuffer(lastLogitsGpu, logitsFlatElems);
                lastLogitsGpu.copyFromDevice(logitsFlat, logitsFlatElems);
            }
            float[] host = new float[rows * vocabSize];
            logitsFlat.copyTo(host, 0, host.length);
            return Tensor.wrap(host, new int[]{batch, seqLen, vocabSize});
        }
    }

    /**
     * Эмбеддинги токенов + позиционные сдвиги на VRAM; без скачивания активаций на хост.
     *
     * @return planar {@code batch*seqLen*dModel} float на GPU (владеет {@link #embeddingScratchGpu})
     */
    public GpuFloatBuffer forwardGpuEmbeddings(Tensor inputTokens) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape.length != 2) {
            throw new IllegalArgumentException("inputTokens must be 2D [batch, seq_len]");
        }
        int batch = inputShape[0];
        int seqLen = inputShape[1];
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException("seq_len " + seqLen + " > max_seq_len " + maxSeqLen);
        }
        if (!gpuResident
                || !tokenEmbedding.hasWeightsGpu()
                || !positionEmbedding.hasWeightsGpu()
                || !TensorOpsGPU.shouldUseGpuElementwise(batch * seqLen * dModel)) {
            throw new IllegalStateException(
                    "forwardGpuEmbeddings требует gpuResident, GPU-веса эмбеддингов и CUDA");
        }
        GpuTensor xGpu = ensureEmbeddingScratchGpu(batch, seqLen);
        tokenEmbedding.forwardGatherToGpuTensor(inputTokens, xGpu);
        TensorOpsGPU.addPositionEmbeddingGpuDevice(
                xGpu.dataBuffer(),
                positionEmbedding.positionWeightsDataBuffer(),
                batch,
                seqLen,
                dModel);
        return xGpu.dataBuffer();
    }

    /**
     * Полный стек декодера на VRAM: вход и выход — planar {@code batch*seqLen*dModel}. Без промежуточного D2H между
     * слоями; кэши {@link BlockActivationCacheDevice} должны быть подготовлены (см. {@link #forward}).
     *
     * <p>При {@link #isDecoderLayerCudaGraphRequested()} и успешном захвате — один {@code cudaGraphExec} на слой
     * ({@link #runDecoderStackLayers}); см. env {@code JGPT_DECODER_LAYER_CUDA_GRAPH}.
     */
    public GpuFloatBuffer forwardGpuDecoder(
            GpuFloatBuffer xDevice, Tensor mask, boolean training, int batch, int seqLen) {
        if (!training) {
            throw new IllegalArgumentException("forwardGpuDecoder: только training=true");
        }
        if (!deviceDecoderBackward || gpuDecoderLayer == null || !TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("forwardGpuDecoder требует deviceDecoderBackward, pipeline и CUDA");
        }
        if (blockCachesDevice == null || blockCachesDevice.length != numLayers) {
            throw new IllegalStateException("forwardGpuDecoder: сначала выделите BlockActivationCacheDevice на все слои");
        }
        long planeLong = (long) batch * seqLen * dModel;
        if (planeLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("batch*seq*dModel overflow");
        }
        int plane = (int) planeLong;
        if (xDevice.numFloats() < plane) {
            throw new IllegalArgumentException("xDevice: ожидается не менее " + plane + " float");
        }
        return runDecoderStackLayers(xDevice, mask, batch, seqLen, true, blockCachesDevice);
    }

    /**
     * Стек декодера на VRAM для инференса: тот же D2D-пайплайн, что {@link #forwardGpuDecoder}, но без
     * {@link BlockActivationCacheDevice} (нет сохранения активаций под backward). PCIe: загрузка токенов в embedding
     * (см. {@link #forwardGpuEmbeddings}) и при необходимости одна выгрузка после последнего слоя.
     *
     * <p>CUDA graph на слой — как у {@link #forwardGpuDecoder} при том же env.
     */
    public GpuFloatBuffer forwardGpuDecoderInfer(GpuFloatBuffer xDevice, Tensor mask, int batch, int seqLen) {
        if (!decoderGpuPipeline || gpuDecoderLayer == null || !TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException(
                    "forwardGpuDecoderInfer требует decoderGpuPipeline, VRAM-веса декодера и CUDA");
        }
        long planeLong = (long) batch * seqLen * dModel;
        if (planeLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("batch*seq*dModel overflow");
        }
        int plane = (int) planeLong;
        if (xDevice.numFloats() < plane) {
            throw new IllegalArgumentException("xDevice: ожидается не менее " + plane + " float");
        }
        return runDecoderStackLayers(xDevice, mask, batch, seqLen, false, null);
    }

    private static GpuFloatBuffer decoderScratchOther(GpuFloatBuffer cur, GpuFloatBuffer a, GpuFloatBuffer b) {
        return cur != a ? a : b;
    }

    /**
     * RMSNorm + LM head: активации до нормы уже на GPU. При {@code fillLastHiddenForBackward} заполняет
     * {@link #xBeforeFinalNormGpu}, {@link #lastHiddenGpu}, {@link #lastLogitsGpu}; хостовые {@link #lastHidden} и
     * {@link #xBeforeFinalNorm} не заполняются. Логиты на хост не копируются — возвращается тензор-заглушка той же
     * формы (обучение с {@link #deviceLogitsEnabled} и CE на device).
     */
    public Tensor forwardGpuLmHeadFromDevice(
            GpuFloatBuffer xBeforeNormDevice, int batch, int seqLen, boolean fillLastHiddenForBackward) {
        if (!gpuResident || gpuResidentHead == null) {
            throw new IllegalStateException("forwardGpuLmHeadFromDevice требует gpuResident=true");
        }
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("forwardGpuLmHeadFromDevice требует CUDA");
        }
        int rows = batch * seqLen;
        int flat = rows * dModel;
        int logitsFlatElems = rows * vocabSize;
        if (xBeforeNormDevice.numFloats() < flat) {
            throw new IllegalArgumentException("xBeforeNormDevice слишком мал");
        }
        float eps = TensorOpsGPU.rmsNormEps();
        lmHeadNormScratchGpu = ensureGpuBuffer(lmHeadNormScratchGpu, flat);
        lastLogitsGpu = ensureGpuBuffer(lastLogitsGpu, logitsFlatElems);
        applyLmHeadFusedPreferredThenSplit(
                xBeforeNormDevice,
                gpuResidentHead.layerNormGammaGpu().dataBuffer(),
                eps,
                lmHeadNormScratchGpu,
                gpuResidentHead.lmHeadGpu().dataBuffer(),
                lastLogitsGpu,
                rows,
                dModel,
                vocabSize);
        if (fillLastHiddenForBackward) {
            xBeforeFinalNormGpu = ensureGpuBuffer(xBeforeFinalNormGpu, flat);
            xBeforeFinalNormGpu.copyFromDevice(xBeforeNormDevice, flat);
            lastHiddenGpu = ensureGpuBuffer(lastHiddenGpu, flat);
            lastHiddenGpu.copyFromDevice(lmHeadNormScratchGpu, flat);
            lastHidden = null;
            xBeforeFinalNorm = null;
        }
        int stubLen = logitsFlatElems;
        if (logitsShapeStub == null || logitsShapeStub.length != stubLen) {
            logitsShapeStub = new float[stubLen];
        }
        return Tensor.wrap(logitsShapeStub, new int[] {batch, seqLen, vocabSize});
    }

    /**
     * Sampled-train-only: финальный RMSNorm + dot по столбцам LM-head только для {@code candidateIds}.
     * Не материализует {@link #lastLogitsGpu} ({@code rows×vocab}); закрывает предыдущие VRAM-буферы полных логитов.
     *
     * <p>Явный split-путь (RMSNorm + kernel кандидатов), без {@code JGPT_FUSED_LM_HEAD}, чтобы не получать полную
     * матрицу логитов как побочный продукт fusion.
     *
     * @return тензор-заглушка формы {@code [batch, seq_len, 1]}: candidate logits и их градиенты остаются на device,
     *     а этот Tensor несёт только метаданные шага для существующего train/backward orchestration.
     */
    public Tensor forwardGpuLmHeadCandidateLogitsFromDevice(
            GpuFloatBuffer xBeforeNormDevice,
            int batch,
            int seqLen,
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateLogitsOut,
            int candidateCount,
            boolean fillLastHiddenForBackward) {
        if (!gpuResident || gpuResidentHead == null) {
            throw new IllegalStateException("forwardGpuLmHeadCandidateLogitsFromDevice требует gpuResident=true");
        }
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("forwardGpuLmHeadCandidateLogitsFromDevice требует CUDA");
        }
        if (candidateIds == null || candidateLogitsOut == null || candidateCount <= 0) {
            throw new IllegalArgumentException("candidateIds/candidateLogitsOut/candidateCount");
        }
        int rows = batch * seqLen;
        int flat = rows * dModel;
        long candElemsLong = (long) rows * candidateCount;
        if (candElemsLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("rows*candidateCount overflow");
        }
        int candElems = (int) candElemsLong;
        if (xBeforeNormDevice.numFloats() < flat) {
            throw new IllegalArgumentException("xBeforeNormDevice слишком мал");
        }
        if (candidateIds.numInts() < candElems) {
            throw new IllegalArgumentException("candidateIds buffer too small");
        }
        if (candidateLogitsOut.numFloats() < candElems) {
            throw new IllegalArgumentException("candidateLogitsOut buffer too small");
        }
        float eps = TensorOpsGPU.rmsNormEps();
        lmHeadNormScratchGpu = ensureGpuBuffer(lmHeadNormScratchGpu, flat);
        lastLogitsGpu = closeGpuBuffer(lastLogitsGpu);
        lastLogitsGradGpu = closeGpuBuffer(lastLogitsGradGpu);

        TensorOpsGPU.rmsNormGpuDevice(
                xBeforeNormDevice,
                gpuResidentHead.layerNormGammaGpu().dataBuffer(),
                eps,
                lmHeadNormScratchGpu,
                rows,
                dModel);
        TensorOpsGPU.lmHeadCandidateLogitsGpuDevice(
                lmHeadNormScratchGpu,
                gpuResidentHead.lmHeadGpu().dataBuffer(),
                candidateIds,
                candidateLogitsOut,
                rows,
                dModel,
                vocabSize,
                candidateCount);

        if (fillLastHiddenForBackward) {
            xBeforeFinalNormGpu = ensureGpuBuffer(xBeforeFinalNormGpu, flat);
            xBeforeFinalNormGpu.copyFromDevice(xBeforeNormDevice, flat);
            lastHiddenGpu = ensureGpuBuffer(lastHiddenGpu, flat);
            lastHiddenGpu.copyFromDevice(lmHeadNormScratchGpu, flat);
            lastHidden = null;
            xBeforeFinalNorm = null;
        }
        int stubLen = rows;
        if (logitsShapeStub == null || logitsShapeStub.length != stubLen) {
            logitsShapeStub = new float[stubLen];
        }
        /*
         * Sampled path keeps candidate logits and grads on device; this tensor is only a lightweight carrier
         * for batch/seq metadata into the existing backward/optimizer orchestration.
         */
        return Tensor.wrap(logitsShapeStub, new int[] {batch, seqLen, 1});
    }

    /**
     * Финальный RMSNorm + LM head на device; логиты один раз копируются на хост ({@code [batch, seq, vocab]}).
     * Для инференса после {@link #forwardGpuDecoderInfer}.
     */
    public Tensor forwardGpuLmHeadFromDeviceToHost(GpuFloatBuffer xBeforeNormDevice, int batch, int seqLen) {
        if (!gpuResident || gpuResidentHead == null) {
            throw new IllegalStateException("forwardGpuLmHeadFromDeviceToHost требует gpuResident=true");
        }
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("forwardGpuLmHeadFromDeviceToHost требует CUDA");
        }
        int rows = batch * seqLen;
        int flat = rows * dModel;
        int logitsFlatElems = rows * vocabSize;
        if (xBeforeNormDevice.numFloats() < flat) {
            throw new IllegalArgumentException("xBeforeNormDevice слишком мал");
        }
        float eps = TensorOpsGPU.rmsNormEps();
        lmHeadNormScratchGpu = ensureGpuBuffer(lmHeadNormScratchGpu, flat);
        try (GpuFloatBuffer logitsFlat = GpuFloatBuffer.allocate(logitsFlatElems)) {
            applyLmHeadFusedPreferredThenSplit(
                    xBeforeNormDevice,
                    gpuResidentHead.layerNormGammaGpu().dataBuffer(),
                    eps,
                    lmHeadNormScratchGpu,
                    gpuResidentHead.lmHeadGpu().dataBuffer(),
                    logitsFlat,
                    rows,
                    dModel,
                    vocabSize);
            float[] host = new float[logitsFlatElems];
            logitsFlat.copyTo(host, 0, logitsFlatElems);
            xBeforeFinalNorm = null;
            lastHidden = null;
            return Tensor.wrap(host, new int[] {batch, seqLen, vocabSize});
        }
    }

    /**
     * LM head backward + final RMSNorm backward entirely on device.
     * Входные device-буферы: {@code dLogitsGrad} [rows×V], {@code dNormedHidden} [rows×d],
     * {@code dXBeforeNorm} [rows×d], {@code dGammaGpu} (final RMSNorm γ).
     *
     * @return {@link GpuFloatBuffer} — ∂L/∂x_before_norm [rows × dModel] на VRAM (вход для decoder backward loop)
     */
    public GpuFloatBuffer backwardFromDeviceLogits(
            GpuFloatBuffer dLogitsGrad,
            GpuFloatBuffer dNormedHidden,
            GpuFloatBuffer dXBeforeNorm,
            int rows,
            boolean zeroGpuGrads) {
        if (!gpuResident || gpuResidentHead == null) {
            throw new IllegalStateException("backwardFromDeviceLogits requires gpuResident=true");
        }
        float eps = TensorOpsGPU.rmsNormEps();

        GpuFloatBuffer lmHeadBuf = gpuResidentHead.lmHeadGpu().dataBuffer();
        GpuFloatBuffer gammaBuf = gpuResidentHead.layerNormGammaGpu().dataBuffer();

        GpuFloatBuffer dHidden = GpuFloatBuffer.allocate((long) rows * dModel);
        TensorOpsGPU.matmulGpuDeviceEx(dLogitsGrad, lmHeadBuf, dHidden, rows, vocabSize, dModel, false, true);

        GpuTensor lmHeadGpu = gpuResidentHead.lmHeadGpu();
        if (!lmHeadGpu.hasGradBuffer() || zeroGpuGrads) {
            lmHeadGpu.zeroGrad();
        }
        GpuFloatBuffer dWLmHead = GpuFloatBuffer.allocate((long) dModel * vocabSize);
        TensorOpsGPU.matmulGpuDeviceEx(dNormedHidden, dLogitsGrad, dWLmHead, dModel, rows, vocabSize, true, false);
        TensorOpsGPU.accumulateAddGpuDevice(lmHeadGpu.gradBuffer(), dWLmHead, lmHead.size());
        dWLmHead.close();

        GpuTensor gammaGpu = gpuResidentHead.layerNormGammaGpu();
        if (!gammaGpu.hasGradBuffer() || zeroGpuGrads) {
            gammaGpu.zeroGrad();
        }
        GpuFloatBuffer dGradBeforeNorm = GpuFloatBuffer.allocate((long) rows * dModel);
        GpuFloatBuffer dGGamma = GpuFloatBuffer.allocate(dModel);
        if (DebugGpuTrain.isEnabled()) {
            TensorOpsGPU.synchronizeStream();
            float[] preGx = new float[4];
            dGradBeforeNorm.copyTo(preGx, 0, Math.min(4, rows * dModel));
            agentLogB39372RmsBuf(
                    "H_uninit",
                    "GPTModel.backwardFromDeviceLogits",
                    "dGradBeforeNorm_before_clear",
                    preGx[0],
                    preGx.length > 1 ? preGx[1] : 0f,
                    preGx.length > 2 ? preGx[2] : 0f,
                    preGx.length > 3 ? preGx[3] : 0f,
                    (long) rows * dModel);
            float[] preGg = new float[1];
            dGGamma.copyTo(preGg, 0, 1);
            agentLogB39372RmsBuf(
                    "H_uninit",
                    "GPTModel.backwardFromDeviceLogits",
                    "dGGamma_before_clear",
                    preGg[0],
                    0f,
                    0f,
                    0f,
                    dModel);
        }
        // rms_norm_bwd accumulates into gX / gGamma; cudaMalloc/async content is undefined.
        dGradBeforeNorm.clear();
        dGGamma.clear();
        TensorOpsGPU.rmsNormBackwardGpuDevice(dHidden, dXBeforeNorm, gammaBuf, eps, dGradBeforeNorm, dGGamma, rows, dModel);
        TensorOpsGPU.accumulateAddGpuDevice(gammaGpu.gradBuffer(), dGGamma, layerNormFinal.size());
        dGGamma.close();

        dHidden.close();
        return dGradBeforeNorm;
    }

    private static boolean resolveUseFp16ActivationCache() {
        return BlockActivationCacheDevice.activationCacheFp16StorageFromEnv();
    }

    /**
     * Forward: logits [batch, seq_len, vocab_size].
     *
     * @param inputTokens индексы токенов [batch, seq_len] (в {@link Tensor} как float)
     */
    public Tensor forward(Tensor inputTokens) {
        return forward(inputTokens, false);
    }

    /**
     * @param training если {@code true}, сохраняются активации для {@link #backward(Tensor)}.
     */
    public Tensor forward(Tensor inputTokens, boolean training) {
        return forward(inputTokens, training, false);
    }

    /**
     * @param useGpuResidentHead при {@code true} (и {@link #isGpuResident()}, доступная CUDA) — финальный RMSNorm
     *     и LM head на GPU без H2D весов; иначе как {@link #forward(Tensor, boolean)} с {@code false}.
     *     <p>При {@code training == false}, {@link #isGpuResident()} и {@link #isDecoderGpuPipeline()} — активации от
     *     GPU-embedding до выхода декодера остаются на VRAM (D2D между слоями); при {@code useGpuResidentHead} — до LM
     *     head включительно, с одним D2H логитов в конце. При {@code useGpuResidentHead == false} — одна выгрузка
     *     плоскости {@code [B,S,d]} перед CPU RMSNorm/LM head.
     */
    public Tensor forward(Tensor inputTokens, boolean training, boolean useGpuResidentHead) {
        return forward(inputTokens, training, useGpuResidentHead, false);
    }

    /**
     * @param inferLogitsOnDevice если {@code true}, {@code training == false}, {@code useGpuResidentHead == true} и
     *     активен VRAM infer-пайплайн — логиты остаются в {@link #deviceLogitsBuffer()} (без D2H плоскости логитов);
     *     возвращается тензор-заглушка той же формы. Для оценочного loss на device см. {@code LLMTrainer.evaluate}.
     */
    public Tensor forward(Tensor inputTokens, boolean training, boolean useGpuResidentHead, boolean inferLogitsOnDevice) {
        if (inferLogitsOnDevice && training) {
            throw new IllegalArgumentException("inferLogitsOnDevice is only valid when training=false");
        }
        if (lastForwardWasTraining != training) {
            closeBlockCachesDevice();
            blockCachesDevice = null;
        }
        try {
        int[] inputShape = inputTokens.getShape();
        if (inputShape.length != 2) {
            throw new IllegalArgumentException("inputTokens must be 2D [batch, seq_len]");
        }

        int batch = inputShape[0];
        int seqLen = inputShape[1];

        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException("seq_len " + seqLen + " > max_seq_len " + maxSeqLen);
        }

        lastInputTokens = inputTokens;
        lastSeqLen = seqLen;

        Tensor mask = getOrCreateCausalMask(seqLen);
        lastMask = mask;

        if (training) {
            boolean deviceOnlyActivations =
                    deviceDecoderBackward
                            && gpuDecoderLayer != null
                            && TensorOpsGPU.isGpuAvailable();
            if (deviceOnlyActivations) {
                blockCaches = null;
                if (blockCachesDevice == null || blockCachesDevice.length != numLayers) {
                    closeBlockCachesDevice();
                    blockCachesDevice = new BlockActivationCacheDevice[numLayers];
                }
                for (int i = 0; i < numLayers; i++) {
                    if (blockCachesDevice[i] == null) {
                        blockCachesDevice[i] = new BlockActivationCacheDevice();
                    }
                    blockCachesDevice[i].ensure(batch, seqLen, dModel, numHeads, dIntermediate);
                }
            } else {
                if (blockCaches == null || blockCaches.length != numLayers) {
                    blockCaches = new BlockActivationCache[numLayers];
                }
                for (int i = 0; i < numLayers; i++) {
                    if (blockCaches[i] == null) {
                        blockCaches[i] = new BlockActivationCache();
                    }
                    blockCaches[i].useFp16ActivationStorage = useFp16ActivationCache;
                    blockCaches[i].preferFp32ForFusedGpuBackwardSlots =
                            gpuResident && TensorOpsGPU.isGpuAvailable();
                }
                closeBlockCachesDevice();
            }
        } else {
            blockCaches = null;
            closeBlockCachesDevice();
        }

        boolean vramInferDecoderLm =
                !training
                        && decoderGpuPipeline
                        && gpuDecoderLayer != null
                        && TensorOpsGPU.isGpuAvailable()
                        && gpuResident
                        && tokenEmbedding.hasWeightsGpu()
                        && positionEmbedding.hasWeightsGpu()
                        && TensorOpsGPU.shouldUseGpuElementwise(batch * seqLen * dModel);
        if (vramInferDecoderLm) {
            GpuFloatBuffer emb = forwardGpuEmbeddings(inputTokens);
            GpuFloatBuffer decOut = forwardGpuDecoderInfer(emb, mask, batch, seqLen);
            if (useGpuResidentHead) {
                if (gpuResidentHead == null) {
                    throw new IllegalArgumentException(
                            "useGpuResidentHead requires model constructed with gpuResident=true");
                }
                xBeforeFinalNorm = null;
                lastHidden = null;
                if (inferLogitsOnDevice) {
                    return forwardGpuLmHeadFromDevice(decOut, batch, seqLen, false);
                }
                return forwardGpuLmHeadFromDeviceToHost(decOut, batch, seqLen);
            }
            int rowsInfer = batch * seqLen;
            int flatInfer = rowsInfer * dModel;
            Tensor xHostInfer = new Tensor(new int[] {batch, seqLen, dModel});
            decOut.copyTo(xHostInfer.internalBuffer(), 0, flatInfer);
            xBeforeFinalNorm = xHostInfer;
            Tensor xNormInfer = TensorOps.rmsNorm(xHostInfer, layerNormFinal, TensorOpsGPU.rmsNormEps());
            lastHidden = xNormInfer;
            Tensor hiddenFlatInfer = Tensor.wrap(xNormInfer.internalBuffer(), new int[] {rowsInfer, dModel});
            Tensor logitsFlatInfer = TensorOps.matmul(hiddenFlatInfer, lmHead);
            return Tensor.wrap(logitsFlatInfer.internalBuffer(), new int[] {batch, seqLen, vocabSize});
        }

        boolean vramEmbedDecoderLm =
                training
                        && useGpuResidentHead
                        && deviceLogitsEnabled
                        && deviceDecoderBackward
                        && gpuDecoderLayer != null
                        && TensorOpsGPU.isGpuAvailable()
                        && gpuResident
                        && tokenEmbedding.hasWeightsGpu()
                        && positionEmbedding.hasWeightsGpu()
                        && TensorOpsGPU.shouldUseGpuElementwise(batch * seqLen * dModel);
        if (vramEmbedDecoderLm) {
            GpuFloatBuffer emb = forwardGpuEmbeddings(inputTokens);
            GpuFloatBuffer decOut = forwardGpuDecoder(emb, mask, true, batch, seqLen);
            xBeforeFinalNorm = null;
            lastHidden = null;
            return forwardGpuLmHeadFromDevice(decOut, batch, seqLen, training);
        }

        Tensor x;
        if (gpuResident
                && tokenEmbedding.hasWeightsGpu()
                && positionEmbedding.hasWeightsGpu()
                && TensorOpsGPU.shouldUseGpuElementwise(batch * seqLen * dModel)) {
            GpuTensor xGpu = ensureEmbeddingScratchGpu(batch, seqLen);
            tokenEmbedding.forwardGatherToGpuTensor(inputTokens, xGpu);
            TensorOpsGPU.addPositionEmbeddingGpuDevice(
                    xGpu.dataBuffer(),
                    positionEmbedding.positionWeightsDataBuffer(),
                    batch,
                    seqLen,
                    dModel);
            x = new Tensor(new int[]{batch, seqLen, dModel});
            xGpu.downloadTo(x.internalBuffer(), 0, batch * seqLen * dModel);
        } else {
            x = tokenEmbedding.forward(inputTokens);
            positionEmbedding.addToActivationsInPlace(x, seqLen, 0);
        }

        x = forwardDecoderStack(x, mask, training, batch, seqLen);

        xBeforeFinalNorm = x;
        if (useGpuResidentHead) {
            if (!gpuResident || gpuResidentHead == null) {
                throw new IllegalArgumentException(
                        "useGpuResidentHead requires model constructed with gpuResident=true");
            }
            if (!TensorOpsGPU.isGpuAvailable()) {
                throw new IllegalStateException("useGpuResidentHead requires CUDA");
            }
            return forwardGpuLmHead(xBeforeFinalNorm, training);
        }
        x = TensorOps.rmsNorm(x, layerNormFinal, TensorOpsGPU.rmsNormEps());
        lastHidden = x;
        Tensor hiddenFlat = Tensor.wrap(x.internalBuffer(), new int[]{batch * seqLen, dModel});
        // matmul пишет в CPU-буфер результата (GPU-путь — через cuBLAS в тот же массив), без лишней копии
        Tensor logitsFlat = TensorOps.matmul(hiddenFlat, lmHead);
        return Tensor.wrap(logitsFlat.internalBuffer(), new int[]{batch, seqLen, vocabSize});
        } finally {
            lastForwardWasTraining = training;
        }
    }

    /**
     * Полный GPU-forward обучения для sampled CE: декодер на VRAM + финальный RMSNorm + кандидатные логиты без
     * материализации {@code rows×vocab} логитов на GPU.
     *
     * <p>Требует тот же контракт, что ветка {@code vramEmbedDecoderLm} в {@link #forward(Tensor, boolean, boolean)}:
     * {@link #deviceLogitsEnabled}, {@link #deviceDecoderBackward}, резидентные эмбеддинги и CUDA.
     *
     * @param candidateIds {@code rows × candidateCount} int на GPU (уже заполнен на хосте→device)
     * @param candidateLogitsOut {@code rows × candidateCount} float на GPU (заполняется)
     */
    public Tensor forwardTrainingDeviceSampled(
            Tensor inputTokens, GpuIntBuffer candidateIds, GpuFloatBuffer candidateLogitsOut, int candidateCount) {
        if (!lastForwardWasTraining) {
            closeBlockCachesDevice();
            blockCachesDevice = null;
        }
        try {
            int[] inputShape = inputTokens.getShape();
            if (inputShape.length != 2) {
                throw new IllegalArgumentException("inputTokens must be 2D [batch, seq_len]");
            }
            int batch = inputShape[0];
            int seqLen = inputShape[1];
            if (seqLen > maxSeqLen) {
                throw new IllegalArgumentException("seq_len " + seqLen + " > max_seq_len " + maxSeqLen);
            }
            boolean vramEmbedDecoderLm =
                    deviceLogitsEnabled
                            && deviceDecoderBackward
                            && gpuDecoderLayer != null
                            && TensorOpsGPU.isGpuAvailable()
                            && gpuResident
                            && tokenEmbedding.hasWeightsGpu()
                            && positionEmbedding.hasWeightsGpu()
                            && TensorOpsGPU.shouldUseGpuElementwise(batch * seqLen * dModel);
            if (!vramEmbedDecoderLm) {
                throw new IllegalStateException(
                        "forwardTrainingDeviceSampled requires deviceLogitsEnabled, deviceDecoderBackward, decoder GPU pipeline and resident embeddings");
            }
            lastInputTokens = inputTokens;
            lastSeqLen = seqLen;
            Tensor mask = getOrCreateCausalMask(seqLen);
            lastMask = mask;

            blockCaches = null;
            if (blockCachesDevice == null || blockCachesDevice.length != numLayers) {
                closeBlockCachesDevice();
                blockCachesDevice = new BlockActivationCacheDevice[numLayers];
            }
            for (int i = 0; i < numLayers; i++) {
                if (blockCachesDevice[i] == null) {
                    blockCachesDevice[i] = new BlockActivationCacheDevice();
                }
                blockCachesDevice[i].ensure(batch, seqLen, dModel, numHeads, dIntermediate);
            }

            GpuFloatBuffer emb = forwardGpuEmbeddings(inputTokens);
            GpuFloatBuffer decOut = forwardGpuDecoder(emb, mask, true, batch, seqLen);
            xBeforeFinalNorm = null;
            lastHidden = null;
            return forwardGpuLmHeadCandidateLogitsFromDevice(
                    decOut, batch, seqLen, candidateIds, candidateLogitsOut, candidateCount, true);
        } finally {
            lastForwardWasTraining = true;
        }
    }

    /**
     * Полный проход по {@code numLayers} декодер-блокам (аттеншн + FFN), с resident GPU-буферами внутри
     * {@link DecoderBlock#forward} при {@link #gpuResident}. Тот же цикл, что между embedding и финальной
     * RMSNorm в {@link #forward(Tensor, boolean, boolean)} — вынесен для профилирования (Nsight) и единой точки
     * входа для будущих fusion между слоями.
     *
     * @param x активации после embedding/позиций [batch, seqLen, dModel]
     * @return активации перед финальной нормой
     */
    Tensor forwardDecoderStack(Tensor x, Tensor mask, boolean training, int batch, int seqLen) {
        boolean deviceOnlyActivations =
                training
                        && deviceDecoderBackward
                        && gpuDecoderLayer != null
                        && TensorOpsGPU.isGpuAvailable();
        for (int i = 0; i < numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].attnBuffers() : null;
            x =
                    blocks[i].forwardGpuPipeline(
                            x,
                            mask,
                            deviceOnlyActivations ? null : (training ? blockCaches[i] : null),
                            deviceOnlyActivations ? blockCachesDevice[i] : null,
                            ffnResident,
                            attnResident);
        }
        return x;
    }

    private Tensor getOrCreateCausalMask(int seqLen) {
        if (cachedCausalMask != null && cachedCausalMaskSeqLen == seqLen) {
            return cachedCausalMask;
        }
        cachedCausalMask = TensorOps.createCausalMask(seqLen);
        cachedCausalMaskSeqLen = seqLen;
        return cachedCausalMask;
    }

    private Tensor getOrCreateLmHeadTranspose() {
        if (cachedLmHeadTranspose != null && cachedLmHeadTransposeValid) {
            return cachedLmHeadTranspose;
        }
        cachedLmHeadTranspose = TensorOpsBackward.transpose(lmHead);
        cachedLmHeadTransposeValid = true;
        return cachedLmHeadTranspose;
    }

    public void onParametersUpdated() {
        cachedLmHeadTransposeValid = false;
        syncGpuResidentWeightsFromHost();
    }

    public void onGpuParametersUpdated() {
        cachedLmHeadTransposeValid = false;
    }

    public void syncWeightsFromGpu(Map<Tensor, GpuTensor> paramMap) {
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt != null && !gt.isClosed()) {
                Tensor p = e.getKey();
                gt.downloadTo(p.internalBuffer(), 0, p.size());
            }
        }
        if (TensorOpsGPU.isGpuAvailable()) {
            /* cudaMemcpyAsync на kTensorCudaStream; следующий forward/шаг должен ждать завершения D2H. */
            TensorOpsGPU.synchronizeStream();
        }
    }

    public boolean hasDeviceLogitsBuffers() {
        return lastLogitsGpu != null
                && !lastLogitsGpu.isClosed()
                && lastHiddenGpu != null
                && !lastHiddenGpu.isClosed()
                && xBeforeFinalNormGpu != null
                && !xBeforeFinalNormGpu.isClosed();
    }

    /**
     * VRAM-активации под backward финального RMSNorm/LM-head без полного буфера логитов — после
     * {@link #forwardTrainingDeviceSampled} / {@link #forwardGpuLmHeadCandidateLogitsFromDevice}.
     */
    public boolean hasDeviceSampledTrainLmHeadActivations() {
        return lastHiddenGpu != null
                && !lastHiddenGpu.isClosed()
                && xBeforeFinalNormGpu != null
                && !xBeforeFinalNormGpu.isClosed();
    }

    public GpuFloatBuffer deviceLogitsBuffer() {
        return lastLogitsGpu;
    }

    /** ∂logits на device при path device-CE; иначе {@code null}. */
    public GpuFloatBuffer deviceLogitsGradBuffer() {
        return lastLogitsGradGpu;
    }

    public GpuFloatBuffer ensureDeviceLogitsGradBuffer(int numFloats) {
        lastLogitsGradGpu = ensureGpuBuffer(lastLogitsGradGpu, numFloats);
        return lastLogitsGradGpu;
    }

    public void setSampledTrainLossGrad(
            GpuIntBuffer candidateIds, GpuFloatBuffer candidateGrad, int candidateCount) {
        lastSampledCandidateIdsGpu = candidateIds;
        lastSampledCandidateGradGpu = candidateGrad;
        lastSampledCandidateCount = candidateCount;
        lastLogitsGradGpu = null;
    }

    public void clearSampledTrainLossGrad() {
        lastSampledCandidateIdsGpu = null;
        lastSampledCandidateGradGpu = null;
        lastSampledCandidateCount = 0;
    }

    public boolean hasSampledTrainLossGrad() {
        return lastSampledCandidateIdsGpu != null
                && lastSampledCandidateGradGpu != null
                && lastSampledCandidateCount > 0;
    }

    /**
     * Prefill с KV-cache (batch=1): заполняет кэш K/V после RoPE по всем слоям.
     *
     * @param ropeOffset сдвиг абсолютных позиций RoPE (0 — как обычный forward; при скользящем окне — {@code startIdx})
     * @return logits {@code [1, seq_len, vocab_size]}
     */
    public Tensor forwardPrefill(Tensor inputTokens, KvCache cache, int ropeOffset) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape[0] != 1) {
            throw new IllegalArgumentException("forwardPrefill supports batch_size=1 only");
        }
        int seqLen = inputShape[1];
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException("seq_len " + seqLen + " > max_seq_len " + maxSeqLen);
        }
        if (cache.numLayers() != numLayers) {
            throw new IllegalArgumentException("KvCache layer count mismatch");
        }

        Tensor x = tokenEmbedding.forward(inputTokens);
        positionEmbedding.addToActivationsInPlace(x, seqLen, ropeOffset);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        for (int i = 0; i < numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].attnBuffers() : null;
            x = blocks[i].forwardKvPrefill(
                    x, mask, null, cache.getK(i), cache.getV(i), ropeOffset, ffnResident, attnResident);
        }

        x = TensorOps.rmsNorm(x, layerNormFinal, TensorOpsGPU.rmsNormEps());
        Tensor logits = new Tensor(new int[]{1, seqLen, vocabSize});
        copyBatchPlane(logits, TensorOps.matmul(sliceBatch3D(x, 0), lmHead), 0);
        cache.setLength(seqLen);
        return logits;
    }

    /**
     * Prefill с KV на VRAM ({@link KvCacheGpu}): без host-тензоров K/V в attention на горячем пути.
     *
     * @see #forwardPrefill(Tensor, KvCache, int)
     */
    public Tensor forwardPrefill(Tensor inputTokens, KvCacheGpu cache, int ropeOffset) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape[0] != 1) {
            throw new IllegalArgumentException("forwardPrefill supports batch_size=1 only");
        }
        int seqLen = inputShape[1];
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException("seq_len " + seqLen + " > max_seq_len " + maxSeqLen);
        }
        if (cache.numLayers() != numLayers) {
            throw new IllegalArgumentException("KvCacheGpu layer count mismatch");
        }
        if (seqLen > cache.maxSeqLen()) {
            throw new IllegalArgumentException("seq_len " + seqLen + " > KvCacheGpu.maxSeqLen " + cache.maxSeqLen());
        }

        Tensor x = tokenEmbedding.forward(inputTokens);
        positionEmbedding.addToActivationsInPlace(x, seqLen, ropeOffset);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        for (int i = 0; i < numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].attnBuffers() : null;
            x =
                    blocks[i].forwardKvPrefillVram(
                            x,
                            mask,
                            null,
                            cache.getK(i),
                            cache.getV(i),
                            cache.maxSeqLen(),
                            ropeOffset,
                            ffnResident,
                            attnResident);
        }

        x = TensorOps.rmsNorm(x, layerNormFinal, TensorOpsGPU.rmsNormEps());
        Tensor logits = new Tensor(new int[] {1, seqLen, vocabSize});
        copyBatchPlane(logits, TensorOps.matmul(sliceBatch3D(x, 0), lmHead), 0);
        cache.setLength(seqLen);
        return logits;
    }

    /**
     * Один шаг декодирования с KV-cache (batch=1). До вызова: {@code cache.length() == cacheLenBefore}.
     *
     * @param inputTokens {@code [1, 1]} — один токен
     * @param ropePosition абсолютный индекс позиции этого токена (RoPE)
     * @return logits {@code [1, 1, vocab_size]}
     */
    public Tensor forwardDecode(Tensor inputTokens, KvCache cache, int cacheLenBefore, int ropePosition) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape[0] != 1 || inputShape[1] != 1) {
            throw new IllegalArgumentException("forwardDecode expects [1, 1] token indices");
        }
        if (cache.length() != cacheLenBefore) {
            throw new IllegalStateException(
                    "KV cache length " + cache.length() + " != cacheLenBefore " + cacheLenBefore);
        }

        Tensor x = tokenEmbedding.forward(inputTokens);
        positionEmbedding.addToActivationsInPlace(x, 1, ropePosition);

        for (int i = 0; i < numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].attnBuffers() : null;
            x = blocks[i].forwardKvDecode(
                    x,
                    null,
                    cache.getK(i),
                    cache.getV(i),
                    cacheLenBefore,
                    ropePosition,
                    ffnResident,
                    attnResident);
        }

        x = TensorOps.rmsNorm(x, layerNormFinal, TensorOpsGPU.rmsNormEps());
        Tensor logits = new Tensor(new int[]{1, 1, vocabSize});
        copyBatchPlane(logits, TensorOps.matmul(sliceBatch3D(x, 0), lmHead), 0);
        cache.setLength(cacheLenBefore + 1);
        return logits;
    }

    /**
     * Decode с KV на VRAM.
     *
     * @see #forwardDecode(Tensor, KvCache, int, int)
     */
    public Tensor forwardDecode(Tensor inputTokens, KvCacheGpu cache, int cacheLenBefore, int ropePosition) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape[0] != 1 || inputShape[1] != 1) {
            throw new IllegalArgumentException("forwardDecode expects [1, 1] token indices");
        }
        if (cache.length() != cacheLenBefore) {
            throw new IllegalStateException(
                    "KV cache length " + cache.length() + " != cacheLenBefore " + cacheLenBefore);
        }
        if (cacheLenBefore >= cache.maxSeqLen()) {
            throw new IllegalStateException("KV cache full (cacheLenBefore " + cacheLenBefore + ")");
        }

        Tensor x = tokenEmbedding.forward(inputTokens);
        positionEmbedding.addToActivationsInPlace(x, 1, ropePosition);

        for (int i = 0; i < numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[i].attnBuffers() : null;
            x =
                    blocks[i].forwardKvDecodeVram(
                            x,
                            null,
                            cache.getK(i),
                            cache.getV(i),
                            cache.maxSeqLen(),
                            cacheLenBefore,
                            ropePosition,
                            ffnResident,
                            attnResident);
        }

        x = TensorOps.rmsNorm(x, layerNormFinal, TensorOpsGPU.rmsNormEps());
        Tensor logits = new Tensor(new int[] {1, 1, vocabSize});
        copyBatchPlane(logits, TensorOps.matmul(sliceBatch3D(x, 0), lmHead), 0);
        cache.setLength(cacheLenBefore + 1);
        return logits;
    }

    /**
     * Полный backward: ожидает {@link Tensor#gradBuffer()} у {@code logits} (например CE+softmax).
     * Перед вызовом нужен {@code forward(..., true)} в том же шаге.
     */
    public void backward(Tensor logits) {
        backward(logits, true, false);
    }

    /**
     * @param zeroParamGrads если {@code true} — обнулить градиенты параметров перед backward (обычный шаг);
     *     если {@code false} — накапливать в существующие буферы (gradient accumulation после первого микробатча).
     */
    public void backward(Tensor logits, boolean zeroParamGrads) {
        backward(logits, zeroParamGrads, false);
    }

    /**
     * @param zeroParamGrads если {@code true} — обнулить градиенты параметров перед backward (обычный шаг);
     *     если {@code false} — накапливать в существующие буферы (gradient accumulation после первого микробатча).
     * @param gpuTrainableGradsAlreadyZero подсказка от тренера: ∂ обучаемых параметров на VRAM уже обнулены после
     *     прошлого шага оптимизатора — можно пропустить {@link #zeroGpuTrainableParameterGrads()} (экономия
     *     {@code cudaMemsetAsync} на параметр). Имеет смысл только при {@link #isGpuResident()} и
     *     {@code zeroParamGrads == true}; иначе игнорируется.
     */
    public void backward(Tensor logits, boolean zeroParamGrads, boolean gpuTrainableGradsAlreadyZero) {
        if (blockCaches == null && blockCachesDevice == null) {
            throw new IllegalStateException(
                    lastForwardWasTraining
                            ? "activation caches missing after forward(training=true)"
                            : "call forward(input, true) before backward — last forward had training=false");
        }
        boolean deviceLmHeadActivations =
                deviceLogitsEnabled
                        && gpuResident
                        && gpuResidentHead != null
                        && lastHiddenGpu != null
                        && !lastHiddenGpu.isClosed()
                        && xBeforeFinalNormGpu != null
                        && !xBeforeFinalNormGpu.isClosed();
        if (!deviceLmHeadActivations && (lastHidden == null || xBeforeFinalNorm == null)) {
            throw new IllegalStateException(
                    "lastHidden or xBeforeFinalNorm is null — run forward(input, true) in the same step before backward");
        }
        boolean hasLogitsGrad =
                logits.hasGrad()
                        || (lastLogitsGradGpu != null && !lastLogitsGradGpu.isClosed())
                        || hasSampledTrainLossGrad();
        if (!hasLogitsGrad) {
            throw new IllegalStateException("logits must have grad buffer (zeroGrad + CE grad)");
        }

        if (zeroParamGrads) {
            /* При device-logits + VRAM хостовые ∂ параметров в этом шаге не используются — не тратить CPU на fill. */
            if (!(deviceLogitsEnabled && gpuResident && gpuResidentHead != null)) {
                zeroGradParameters();
            }
            if (gpuResident) {
                /* После успешного GPU step тренер уже обнулил ∂ на VRAM — повторный memset в начале шага избыточен. */
                if (!gpuTrainableGradsAlreadyZero) {
                    zeroGpuTrainableParameterGrads();
                }
            }
        }

        int[] ls = logits.getShape();
        int batch = ls[0];
        int seqLen = ls[1];

        if (deviceLogitsEnabled && gpuResident && gpuResidentHead != null) {
            backwardDeviceLogitsPath(logits, batch, seqLen, zeroParamGrads);
            return;
        }

        if (zeroParamGrads) {
            lmHead.zeroGrad();
            layerNormFinal.zeroGrad();
        }

        Tensor gradHidden = ensureReusableLike(backwardGradHidden, lastHidden);
        backwardGradHidden = gradHidden;
        gradHidden.zeroGrad();
        Tensor hiddenFlat = Tensor.wrap(lastHidden.internalBuffer(), new int[]{batch * seqLen, dModel});
        Tensor logitsGradFlat = Tensor.wrap(logits.gradBuffer(), new int[]{batch * seqLen, vocabSize});
        Tensor lmHeadT = getOrCreateLmHeadTranspose();
        Tensor gradHiddenFlat = TensorOps.matmul(logitsGradFlat, lmHeadT);
        Tensor dW = TensorOps.matmul(TensorOpsBackward.transpose(hiddenFlat), logitsGradFlat);
        TensorOpsBackward.accumulateGradientInto(lmHead, dW);
        System.arraycopy(
                gradHiddenFlat.internalBuffer(),
                0,
                gradHidden.gradBuffer(),
                0,
                gradHiddenFlat.internalBuffer().length);

        Tensor gradBeforeNorm = ensureReusableLike(backwardGradBeforeNorm, xBeforeFinalNorm);
        backwardGradBeforeNorm = gradBeforeNorm;
        gradBeforeNorm.zeroGrad();
        TensorOpsBackward.rmsNormBackward(gradHidden, xBeforeFinalNorm, layerNormFinal, TensorOpsGPU.rmsNormEps(), gradBeforeNorm, layerNormFinal);

        if (deviceDecoderBackward && gpuDecoderLayer != null) {
            try (GpuFloatBuffer dGradBeforeNorm = GpuFloatBuffer.allocate((long) batch * seqLen * dModel)) {
                dGradBeforeNorm.copyFrom(gradBeforeNorm.gradBuffer(), 0, batch * seqLen * dModel);
                backwardDecoderLayersDevice(dGradBeforeNorm, batch, seqLen, zeroParamGrads);
            }
        } else {
            if (gpuResident && TensorOpsGPU.isGpuAvailable()) {
                throw new IllegalStateException(
                        "GPU-resident model with CUDA requires device decoder backward (setDeviceLogitsEnabled(true) then "
                                + "setDeviceDecoderBackward(true), decoder pipeline env/prop); LLMTrainer + "
                                + "LLMConfig.toTrainingConfig enable this for training.");
            }
            backwardDecoderLayersHost(gradBeforeNorm, batch, seqLen, zeroParamGrads);
        }
    }

    /**
     * Device logits backward path: LM head + RMSNorm backward on GPU, then decoder layers
     * (device or host depending on {@link #deviceDecoderBackward}).
     */
    private void backwardDeviceLogitsPath(Tensor logits, int batch, int seqLen, boolean zeroParamGrads) {
        int rows = batch * seqLen;
        int flat = rows * dModel;
        int logitsFlat = rows * vocabSize;

        if (hasSampledTrainLossGrad()) {
            GpuFloatBuffer dNormedHidden = lastHiddenGpu;
            GpuFloatBuffer dXBeforeNorm = xBeforeFinalNormGpu;
            GpuFloatBuffer tmpNormedHidden = null;
            GpuFloatBuffer tmpXBeforeNorm = null;
            try {
                if (dNormedHidden == null || dNormedHidden.isClosed()) {
                    tmpNormedHidden = GpuFloatBuffer.allocate(flat);
                    tmpNormedHidden.copyFrom(lastHidden.internalBuffer(), 0, flat);
                    dNormedHidden = tmpNormedHidden;
                }
                if (dXBeforeNorm == null || dXBeforeNorm.isClosed()) {
                    tmpXBeforeNorm = GpuFloatBuffer.allocate(flat);
                    tmpXBeforeNorm.copyFrom(xBeforeFinalNorm.internalBuffer(), 0, flat);
                    dXBeforeNorm = tmpXBeforeNorm;
                }
                // Buffer belongs to thread-local workspace — must NOT close it here
                GpuFloatBuffer dGradBeforeNorm =
                        backwardFromSampledDeviceLogits(
                                lastSampledCandidateIdsGpu,
                                lastSampledCandidateGradGpu,
                                dNormedHidden,
                                dXBeforeNorm,
                                rows,
                                lastSampledCandidateCount,
                                zeroParamGrads);
                if (!deviceDecoderBackward || gpuDecoderLayer == null || blockCachesDevice == null) {
                    throw new IllegalStateException(
                            "device logits backward requires deviceDecoderBackward, gpuDecoderLayer and "
                                    + "BlockActivationCacheDevice from forward(training=true); check TrainingConfig / LLMConfig.");
                }
                backwardDecoderLayersDevice(dGradBeforeNorm, batch, seqLen, zeroParamGrads);
            } finally {
                closeGpuBuffer(tmpNormedHidden);
                closeGpuBuffer(tmpXBeforeNorm);
                clearSampledTrainLossGrad();
            }
            return;
        }

        GpuFloatBuffer dLogitsGrad = lastLogitsGradGpu;
        GpuFloatBuffer dNormedHidden = lastHiddenGpu;
        GpuFloatBuffer dXBeforeNorm = xBeforeFinalNormGpu;
        GpuFloatBuffer tmpLogitsGrad = null;
        GpuFloatBuffer tmpNormedHidden = null;
        GpuFloatBuffer tmpXBeforeNorm = null;
        try {
            if (dLogitsGrad == null || dLogitsGrad.isClosed()) {
                tmpLogitsGrad = GpuFloatBuffer.allocate(logitsFlat);
                tmpLogitsGrad.copyFrom(logits.gradBuffer(), 0, logitsFlat);
                dLogitsGrad = tmpLogitsGrad;
            }
            if (dNormedHidden == null || dNormedHidden.isClosed()) {
                tmpNormedHidden = GpuFloatBuffer.allocate(flat);
                tmpNormedHidden.copyFrom(lastHidden.internalBuffer(), 0, flat);
                dNormedHidden = tmpNormedHidden;
            }
            if (dXBeforeNorm == null || dXBeforeNorm.isClosed()) {
                tmpXBeforeNorm = GpuFloatBuffer.allocate(flat);
                tmpXBeforeNorm.copyFrom(xBeforeFinalNorm.internalBuffer(), 0, flat);
                dXBeforeNorm = tmpXBeforeNorm;
            }

            try (GpuFloatBuffer dGradBeforeNorm =
                    backwardFromDeviceLogits(dLogitsGrad, dNormedHidden, dXBeforeNorm, rows, zeroParamGrads)) {
                if (!deviceDecoderBackward || gpuDecoderLayer == null || blockCachesDevice == null) {
                    throw new IllegalStateException(
                            "device logits backward requires deviceDecoderBackward, gpuDecoderLayer and "
                                    + "BlockActivationCacheDevice from forward(training=true); check TrainingConfig / LLMConfig.");
                }
                backwardDecoderLayersDevice(dGradBeforeNorm, batch, seqLen, zeroParamGrads);
            }
        } finally {
            closeGpuBuffer(tmpLogitsGrad);
            closeGpuBuffer(tmpNormedHidden);
            closeGpuBuffer(tmpXBeforeNorm);
        }
    }

    public GpuFloatBuffer backwardFromSampledDeviceLogits(
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateGrad,
            GpuFloatBuffer dNormedHidden,
            GpuFloatBuffer dXBeforeNorm,
            int rows,
            int candidateCount,
            boolean zeroGpuGrads) {
        if (!gpuResident || gpuResidentHead == null) {
            throw new IllegalStateException("backwardFromSampledDeviceLogits requires gpuResident=true");
        }
        if (candidateIds == null || candidateGrad == null || candidateCount <= 0) {
            throw new IllegalArgumentException("sampled logits backward requires candidate ids/grad");
        }
        GpuFloatBuffer lmHeadBuf = gpuResidentHead.lmHeadGpu().dataBuffer();
        GpuFloatBuffer gammaBuf = gpuResidentHead.layerNormGammaGpu().dataBuffer();

        GpuTensor lmHeadGpu = gpuResidentHead.lmHeadGpu();
        if (!lmHeadGpu.hasGradBuffer() || zeroGpuGrads) {
            lmHeadGpu.zeroGrad();
        }

        GpuTensor gammaGpu = gpuResidentHead.layerNormGammaGpu();
        if (!gammaGpu.hasGradBuffer() || zeroGpuGrads) {
            gammaGpu.zeroGrad();
        }

        // Workspace-backed: no cudaMalloc per step (dHidden, dGradBeforeNorm, dGGamma)
        return TransformerBackward.backwardSampledLmHeadDevice(
                candidateIds,
                candidateGrad,
                dNormedHidden,
                dXBeforeNorm,
                lmHeadBuf,
                lmHeadGpu.gradBuffer(),
                gammaBuf,
                gammaGpu.gradBuffer(),
                layerNormFinal.size(),
                rows,
                dModel,
                vocabSize,
                candidateCount);
    }

    private void backwardDecoderLayersHost(Tensor gradBeforeNorm, int batch, int seqLen, boolean zeroParamGrads) {
        Tensor gradX = gradBeforeNorm;
        for (int layer = numLayers - 1; layer >= 0; layer--) {
            DecoderBlock blk = blocks[layer];
            if (zeroParamGrads) {
                blk.zeroGradTensors();
            }
            Tensor gradXIn = gradBufferOppositePingPong(gradX);
            gradXIn.zeroGrad();
            TensorOps.GpuAttnResidentBuffers attnResidentBwd =
                    gpuDecoderLayer != null ? gpuDecoderLayer[layer].attnBuffers() : null;
            TensorOps.GpuFfnResidentBuffers ffnResidentBwd =
                    gpuDecoderLayer != null ? gpuDecoderLayer[layer].ffnBuffers() : null;
            TransformerBackward.transformerBlockBackward(
                    gradX,
                    blockCaches[layer],
                    blk.getWq(),
                    blk.getWk(),
                    blk.getWv(),
                    blk.getWo(),
                    blk.getW1(),
                    blk.getW2(),
                    blk.getW3(),
                    blk.getNorm1(),
                    blk.getNorm2(),
                    numHeads,
                    lastMask,
                    true,
                    gradXIn,
                    blk.getWq(),
                    blk.getWk(),
                    blk.getWv(),
                    blk.getWo(),
                    blk.getW1(),
                    blk.getW2(),
                    blk.getW3(),
                    blk.getNorm1(),
                    blk.getNorm2(),
                    attnResidentBwd,
                    ffnResidentBwd);
            gradX = gradXIn;
        }
        tokenEmbedding.backwardScatter(lastInputTokens, gradX);
        positionEmbedding.backwardAccumulate(gradX, seqLen);
    }

    /**
     * Decoder layers backward with gradient ping-pong on device.
     * Эмбеддинги: backward на device по градиенту без D2H {@code [B,S,d]}.
     */
    private void backwardDecoderLayersDevice(GpuFloatBuffer dGradBeforeNorm, int batch, int seqLen, boolean zeroParamGrads) {
        int flat = batch * seqLen * dModel;
        if (!deviceDecoderBackward || blockCachesDevice == null || !TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException(
                    "decoder backward on VRAM requires deviceDecoderBackward, CUDA, and device activation cache "
                            + "from forward(training=true); misconfiguration or missing GPU pipeline.");
        }
        decoderBwdGradPing = ensureGpuBuffer(decoderBwdGradPing, flat);
        decoderBwdGradPong = ensureGpuBuffer(decoderBwdGradPong, flat);

        GpuFloatBuffer gradCur = dGradBeforeNorm;
        GpuFloatBuffer gradNext = decoderBwdGradPing;

        for (int layer = numLayers - 1; layer >= 0; layer--) {
            DecoderBlock blk = blocks[layer];
            if (zeroParamGrads) {
                blk.zeroGradTensors();
            }
            TensorOps.GpuAttnResidentBuffers attnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[layer].attnBuffers() : null;
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    gpuDecoderLayer != null ? gpuDecoderLayer[layer].ffnBuffers() : null;
            TransformerBackward.transformerBlockBackwardGpuDevice(
                    gradCur,
                    blockCachesDevice[layer],
                    blk.getWq(),
                    blk.getWk(),
                    blk.getWv(),
                    blk.getWo(),
                    blk.getW1(),
                    blk.getW2(),
                    blk.getW3(),
                    blk.getNorm1(),
                    blk.getNorm2(),
                    numHeads,
                    lastMask,
                    true,
                    gradNext,
                    blk.getWq(),
                    blk.getWk(),
                    blk.getWv(),
                    blk.getWo(),
                    blk.getW1(),
                    blk.getW2(),
                    blk.getW3(),
                    blk.getNorm1(),
                    blk.getNorm2(),
                    attnResident,
                    ffnResident);

            if (DebugGpuTrain.perLayerFiniteCheck()) {
                if (TensorOpsGPU.anyNonFiniteGpuDevice(gradNext, flat)) {
                    log.warn(
                            "Non-finite gradient after decoder layer {} ({} elements). "
                                    + "Enable JGPT_DEBUG_GPU_TRAIN=1 for JSONL diagnostics; "
                                    + "JGPT_BWD_LAYER_FINITE_CHECK=0 and no debug = one check after stack only.",
                            layer,
                            flat);
                    agentLogB39372LayerGrad(layer, flat);
                }
            }

            GpuFloatBuffer tmp = gradCur;
            gradCur = gradNext;
            gradNext = tmp;
        }

        if (!DebugGpuTrain.perLayerFiniteCheck()) {
            if (TensorOpsGPU.anyNonFiniteGpuDevice(gradCur, flat)) {
                log.warn(
                        "Non-finite gradient after decoder stack ({} elements). "
                                + "Enable JGPT_DEBUG_GPU_TRAIN=1 for JSONL or JGPT_BWD_LAYER_FINITE_CHECK=1 "
                                + "to localize by layer.",
                        flat);
                agentLogB39372LayerGrad(-1, flat);
            }
        }

        tokenEmbedding.backwardScatterFromDeviceGrad(lastInputTokens, gradCur, batch, seqLen);
        positionEmbedding.backwardAccumulateFromDeviceGrad(gradCur, batch, seqLen);
    }

    private static Tensor ensureReusableLike(Tensor reusable, Tensor like) {
        int[] shape = like.getShape();
        if (reusable != null && Arrays.equals(reusable.getShape(), shape)) {
            return reusable;
        }
        return new Tensor(shape);
    }

    /**
     * Буфер для ∂L/∂x на входе блока: всегда отличен от {@code gradX}, чтобы не обнулять вход при
     * {@code gradXIn.zeroGrad()} (иначе при чётном числе слоёв возможен gradX == gradXIn).
     */
    private Tensor gradBufferOppositePingPong(Tensor gradX) {
        if (gradX == backwardGradPing) {
            backwardGradPong = ensureReusableLike(backwardGradPong, gradX);
            return backwardGradPong;
        }
        if (gradX == backwardGradPong) {
            backwardGradPing = ensureReusableLike(backwardGradPing, gradX);
            return backwardGradPing;
        }
        backwardGradPing = ensureReusableLike(backwardGradPing, gradX);
        return backwardGradPing;
    }

    /** Обнуляет буферы градиентов у всех параметров (перед backward). */
    public void zeroGradParameters() {
        for (Tensor p : getParameters()) {
            p.zeroGrad();
        }
    }

    /**
     * Авторегрессивная генерация (batch=1). Возвращает буфер длины {@code seq_len + maxNewTokens};
     * неиспользуемый хвост остаётся 0 (при раннем EOS).
     * <p>
     * При {@code temperature <= 0} — жадный выбор (argmax по logit; при равенстве — меньший индекс), без сэмплирования.
     * <p>
     * <b>Скользящее окно (когда длина контекста превышает {@link #getMaxSeqLen()}):</b> KV-cache сбрасывается и
     * выполняется полный prefill по срезу подсказки с подходящим RoPE-сдвигом. Это корректно, но даёт дополнительную
     * работу порядка квадрата окна на каждое срабатывание; для очень длинных генераций в production обычно используют
     * rolling KV / paged attention вместо полного пересчёта.
     */
    public Tensor generate(Tensor inputTokens, int maxNewTokens, float temperature, int topK) {
        int[] inputShape = inputTokens.getShape();
        int batch = inputShape[0];
        int seqLen = inputShape[1];

        if (batch != 1) {
            throw new IllegalArgumentException("generate currently supports batch_size=1 only");
        }

        Tensor output = new Tensor(new int[]{1, seqLen + maxNewTokens});
        float[] outData = output.internalBuffer();
        float[] inData = inputTokens.internalBuffer();
        System.arraycopy(inData, 0, outData, 0, seqLen);

        int dHead = dModel / numHeads;
        KvCache cache = new KvCache(numLayers, numHeads, dHead, maxSeqLen);

        Tensor logitsPrefill = forwardPrefill(inputTokens, cache, 0);
        Tensor lastPlane = sliceBatch3D(logitsPrefill, 0);
        float[] lastLogitData = lastPlane.internalBuffer();
        int lastRowOffset = (seqLen - 1) * vocabSize;

        int nextToken =
                sampleNextToken(lastLogitData, lastRowOffset, vocabSize, temperature, topK);
        outData[seqLen] = nextToken;
        if (nextToken == 0) {
            return output;
        }

        for (int j = 1; j < maxNewTokens; j++) {
            int currentLen = seqLen + j;

            if (currentLen > maxSeqLen) {
                int startIdx = currentLen - maxSeqLen;
                // forwardRange(ropeOffset, seqLen) требует startIdx + seqLen <= число строк PE (= maxSeqLen)
                int sliceLen = maxSeqLen - startIdx;
                if (sliceLen <= 0) {
                    throw new IllegalStateException(
                            "sliding window: startIdx="
                                    + startIdx
                                    + " maxSeqLen="
                                    + maxSeqLen
                                    + " (увеличьте max_seq_len модели или уменьшите длину генерации)");
                }
                cache.clear();
                log.warn(
                        "Скользящее окно KV (кэш на хосте): полный prefill по {} токенам (позиции {}..{}). "
                                + "Каждое срабатывание — O(окно²); для длинных прогонов увеличьте max_seq_len или используйте paged/rolling KV.",
                        sliceLen,
                        startIdx,
                        startIdx + sliceLen - 1);
                if (reusableSlidingPrefillInput == null
                        || reusableSlidingPrefillInput.getShape()[0] != 1
                        || reusableSlidingPrefillInput.getShape()[1] != sliceLen) {
                    reusableSlidingPrefillInput = new Tensor(new int[] {1, sliceLen});
                }
                float[] sliceData = reusableSlidingPrefillInput.internalBuffer();
                for (int t = 0; t < sliceLen; t++) {
                    sliceData[t] = outData[startIdx + t];
                }
                logitsPrefill = forwardPrefill(reusableSlidingPrefillInput, cache, startIdx);
                lastPlane = sliceBatch3D(logitsPrefill, 0);
                lastLogitData = lastPlane.internalBuffer();
                lastRowOffset = (sliceLen - 1) * vocabSize;
                nextToken =
                        sampleNextToken(lastLogitData, lastRowOffset, vocabSize, temperature, topK);
                outData[currentLen] = nextToken;
                if (nextToken == 0) {
                    break;
                }
                continue;
            }

            if (reusableDecodeOneToken == null) {
                reusableDecodeOneToken = new Tensor(new int[]{1, 1});
            }
            reusableDecodeOneToken.internalBuffer()[0] = outData[seqLen + j - 1];
            Tensor logitsDec = forwardDecode(reusableDecodeOneToken, cache, cache.length(), seqLen + j - 1);
            lastPlane = sliceBatch3D(logitsDec, 0);
            lastLogitData = lastPlane.internalBuffer();
            nextToken = sampleNextToken(lastLogitData, 0, vocabSize, temperature, topK);
            outData[currentLen] = nextToken;
            if (nextToken == 0) {
                break;
            }
        }

        return output;
    }

    /**
     * Авторегрессивная генерация (batch=1), как {@link #generate(Tensor, int, float, int)}, но KV хранится в
     * {@link KvCacheGpu} (путь attention без host-тензоров K/V). Тот же контракт по {@code temperature} / {@code
     * topK}, что и {@link #generate}.
     * <p>
     * При превышении {@link #getMaxSeqLen()} действует тот же механизм скользящего окна, что и в {@link #generate}, с
     * полным reprefill среза (см. javadoc {@link #generate}).
     * <p>
     * Требует {@link #isGpuResident()} и доступной CUDA ({@link TensorOpsGPU#isGpuAvailable()}).
     */
    public Tensor generateGpuKv(Tensor inputTokens, int maxNewTokens, float temperature, int topK) {
        if (!isGpuResident()) {
            throw new IllegalStateException("generateGpuKv requires GPU-resident weights");
        }
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("generateGpuKv requires CUDA");
        }
        int[] inputShape = inputTokens.getShape();
        int batch = inputShape[0];
        int seqLen = inputShape[1];

        if (batch != 1) {
            throw new IllegalArgumentException("generateGpuKv currently supports batch_size=1 only");
        }

        Tensor output = new Tensor(new int[] {1, seqLen + maxNewTokens});
        float[] outData = output.internalBuffer();
        float[] inData = inputTokens.internalBuffer();
        System.arraycopy(inData, 0, outData, 0, seqLen);

        int dHead = dModel / numHeads;
        try (KvCacheGpu cache = new KvCacheGpu(numLayers, numHeads, dHead, maxSeqLen)) {
            try {
            Tensor logitsPrefill = forwardPrefill(inputTokens, cache, 0);
            Tensor lastPlane = sliceBatch3D(logitsPrefill, 0);
            float[] lastLogitData = lastPlane.internalBuffer();
            int lastRowOffset = (seqLen - 1) * vocabSize;

            int nextToken =
                    sampleNextToken(lastLogitData, lastRowOffset, vocabSize, temperature, topK);
            outData[seqLen] = nextToken;
            if (nextToken == 0) {
                return output;
            }

            for (int j = 1; j < maxNewTokens; j++) {
                int currentLen = seqLen + j;

                if (currentLen > maxSeqLen) {
                    int startIdx = currentLen - maxSeqLen;
                    int sliceLen = maxSeqLen - startIdx;
                    if (sliceLen <= 0) {
                        throw new IllegalStateException(
                                "sliding window: startIdx="
                                        + startIdx
                                        + " maxSeqLen="
                                        + maxSeqLen
                                        + " (увеличьте max_seq_len модели или уменьшите длину генерации)");
                    }
                    cache.clear();
                    log.warn(
                            "Скользящее окно KV (кэш в VRAM): полный prefill по {} токенам (позиции {}..{}). "
                                    + "Каждое срабатывание — O(окно²); для длинных прогонов увеличьте max_seq_len или используйте paged/rolling KV.",
                            sliceLen,
                            startIdx,
                            startIdx + sliceLen - 1);
                    if (reusableSlidingPrefillInput == null
                            || reusableSlidingPrefillInput.getShape()[0] != 1
                            || reusableSlidingPrefillInput.getShape()[1] != sliceLen) {
                        reusableSlidingPrefillInput = new Tensor(new int[] {1, sliceLen});
                    }
                    float[] sliceData = reusableSlidingPrefillInput.internalBuffer();
                    for (int t = 0; t < sliceLen; t++) {
                        sliceData[t] = outData[startIdx + t];
                    }
                    logitsPrefill = forwardPrefill(reusableSlidingPrefillInput, cache, startIdx);
                    lastPlane = sliceBatch3D(logitsPrefill, 0);
                    lastLogitData = lastPlane.internalBuffer();
                    lastRowOffset = (sliceLen - 1) * vocabSize;
                    nextToken =
                            sampleNextToken(lastLogitData, lastRowOffset, vocabSize, temperature, topK);
                    outData[currentLen] = nextToken;
                    if (nextToken == 0) {
                        break;
                    }
                    continue;
                }

                if (reusableDecodeOneToken == null) {
                    reusableDecodeOneToken = new Tensor(new int[] {1, 1});
                }
                reusableDecodeOneToken.internalBuffer()[0] = outData[seqLen + j - 1];
                Tensor logitsDec =
                        forwardDecode(reusableDecodeOneToken, cache, cache.length(), seqLen + j - 1);
                lastPlane = sliceBatch3D(logitsDec, 0);
                lastLogitData = lastPlane.internalBuffer();
                nextToken = sampleNextToken(lastLogitData, 0, vocabSize, temperature, topK);
                outData[currentLen] = nextToken;
                if (nextToken == 0) {
                    break;
                }
            }
            } finally {
                TensorOpsGPU.synchronizeStream();
            }
        }

        return output;
    }

    private int sampleNextToken(
            float[] logits, int offset, int vocabSize, float temperature, int topK) {
        if (sampleLogitsScratch == null || sampleLogitsScratch.length < vocabSize) {
            sampleLogitsScratch = new float[vocabSize];
        }
        System.arraycopy(logits, offset, sampleLogitsScratch, 0, vocabSize);

        if (temperature != 1.0f && temperature > 0) {
            for (int i = 0; i < vocabSize; i++) {
                sampleLogitsScratch[i] /= temperature;
            }
        }

        if (topK > 0 && topK < vocabSize) {
            // O(n log k): min-heap из k «худших» среди текущего топа; при равных logit хуже больший индекс
            // (как при стабильной сортировке по убыванию logit на Integer[] 0..v-1).
            PriorityQueue<Integer> worstOfTop =
                    new PriorityQueue<>(
                            topK,
                            (a, b) -> {
                                int c = Float.compare(sampleLogitsScratch[a], sampleLogitsScratch[b]);
                                if (c != 0) {
                                    return c;
                                }
                                return Integer.compare(b, a);
                            });
            for (int i = 0; i < vocabSize; i++) {
                if (worstOfTop.size() < topK) {
                    worstOfTop.offer(i);
                } else {
                    int w = worstOfTop.peek();
                    if (isBetterLogit(sampleLogitsScratch, i, w)) {
                        worstOfTop.poll();
                        worstOfTop.offer(i);
                    }
                }
            }
            if (sampleTopKMember == null || sampleTopKMember.length < vocabSize) {
                sampleTopKMember = new boolean[vocabSize];
            }
            Arrays.fill(sampleTopKMember, 0, vocabSize, false);
            while (!worstOfTop.isEmpty()) {
                sampleTopKMember[worstOfTop.poll()] = true;
            }
            for (int i = 0; i < vocabSize; i++) {
                if (!sampleTopKMember[i]) {
                    sampleLogitsScratch[i] = Float.NEGATIVE_INFINITY;
                }
            }
        }

        if (temperature <= 0f) {
            return argmaxLogitsGreedy(sampleLogitsScratch, vocabSize);
        }

        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocabSize; i++) {
            max = Math.max(max, sampleLogitsScratch[i]);
        }

        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            float e = (float) Math.exp(sampleLogitsScratch[i] - max);
            sampleLogitsScratch[i] = e;
            sum += e;
        }
        for (int i = 0; i < vocabSize; i++) {
            sampleLogitsScratch[i] /= sum;
        }

        float rand = (float) ThreadLocalRandom.current().nextDouble();
        float cumsum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            cumsum += sampleLogitsScratch[i];
            if (rand <= cumsum) {
                return i;
            }
        }

        return vocabSize - 1;
    }

    /** Argmax по logit; при равенстве — минимальный индекс (детерминированно). */
    private static int argmaxLogitsGreedy(float[] logits, int vocabSize) {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < vocabSize; i++) {
            float v = logits[i];
            if (v > bestVal || (v == bestVal && i < best)) {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }

    /** Сравнение как у top-k после сортировки по убыванию logit; при равенстве предпочтительнее меньший индекс. */
    private static boolean isBetterLogit(float[] vals, int i, int j) {
        int c = Float.compare(vals[i], vals[j]);
        if (c != 0) {
            return c > 0;
        }
        return i < j;
    }

    public long countParameters() {
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

    /** Все обучаемые веса (для оптимизатора). */
    public List<Tensor> getParameters() {
        List<Tensor> params = new ArrayList<>();
        tokenEmbedding.collectParameters(params);
        positionEmbedding.collectParameters(params);
        for (DecoderBlock b : blocks) {
            b.collectParameters(params);
        }
        params.add(layerNormFinal);
        params.add(lmHead);
        return Collections.unmodifiableList(params);
    }

    /**
     * Загружает веса из файла, сохранённого {@link LLMTrainer#saveModelWeights(String)} (тот же порядок
     * тензоров, что у {@link #getParameters()}).
     */
    public void loadWeights(String path) throws IOException, ClassNotFoundException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(path), 1 << 20)) {
            bis.mark(8);
            int b0 = bis.read();
            int b1 = bis.read();
            bis.reset();
            if (b0 == 0xac && b1 == 0xed) {
                try (ObjectInputStream in = new ObjectInputStream(bis)) {
                    loadWeightsFromJavaSerialization(in);
                }
            } else {
                DataInputStream dis = new DataInputStream(bis);
                String tag = dis.readUTF();
                if (!MODEL_WEIGHTS_FORMAT_V1.equals(tag)) {
                    throw new IOException("unknown model weights tag: " + tag);
                }
                loadWeightsFromBinaryV1(dis);
            }
        }
        syncGpuResidentWeightsFromHost();
        log.info("Веса модели загружены из файла: {}", path);
    }

    private void loadWeightsFromJavaSerialization(ObjectInputStream in) throws IOException, ClassNotFoundException {
        int n = in.readInt();
        List<Tensor> params = getParameters();
        if (n != params.size()) {
            throw new IOException(
                    "saved parameter count " + n + " != model parameter count " + params.size());
        }
        for (int i = 0; i < n; i++) {
            int[] shape = (int[]) in.readObject();
            float[] data = (float[]) in.readObject();
            copyLoadedParamSlice(i, params, shape, data);
        }
    }

    private void loadWeightsFromBinaryV1(DataInputStream in) throws IOException {
        int n = in.readInt();
        List<Tensor> params = getParameters();
        if (n != params.size()) {
            throw new IOException(
                    "saved parameter count " + n + " != model parameter count " + params.size());
        }
        for (int i = 0; i < n; i++) {
            int rank = in.readInt();
            int[] shape = new int[rank];
            for (int r = 0; r < rank; r++) {
                shape[r] = in.readInt();
            }
            int expected = 1;
            for (int d : shape) {
                expected *= d;
            }
            float[] data = new float[expected];
            if (expected > 0) {
                byte[] raw = new byte[expected * 4];
                in.readFully(raw);
                ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN).asFloatBuffer().get(data);
            }
            copyLoadedParamSlice(i, params, shape, data);
        }
    }

    private static void copyLoadedParamSlice(int i, List<Tensor> params, int[] shape, float[] data)
            throws IOException {
        Tensor p = params.get(i);
        if (!Arrays.equals(shape, p.getShape())) {
            throw new IOException(
                    "shape mismatch at param "
                            + i
                            + ": saved "
                            + Arrays.toString(shape)
                            + " vs model "
                            + Arrays.toString(p.getShape()));
        }
        float[] dst = p.internalBuffer();
        if (data.length != dst.length) {
            throw new IOException("length mismatch at param " + i);
        }
        System.arraycopy(data, 0, dst, 0, dst.length);
    }

    public Tensor getLastHidden() {
        return lastHidden;
    }

    public Tensor getLmHead() {
        return lmHead;
    }

    private static final class TokenEmbedding {
        private final Tensor weights;
        /** Копия {@link #weights} на VRAM при GPU-резидентной модели; gather без H2D весов на каждый forward. */
        private final GpuTensor weightsGpu;

        TokenEmbedding(int vocabSize, int dModel, float scale, boolean gpuResidentEmbedding) {
            this.weights = TensorOps.randomTensor(new int[]{vocabSize, dModel}, scale);
            if (gpuResidentEmbedding) {
                this.weightsGpu = GpuTensor.fromHostTensor(weights);
            } else {
                this.weightsGpu = null;
            }
        }

        void syncGpuWeightsFromHost() {
            if (weightsGpu != null) {
                weightsGpu.uploadFrom(weights.internalBuffer(), 0, weights.size());
            }
        }

        void closeGpuWeights() {
            if (weightsGpu != null) {
                weightsGpu.close();
            }
        }

        boolean hasWeightsGpu() {
            return weightsGpu != null;
        }

        /**
         * Gather в {@code out} на device; веса — {@link #weightsGpu}. Форма {@code out} — {@code [batch, seq, dModel]}.
         */
        void forwardGatherToGpuTensor(Tensor tokens, GpuTensor out) {
            if (weightsGpu == null) {
                throw new IllegalStateException("weightsGpu is null");
            }
            int[] shape = tokens.getShape();
            int batch = shape[0];
            int seqLen = shape[1];
            int dModel = weights.getShape()[1];
            int vocabSize = weights.getShape()[0];
            int[] os = out.getShape();
            if (os.length != 3 || os[0] != batch || os[1] != seqLen || os[2] != dModel) {
                throw new IllegalArgumentException(
                        "out must be [batch, seq, d_model], got " + Arrays.toString(os));
            }
            int nTok = batch * seqLen;
            for (int i = 0; i < nTok; i++) {
                int tokenIdx = (int) tokens.getLinear(i);
                if (tokenIdx < 0 || tokenIdx >= vocabSize) {
                    throw new IllegalArgumentException("token index out of range: " + tokenIdx);
                }
            }
            java.nio.ByteBuffer tokenByteBuf = tokens.directByteBuffer();
            if (tokenByteBuf != null && tokenByteBuf.isDirect()) {
                TensorOpsGPU.embeddingTokenForwardGpuDirectDeviceWeightsToDevice(
                        tokenByteBuf, weightsGpu.devicePointer(), out.dataBuffer(), batch, seqLen, dModel, vocabSize);
            } else {
                float[] tokenData = tokens.internalBuffer();
                TensorOpsGPU.embeddingTokenForwardGpuDeviceWeightsToDevice(
                        tokenData, weightsGpu.devicePointer(), out.dataBuffer(), batch, seqLen, dModel, vocabSize);
            }
        }

        void collectParameters(List<Tensor> out) {
            out.add(weights);
        }

        /** Градиент по таблице эмбеддингов: scatter по индексам токенов. */
        void backwardScatter(Tensor tokens, Tensor gradOut) {
            int[] shape = tokens.getShape();
            int batch = shape[0];
            int seqLen = shape[1];
            int dm = weights.getShape()[1];
            int vocab = weights.getShape()[0];
            int wSize = vocab * dm;
            if (!weights.hasGrad()) {
                weights.zeroGrad();
            }
            float[] gw = weights.gradBuffer();
            float[] tok = tokens.internalBuffer();
            float[] g = gradOut.gradBuffer();
            int n = batch * seqLen * dm;
            if (n <= 0) {
                return;
            }
            if (weightsGpu != null) {
                TensorOpsGPU.requireCuda("TokenEmbedding.backwardScatter");
                if (!weightsGpu.hasGradBuffer()) {
                    weightsGpu.zeroGrad();
                }
                weightsGpu.gradBuffer().copyFrom(gw, 0, wSize);
                TensorOpsGPU.embeddingTokenBackwardGPUDeviceGradWeights(
                        tok, g, batch, seqLen, dm, vocab, weightsGpu.gradDevicePointer());
                weightsGpu.gradBuffer().copyTo(gw, 0, wSize);
                weightsGpu.zeroGrad();
                return;
            }
            TensorOpsGPU.requireCuda("TokenEmbedding.backwardScatter");
            TensorOpsGPU.embeddingTokenBackwardGPU(tok, g, gw, batch, seqLen, dm, vocab);
        }

        /**
         * Как {@link #backwardScatter}, но градиент на входе эмбеддинга уже на device (без D2H всего
         * {@code [B,S,d]}).
         */
        void backwardScatterFromDeviceGrad(Tensor tokens, GpuFloatBuffer gradDevice, int batch, int seqLen) {
            int dm = weights.getShape()[1];
            int vocab = weights.getShape()[0];
            if (!weights.hasGrad()) {
                weights.zeroGrad();
            }
            float[] tok = tokens.internalBuffer();
            if (weightsGpu == null) {
                throw new IllegalStateException(
                        "backwardScatterFromDeviceGrad requires GPU-resident token embedding tables");
            }
            TensorOpsGPU.requireCuda("TokenEmbedding.backwardScatterFromDeviceGrad");
            if (!weightsGpu.hasGradBuffer()) {
                weightsGpu.zeroGrad();
            }
            /* Накопление только в VRAM (atomicAdd в CUDA); хостовый буфер весов не обновляется. Между микробатчами ∂W
             * на GPU не обнуляем (полный GPU-шаг оптимизатора читает только grad на VRAM). */
            TensorOpsGPU.embeddingTokenBackwardGPUDeviceGradWeightsDeviceGrad(
                    tok,
                    gradDevice.devicePointer(),
                    batch,
                    seqLen,
                    dm,
                    vocab,
                    weightsGpu.gradDevicePointer());
        }

        Tensor forward(Tensor tokens) {
            int[] shape = tokens.getShape();
            int batch = shape[0];
            int seqLen = shape[1];
            int dModel = weights.getShape()[1];
            int vocabSize = weights.getShape()[0];

            Tensor output = new Tensor(new int[]{batch, seqLen, dModel});
            float[] outData = output.internalBuffer();
            float[] weightData = weights.internalBuffer();

            int nTok = batch * seqLen;
            for (int i = 0; i < nTok; i++) {
                int tokenIdx = (int) tokens.getLinear(i);
                if (tokenIdx < 0 || tokenIdx >= vocabSize) {
                    throw new IllegalArgumentException("token index out of range: " + tokenIdx);
                }
            }

            int n = batch * seqLen * dModel;
            if (n <= 0) {
                return output;
            }
            TensorOpsGPU.requireCuda("TokenEmbedding.forward");
            java.nio.ByteBuffer tokenByteBuf = tokens.directByteBuffer();
            if (weightsGpu != null) {
                long dW = weightsGpu.devicePointer();
                if (tokenByteBuf != null && tokenByteBuf.isDirect()) {
                    TensorOpsGPU.embeddingTokenForwardGpuDirectDeviceWeights(
                            tokenByteBuf, dW, outData, batch, seqLen, dModel, vocabSize);
                    return output;
                }
                float[] tokenData = tokens.internalBuffer();
                TensorOpsGPU.embeddingTokenForwardGpuDeviceWeights(
                        tokenData, dW, outData, batch, seqLen, dModel, vocabSize);
                return output;
            }
            if (tokenByteBuf != null && tokenByteBuf.isDirect()) {
                TensorOpsGPU.embeddingTokenForwardGpuDirect(
                        tokenByteBuf, weightData, outData, batch, seqLen, dModel, vocabSize);
                return output;
            }
            float[] tokenData = tokens.internalBuffer();
            TensorOpsGPU.embeddingTokenForwardGpu(tokenData, weightData, outData, batch, seqLen, dModel, vocabSize);
            return output;
        }
    }

    private static final class PositionEmbedding {
        private final Tensor weights;
        /** Копия {@link #weights} на VRAM при GPU-резидентной модели; backward scatter ∂ без alloc всей таблицы. */
        private final GpuTensor weightsGpu;

        PositionEmbedding(int maxSeqLen, int dModel, float scale, boolean gpuResidentEmbedding) {
            this.weights = TensorOps.randomTensor(new int[]{maxSeqLen, dModel}, scale);
            if (gpuResidentEmbedding) {
                this.weightsGpu = GpuTensor.fromHostTensor(weights);
            } else {
                this.weightsGpu = null;
            }
        }

        void syncGpuWeightsFromHost() {
            if (weightsGpu != null) {
                weightsGpu.uploadFrom(weights.internalBuffer(), 0, weights.size());
            }
        }

        void closeGpuWeights() {
            if (weightsGpu != null) {
                weightsGpu.close();
            }
        }

        boolean hasWeightsGpu() {
            return weightsGpu != null;
        }

        GpuFloatBuffer positionWeightsDataBuffer() {
            if (weightsGpu == null) {
                throw new IllegalStateException("weightsGpu is null");
            }
            return weightsGpu.dataBuffer();
        }

        /**
         * In-place: {@code x[b,s,:] += table[posRowStart+s,:]}. Таблица позиций — {@code [maxSeq, dModel]} на host или
         * VRAM.
         */
        void addToActivationsInPlace(Tensor x, int seqLen, int posRowStart) {
            int[] sh = x.getShape();
            if (sh.length != 3 || sh[1] != seqLen) {
                throw new IllegalArgumentException(
                        "x must be [batch, seqLen=" + seqLen + ", d_model], got " + Arrays.toString(sh));
            }
            int batch = sh[0];
            int dm = sh[2];
            int maxPos = weights.getShape()[0];
            int wDm = weights.getShape()[1];
            if (dm != wDm) {
                throw new IllegalArgumentException("d_model mismatch: x has " + dm + ", table has " + wDm);
            }
            if (posRowStart < 0 || posRowStart + seqLen > maxPos) {
                throw new IllegalArgumentException(
                        "position rows ["
                                + posRowStart
                                + ", "
                                + (posRowStart + seqLen)
                                + ") out of table rows [0,"
                                + maxPos
                                + ")");
            }
            int n = batch * seqLen * dm;
            if (n == 0) {
                return;
            }
            TensorOpsGPU.requireCuda("PositionEmbedding.addToActivationsInPlace");
            float[] xb = x.internalBuffer();
            if (weightsGpu != null) {
                TensorOpsGPU.addPositionEmbeddingGPUDeviceWeights(
                        xb, weightsGpu.devicePointer(), batch, seqLen, dm, posRowStart);
            } else {
                float[] slice = new float[seqLen * dm];
                float[] w = weights.internalBuffer();
                int rowStride = weights.stridesInternal()[0];
                for (int s = 0; s < seqLen; s++) {
                    System.arraycopy(w, (posRowStart + s) * rowStride, slice, s * dm, dm);
                }
                TensorOpsGPU.addPositionEmbeddingInPlaceHostSlice(xb, slice, batch, seqLen, dm);
            }
        }

        void collectParameters(List<Tensor> out) {
            out.add(weights);
        }

        /** Сумма градиентов по батчу для позиций 0..seqLen-1. */
        void backwardAccumulate(Tensor gradCombined, int seqLen) {
            int dm = weights.getShape()[1];
            int[] gs = gradCombined.getShape();
            int batch = gs[0];
            int wPlane = seqLen * dm;
            if (!weights.hasGrad()) {
                weights.zeroGrad();
            }
            float[] gw = weights.gradBuffer();
            float[] g = gradCombined.gradBuffer();
            int n = batch * seqLen * dm;
            if (n <= 0) {
                return;
            }
            if (weightsGpu != null) {
                TensorOpsGPU.requireCuda("PositionEmbedding.backwardAccumulate");
                if (!weightsGpu.hasGradBuffer()) {
                    weightsGpu.zeroGrad();
                }
                weightsGpu.gradBuffer().copyFrom(gw, 0, wPlane);
                TensorOpsGPU.embeddingPositionBackwardGPUDeviceGradWeights(
                        g, batch, seqLen, dm, weightsGpu.gradDevicePointer());
                weightsGpu.gradBuffer().copyTo(gw, 0, wPlane);
                weightsGpu.zeroGrad();
                return;
            }
            TensorOpsGPU.requireCuda("PositionEmbedding.backwardAccumulate");
            TensorOpsGPU.embeddingPositionBackwardGPU(g, gw, batch, seqLen, dm);
        }

        /** Как {@link #backwardAccumulate}, но градиент объединённого входа уже на device. */
        void backwardAccumulateFromDeviceGrad(GpuFloatBuffer gradDevice, int batch, int seqLen) {
            int dm = weights.getShape()[1];
            if (!weights.hasGrad()) {
                weights.zeroGrad();
            }
            if (weightsGpu == null) {
                throw new IllegalStateException(
                        "backwardAccumulateFromDeviceGrad requires GPU-resident position embedding");
            }
            TensorOpsGPU.requireCuda("PositionEmbedding.backwardAccumulateFromDeviceGrad");
            if (!weightsGpu.hasGradBuffer()) {
                weightsGpu.zeroGrad();
            }
            /* ∂ позиций 0..seqLen-1 накопление в VRAM; буфер размера maxSeq×d строки s≥seqLen не трогаются ядром. */
            TensorOpsGPU.embeddingPositionBackwardGPUDeviceGradWeightsDeviceGrad(
                    gradDevice.devicePointer(), batch, seqLen, dm, weightsGpu.gradDevicePointer());
        }

        Tensor forward(int seqLen) {
            int dModel = weights.getShape()[1];
            Tensor out = new Tensor(new int[]{1, seqLen, dModel});
            float[] w = weights.internalBuffer();
            float[] o = out.internalBuffer();
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(w, s * dModel, o, s * dModel, dModel);
            }
            return out;
        }

        /** Одна строка таблицы позиций: {@code [1, 1, d_model]}. */
        Tensor forwardOne(int position) {
            int maxPos = weights.getShape()[0];
            int dModel = weights.getShape()[1];
            if (position < 0 || position >= maxPos) {
                throw new IllegalArgumentException("position " + position + " out of range [0," + maxPos + ")");
            }
            Tensor out = new Tensor(new int[]{1, 1, dModel});
            float[] w = weights.internalBuffer();
            System.arraycopy(w, position * dModel, out.internalBuffer(), 0, dModel);
            return out;
        }

        /** Строки {@code [start, start+len)} → {@code [1, len, d_model]}. */
        Tensor forwardRange(int start, int len) {
            int maxPos = weights.getShape()[0];
            int dModel = weights.getShape()[1];
            if (start < 0 || len < 0 || start + len > maxPos) {
                throw new IllegalArgumentException(
                        "range [" + start + ", " + (start + len) + ") out of [0," + maxPos + ")");
            }
            Tensor out = new Tensor(new int[]{1, len, dModel});
            float[] w = weights.internalBuffer();
            float[] o = out.internalBuffer();
            for (int s = 0; s < len; s++) {
                System.arraycopy(w, (start + s) * dModel, o, s * dModel, dModel);
            }
            return out;
        }
    }

    /**
     * Декодер-блок (backward на CPU). При {@link GPTModel#gpuResident}: аттеншн и FFN без H2D весов на шаг
     * (см. {@link TensorOps#tryMultiHeadAttentionWithRoPEGpuResident} и fused FFN).
     */
    private static final class DecoderBlock {
        private final Tensor Wq;
        private final Tensor Wk;
        private final Tensor Wv;
        private final Tensor Wo;
        private final Tensor W1;
        private final Tensor W2;
        private final Tensor W3;
        private final Tensor norm1;
        private final Tensor norm2;
        private final int numHeads;

        DecoderBlock(int dModel, int numHeads, int dIntermediate, float projScale, float ffnScale) {
            this.numHeads = numHeads;
            this.Wq = TensorOps.randomTensor(new int[]{dModel, dModel}, projScale);
            this.Wk = TensorOps.randomTensor(new int[]{dModel, dModel}, projScale);
            this.Wv = TensorOps.randomTensor(new int[]{dModel, dModel}, projScale);
            this.Wo = TensorOps.randomTensor(new int[]{dModel, dModel}, projScale);
            this.W1 = TensorOps.randomTensor(new int[]{dModel, dIntermediate}, ffnScale);
            this.W2 = TensorOps.randomTensor(new int[]{dIntermediate, dModel}, ffnScale);
            this.W3 = TensorOps.randomTensor(new int[]{dModel, dIntermediate}, ffnScale);
            this.norm1 = TensorOps.onesTensor(new int[]{dModel});
            this.norm2 = TensorOps.onesTensor(new int[]{dModel});
        }

        void collectParameters(List<Tensor> out) {
            out.add(Wq);
            out.add(Wk);
            out.add(Wv);
            out.add(Wo);
            out.add(W1);
            out.add(W2);
            out.add(W3);
            out.add(norm1);
            out.add(norm2);
        }

        void zeroGradTensors() {
            Wq.zeroGrad();
            Wk.zeroGrad();
            Wv.zeroGrad();
            Wo.zeroGrad();
            W1.zeroGrad();
            W2.zeroGrad();
            W3.zeroGrad();
            norm1.zeroGrad();
            norm2.zeroGrad();
        }

        Tensor getWq() {
            return Wq;
        }

        Tensor getWk() {
            return Wk;
        }

        Tensor getWv() {
            return Wv;
        }

        Tensor getWo() {
            return Wo;
        }

        Tensor getW1() {
            return W1;
        }

        Tensor getW2() {
            return W2;
        }

        Tensor getW3() {
            return W3;
        }

        Tensor getNorm1() {
            return norm1;
        }

        Tensor getNorm2() {
            return norm2;
        }

        Tensor forward(Tensor x, Tensor mask, boolean useRoPE) {
            return forward(x, mask, useRoPE, null, null, null, null);
        }

        Tensor forward(
                Tensor x,
                Tensor mask,
                boolean useRoPE,
                BlockActivationCache cache) {
            return forward(x, mask, useRoPE, cache, null, null, null);
        }

        Tensor forward(
                Tensor x,
                Tensor mask,
                boolean useRoPE,
                BlockActivationCache cache,
                TensorOps.GpuFfnResidentBuffers ffnResident) {
            return forward(x, mask, useRoPE, cache, null, ffnResident, null);
        }

        Tensor forward(
                Tensor x,
                Tensor mask,
                boolean useRoPE,
                BlockActivationCache cache,
                TensorOps.GpuFfnResidentBuffers ffnResident,
                TensorOps.GpuAttnResidentBuffers attnResident) {
            return forward(x, mask, useRoPE, cache, null, ffnResident, attnResident);
        }

        Tensor forward(
                Tensor x,
                Tensor mask,
                boolean useRoPE,
                BlockActivationCache hostCache,
                BlockActivationCacheDevice deviceCache,
                TensorOps.GpuFfnResidentBuffers ffnResident,
                TensorOps.GpuAttnResidentBuffers attnResident) {
            if (hostCache != null && deviceCache != null) {
                throw new IllegalArgumentException("host BlockActivationCache and BlockActivationCacheDevice are mutually exclusive");
            }
            final float eps = TensorOpsGPU.rmsNormEps();
            TensorOps.AttnGpuResidentResult ar = null;
            if (attnResident != null) {
                ar = TensorOps.tryMultiHeadAttentionWithRoPEGpuResident(
                        x, eps, attnResident, numHeads, mask, useRoPE, hostCache, deviceCache);
            }
            Tensor xNorm1;
            Tensor attnOut;
            if (ar != null) {
                attnOut = ar.out();
                xNorm1 = ar.xNorm1();
            } else {
                xNorm1 = TensorOps.rmsNorm(x, norm1, eps);
                attnOut =
                        useRoPE
                                ? TensorOps.multiHeadAttentionWithRoPE(
                                        xNorm1, Wq, Wk, Wv, Wo, numHeads, mask, true, hostCache)
                                : TensorOps.multiHeadAttention(xNorm1, Wq, Wk, Wv, Wo, numHeads, mask);
            }
            Tensor xRes1 = TensorOps.add(x, attnOut);
            Tensor xNorm2;
            Tensor ffnOut;
            Tensor out;
            TensorOps.FfnForwardResult fusedFfn =
                    ffnResident != null
                            ? TensorOps.tryFusedNormResidualSwiGLUForwardGpuResident(
                                    xRes1, ffnResident, hostCache, deviceCache)
                            : null;
            if (fusedFfn == null) {
                fusedFfn = TensorOps.tryFusedNormResidualSwiGLUForwardGpu(xRes1, norm2, W1, W2, W3, hostCache);
            }
            if (fusedFfn != null) {
                xNorm2 = fusedFfn.xNorm2;
                ffnOut = fusedFfn.ffnOut;
                out = fusedFfn.out;
            } else {
                xNorm2 = TensorOps.rmsNorm(xRes1, norm2, eps);
                ffnOut = TensorOps.feedForwardSwiGLU(xNorm2, W1, W2, W3, hostCache);
                out = TensorOps.add(xRes1, ffnOut);
            }
            int[] xShape = x.getShape();
            int rows = xShape[0] * xShape[1];
            int plane = rows * xShape[2];
            if (hostCache != null) {
                boolean fp16Slot = hostCache.useFp16ActivationStorage;
                boolean fp16Fused = hostCache.fp16ForFusedGpuBackwardConsumptionSlots();
                hostCache.xIn.store(x, fp16Slot);
                hostCache.xNorm1.store(xNorm1, fp16Fused);
                hostCache.attnOut.store(attnOut, fp16Slot);
                hostCache.xRes1.store(xRes1, fp16Fused);
                if (fusedFfn == null) {
                    hostCache.xNorm2.store(xNorm2, fp16Fused);
                    hostCache.ffnOut.store(ffnOut, fp16Fused);
                }
                hostCache.xOut.store(out, fp16Slot);
            }
            if (deviceCache != null) {
                deviceCache.copySlotFromHostFloat(BlockActivationCacheDevice.SlotId.X_IN, x.internalBuffer(), 0, plane);
                deviceCache.copySlotFromHostFloat(
                        BlockActivationCacheDevice.SlotId.X_RES1, xRes1.internalBuffer(), 0, plane);
                deviceCache.copySlotFromHostFloat(BlockActivationCacheDevice.SlotId.X_OUT, out.internalBuffer(), 0, plane);
            }
            return out;
        }

        /**
         * Явный вход «GPU-пайплайн блока» (resident attention/FFN при ненулевых буферах); эквивалентно
         * {@link #forward(Tensor, Tensor, boolean, BlockActivationCache, BlockActivationCacheDevice, TensorOps.GpuFfnResidentBuffers, TensorOps.GpuAttnResidentBuffers)}
         * с RoPE.
         */
        Tensor forwardGpuPipeline(
                Tensor x,
                Tensor mask,
                BlockActivationCache hostCache,
                BlockActivationCacheDevice deviceCache,
                TensorOps.GpuFfnResidentBuffers ffnResident,
                TensorOps.GpuAttnResidentBuffers attnResident) {
            return forward(x, mask, true, hostCache, deviceCache, ffnResident, attnResident);
        }

        Tensor forwardKvPrefill(
                Tensor x,
                Tensor mask,
                BlockActivationCache cache,
                Tensor kCacheLayer,
                Tensor vCacheLayer,
                int ropeOffset) {
            return forwardKvPrefill(x, mask, cache, kCacheLayer, vCacheLayer, ropeOffset, null, null);
        }

        Tensor forwardKvPrefill(
                Tensor x,
                Tensor mask,
                BlockActivationCache cache,
                Tensor kCacheLayer,
                Tensor vCacheLayer,
                int ropeOffset,
                TensorOps.GpuFfnResidentBuffers ffnResident) {
            return forwardKvPrefill(
                    x, mask, cache, kCacheLayer, vCacheLayer, ropeOffset, ffnResident, null);
        }

        Tensor forwardKvPrefill(
                Tensor x,
                Tensor mask,
                BlockActivationCache cache,
                Tensor kCacheLayer,
                Tensor vCacheLayer,
                int ropeOffset,
                TensorOps.GpuFfnResidentBuffers ffnResident,
                TensorOps.GpuAttnResidentBuffers attnResident) {
            final float eps = TensorOpsGPU.rmsNormEps();
            Tensor attnOut = null;
            if (attnResident != null) {
                attnOut =
                        TensorOps.tryMultiHeadAttentionWithRoPEPrefillGpuResident(
                                x,
                                eps,
                                attnResident,
                                numHeads,
                                mask,
                                kCacheLayer,
                                vCacheLayer,
                                ropeOffset);
            }
            if (attnOut == null) {
                Tensor xNorm1 = TensorOps.rmsNorm(x, norm1, eps);
                attnOut =
                        TensorOps.multiHeadAttentionWithRoPEPrefill(
                                xNorm1,
                                Wq,
                                Wk,
                                Wv,
                                Wo,
                                numHeads,
                                mask,
                                kCacheLayer,
                                vCacheLayer,
                                ropeOffset);
            }
            Tensor xRes1 = TensorOps.add(x, attnOut);
            TensorOps.FfnForwardResult fusedFfn =
                    ffnResident != null
                            ? TensorOps.tryFusedNormResidualSwiGLUForwardGpuResident(xRes1, ffnResident, cache)
                            : null;
            if (fusedFfn == null) {
                fusedFfn = TensorOps.tryFusedNormResidualSwiGLUForwardGpu(xRes1, norm2, W1, W2, W3, cache);
            }
            if (fusedFfn != null) {
                return fusedFfn.out;
            }
            Tensor xNorm2 = TensorOps.rmsNorm(xRes1, norm2, eps);
            Tensor ffnOut = TensorOps.feedForwardSwiGLU(xNorm2, W1, W2, W3, cache);
            return TensorOps.add(xRes1, ffnOut);
        }

        Tensor forwardKvDecode(
                Tensor x,
                BlockActivationCache cache,
                Tensor kCacheLayer,
                Tensor vCacheLayer,
                int cacheLenBefore,
                int ropePosition) {
            return forwardKvDecode(
                    x, cache, kCacheLayer, vCacheLayer, cacheLenBefore, ropePosition, null, null);
        }

        Tensor forwardKvDecode(
                Tensor x,
                BlockActivationCache cache,
                Tensor kCacheLayer,
                Tensor vCacheLayer,
                int cacheLenBefore,
                int ropePosition,
                TensorOps.GpuFfnResidentBuffers ffnResident) {
            return forwardKvDecode(
                    x,
                    cache,
                    kCacheLayer,
                    vCacheLayer,
                    cacheLenBefore,
                    ropePosition,
                    ffnResident,
                    null);
        }

        Tensor forwardKvDecode(
                Tensor x,
                BlockActivationCache cache,
                Tensor kCacheLayer,
                Tensor vCacheLayer,
                int cacheLenBefore,
                int ropePosition,
                TensorOps.GpuFfnResidentBuffers ffnResident,
                TensorOps.GpuAttnResidentBuffers attnResident) {
            final float eps = TensorOpsGPU.rmsNormEps();
            Tensor attnOut = null;
            if (attnResident != null) {
                attnOut =
                        TensorOps.tryMultiHeadAttentionWithRoPEDecodeGpuResident(
                                x,
                                eps,
                                attnResident,
                                numHeads,
                                kCacheLayer,
                                vCacheLayer,
                                cacheLenBefore,
                                ropePosition);
            }
            if (attnOut == null) {
                Tensor xNorm1 = TensorOps.rmsNorm(x, norm1, eps);
                attnOut =
                        TensorOps.multiHeadAttentionWithRoPEDecode(
                                xNorm1,
                                Wq,
                                Wk,
                                Wv,
                                Wo,
                                numHeads,
                                kCacheLayer,
                                vCacheLayer,
                                cacheLenBefore,
                                ropePosition);
            }
            Tensor xRes1 = TensorOps.add(x, attnOut);
            TensorOps.FfnForwardResult fusedFfn =
                    ffnResident != null
                            ? TensorOps.tryFusedNormResidualSwiGLUForwardGpuResident(xRes1, ffnResident, cache)
                            : null;
            if (fusedFfn == null) {
                fusedFfn = TensorOps.tryFusedNormResidualSwiGLUForwardGpu(xRes1, norm2, W1, W2, W3, cache);
            }
            if (fusedFfn != null) {
                return fusedFfn.out;
            }
            Tensor xNorm2 = TensorOps.rmsNorm(xRes1, norm2, eps);
            Tensor ffnOut = TensorOps.feedForwardSwiGLU(xNorm2, W1, W2, W3, cache);
            return TensorOps.add(xRes1, ffnOut);
        }

        Tensor forwardKvPrefillVram(
                Tensor x,
                Tensor mask,
                BlockActivationCache cache,
                GpuFloatBuffer kGpu,
                GpuFloatBuffer vGpu,
                int maxSeqLen,
                int ropeOffset,
                TensorOps.GpuFfnResidentBuffers ffnResident,
                TensorOps.GpuAttnResidentBuffers attnResident) {
            final float eps = TensorOpsGPU.rmsNormEps();
            if (attnResident == null) {
                throw new IllegalArgumentException("forwardKvPrefillVram requires GpuAttnResidentBuffers");
            }
            Tensor attnOut =
                    TensorOps.tryMultiHeadAttentionWithRoPEPrefillGpuResident(
                            x, eps, attnResident, numHeads, mask, kGpu, vGpu, maxSeqLen, ropeOffset);
            if (attnOut == null) {
                throw new IllegalStateException("GPU KV VRAM prefill path failed");
            }
            Tensor xRes1 = TensorOps.add(x, attnOut);
            TensorOps.FfnForwardResult fusedFfn =
                    ffnResident != null
                            ? TensorOps.tryFusedNormResidualSwiGLUForwardGpuResident(xRes1, ffnResident, cache)
                            : null;
            if (fusedFfn == null) {
                fusedFfn = TensorOps.tryFusedNormResidualSwiGLUForwardGpu(xRes1, norm2, W1, W2, W3, cache);
            }
            if (fusedFfn != null) {
                return fusedFfn.out;
            }
            Tensor xNorm2 = TensorOps.rmsNorm(xRes1, norm2, eps);
            Tensor ffnOut = TensorOps.feedForwardSwiGLU(xNorm2, W1, W2, W3, cache);
            return TensorOps.add(xRes1, ffnOut);
        }

        Tensor forwardKvDecodeVram(
                Tensor x,
                BlockActivationCache cache,
                GpuFloatBuffer kGpu,
                GpuFloatBuffer vGpu,
                int maxSeqLen,
                int cacheLenBefore,
                int ropePosition,
                TensorOps.GpuFfnResidentBuffers ffnResident,
                TensorOps.GpuAttnResidentBuffers attnResident) {
            final float eps = TensorOpsGPU.rmsNormEps();
            if (attnResident == null) {
                throw new IllegalArgumentException("forwardKvDecodeVram requires GpuAttnResidentBuffers");
            }
            Tensor attnOut =
                    TensorOps.tryMultiHeadAttentionWithRoPEDecodeGpuResident(
                            x,
                            eps,
                            attnResident,
                            numHeads,
                            kGpu,
                            vGpu,
                            maxSeqLen,
                            cacheLenBefore,
                            ropePosition);
            if (attnOut == null) {
                throw new IllegalStateException("GPU KV VRAM decode path failed");
            }
            Tensor xRes1 = TensorOps.add(x, attnOut);
            TensorOps.FfnForwardResult fusedFfn =
                    ffnResident != null
                            ? TensorOps.tryFusedNormResidualSwiGLUForwardGpuResident(xRes1, ffnResident, cache)
                            : null;
            if (fusedFfn == null) {
                fusedFfn = TensorOps.tryFusedNormResidualSwiGLUForwardGpu(xRes1, norm2, W1, W2, W3, cache);
            }
            if (fusedFfn != null) {
                return fusedFfn.out;
            }
            Tensor xNorm2 = TensorOps.rmsNorm(xRes1, norm2, eps);
            Tensor ffnOut = TensorOps.feedForwardSwiGLU(xNorm2, W1, W2, W3, cache);
            return TensorOps.add(xRes1, ffnOut);
        }
    }

    /** [batch, seq, d] → плоскость [seq, d]. */
    private static Tensor sliceBatch3D(Tensor t, int batchIdx) {
        int[] shape = t.getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("expected 3D tensor");
        }
        Tensor slice = new Tensor(new int[]{shape[1], shape[2]});
        float[] src = t.internalBuffer();
        float[] dst = slice.internalBuffer();
        int[] s = t.stridesInternal();
        int srcBase = batchIdx * s[0];
        if (s[2] == 1) {
            for (int i = 0; i < shape[1]; i++) {
                System.arraycopy(src, srcBase + i * s[1], dst, i * shape[2], shape[2]);
            }
        } else {
            for (int i = 0; i < shape[1]; i++) {
                for (int j = 0; j < shape[2]; j++) {
                    dst[i * shape[2] + j] = src[srcBase + i * s[1] + j * s[2]];
                }
            }
        }
        return slice;
    }

    /** Копия [seq, last] в dest[batchIdx, :, :]. */
    private static void copyBatchPlane(Tensor dest, Tensor src, int batchIdx) {
        int[] dShape = dest.getShape();
        int[] sShape = src.getShape();
        if (dShape.length != 3
                || sShape.length != 2
                || sShape[0] != dShape[1]
                || sShape[1] != dShape[2]) {
            throw new IllegalArgumentException(
                    "copyBatchPlane: dest " + Arrays.toString(dShape) + ", src " + Arrays.toString(sShape));
        }
        float[] d = dest.internalBuffer();
        float[] sbuf = src.internalBuffer();
        int[] ds = dest.stridesInternal();
        int[] ss = src.stridesInternal();
        int destBase = batchIdx * ds[0];
        if (ds[2] == 1 && ss[1] == 1) {
            for (int i = 0; i < sShape[0]; i++) {
                System.arraycopy(sbuf, i * ss[0], d, destBase + i * ds[1], sShape[1]);
            }
        } else {
            for (int i = 0; i < sShape[0]; i++) {
                for (int j = 0; j < sShape[1]; j++) {
                    d[destBase + i * ds[1] + j * ds[2]] = sbuf[i * ss[0] + j * ss[1]];
                }
            }
        }
    }

}
