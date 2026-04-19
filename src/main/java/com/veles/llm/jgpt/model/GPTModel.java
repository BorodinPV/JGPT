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
import com.veles.llm.jgpt.util.CursorDebugB39372;
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
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

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

    /** Бинарный формат {@link com.veles.llm.jgpt.training.LLMTrainer#saveModelWeights(String)} (UTF-тег + raw float). */
    public static final String MODEL_WEIGHTS_FORMAT_V1 = "veles.weights.v1";

    final int vocabSize;
    final int maxSeqLen;
    final int dModel;
    final int numHeads;
    final int numLayers;
    final int dIntermediate;

    final TokenEmbedding tokenEmbedding;
    final PositionEmbedding positionEmbedding;
    final DecoderBlock[] blocks;
    final Tensor layerNormFinal;
    final Tensor lmHead;

    /**
     * При {@code true} (и доступной CUDA) дубликаты {@link #layerNormFinal} и {@link #lmHead} лежат в
     * {@link GptGpuWeights} для путей вроде {@link #forwardGpuLmHead(Tensor)} без H2D весов на каждый шаг.
     */
    private final boolean gpuResident;
    /** Dropout probability for residual connections (applied after attention and FFN). */
    private float residualDropout = 0f;
    /** Dropout probability for attention weights (after softmax). */
    private float attentionDropout = 0f;
    /** Dropout probability for embedding (after gather). */
    private float embeddingDropout = 0f;

    private final GptGpuWeights gpuResidentHead;

    /** VRAM-копии весов декодер-слоя (pre-norm + Q/K/V/O + FFN); без H2D весов в fused/attention путях. */
    final GptGpuDecoderLayerGpuWeights[] gpuDecoderLayer;

    /**
     * {@code true} если GPU-резидентно и decoder-pipeline разрешён (env {@code JGPT_DECODER_GPU_PIPELINE=1}
     * или prop {@code jgpt.decoder.gpu.pipeline=true}).
     */
    private final boolean decoderGpuPipeline;

    /**
     * Запрошен env {@code JGPT_DECODER_LAYER_CUDA_GRAPH}: при успешном захвате — один graph exec на слой (MHA+FFN).
     */
    final boolean decoderLayerCudaGraphWanted;

    /** native {@code cudaGraphExec_t} на слой; {@code 0} — ещё не захвачен или сброшен. */
    final long[] decoderLayerGraphExec;

    int decoderLayerGraphCaptureKey = Integer.MIN_VALUE;

    boolean decoderLayerGraphRuntimeDisabled;

    /**
     * Env {@code JGPT_DECODER_LAYER_CUDA_GRAPH_LOG=1}: логи указателей и сравнение снимка с момента capture (см.
     * {@link LLMConfig#decoderLayerCudaGraphDebugLogFromEnvOrProp()}).
     */
    final boolean decoderLayerCudaGraphDebugLog;

    /**
     * При debug: числовой снимок device pointer'ов после успешного {@code cudaStreamEndCapture} для слоя; иначе
     * {@code null}.
     */
    long[][] decoderLayerGraphDebugCaptureSnapshot;

    /**
     * Exec-объекты промежуточных слоёв, отложенные до конца {@link #runDecoderStackLayers}: один batched destroy +
     * trim в {@link #flushPendingDecoderGraphExecDestroy()}.
     */
    long[] decoderLayerGraphExecPendingDestroy;

    int decoderLayerGraphExecPendingDestroyCount;

    /** Счётчик вызовов {@link #forwardGpuDecoder} для {@link LLMConfig#trainVramStepProbeFromEnvOrProp()}. */
    private static final AtomicLong trainDecoderVramProbeSeq = new AtomicLong();

    /** CE + LM head backward на device; включается через {@link #setDeviceLogitsEnabled(boolean)}. */
    private boolean deviceLogitsEnabled;

    /** Decoder backward на VRAM; включается через {@link #setDeviceDecoderBackward(boolean)}. */
    boolean deviceDecoderBackward;

    /** Скрытое состояние перед LM head (после финального RMSNorm); для backward последнего слоя. */
    private Tensor lastHidden;

    /** Вход в финальный RMSNorm (до γ); нужен для backward. */
    private Tensor xBeforeFinalNorm;

    /** Кэш активаций по слоям при {@link #forward(Tensor, boolean)} с training=true. */
    private volatile BlockActivationCache[] blockCaches;

    /** Device-копии активаций по слоям для полного decoder backward без host round-trip. */
    volatile BlockActivationCacheDevice[] blockCachesDevice;

    /**
     * Сохранять активации блока в FP16 в {@link BlockActivationCache} (env {@code
     * JGPT_ACTIVATION_CACHE_FP16=1} или {@code -Djgpt.activationCache.fp16=true}).
     */
    private final boolean useFp16ActivationCache;

    /** Последний вызов {@link #forward(Tensor, boolean)}: было ли {@code training=true}. */
    private boolean lastForwardWasTraining;

    Tensor lastMask;
    private Tensor cachedCausalMask;
    private int cachedCausalMaskSeqLen = -1;
    private Tensor cachedLmHeadTranspose;
    private boolean cachedLmHeadTransposeValid;

    /** Кэш {@link #gpuTensorByTrainableParameter()}: пересоздаётся только при явном сбросе. */
    private Map<Tensor, GpuTensor> cachedGpuParamMap;
    private boolean cachedGpuParamMapValid;
    Tensor lastInputTokens;
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
    GpuFloatBuffer decoderBwdGradPing;
    GpuFloatBuffer decoderBwdGradPong;

    /**
     * [batch, seq, d_model] на VRAM: gather токенов + позиции без D2H между ними (см. {@link #ensureEmbeddingScratchGpu}).
     */
    private GpuTensor embeddingScratchGpu;

    private int embeddingScratchBatch = -1;
    private int embeddingScratchSeqLen = -1;

    /** Цепочка decoder: ping-pong на VRAM без D2H между слоями ({@link #forwardGpuDecoder}). */
    GpuFloatBuffer decoderChainPing;

    GpuFloatBuffer decoderChainPong;

    /**
     * Strided-batched QKV/FFN pack на VRAM при {@link #decoderLayerCudaGraphWanted}: фиксированные указатели для
     * {@link TensorOpsGPU#setStridedBatchedPackOverride} на время {@link #runDecoderStackLayers} (инвариант CUDA graph).
     */
    GpuFloatBuffer decoderGraphStridedPackW;

    GpuFloatBuffer decoderGraphStridedPackC;

    /** RMSNorm перед LM head на VRAM ({@link #forwardGpuLmHeadFromDevice}). */
    private GpuFloatBuffer lmHeadNormScratchGpu;

    /**
     * Заглушка формы логитов для API {@link Tensor} при обучении с логитами только на GPU (данные не читаются —
     * CE и backward идут по {@link #lastLogitsGpu}).
     */
    private float[] logitsShapeStub;

    /** Одна строка логитов для {@link GptAutoregressiveGenerator#sampleNextToken} без порчи буфера logits. */
    float[] sampleLogitsScratch;

    /** Маска «входит в top-k» при сэмплинге; совпадает с порогом стабильной сортировки по (logit↓, индекс↑). */
    boolean[] sampleTopKMember;

    /** Переиспользование {@code [1,1]} в {@link #generate} (декодирование одного токена). */
    Tensor reusableDecodeOneToken;

    /**
     * Слайс токенов {@code [1, sliceLen]} при скользящем KV-окне: один экземпляр на ту же длину {@code sliceLen},
     * чтобы не аллоцировать тензор на каждом срабатывании (длина может меняться между шагами).
     */
    Tensor reusableSlidingPrefillInput;

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
            this.decoderLayerCudaGraphDebugLog = LLMConfig.decoderLayerCudaGraphDebugLogFromEnvOrProp();
            this.decoderLayerGraphDebugCaptureSnapshot =
                    this.decoderLayerCudaGraphDebugLog ? new long[numLayers][] : null;
        } else {
            this.decoderLayerCudaGraphWanted = false;
            this.decoderLayerGraphExec = null;
            this.decoderLayerCudaGraphDebugLog = false;
            this.decoderLayerGraphDebugCaptureSnapshot = null;
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
        GpuTensor tokEmbGpu = tokenEmbedding.deviceWeightsOrNull();
        if (tokEmbGpu != null) {
            map.put(tokenEmbedding.hostWeights(), tokEmbGpu);
        }
        GpuTensor posEmbGpu = positionEmbedding.deviceWeightsOrNull();
        if (posEmbGpu != null) {
            map.put(positionEmbedding.hostWeights(), posEmbGpu);
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
        GptDecoderStackRunner.destroyDecoderLayerCudaGraphs(this);
    }

    private GpuFloatBuffer runDecoderStackLayers(
            GpuFloatBuffer xDevice,
            Tensor mask,
            int batch,
            int seqLen,
            boolean trainingStep,
            BlockActivationCacheDevice[] cachesPerLayer) {
        return GptDecoderStackRunner.runDecoderStackLayers(
                this, xDevice, mask, batch, seqLen, trainingStep, cachesPerLayer);
    }

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

    /**
     * Задаёт вероятности dropout для обучения. Вызывается из {@link LLMTrainer} после создания модели.
     *
     * @param residualDropout вероятность dropout для residual connections
     * @param attentionDropout вероятность dropout для attention weights
     * @param embeddingDropout вероятность dropout для embedding
     */
    public void setDropout(float residualDropout, float attentionDropout, float embeddingDropout) {
        this.residualDropout = Math.max(0f, Math.min(1f, residualDropout));
        this.attentionDropout = Math.max(0f, Math.min(1f, attentionDropout));
        this.embeddingDropout = Math.max(0f, Math.min(1f, embeddingDropout));
        for (int i = 0; i < blocks.length; i++) {
            blocks[i].setDropout(residualDropout, attentionDropout, i);
        }
    }
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
        /*
         * CUDA graph capture записывает указатели на слоты {@link BlockActivationCacheDevice}. После close()/пула
         * буферы на новых адресах при том же decoderLayerGraphKey (batch/seq/mask/режим) replay давал бы
         * cudaGraphLaunch / illegal access — обязательно сбрасываем exec.
         */
        destroyDecoderLayerCudaGraphs();
        decoderLayerGraphCaptureKey = Integer.MIN_VALUE;
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
        /* training-кэш мог быть уже null (чистый infer); графы декодера всё равно нужно сбросить — см. scratch-handoff */
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
        GptModelDebugLog.scratchHandoff(m);
    }

    static GpuFloatBuffer closeGpuBuffer(GpuFloatBuffer buf) {
        if (buf != null && !buf.isClosed()) {
            buf.close();
        }
        return null;
    }

    static GpuFloatBuffer ensureGpuBuffer(GpuFloatBuffer buf, long size) {
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
        boolean vramStepProbe = LLMConfig.trainVramStepProbeFromEnvOrProp();
        long probeSeq = 0L;
        boolean logThisStep = false;
        if (vramStepProbe) {
            probeSeq = trainDecoderVramProbeSeq.incrementAndGet();
            int every = LLMConfig.trainVramStepProbeEveryFromEnvOrProp();
            logThisStep = every > 0 && (probeSeq % every) == 0L;
            if (logThisStep) {
                TensorOpsGPU.synchronizeDevice();
                // #region agent log
                long totB = TensorOpsGPU.getGpuMemoryReserved();
                long usedB = TensorOpsGPU.getGpuMemoryAllocated();
                CursorDebugB39372.appendJson(
                        "H-vramTrainStep",
                        "GPTModel.forwardGpuDecoder",
                        "decoderBefore",
                        String.format(
                                Locale.ROOT,
                                "\"seq\":%d,\"used\":%d,\"free\":%d,\"total\":%d",
                                probeSeq,
                                usedB,
                                totB - usedB,
                                totB));
                // #endregion
            }
        }
        GpuFloatBuffer out = runDecoderStackLayers(xDevice, mask, batch, seqLen, true, blockCachesDevice);
        if (logThisStep) {
            TensorOpsGPU.synchronizeDevice();
            // #region agent log
            long totA = TensorOpsGPU.getGpuMemoryReserved();
            long usedA = TensorOpsGPU.getGpuMemoryAllocated();
            CursorDebugB39372.appendJson(
                    "H-vramTrainStep",
                    "GPTModel.forwardGpuDecoder",
                    "decoderAfter",
                    String.format(
                            Locale.ROOT,
                            "\"seq\":%d,\"used\":%d,\"free\":%d,\"total\":%d",
                            probeSeq,
                            usedA,
                            totA - usedA,
                            totA));
            // #endregion
        }
        return out;
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

    static GpuFloatBuffer decoderScratchOther(GpuFloatBuffer cur, GpuFloatBuffer a, GpuFloatBuffer b) {
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
        GptLmHeadGpu.applyFusedPreferredThenSplit(
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
            GptLmHeadGpu.applyFusedPreferredThenSplit(
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
            GptModelDebugLog.rmsBuf(
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
            GptModelDebugLog.rmsBuf(
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
        /* D2H в GpuFloatBuffer делает cudaStreamSynchronize; при незавершённом cudaStreamBeginCapture (shutdown между
         * слоями и т.п.) это ломает захват. Сбрасываем capture до любых downloadTo. */
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.abortCudaStreamCaptureIfActive();
        }
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
        return GptKvForward.forwardPrefillHost(this, inputTokens, cache, ropeOffset);
    }

    /**
     * Prefill с KV на VRAM ({@link KvCacheGpu}): без host-тензоров K/V в attention на горячем пути.
     *
     * @see #forwardPrefill(Tensor, KvCache, int)
     */
    public Tensor forwardPrefill(Tensor inputTokens, KvCacheGpu cache, int ropeOffset) {
        return GptKvForward.forwardPrefillGpu(this, inputTokens, cache, ropeOffset);
    }

    /**
     * Один шаг декодирования с KV-cache (batch=1). До вызова: {@code cache.length() == cacheLenBefore}.
     *
     * @param inputTokens {@code [1, 1]} — один токен
     * @param ropePosition абсолютный индекс позиции этого токена (RoPE)
     * @return logits {@code [1, 1, vocab_size]}
     */
    public Tensor forwardDecode(Tensor inputTokens, KvCache cache, int cacheLenBefore, int ropePosition) {
        return GptKvForward.forwardDecodeHost(this, inputTokens, cache, cacheLenBefore, ropePosition);
    }

    /**
     * Decode с KV на VRAM.
     *
     * @see #forwardDecode(Tensor, KvCache, int, int)
     */
    public Tensor forwardDecode(Tensor inputTokens, KvCacheGpu cache, int cacheLenBefore, int ropePosition) {
        return GptKvForward.forwardDecodeGpu(this, inputTokens, cache, cacheLenBefore, ropePosition);
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
                GptDecoderBackward.backwardDecoderLayersDevice(this, dGradBeforeNorm, batch, seqLen, zeroParamGrads);
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
                GptDecoderBackward.backwardDecoderLayersDevice(this, dGradBeforeNorm, batch, seqLen, zeroParamGrads);
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
                GptDecoderBackward.backwardDecoderLayersDevice(this, dGradBeforeNorm, batch, seqLen, zeroParamGrads);
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
     * <b>Скользящее окно (когда длина контекста превышает :</b> KV-cache сбрасывается и
     * выполняется полный prefill по срезу подсказки с подходящим RoPE-сдвигом. Это корректно, но даёт дополнительную
     * работу порядка квадрата окна на каждое срабатывание; для очень длинных генераций в production обычно используют
     * rolling KV / paged attention вместо полного пересчёта.
     */
    public Tensor generate(Tensor inputTokens, int maxNewTokens, float temperature, int topK) {
        return GptAutoregressiveGenerator.generateHost(this, inputTokens, maxNewTokens, temperature, topK);
    }

    /**
     * Авторегрессивная генерация (batch=1), как {@link #generate(Tensor, int, float, int)}, но KV хранится в
     * {@link KvCacheGpu} (путь attention без host-тензоров K/V). Тот же контракт по {@code temperature} / {@code
     * topK}, что и {@link #generate}.
     * <p>
     * При превышении действует тот же механизм скользящего окна, что и в {@link #generate}, с
     * полным reprefill среза (см. javadoc {@link #generate}).
     * <p>
     * Требует {@link #isGpuResident()} и доступной CUDA ({@link TensorOpsGPU#isGpuAvailable()}).
     */
    public Tensor generateGpuKv(Tensor inputTokens, int maxNewTokens, float temperature, int topK) {
        return GptAutoregressiveGenerator.generateGpuKv(this, inputTokens, maxNewTokens, temperature, topK);
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
                    GptModelWeightsLoader.loadFromJavaSerialization(numLayers, getParameters(), in);
                }
            } else {
                DataInputStream dis = new DataInputStream(bis);
                String tag = dis.readUTF();
                if (!MODEL_WEIGHTS_FORMAT_V1.equals(tag)) {
                    throw new IOException("unknown model weights tag: " + tag);
                }
                GptModelWeightsLoader.loadFromBinaryV1(numLayers, getParameters(), dis);
            }
        }
        syncGpuResidentWeightsFromHost();
        log.info("Веса модели загружены из файла: {}", path);
    }

    public Tensor getLastHidden() {
        return lastHidden;
    }

    public Tensor getLmHead() {
        return lmHead;
    }
}
