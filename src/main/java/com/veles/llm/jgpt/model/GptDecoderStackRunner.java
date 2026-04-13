package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;
import com.veles.llm.jgpt.training.LLMConfig;
import com.veles.llm.jgpt.util.CursorDebugB39372;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CUDA graph + eager стек декодера; поля состояния на {@link GPTModel} (package-private доступ).
 */
final class GptDecoderStackRunner {

    private static final Logger log = LoggerFactory.getLogger(GptDecoderStackRunner.class);
    private static final int DECODER_GRAPH_DEVICE_SNAPSHOT_LEN = 15;

    private GptDecoderStackRunner() {}

    static void destroyDecoderLayerCudaGraphs(GPTModel m) {
        destroyPendingDecoderGraphExecQueuedHandles(m);
        if (m.decoderLayerGraphExec == null) {
            return;
        }
        for (int i = 0; i < m.decoderLayerGraphExec.length; i++) {
            if (m.decoderLayerGraphExec[i] != 0L) {
                TensorOpsGPU.cudaGraphExecDestroy(m.decoderLayerGraphExec[i]);
                m.decoderLayerGraphExec[i] = 0L;
            }
        }
        if (m.decoderLayerGraphDebugCaptureSnapshot != null) {
            Arrays.fill(m.decoderLayerGraphDebugCaptureSnapshot, null);
        }
    }

    /**
     * Отложенные освобождения Java GPU-буферов ({@link TensorOpsGPU#drainDeferredGpuBuffers()}), затем trim async
     * memory pools — тот же порядок, что в {@link com.veles.llm.jgpt.training.LLMTrainer} перед
     * {@link TensorOpsGPU#cudaTrimDeviceMemoryPoolsBestEffort()}.
     */
    private static void drainDeferredGpuBuffersThenTrimPools() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        TensorOpsGPU.drainDeferredGpuBuffers();
        TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
    }

    /**
     * После {@link #destroyDecoderLayerCudaGraphs(m)} при смене ключа захвата: полная синхронизация устройства и trim
     * memory pools — иначе graph memory pool (CUDA 12+) может давать монотонный рост «использованной» VRAM в
     * {@code cudaMemGetInfo} до отложенного reclaim.
     */
    private static void trimDecoderGraphMemoryAfterExecDestroy(GPTModel m) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        TensorOpsGPU.synchronizeDevice();
        drainDeferredGpuBuffersThenTrimPools();

    }

    /** Уничтожает handle'ы в очереди pending без trim (см. {@link #flushPendingDecoderGraphExecDestroy}). */
    private static void destroyPendingDecoderGraphExecQueuedHandles(GPTModel m) {
        if (m.decoderLayerGraphExecPendingDestroyCount == 0) {
            return;
        }
        for (int i = 0; i < m.decoderLayerGraphExecPendingDestroyCount; i++) {
            long ex = m.decoderLayerGraphExecPendingDestroy[i];
            if (ex != 0L) {
                TensorOpsGPU.cudaGraphExecDestroy(ex);
            }
            m.decoderLayerGraphExecPendingDestroy[i] = 0L;
        }
        m.decoderLayerGraphExecPendingDestroyCount = 0;
    }

    /**
     * Откладывает destroy exec текущего слоя до конца forward: все накопленные exec уничтожаются одним блоком в
     * {@link #flushPendingDecoderGraphExecDestroy(m)} с одним trim вместо N отдельных.
     */
    private static void scheduleDecoderGraphExecDestroyIfNotLast(GPTModel m, int layerIndex, int numLayers) {
        if (m.decoderLayerGraphExec == null || layerIndex >= numLayers - 1) {
            return;
        }
        long ex = m.decoderLayerGraphExec[layerIndex];
        if (ex == 0L) {
            return;
        }
        if (m.decoderLayerGraphExecPendingDestroy == null
                || m.decoderLayerGraphExecPendingDestroy.length < numLayers) {
            m.decoderLayerGraphExecPendingDestroy = new long[numLayers];
        }
        m.decoderLayerGraphExecPendingDestroy[m.decoderLayerGraphExecPendingDestroyCount++] = ex;
        m.decoderLayerGraphExec[layerIndex] = 0L;
        if (m.decoderLayerGraphDebugCaptureSnapshot != null) {
            m.decoderLayerGraphDebugCaptureSnapshot[layerIndex] = null;
        }
    }

    /**
     * Уничтожает все exec, накопленные за этот forward, и делает один trim — вместо N отдельных synchronize + trim
     * после каждого слоя.
     */
    private static void flushPendingDecoderGraphExecDestroy(GPTModel m) {
        if (m.decoderLayerGraphExecPendingDestroyCount == 0) {
            return;
        }
        destroyPendingDecoderGraphExecQueuedHandles(m);
        trimDecoderGraphMemoryAfterExecDestroy(m);
    }

    /** Сколько кэшей реально освободили staging; см. {@link BlockActivationCacheDevice#releaseTransientFloatStagingBuffers()}. */
    private static int releaseTransientFloatStagingForAllCaches(BlockActivationCacheDevice[] caches) {
        int n = 0;
        if (caches == null) {
            return 0;
        }
        for (BlockActivationCacheDevice c : caches) {
            if (c != null && c.releaseTransientFloatStagingBuffers()) {
                n++;
            }
        }
        return n;
    }

    private static boolean decoderGraphDeviceSnapshotsEqual(long[] a, long[] b) {
        if (a == null || b == null || a.length != b.length) {
            return false;
        }
        return Arrays.equals(a, b);
    }

    private static long[] computeDecoderGraphDeviceSnapshot(
            GPTModel m,
            GpuFloatBuffer cur,
            GpuFloatBuffer attnOut,
            GpuFloatBuffer blockOut,
            BlockActivationCacheDevice layerCache) {
        long[] nat = TensorOpsGPU.decoderGraphDebugNativeAuxSnapshot();
        long[] s = new long[DECODER_GRAPH_DEVICE_SNAPSHOT_LEN];
        s[0] = m.decoderGraphStridedPackW != null && !m.decoderGraphStridedPackW.isClosed()
                ? m.decoderGraphStridedPackW.devicePointer()
                : 0L;
        s[1] = m.decoderGraphStridedPackC != null && !m.decoderGraphStridedPackC.isClosed()
                ? m.decoderGraphStridedPackC.devicePointer()
                : 0L;
        s[2] = m.decoderChainPing != null && !m.decoderChainPing.isClosed() ? m.decoderChainPing.devicePointer() : 0L;
        s[3] = m.decoderChainPong != null && !m.decoderChainPong.isClosed() ? m.decoderChainPong.devicePointer() : 0L;
        s[4] = cur != null && !cur.isClosed() ? cur.devicePointer() : 0L;
        s[5] = attnOut != null && !attnOut.isClosed() ? attnOut.devicePointer() : 0L;
        s[6] = blockOut != null && !blockOut.isClosed() ? blockOut.devicePointer() : 0L;
        s[7] = layerCache != null ? layerCache.graphCaptureGeneration() : 0L;
        s[8] = 0L;
        s[9] = 0L;
        s[10] = 0L;
        if (layerCache != null) {
            layerCache.fillDecoderGraphDebugSnapshotSlots(s);
        }
        s[11] = nat[0];
        s[12] = nat[1];
        s[13] = nat[2];
        s[14] = nat[3];
        return s;
    }

    private static String formatDecoderGraphPointerLine(
            GPTModel m,
            String phase,
            int layer,
            long exec,
            GpuFloatBuffer cur,
            GpuFloatBuffer attnOut,
            GpuFloatBuffer blockOut,
            BlockActivationCacheDevice layerCache) {
        StringBuilder sb = new StringBuilder(320);
        sb.append(phase)
                .append(" layer=")
                .append(layer)
                .append(" exec=0x")
                .append(Long.toHexString(exec))
                .append(" captureKey=")
                .append(m.decoderLayerGraphCaptureKey);
        if (m.decoderGraphStridedPackW != null && !m.decoderGraphStridedPackW.isClosed()) {
            sb.append(" packW=0x").append(Long.toHexString(m.decoderGraphStridedPackW.devicePointer()));
        }
        if (m.decoderGraphStridedPackC != null && !m.decoderGraphStridedPackC.isClosed()) {
            sb.append(" packC=0x").append(Long.toHexString(m.decoderGraphStridedPackC.devicePointer()));
        }
        if (m.decoderChainPing != null && !m.decoderChainPing.isClosed()) {
            sb.append(" ping=0x").append(Long.toHexString(m.decoderChainPing.devicePointer()));
        }
        if (m.decoderChainPong != null && !m.decoderChainPong.isClosed()) {
            sb.append(" pong=0x").append(Long.toHexString(m.decoderChainPong.devicePointer()));
        }
        if (cur != null && !cur.isClosed()) {
            sb.append(" cur=0x").append(Long.toHexString(cur.devicePointer()));
        }
        if (attnOut != null && !attnOut.isClosed()) {
            sb.append(" attnOut=0x").append(Long.toHexString(attnOut.devicePointer()));
        }
        if (blockOut != null && !blockOut.isClosed()) {
            sb.append(" blockOut=0x").append(Long.toHexString(blockOut.devicePointer()));
        }
        if (layerCache != null) {
            sb.append(' ');
            layerCache.appendDecoderGraphDevicePointers(sb);
        }
        long[] na = TensorOpsGPU.decoderGraphDebugNativeAuxSnapshot();
        sb.append(" nativeAuxNonGraph=0x")
                .append(Long.toHexString(na[0]))
                .append(" sz=")
                .append(na[2])
                .append(" nativeAuxGraph=0x")
                .append(Long.toHexString(na[1]))
                .append(" sz=")
                .append(na[3]);
        return sb.toString();
    }

    // #region agent log
    private static void logDecoderGraphB39372Failure(GPTModel m, String phase, int layer, long exec, int batch, int seqLen) {
        long[] pr = TensorOpsGPU.decoderGraphLaunchProbe(exec);
        long[] na = TensorOpsGPU.decoderGraphDebugNativeAuxSnapshot();
        CursorDebugB39372.appendJson(
                "H5",
                "GPTModel.runDecoderStackLayers",
                "graphLaunchFailed",
                String.format(
                        Locale.ROOT,
                        "\"phase\":\"%s\",\"layer\":%d,\"batch\":%d,\"seqLen\":%d,\"exec\":%d,\"captureKey\":%d,"
                                + "\"threadId\":%d,\"flash\":%b,\"probeDev\":%d,\"probeCap\":%d,\"probeQuery\":%d,"
                                + "\"probeNoPreSyncEnv\":%d,\"probeDriver\":%d,\"probeRuntime\":%d,\"probeExecFlags\":%d,"
                                + "\"probeStream\":%d,\"nativeAuxNonGraph\":%d,\"nativeAuxGraph\":%d,"
                                + "\"nativeAuxNonGraphSz\":%d,\"nativeAuxGraphSz\":%d",
                        phase,
                        layer,
                        batch,
                        seqLen,
                        exec,
                        m.decoderLayerGraphCaptureKey,
                        Thread.currentThread().threadId(),
                        TensorOpsGPU.FLASH_ATTENTION,
                        pr.length > 0 ? pr[0] : -1L,
                        pr.length > 1 ? pr[1] : -1L,
                        pr.length > 2 ? pr[2] : -1L,
                        pr.length > 3 ? pr[3] : -1L,
                        pr.length > 4 ? pr[4] : -1L,
                        pr.length > 5 ? pr[5] : -1L,
                        pr.length > 6 ? pr[6] : -1L,
                        pr.length > 7 ? pr[7] : -1L,
                        na.length > 0 ? na[0] : -1L,
                        na.length > 1 ? na[1] : -1L,
                        na.length > 2 ? na[2] : -1L,
                        na.length > 3 ? na[3] : -1L));
    }

    private static void logDecoderGraphB39372PreLaunch(GPTModel m, int layer, long exec, int batch, int seqLen) {
        if (!CursorDebugB39372.verboseDecoderGraph()) {
            return;
        }
        long[] pr = TensorOpsGPU.decoderGraphLaunchProbe(exec);
        CursorDebugB39372.appendJson(
                "H2",
                "GPTModel.runDecoderStackLayers",
                "preGraphLaunch",
                String.format(
                        Locale.ROOT,
                        "\"layer\":%d,\"batch\":%d,\"seqLen\":%d,\"exec\":%d,\"captureKey\":%d,\"probeDev\":%d,"
                                + "\"probeCap\":%d,\"probeQuery\":%d,\"probeNoPreSyncEnv\":%d,\"probeDriver\":%d,"
                                + "\"probeRuntime\":%d,\"probeExecFlags\":%d,\"probeStream\":%d",
                        layer,
                        batch,
                        seqLen,
                        exec,
                        m.decoderLayerGraphCaptureKey,
                        pr.length > 0 ? pr[0] : -1L,
                        pr.length > 1 ? pr[1] : -1L,
                        pr.length > 2 ? pr[2] : -1L,
                        pr.length > 3 ? pr[3] : -1L,
                        pr.length > 4 ? pr[4] : -1L,
                        pr.length > 5 ? pr[5] : -1L,
                        pr.length > 6 ? pr[6] : -1L,
                        pr.length > 7 ? pr[7] : -1L));
    }
    // #endregion

    private static void disableDecoderLayerCudaGraph(GPTModel m, String reason) {
        if (m.decoderLayerGraphRuntimeDisabled) {
            return;
        }
        m.decoderLayerGraphRuntimeDisabled = true;
        destroyDecoderLayerCudaGraphs(m);
        m.decoderLayerGraphCaptureKey = Integer.MIN_VALUE;
        if (m.decoderLayerCudaGraphWanted) {
            log.warn("JGPT_DECODER_LAYER_CUDA_GRAPH: отключён ({}).", reason);
        }
    }

    private static int decoderLayerGraphKey(
            GPTModel m,
            Tensor mask, boolean trainingStep, int batch, int seqLen, long decoderStackInputDevicePtr) {
        int fp16Mm = TensorOpsGPU.useFp16Matmul() ? 1 : 0;
        int maskId = mask == null ? 0 : System.identityHashCode(mask);
        int cacheFp16 = 0;
        long cacheGen = 0L;
        if (trainingStep && m.blockCachesDevice != null && m.blockCachesDevice.length > 0) {
            if (m.blockCachesDevice[0] != null) {
                cacheFp16 = m.blockCachesDevice[0].isFp16ActivationStorage() ? 1 : 0;
            }
            for (BlockActivationCacheDevice c : m.blockCachesDevice) {
                if (c != null) {
                    cacheGen = cacheGen * 31L + c.graphCaptureGeneration();
                }
            }
        }
        /*
         * Обязательно различать training vs infer в ключе: при cacheFp16==0 (нет FP16-хранения в кэше) и одинаковых
         * batch/seq/mask ключ совпадал бы с {@link #forwardGpuDecoderInfer} (layerCache=null), и следующий train-step
         * мог бы replay графа без записи активаций в {@link BlockActivationCacheDevice} → битый backward / non-finite.
         *
         * cacheGen + flash: при перевыделении VRAM-слотов / LSE+O_heads (Flash) или пуле кэша указатели в графе
         * устаревают — ключ должен измениться до cudaGraphExecLaunch.
         *
         * decoderStackInputDevicePtr: для слоя 0 в граф попадает {@code cur} = вход стека (выход
         * {@link #forwardGpuEmbeddings} / {@link #ensureEmbeddingScratchGpu}). При смене batch/seq буфер
         * эмбеддинга перевыделяется — тот же (batch,seq,…) давал бы stale pointers и cudaGraphLaunch invalid argument.
         *
         * <p>Native thread-local буферы (SDPA graph aux, prewarm {@code tl_graph_sdpa_warmup}, Flash {@code tl_fa_D},
         * QKV pack override/TL — см. {@link TensorOpsGPU#decoderGraphNativeStabilityToken()}): при перевыделении указатели
         * в {@code cudaGraphExec} устаревают. Пересчёт ключа после {@link TensorOps#primeDecoderGraphLayerWorkspaces}.
         */
        int trainMode = trainingStep ? 1 : 0;
        int flash = TensorOpsGPU.FLASH_ATTENTION ? 1 : 0;
        long nativeGraphAuxToken = TensorOpsGPU.decoderGraphNativeStabilityToken();
        return Objects.hash(
                batch,
                seqLen,
                m.dModel,
                m.numHeads,
                maskId,
                fp16Mm,
                cacheFp16,
                trainMode,
                flash,
                cacheGen,
                decoderStackInputDevicePtr,
                nativeGraphAuxToken);
    }

    /**
     * Гарантирует pack-буферы под decoder CUDA graph; при росте — сбрасывает exec (новые указатели).
     */
    private static void ensureDecoderGraphStridedPacks(GPTModel m, long rows) {
        long[] need = TensorOpsGPU.stridedBatchedPackNeed(rows, m.dModel, m.dIntermediate);
        long wNeed = need[0];
        long cNeed = need[1];
        boolean needGrow =
                m.decoderGraphStridedPackW == null
                        || m.decoderGraphStridedPackW.isClosed()
                        || m.decoderGraphStridedPackW.numFloats() < wNeed
                        || m.decoderGraphStridedPackC == null
                        || m.decoderGraphStridedPackC.isClosed()
                        || m.decoderGraphStridedPackC.numFloats() < cNeed;
        if (!needGrow) {
            return;
        }
        destroyDecoderLayerCudaGraphs(m);
        /* Не сбрасывать m.decoderLayerGraphCaptureKey: он уже совпадает с текущим nk из runDecoderStackLayers;
         * иначе на следующем forward сработает ложное «ключ изменился» и графы пересоздадутся дважды подряд. */
        m.decoderGraphStridedPackW = GPTModel.closeGpuBuffer(m.decoderGraphStridedPackW);
        m.decoderGraphStridedPackC = GPTModel.closeGpuBuffer(m.decoderGraphStridedPackC);
        m.decoderGraphStridedPackW = GpuFloatBuffer.allocate(wNeed);
        m.decoderGraphStridedPackC = GpuFloatBuffer.allocate(cNeed);
    }

    private static boolean runDecoderLayerResidentEager(
            GPTModel m,
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
                m.gpuDecoderLayer[layer].attnBuffers(),
                batch,
                seqLen,
                m.dModel,
                m.numHeads,
                mask,
                true,
                devCache)) {
            return false;
        }
        return TensorOps.fusedNormResidualSwiGLUResidentDeviceToDevice(
                attnOut,
                blockOut,
                m.gpuDecoderLayer[layer].ffnBuffers(),
                batch,
                seqLen,
                m.dModel,
                devCache);
    }

    /**
     * @return активации после последнего блока (planar на GPU)
     */
    static GpuFloatBuffer runDecoderStackLayers(
            GPTModel m,
            GpuFloatBuffer xDevice,
            Tensor mask,
            int batch,
            int seqLen,
            boolean trainingStep,
            BlockActivationCacheDevice[] cachesPerLayer) {
        long planeLong = (long) batch * seqLen * m.dModel;
        int plane = (int) planeLong;
        m.decoderChainPing = GPTModel.ensureGpuBuffer(m.decoderChainPing, plane);
        m.decoderChainPong = GPTModel.ensureGpuBuffer(m.decoderChainPong, plane);

        boolean wantGraph =
                m.decoderLayerCudaGraphWanted
                        && !m.decoderLayerGraphRuntimeDisabled
                        && m.decoderLayerGraphExec != null
                        && TensorOpsGPU.isGpuAvailable();
        if (wantGraph) {
            long inPtr =
                    xDevice != null && !xDevice.isClosed() ? xDevice.devicePointer() : 0L;
            int nk = decoderLayerGraphKey(m, mask, trainingStep, batch, seqLen, inPtr);
            if (m.decoderLayerGraphCaptureKey != nk) {
                destroyDecoderLayerCudaGraphs(m);
                trimDecoderGraphMemoryAfterExecDestroy(m);
                m.decoderLayerGraphCaptureKey = nk;
            }
        }

        boolean stridedPackOverride = false;
        if (wantGraph) {
            ensureDecoderGraphStridedPacks(m, (long) batch * seqLen);
            TensorOpsGPU.setStridedBatchedPackOverride(
                    m.decoderGraphStridedPackW.devicePointer(),
                    m.decoderGraphStridedPackC.devicePointer(),
                    m.decoderGraphStridedPackW.numFloats(),
                    m.decoderGraphStridedPackC.numFloats());
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
            if (wantGraph && !m.decoderLayerGraphRuntimeDisabled) {
                /*
                 * Один проход prime до всех launch/capture: иначе prime только перед слоем с ex==0 перевыделяет
                 * thread-local / нативные aux под SDPA, и уже захваченные графы других слоёв получают stale pointers.
                 */
                TensorOpsGPU.ensureStridedBatchedPackScratch((long) batch * seqLen, m.dModel, m.dIntermediate);
                TensorOps.primeDecoderGraphLayerWorkspaces(
                        batch, seqLen, m.dModel, m.numHeads, m.dIntermediate, mask, cachesPerLayer);
                /*
                 * Ключ до prime не видит перевыделение tl_attn_fwd_graph_aux → иначе replay с устаревшими указателями
                 * (cudaGraphLaunch cudaErrorInvalidValue при Flash).
                 */
                long inPtrPost =
                        xDevice != null && !xDevice.isClosed() ? xDevice.devicePointer() : 0L;
                int nkPost = decoderLayerGraphKey(m, mask, trainingStep, batch, seqLen, inPtrPost);
                if (m.decoderLayerGraphCaptureKey != nkPost) {
                    // #region agent log
                    CursorDebugB39372.appendJson(
                            "postPrimeKey",
                            "GPTModel.runDecoderStackLayers",
                            "invalidateGraphsAfterPrimeNativeAux",
                            String.format(
                                    Locale.ROOT,
                                    "\"prevCaptureKey\":%d,\"nkPost\":%d,\"hadExecLayer0\":%b,\"nativeStabilityToken\":%d",
                                    m.decoderLayerGraphCaptureKey,
                                    nkPost,
                                    m.decoderLayerGraphExec[0] != 0L,
                                    TensorOpsGPU.decoderGraphNativeStabilityToken()));
                    // #endregion
                    destroyDecoderLayerCudaGraphs(m);
                    trimDecoderGraphMemoryAfterExecDestroy(m);
                    m.decoderLayerGraphCaptureKey = nkPost;
                }
            }
            if (wantGraph
                    && !m.decoderLayerGraphRuntimeDisabled
                    && LLMConfig.decoderCudaGraphMemLogFromEnvOrProp()) {
                long used = TensorOpsGPU.getGpuMemoryAllocated();
                long tot = TensorOpsGPU.getGpuMemoryReserved();
                log.info(
                        "[DECODER_CUDA_GRAPH] VRAM перед циклом слоёв (graph): used≈{} MiB / total≈{} MiB",
                        used / (1024L * 1024L),
                        tot / (1024L * 1024L));
            }
            /*
             * После OOM на cudaGraphLaunch дальнейшие попытки capture+instantiate на каждом слое (при уже низком
             * cudaMemGetInfo free) в логах дают идентичные сбои 12–19 и лишь нагружают контекст; до конца этого
             * forward используем только eager. Глобально graph не выключаем — следующий вызов runDecoderStackLayers
             * снова попробует graph.
             */
            boolean decoderGraphSkipUntilEndOfForward = false;
            for (int i = 0; i < m.numLayers; i++) {
                GpuFloatBuffer attnOut = GPTModel.decoderScratchOther(cur, m.decoderChainPing, m.decoderChainPong);
                GpuFloatBuffer blockOut = GPTModel.decoderScratchOther(attnOut, m.decoderChainPing, m.decoderChainPong);
                BlockActivationCacheDevice layerCache = cachesPerLayer != null ? cachesPerLayer[i] : null;

                boolean executed = false;
                boolean graphAllowedThisLayer =
                        wantGraph && !m.decoderLayerGraphRuntimeDisabled && !decoderGraphSkipUntilEndOfForward;
                if (graphAllowedThisLayer) {
                    long minFreeB = LLMConfig.decoderGraphMinFreeBytesFromEnvOrProp();
                    if (minFreeB > 0L) {
                        TensorOpsGPU.synchronizeDevice();
                        long totF = TensorOpsGPU.getGpuMemoryReserved();
                        long usedF = TensorOpsGPU.getGpuMemoryAllocated();
                        long freeF = totF - usedF;
                        if (freeF < minFreeB) {
                            decoderGraphSkipUntilEndOfForward = true;
                            graphAllowedThisLayer = false;
                            log.warn(
                                    "JGPT_DECODER_LAYER_CUDA_GRAPH: cudaMemGetInfo free={} B < JGPT_DECODER_GRAPH_MIN_FREE_MIB ({} MiB); "
                                            + "graph отключён до конца forward (проактивно).",
                                    freeF,
                                    LLMConfig.decoderGraphMinFreeMibFromEnvOrProp());
                            // #region agent log
                            CursorDebugB39372.appendJson(
                                    "H-proactiveGraphSkipLowFree",
                                    "GPTModel.runDecoderStackLayers",
                                    "beforeLayer",
                                    String.format(
                                            Locale.ROOT,
                                            "\"layer\":%d,\"free\":%d,\"minFree\":%d",
                                            i,
                                            freeF,
                                            minFreeB));
                            // #endregion
                            drainDeferredGpuBuffersThenTrimPools();
                        }
                    }
                }
                if (graphAllowedThisLayer) {
                    long ex = m.decoderLayerGraphExec[i];
                    if (ex != 0L) {
                        if (m.decoderLayerCudaGraphDebugLog) {
                            long[] now =
                                    computeDecoderGraphDeviceSnapshot(m, cur, attnOut, blockOut, layerCache);
                            long[] cap = m.decoderLayerGraphDebugCaptureSnapshot[i];
                            log.info(
                                    "[DECODER_CUDA_GRAPH] {}",
                                    formatDecoderGraphPointerLine(m, 
                                            "preLaunch", i, ex, cur, attnOut, blockOut, layerCache));
                            if (cap != null && !decoderGraphDeviceSnapshotsEqual(cap, now)) {
                                log.warn(
                                        "[DECODER_CUDA_GRAPH] layer {} snapshot mismatch capture vs preLaunch\ncapture={}\nnow    ={}",
                                        i,
                                        Arrays.toString(cap),
                                        Arrays.toString(now));
                            }
                        }
                        logDecoderGraphB39372PreLaunch(m, i, ex, batch, seqLen);
                        if (TensorOpsGPU.cudaGraphExecLaunch(ex)) {
                            executed = true;
                            TensorOpsGPU.synchronizeDevice();
                            scheduleDecoderGraphExecDestroyIfNotLast(m, i, m.numLayers);
                        } else {
                            logDecoderGraphB39372Failure(m, "replay", i, ex, batch, seqLen);
                            log.warn(
                                    "[DECODER_CUDA_GRAPH] cudaGraphExecLaunch failed {}",
                                    formatDecoderGraphPointerLine(m, 
                                            "launchFailed", i, ex, cur, attnOut, blockOut, layerCache));
                            int lastErr = TensorOpsGPU.decoderGraphExecLaunchLastCudaError();
                            if (lastErr == TensorOpsGPU.CUDA_ERROR_MEMORY_ALLOCATION) {
                                /*
                                 * Накопленные cudaGraphExec нижних слоёв держат заметный VRAM; сбрасываем все exec —
                                 * текущий forward для слоёв > i пойдёт через capture/eager без старых графов.
                                 */
                                log.warn(
                                        "JGPT_DECODER_LAYER_CUDA_GRAPH: слой {} — OOM при replay; уничтожаем все decoder graph exec "
                                                + "(освобождение VRAM), этот шаг — eager для оставшихся слоёв.",
                                        i);
                                // #region agent log
                                CursorDebugB39372.appendJson(
                                        "H8-destroyAllDecoderGraphsOnOOM",
                                        "GPTModel.runDecoderStackLayers",
                                        "replayOOM",
                                        String.format(Locale.ROOT, "\"layer\":%d", i));
                                // #endregion
                                destroyDecoderLayerCudaGraphs(m);
                                if (m.decoderLayerGraphDebugCaptureSnapshot != null) {
                                    Arrays.fill(m.decoderLayerGraphDebugCaptureSnapshot, null);
                                }
                                int cachesReleasedStaging = releaseTransientFloatStagingForAllCaches(cachesPerLayer);
                                TensorOpsGPU.synchronizeDevice();
                                drainDeferredGpuBuffersThenTrimPools();
                                // #region agent log
                                {
                                    long uTrim = TensorOpsGPU.getGpuMemoryAllocated();
                                    long tTrim = TensorOpsGPU.getGpuMemoryReserved();
                                    CursorDebugB39372.appendJson(
                                            "H14-releaseStagingAfterGraphOOM",
                                            "GPTModel.runDecoderStackLayers",
                                            "afterReplayOOM",
                                            String.format(
                                                    Locale.ROOT,
                                                    "\"layer\":%d,\"cachesReleasedStaging\":%d,\"used\":%d,\"total\":%d,\"free\":%d",
                                                    i,
                                                    cachesReleasedStaging,
                                                    uTrim,
                                                    tTrim,
                                                    tTrim - uTrim));
                                    CursorDebugB39372.appendJson(
                                            "H9-trimAfterGraphOOM",
                                            "GPTModel.runDecoderStackLayers",
                                            "afterReplayOOMTrim",
                                            String.format(
                                                    Locale.ROOT,
                                                    "\"layer\":%d,\"used\":%d,\"total\":%d,\"free\":%d",
                                                    i,
                                                    uTrim,
                                                    tTrim,
                                                    tTrim - uTrim));
                                }
                                // #endregion
                                decoderGraphSkipUntilEndOfForward = true;
                                log.warn(
                                        "JGPT_DECODER_LAYER_CUDA_GRAPH: до конца этого forward graph отключён (только eager); "
                                                + "следующий forward снова попробует replay/capture.");
                                // #region agent log
                                CursorDebugB39372.appendJson(
                                        "H-skipGraphRestOfForward",
                                        "GPTModel.runDecoderStackLayers",
                                        "afterReplayOOM",
                                        String.format(Locale.ROOT, "\"layer\":%d", i));
                                // #endregion
                                executed = false;
                            } else {
                                disableDecoderLayerCudaGraph(m, "cudaGraphLaunch failed");
                                /*
                                 * После ошибки запуска графа (часто illegal memory) дальнейшие kernel'ы в этом процессе
                                 * ненадёжны; eager-fallback лишь множит ошибки (cuBLAS 14, FFN и т.д.). Требуется новый JVM.
                                 */
                                throw new IllegalStateException(
                                        "cudaGraphLaunch не удался — контекст CUDA считается недействительным. "
                                                + "Перезапустите JVM. Чтобы не использовать decoder CUDA graph: не задавайте "
                                                + "JGPT_DECODER_LAYER_CUDA_GRAPH=1 (графы включаются только явно).");
                            }
                        }
                    } else {
                        /*
                         * Перед захватом нового слоя: pending exec предыдущих слоёв + trim — иначе graph memory pool
                         * всё ещё держит aux до {@link #flushPendingDecoderGraphExecDestroy(m)} в finally (слишком поздно для
                         * следующего {@code cudaStreamEndCaptureAndInstantiate}).
                         */
                        flushPendingDecoderGraphExecDestroy(m);
                        /*
                         * Полная синхронизация устройства перед захватом: снижает пик VRAM при цепочке
                         * capture→instantiate→launch по слоям (OOM на поздних слоях при stream-only sync).
                         */
                        TensorOpsGPU.synchronizeDevice();
                        if (!TensorOpsGPU.cudaStreamBeginCapture()) {
                            disableDecoderLayerCudaGraph(m, "cudaStreamBeginCapture failed");
                        } else {
                            TensorOps.setDecoderGraphCaptureSkipAttentionMaskHostUpload(true);
                            boolean captureStillActive = true;
                            try {
                                boolean ok =
                                        runDecoderLayerResidentEager(m, 
                                                i, cur, attnOut, blockOut, mask, batch, seqLen, layerCache);
                                long nexec = TensorOpsGPU.cudaStreamEndCaptureAndInstantiate();
                                captureStillActive = false;
                                if (!ok) {
                                    disableDecoderLayerCudaGraph(m, "layer forward failed during capture");
                                    TensorOpsGPU.synchronizeStream();
                                    throw new IllegalStateException(
                                            "decoder layer " + i + " during CUDA graph capture");
                                }
                                if (nexec == 0L) {
                                    disableDecoderLayerCudaGraph(m, "cudaStreamEndCapture/instantiate failed");
                                    TensorOpsGPU.synchronizeStream();
                                    /* граф не создан — захват не исполнял слой; нужен eager */
                                    executed = false;
                                } else {
                                    m.decoderLayerGraphExec[i] = nexec;
                                    if (m.decoderLayerCudaGraphDebugLog) {
                                        m.decoderLayerGraphDebugCaptureSnapshot[i] =
                                                computeDecoderGraphDeviceSnapshot(m, 
                                                        cur, attnOut, blockOut, layerCache);
                                        log.info(
                                                "[DECODER_CUDA_GRAPH] captureOk {} arraysnap={}",
                                                formatDecoderGraphPointerLine(m, 
                                                        "captureOk", i, nexec, cur, attnOut, blockOut, layerCache),
                                                Arrays.toString(m.decoderLayerGraphDebugCaptureSnapshot[i]));
                                    }
                                    /* Захват только записывает ядра, не исполняет их.
                                     * Немедленно запускаем только что захваченный граф,
                                     * чтобы blockOut был заполнен реальными данными. */
                                    logDecoderGraphB39372PreLaunch(m, i, nexec, batch, seqLen);
                                    if (TensorOpsGPU.cudaGraphExecLaunch(nexec)) {
                                        executed = true;
                                        /*
                                         * Завершить отложенные free/аллокации графа до захвата следующего слоя —
                                         * иначе OOM на поздних слоях при том же объёме модели.
                                         */
                                        TensorOpsGPU.synchronizeDevice();
                                        if (LLMConfig.decoderCudaGraphMemLogFromEnvOrProp()) {
                                            long u = TensorOpsGPU.getGpuMemoryAllocated();
                                            long t = TensorOpsGPU.getGpuMemoryReserved();
                                            log.info(
                                                    "[DECODER_CUDA_GRAPH] VRAM после первого launch графа слоя {}: used≈{} MiB / total≈{} MiB",
                                                    i,
                                                    u / (1024L * 1024L),
                                                    t / (1024L * 1024L));
                                        }
                                        scheduleDecoderGraphExecDestroyIfNotLast(m, i, m.numLayers);
                                    } else {
                                        logDecoderGraphB39372Failure(m, "postCapture", i, nexec, batch, seqLen);
                                        log.warn(
                                                "[DECODER_CUDA_GRAPH] postCapture launch failed {}",
                                                formatDecoderGraphPointerLine(m, 
                                                        "postCaptureLaunchFailed",
                                                        i,
                                                        nexec,
                                                        cur,
                                                        attnOut,
                                                        blockOut,
                                                        layerCache));
                                        int lastErr = TensorOpsGPU.decoderGraphExecLaunchLastCudaError();
                                        if (lastErr == TensorOpsGPU.CUDA_ERROR_MEMORY_ALLOCATION) {
                                            log.warn(
                                                    "JGPT_DECODER_LAYER_CUDA_GRAPH: слой {} — OOM при первом launch после capture; "
                                                            + "уничтожаем все decoder graph exec (нижние слои держали VRAM). "
                                                            + "Дальше в этом forward — только eager (повторный capture отключён).",
                                                    i);
                                            // #region agent log
                                            CursorDebugB39372.appendJson(
                                                    "H8-destroyAllDecoderGraphsOnOOM",
                                                    "GPTModel.runDecoderStackLayers",
                                                    "postCaptureOOM",
                                                    String.format(Locale.ROOT, "\"layer\":%d", i));
                                            // #endregion
                                            destroyDecoderLayerCudaGraphs(m);
                                            if (m.decoderLayerGraphDebugCaptureSnapshot != null) {
                                                Arrays.fill(m.decoderLayerGraphDebugCaptureSnapshot, null);
                                            }
                                            int cachesReleasedStaging =
                                                    releaseTransientFloatStagingForAllCaches(cachesPerLayer);
                                            TensorOpsGPU.synchronizeDevice();
                                            drainDeferredGpuBuffersThenTrimPools();
                                            // #region agent log
                                            {
                                                long uTrim = TensorOpsGPU.getGpuMemoryAllocated();
                                                long tTrim = TensorOpsGPU.getGpuMemoryReserved();
                                                CursorDebugB39372.appendJson(
                                                        "H14-releaseStagingAfterGraphOOM",
                                                        "GPTModel.runDecoderStackLayers",
                                                        "afterPostCaptureOOM",
                                                        String.format(
                                                                Locale.ROOT,
                                                                "\"layer\":%d,\"cachesReleasedStaging\":%d,\"used\":%d,\"total\":%d,\"free\":%d",
                                                                i,
                                                                cachesReleasedStaging,
                                                                uTrim,
                                                                tTrim,
                                                                tTrim - uTrim));
                                                CursorDebugB39372.appendJson(
                                                        "H9-trimAfterGraphOOM",
                                                        "GPTModel.runDecoderStackLayers",
                                                        "afterPostCaptureOOMTrim",
                                                        String.format(
                                                                Locale.ROOT,
                                                                "\"layer\":%d,\"used\":%d,\"total\":%d,\"free\":%d",
                                                                i,
                                                                uTrim,
                                                                tTrim,
                                                                tTrim - uTrim));
                                            }
                                            // #endregion
                                            decoderGraphSkipUntilEndOfForward = true;
                                            log.warn(
                                                    "JGPT_DECODER_LAYER_CUDA_GRAPH: до конца этого forward graph отключён (только eager); "
                                                            + "следующий forward снова попробует capture.");
                                            // #region agent log
                                            CursorDebugB39372.appendJson(
                                                    "H-skipGraphRestOfForward",
                                                    "GPTModel.runDecoderStackLayers",
                                                    "afterPostCaptureOOM",
                                                    String.format(Locale.ROOT, "\"layer\":%d", i));
                                            // #endregion
                                        } else {
                                            TensorOpsGPU.cudaGraphExecDestroy(nexec);
                                            m.decoderLayerGraphExec[i] = 0L;
                                            if (m.decoderLayerGraphDebugCaptureSnapshot != null) {
                                                m.decoderLayerGraphDebugCaptureSnapshot[i] = null;
                                            }
                                            disableDecoderLayerCudaGraph(m, "cudaGraphLaunch failed on capture step");
                                        }
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
                    if (!runDecoderLayerResidentEager(m, i, cur, attnOut, blockOut, mask, batch, seqLen, layerCache)) {
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
            flushPendingDecoderGraphExecDestroy(m);
        }
    }

}
