package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.core.QuantizedTensor;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.model.BlockActivationCache;
import com.veles.llm.jgpt.model.BlockActivationCacheDevice;

import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.training.LLMConfig;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Операции над тензорами: реализации через {@link TensorOpsGPU} (CUDA), без дублирующего CPU-пути там,
 * где натив уже есть. Требуется {@link TensorOpsGPU#requireCuda(String)} в точке входа процесса.
 * <p>
 * Редукции вроде {@link #sum} без отдельного GPU-kernel остаются на хосте (Vector API при наличии).
 */
public final class TensorOps {

    private static final Logger log = LoggerFactory.getLogger(TensorOps.class);

    /**
     * H2D маски attention с Java heap недопустим внутри {@code cudaStreamBeginCapture} — граф необходимо строить с уже
     * заполненным {@link GpuAttentionResidentWorkspace#getMaskDev()}. Перед захватом: {@link
     * #primeDecoderGraphAttentionMaskDevice}; между begin/end capture: {@link
     * #setDecoderGraphCaptureSkipAttentionMaskHostUpload}{@code (true)} (см. {@code GPTModel#runDecoderStackLayers}).
     */
    private static final ThreadLocal<Boolean> decoderGraphCaptureSkipAttentionMaskH2D =
            ThreadLocal.withInitial(() -> Boolean.FALSE);

    /**
     * Загрузить маску на device в thread-local attention workspace до {@code cudaStreamBeginCapture}. Безопасно
     * вызывать несколько раз для той же {@code seqLen} (буфер переиспользуется).
     */
    public static void primeDecoderGraphAttentionMaskDevice(Tensor mask, int seqLen) {
        if (mask == null) {
            return;
        }
        int[] mShape = mask.getShape();
        if (mShape.length != 2 || mShape[0] != seqLen || mShape[1] != seqLen) {
            log.warn(
                    "primeDecoderGraphAttentionMaskDevice: ожидается маска [{}, {}], получено {} — H2D пропущен; "
                            + "replay CUDA graph читал бы устаревший maskDev",
                    seqLen,
                    seqLen,
                    Arrays.toString(mShape));
            return;
        }
        GpuAttentionResidentWorkspace ws = GpuAttentionResidentWorkspace.local();
        synchronized (ws.exclusiveUseLock()) {
            ws.ensureMask(seqLen);
            int n = Math.multiplyExact(seqLen, seqLen);
            ws.getMaskDev().copyFrom(mask.internalBuffer(), 0, n);
        }
    }

    /**
     * Выделить thread-local attention/FFN workspace, нативный SDPA aux и FP16 staging под probs до любого
     * {@code cudaStreamBeginCapture}/{@code cudaGraphExecLaunch} в этом decoder-pass.
     *
     * <p><b>Обязательно вызывать один раз до цикла по слоям</b>: при частичном захвате (часть {@code exec} уже есть)
     * повторный prime только перед слоем N перевыделил бы общие TL-буферы и инвалидировал графы слоёв
     * {@code 0..N-1} (устаревшие указатели в CUDA graph → {@code cudaGraphLaunch} / illegal access).
     *
     * <p>Во время захвата недопустимы {@code cudaMalloc}/{@code cudaFree} и синхронизация основного stream.
     */
    public static void primeDecoderGraphLayerWorkspaces(
            int batch,
            int seqLen,
            int dModel,
            int numHeads,
            int dIntermediate,
            Tensor mask,
            BlockActivationCacheDevice[] cachesPerLayer) {
        int rows = Math.multiplyExact(batch, seqLen);
        GpuAttentionResidentWorkspace aw = GpuAttentionResidentWorkspace.local();
        synchronized (aw.exclusiveUseLock()) {
            aw.ensure(rows, dModel);
            if (mask != null) {
                aw.ensureMask(seqLen);
            }
        }
        GpuForwardBlockWorkspace fw = GpuForwardBlockWorkspace.local();
        synchronized (fw.exclusiveUseLock()) {
            fw.ensureFfnNormResidual(rows, dModel, dIntermediate);
        }
        if (cachesPerLayer != null) {
            int bAttn = Math.multiplyExact(batch, numHeads);
            int probFloats = Math.multiplyExact(Math.multiplyExact(bAttn, seqLen), seqLen);
            for (BlockActivationCacheDevice c : cachesPerLayer) {
                if (c != null) {
                    c.ensureAttentionProbsFloatStageForGraphCapture(probFloats);
                }
            }
        }
        TensorOpsGPU.decoderGraphPrewarmDeviceOps(batch, seqLen, dModel, numHeads, dIntermediate);
    }

    /**
     * Вкл/выкл пропуск H2D маски в {@link #multiHeadAttentionResidentDeviceToDevice}. Только на границе CUDA graph
     * capture в декодере.
     */
    public static void setDecoderGraphCaptureSkipAttentionMaskHostUpload(boolean skip) {
        decoderGraphCaptureSkipAttentionMaskH2D.set(skip);
    }

    private static boolean isDecoderGraphCaptureSkipAttentionMaskHostUpload() {
        return Boolean.TRUE.equals(decoderGraphCaptureSkipAttentionMaskH2D.get());
    }

    private static void rmsNormThenFfnW1W3ProjectionsGpu(
            GpuFloatBuffer xRes1,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer xNorm2,
            GpuFloatBuffer w1,
            GpuFloatBuffer w3,
            GpuFloatBuffer h1,
            GpuFloatBuffer gate,
            int rows,
            int dModel,
            int dInt) {
        if (LLMConfig.fusedFfnRmsW1W3FromEnvOrProp()) {
            TensorOpsGPU.rmsNormMatmulFfnW1W3GpuDevice(
                    xRes1,
                    gamma,
                    eps,
                    xNorm2,
                    w1,
                    w3,
                    h1,
                    gate,
                    rows,
                    dModel,
                    dInt,
                    TensorOpsGPU.useFp16Matmul());
        } else {
            TensorOpsGPU.rmsNormGpuDevice(xRes1, gamma, eps, xNorm2, rows, dModel);
            TensorOpsGPU.matmulGpuDeviceFfnW1W3Projections(xNorm2, w1, w3, h1, gate, rows, dModel, dInt);
        }
    }

    public static final class AttentionForwardResult {
        public final Tensor output;
        public final Tensor attentionWeights;

        public AttentionForwardResult(Tensor output, Tensor attentionWeights) {
            this.output = output;
            this.attentionWeights = attentionWeights;
        }
    }

    public static final class FfnForwardResult {
        public final Tensor xNorm2;
        public final Tensor ffnOut;
        public final Tensor out;

        public FfnForwardResult(Tensor xNorm2, Tensor ffnOut, Tensor out) {
            this.xNorm2 = xNorm2;
            this.ffnOut = ffnOut;
            this.out = out;
        }
    }

    /**
     * Указатели на уже загруженные на device веса FFN (γ второго RMSNorm, W1, W2, W3) — без H2D копий на каждый
     * forward (см. {@link #tryFusedNormResidualSwiGLUForwardGpuResident}).
     */
    public record GpuFfnResidentBuffers(
            GpuFloatBuffer normGamma, GpuFloatBuffer w1, GpuFloatBuffer w2, GpuFloatBuffer w3) {}

    /**
     * Устройственные веса первого RMSNorm и Q/K/V/O для {@link #tryMultiHeadAttentionWithRoPEGpuResident}.
     */
    public record GpuAttnResidentBuffers(
            GpuFloatBuffer normGamma,
            GpuFloatBuffer wq,
            GpuFloatBuffer wk,
            GpuFloatBuffer wv,
            GpuFloatBuffer wo) {}

    /**
     * Результат resident-attention: выход MHA и (если нужен кэш) x после первого RMSNorm на хосте.
     */
    public record AttnGpuResidentResult(Tensor out, Tensor xNorm1) {}

    /** Кэш {@link #createCausalMask(int)} по {@code seqLen}; не изменять возвращённый тензор на месте. */
    private static final ConcurrentHashMap<Integer, Tensor> CAUSAL_MASK_CACHE = new ConcurrentHashMap<>();

    private static final VectorSpecies<Float> VECTOR_SPECIES;
    private static final int VECTOR_LANE_COUNT;
    private static final boolean VECTOR_SUPPORTED;

    static {
        VectorSpecies<Float> species;
        boolean supported;
        try {
            species = FloatVector.SPECIES_PREFERRED;
            supported = species.length() > 0;
            if (supported) {
                log.info("Vector API (SIMD) включён: species={}, число линий={}", species, species.length());
            }
        } catch (Exception | Error e) {
            species = FloatVector.SPECIES_128;
            supported = false;
            log.warn("Vector API недоступен ({}), математика — скалярная", e.getClass().getSimpleName());
        }
        VECTOR_SPECIES = species;
        VECTOR_LANE_COUNT = species.length();
        VECTOR_SUPPORTED = supported;
    }

    public static boolean vectorApiSupported() {
        return VECTOR_SUPPORTED;
    }

    public static int vectorLaneCount() {
        return VECTOR_LANE_COUNT;
    }

    public static VectorSpecies<Float> vectorSpecies() {
        return VECTOR_SPECIES;
    }

    /** @deprecated Плитка CPU-GEMM не используется; только проверка диапазона. */
    @Deprecated
    public static void setBlockSize(int size) {
        if (size < 1 || size > 4096) {
            throw new IllegalArgumentException("blockSize must be in [1, 4096], got " + size);
        }
    }

    /** @deprecated Всегда 64. */
    @Deprecated
    public static int getBlockSize() {
        return 64;
    }

    /** @deprecated Всегда 64. */
    @Deprecated
    public static int getEffectiveMatmulBlockSize(int m, int k, int n) {
        return 64;
    }

    /** @deprecated Всегда 64. */
    @Deprecated
    public static int getEffectiveBlockSize(int m, int k, int n) {
        return 64;
    }

    public static Tensor add(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a cannot be null");
        Objects.requireNonNull(b, "b cannot be null");
        validateSameShape(a, b);
        Tensor result = new Tensor(a.getShape());
        float[] da = a.internalBuffer();
        float[] db = b.internalBuffer();
        float[] dr = result.internalBuffer();
        int n = da.length;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.add");
        TensorOpsGPU.addGPU(da, db, dr, n);
        return result;
    }

    public static Tensor subtract(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a cannot be null");
        Objects.requireNonNull(b, "b cannot be null");
        validateSameShape(a, b);
        Tensor result = new Tensor(a.getShape());
        float[] da = a.internalBuffer();
        float[] db = b.internalBuffer();
        float[] dr = result.internalBuffer();
        int n = da.length;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.subtract");
        TensorOpsGPU.subtractGPU(da, db, dr, n);
        return result;
    }

    public static Tensor multiply(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a cannot be null");
        Objects.requireNonNull(b, "b cannot be null");
        validateSameShape(a, b);
        Tensor result = new Tensor(a.getShape());
        float[] da = a.internalBuffer();
        float[] db = b.internalBuffer();
        float[] dr = result.internalBuffer();
        int n = da.length;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.multiply");
        TensorOpsGPU.multiplyGPU(da, db, dr, n);
        return result;
    }

    public static Tensor multiplyScalar(Tensor a, float scalar) {
        Objects.requireNonNull(a, "a cannot be null");
        Tensor result = new Tensor(a.getShape());
        float[] da = a.internalBuffer();
        float[] dr = result.internalBuffer();
        int n = da.length;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.multiplyScalar");
        TensorOpsGPU.multiplyScalarGPU(da, dr, n, scalar);
        return result;
    }

    public static Tensor relu(Tensor a) {
        Objects.requireNonNull(a, "a cannot be null");
        Tensor result = new Tensor(a.getShape());
        float[] da = a.internalBuffer();
        float[] dr = result.internalBuffer();
        if (da.length <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.relu");
        TensorOpsGPU.reluGPU(da, dr, da.length);
        return result;
    }

    public static Tensor sigmoid(Tensor a) {
        Objects.requireNonNull(a, "a cannot be null");
        Tensor result = new Tensor(a.getShape());
        float[] da = a.internalBuffer();
        float[] dr = result.internalBuffer();
        if (da.length <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.sigmoid");
        TensorOpsGPU.sigmoidGPU(da, dr, da.length);
        return result;
    }

    /**
     * GELU: {@code x * 0.5 * (1 + erf(x / √2))}. Аппроксимация:
     * {@code 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))}.
     */
    public static Tensor gelu(Tensor a) {
        Objects.requireNonNull(a, "a cannot be null");
        Tensor result = new Tensor(a.getShape());
        float[] src = a.internalBuffer();
        float[] dst = result.internalBuffer();
        if (src.length <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.gelu");
        TensorOpsGPU.geluGPU(src, dst, src.length);
        return result;
    }

    /**
     * SwiGLU: {@code (x W₁) ⊙ SiLU(x W₃)} {@code W₂}, где {@code SiLU(t)=t·σ(t)}.
     *
     * @param x  [batch, seq_len, d_model]
     * @param W1 [d_model, d_intermediate]
     * @param W2 [d_intermediate, d_model]
     * @param W3 [d_model, d_intermediate]
     */
    public static Tensor feedForwardSwiGLU(Tensor x, Tensor W1, Tensor W2, Tensor W3) {
        return feedForwardSwiGLU(x, W1, W2, W3, null);
    }

    /**
     * SwiGLU FFN; при непустом {@code cache} сохраняет промежуточные активации для backward без повторных matmul.
     */
    public static Tensor feedForwardSwiGLU(
            Tensor x, Tensor W1, Tensor W2, Tensor W3, BlockActivationCache cache) {
        Objects.requireNonNull(x, "x cannot be null");
        Objects.requireNonNull(W1, "W1 cannot be null");
        Objects.requireNonNull(W2, "W2 cannot be null");
        Objects.requireNonNull(W3, "W3 cannot be null");
        int[] xShape = x.getShape();
        if (xShape.length != 3) {
            throw new IllegalArgumentException("x must be 3D [batch, seq_len, d_model]");
        }
        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];

        validateFFNWeights(dModel, W1, W2, "feedForwardSwiGLU");
        int[] w3s = W3.getShape();
        if (w3s.length != 2 || w3s[0] != dModel || w3s[1] != W1.getShape()[1]) {
            throw new IllegalArgumentException(
                    "W3 must be [d_model, d_intermediate], got " + Arrays.toString(w3s));
        }

        int dIntermediate = W1.getShape()[1];
        Tensor gpuResult = tryFeedForwardSwiGLUGpu(x, W1, W2, W3, cache, batch, seqLen, dModel, dIntermediate);
        if (gpuResult != null) {
            return gpuResult;
        }
        Tensor xFlat = Tensor.wrap(x.internalBuffer(), new int[]{batch * seqLen, dModel});
        Tensor h1Flat = new Tensor(new int[]{batch * seqLen, dIntermediate});
        matmulInto(xFlat, W1, h1Flat);
        Tensor h1 = Tensor.wrap(h1Flat.internalBuffer(), new int[]{batch, seqLen, dIntermediate});
        Tensor gateFlat = new Tensor(new int[]{batch * seqLen, dIntermediate});
        matmulInto(xFlat, W3, gateFlat);
        Tensor gate = Tensor.wrap(gateFlat.internalBuffer(), new int[]{batch, seqLen, dIntermediate});

        Tensor sig = sigmoid(gate);
        Tensor gateSwish = multiply(gate, sig);
        Tensor hActivated = multiply(h1, gateSwish);

        if (cache != null) {
            boolean fp16 = cache.fp16ForFusedGpuBackwardConsumptionSlots();
            cache.ffnH1.store(h1, fp16);
            cache.ffnGate.store(gate, fp16);
            cache.ffnSig.store(sig, fp16);
            cache.ffnGateSwish.store(gateSwish, fp16);
            cache.ffnHActivated.store(hActivated, fp16);
        }

        Tensor outFlat = new Tensor(new int[]{batch * seqLen, dModel});
        matmulInto(
                Tensor.wrap(hActivated.internalBuffer(), new int[]{batch * seqLen, dIntermediate}),
                W2,
                outFlat);
        return Tensor.wrap(outFlat.internalBuffer(), xShape);
    }

    public static FfnForwardResult tryFusedNormResidualSwiGLUForwardGpu(
            Tensor xRes1, Tensor norm2, Tensor W1, Tensor W2, Tensor W3, BlockActivationCache cache) {
        int[] xShape = xRes1.getShape();
        if (xShape.length != 3) {
            return null;
        }
        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];
        validateFFNWeights(dModel, W1, W2, "tryFusedNormResidualSwiGLUForwardGpu");
        int[] w3s = W3.getShape();
        if (w3s.length != 2 || w3s[0] != dModel || w3s[1] != W1.getShape()[1]) {
            return null;
        }
        int dInt = W1.getShape()[1];
        int rows = batch * seqLen;
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dInt)
                || !TensorOpsGPU.shouldUseGpuElementwise(rows * dInt)
                || !TensorOpsGPU.shouldUseGpuElementwise(rows * dModel)) {
            return null;
        }

        GpuForwardBlockWorkspace ws = GpuForwardBlockWorkspace.local();
        boolean needCache = cache != null;
        boolean fp16 = needCache && cache.fp16ForFusedGpuBackwardConsumptionSlots();
        Tensor xNorm2 = null;
        Tensor ffnOut = null;
        Tensor out;
        synchronized (ws.exclusiveUseLock()) {
            ws.ensureFfnNormResidual(rows, dModel, dInt);

            ws.getXRes1().copyFrom(xRes1.internalBuffer(), 0, rows * dModel);
            ws.getNormGamma().copyFrom(norm2.internalBuffer(), 0, dModel);
            ws.getW1().copyFrom(W1.internalBuffer(), 0, dModel * dInt);
            ws.getW2().copyFrom(W2.internalBuffer(), 0, dInt * dModel);
            ws.getW3().copyFrom(W3.internalBuffer(), 0, dModel * dInt);

            rmsNormThenFfnW1W3ProjectionsGpu(
                    ws.getXRes1(),
                    ws.getNormGamma(),
                    TensorOpsGPU.rmsNormEps(),
                    ws.getXNorm2(),
                    ws.getW1(),
                    ws.getW3(),
                    ws.getH1(),
                    ws.getGate(),
                    rows,
                    dModel,
                    dInt);
            TensorOpsGPU.sigmoidGpuDevice(ws.getGate(), ws.getSig(), rows * dInt);
            TensorOpsGPU.multiplyGpuDevice(ws.getGate(), ws.getSig(), ws.getGateSwish(), rows * dInt);
            TensorOpsGPU.multiplyGpuDevice(ws.getH1(), ws.getGateSwish(), ws.getHAct(), rows * dInt);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getHAct(), ws.getW2(), ws.getFfnOut(), rows, dInt, dModel, false, false);
            ws.getOut().copyFromDevice(ws.getXRes1(), rows * dModel);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getOut(), ws.getFfnOut(), rows * dModel);

            ws.getOut().copyTo(ws.getHostOut(), 0, rows * dModel);
            out = Tensor.wrap(Arrays.copyOf(ws.getHostOut(), rows * dModel), xShape);

            if (needCache) {
                Tensor xNorm2Buf = cache.xNorm2.ensureTensor(xShape);
                Tensor ffnOutBuf = cache.ffnOut.ensureTensor(xShape);
                Tensor ffnH1Buf = cache.ffnH1.ensureTensor(new int[]{batch, seqLen, dInt});
                Tensor ffnGateBuf = cache.ffnGate.ensureTensor(new int[]{batch, seqLen, dInt});

                ws.getXNorm2().copyTo(xNorm2Buf.internalBuffer(), 0, rows * dModel);
                ws.getFfnOut().copyTo(ffnOutBuf.internalBuffer(), 0, rows * dModel);
                ws.getH1().copyTo(ffnH1Buf.internalBuffer(), 0, rows * dInt);
                ws.getGate().copyTo(ffnGateBuf.internalBuffer(), 0, rows * dInt);
                cache.ffnSig.clear();
                cache.ffnGateSwish.clear();
                cache.ffnHActivated.clear();

                cache.xNorm2.finalizeAfterWrite(fp16);
                cache.ffnOut.finalizeAfterWrite(fp16);
                cache.ffnH1.finalizeAfterWrite(fp16);
                cache.ffnGate.finalizeAfterWrite(fp16);

                xNorm2 = cache.xNorm2.getTensor();
                ffnOut = cache.ffnOut.getTensor();
            }
        }
        return new FfnForwardResult(xNorm2, ffnOut, out);
    }

    /**
     * Как {@link #tryFusedNormResidualSwiGLUForwardGpu}, но веса FFN берутся из {@link GpuFfnResidentBuffers} без
     * копирования с хоста в thread-local workspace (активации по-прежнему H2D из {@code xRes1}).
     */
    public static FfnForwardResult tryFusedNormResidualSwiGLUForwardGpuResident(
            Tensor xRes1, GpuFfnResidentBuffers resident, BlockActivationCache cache) {
        return tryFusedNormResidualSwiGLUForwardGpuResident(xRes1, resident, cache, null);
    }

    /**
     * Как {@link #tryFusedNormResidualSwiGLUForwardGpuResident(Tensor, GpuFfnResidentBuffers, BlockActivationCache)}:
     * ровно один из {@code cache} / {@code devCache} не {@code null} при необходимости записи активаций для backward.
     */
    public static FfnForwardResult tryFusedNormResidualSwiGLUForwardGpuResident(
            Tensor xRes1, GpuFfnResidentBuffers resident, BlockActivationCache cache, BlockActivationCacheDevice devCache) {
        Objects.requireNonNull(resident, "resident");
        if (cache != null && devCache != null) {
            throw new IllegalArgumentException("pass either BlockActivationCache or BlockActivationCacheDevice, not both");
        }
        int[] xShape = xRes1.getShape();
        if (xShape.length != 3) {
            return null;
        }
        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];
        int rows = batch * seqLen;
        GpuFloatBuffer normGammaGpu = resident.normGamma();
        GpuFloatBuffer w1Gpu = resident.w1();
        GpuFloatBuffer w2Gpu = resident.w2();
        GpuFloatBuffer w3Gpu = resident.w3();
        if (normGammaGpu.numFloats() < dModel) {
            return null;
        }
        long needW1 = (long) dModel * (w1Gpu.numFloats() / dModel);
        if (w1Gpu.numFloats() % dModel != 0 || w1Gpu.numFloats() <= 0) {
            return null;
        }
        int dInt = (int) (w1Gpu.numFloats() / dModel);
        if (w2Gpu.numFloats() < (long) dInt * dModel || w3Gpu.numFloats() < (long) dModel * dInt) {
            return null;
        }
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dInt)
                || !TensorOpsGPU.shouldUseGpuElementwise(rows * dInt)
                || !TensorOpsGPU.shouldUseGpuElementwise(rows * dModel)) {
            return null;
        }

        GpuForwardBlockWorkspace ws = GpuForwardBlockWorkspace.local();
        boolean needHostCache = cache != null;
        boolean needDevCache = devCache != null;
        boolean needCache = needHostCache || needDevCache;
        boolean fp16 = needHostCache && cache.fp16ForFusedGpuBackwardConsumptionSlots();
        Tensor xNorm2 = null;
        Tensor ffnOut = null;
        Tensor out;
        synchronized (ws.exclusiveUseLock()) {
            ws.ensureFfnNormResidual(rows, dModel, dInt);

            ws.getXRes1().copyFrom(xRes1.internalBuffer(), 0, rows * dModel);

            rmsNormThenFfnW1W3ProjectionsGpu(
                    ws.getXRes1(),
                    normGammaGpu,
                    TensorOpsGPU.rmsNormEps(),
                    ws.getXNorm2(),
                    w1Gpu,
                    w3Gpu,
                    ws.getH1(),
                    ws.getGate(),
                    rows,
                    dModel,
                    dInt);
            TensorOpsGPU.sigmoidGpuDevice(ws.getGate(), ws.getSig(), rows * dInt);
            TensorOpsGPU.multiplyGpuDevice(ws.getGate(), ws.getSig(), ws.getGateSwish(), rows * dInt);
            TensorOpsGPU.multiplyGpuDevice(ws.getH1(), ws.getGateSwish(), ws.getHAct(), rows * dInt);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getHAct(), w2Gpu, ws.getFfnOut(), rows, dInt, dModel, false, false);
            ws.getOut().copyFromDevice(ws.getXRes1(), rows * dModel);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getOut(), ws.getFfnOut(), rows * dModel);

            ws.getOut().copyTo(ws.getHostOut(), 0, rows * dModel);
            out = Tensor.wrap(Arrays.copyOf(ws.getHostOut(), rows * dModel), xShape);

            if (needHostCache) {
                Tensor xNorm2Buf = cache.xNorm2.ensureTensor(xShape);
                Tensor ffnOutBuf = cache.ffnOut.ensureTensor(xShape);
                Tensor ffnH1Buf = cache.ffnH1.ensureTensor(new int[]{batch, seqLen, dInt});
                Tensor ffnGateBuf = cache.ffnGate.ensureTensor(new int[]{batch, seqLen, dInt});

                ws.getXNorm2().copyTo(xNorm2Buf.internalBuffer(), 0, rows * dModel);
                ws.getFfnOut().copyTo(ffnOutBuf.internalBuffer(), 0, rows * dModel);
                ws.getH1().copyTo(ffnH1Buf.internalBuffer(), 0, rows * dInt);
                ws.getGate().copyTo(ffnGateBuf.internalBuffer(), 0, rows * dInt);
                cache.ffnSig.clear();
                cache.ffnGateSwish.clear();
                cache.ffnHActivated.clear();

                cache.xNorm2.finalizeAfterWrite(fp16);
                cache.ffnOut.finalizeAfterWrite(fp16);
                cache.ffnH1.finalizeAfterWrite(fp16);
                cache.ffnGate.finalizeAfterWrite(fp16);

                xNorm2 = cache.xNorm2.getTensor();
                ffnOut = cache.ffnOut.getTensor();
            } else if (needDevCache) {
                int plane = rows * dModel;
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.X_NORM2, ws.getXNorm2(), plane);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.FFN_OUT, ws.getFfnOut(), plane);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.FFN_H1, ws.getH1(), rows * dInt);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.FFN_GATE, ws.getGate(), rows * dInt);
            }
        }
        return new FfnForwardResult(xNorm2, ffnOut, out);
    }

    private static Tensor tryFeedForwardSwiGLUGpu(
            Tensor x,
            Tensor W1,
            Tensor W2,
            Tensor W3,
            BlockActivationCache cache,
            int batch,
            int seqLen,
            int dModel,
            int dIntermediate) {
        int rows = batch * seqLen;
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dIntermediate)
                || !TensorOpsGPU.shouldUseGpuElementwise(rows * dIntermediate)) {
            return null;
        }

        GpuForwardBlockWorkspace ws = GpuForwardBlockWorkspace.local();
        boolean needCache = cache != null;
        boolean fp16 = needCache && cache.fp16ForFusedGpuBackwardConsumptionSlots();
        Tensor result;
        synchronized (ws.exclusiveUseLock()) {
            ws.ensureFfnNormResidual(rows, dModel, dIntermediate);

            ws.getXNorm2().copyFrom(x.internalBuffer(), 0, rows * dModel);
            ws.getW1().copyFrom(W1.internalBuffer(), 0, dModel * dIntermediate);
            ws.getW2().copyFrom(W2.internalBuffer(), 0, dIntermediate * dModel);
            ws.getW3().copyFrom(W3.internalBuffer(), 0, dModel * dIntermediate);

            TensorOpsGPU.matmulGpuDeviceFfnW1W3Projections(
                    ws.getXNorm2(), ws.getW1(), ws.getW3(), ws.getH1(), ws.getGate(), rows, dModel, dIntermediate);
            TensorOpsGPU.sigmoidGpuDevice(ws.getGate(), ws.getSig(), rows * dIntermediate);
            TensorOpsGPU.multiplyGpuDevice(ws.getGate(), ws.getSig(), ws.getGateSwish(), rows * dIntermediate);
            TensorOpsGPU.multiplyGpuDevice(ws.getH1(), ws.getGateSwish(), ws.getHAct(), rows * dIntermediate);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getHAct(), ws.getW2(), ws.getFfnOut(), rows, dIntermediate, dModel, false, false);

            ws.getFfnOut().copyTo(ws.getHostOut(), 0, rows * dModel);
            result = Tensor.wrap(Arrays.copyOf(ws.getHostOut(), rows * dModel), new int[]{batch, seqLen, dModel});
            if (needCache) {
                Tensor h1b = cache.ffnH1.ensureTensor(new int[]{batch, seqLen, dIntermediate});
                Tensor gateb = cache.ffnGate.ensureTensor(new int[]{batch, seqLen, dIntermediate});

                ws.getH1().copyTo(h1b.internalBuffer(), 0, rows * dIntermediate);
                ws.getGate().copyTo(gateb.internalBuffer(), 0, rows * dIntermediate);
                cache.ffnSig.clear();
                cache.ffnGateSwish.clear();
                cache.ffnHActivated.clear();

                cache.ffnH1.finalizeAfterWrite(fp16);
                cache.ffnGate.finalizeAfterWrite(fp16);
            }
        }
        return result;
    }

    /**
     * FFN с GELU: {@code GELU(x W₁) W₂}.
     *
     * @param x  [batch, seq_len, d_model]
     * @param W1 [d_model, d_intermediate]
     * @param W2 [d_intermediate, d_model]
     */
    public static Tensor feedForwardGELU(Tensor x, Tensor W1, Tensor W2) {
        int[] xShape = x.getShape();
        if (xShape.length != 3) {
            throw new IllegalArgumentException("x must be 3D [batch, seq_len, d_model]");
        }
        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];

        validateFFNWeights(dModel, W1, W2, "feedForwardGELU");
        int dIntermediate = W1.getShape()[1];

        Tensor h = new Tensor(new int[]{batch, seqLen, dIntermediate});
        for (int b = 0; b < batch; b++) {
            Tensor xSlice = sliceBatch(x, b);
            copyInto(h, matmul(xSlice, W1), b);
        }

        h = gelu(h);

        Tensor output = new Tensor(xShape);
        for (int b = 0; b < batch; b++) {
            Tensor hSlice = sliceBatch(h, b);
            copyInto(output, matmul(hSlice, W2), b);
        }

        return output;
    }

    private static void validateFFNWeights(int dModel, Tensor W1, Tensor W2, String ctx) {
        int[] w1 = W1.getShape();
        int[] w2 = W2.getShape();
        if (w1.length != 2 || w1[0] != dModel) {
            throw new IllegalArgumentException(
                    ctx + ": W1 must be [d_model, d_intermediate], got " + Arrays.toString(w1));
        }
        int dInt = w1[1];
        if (w2.length != 2 || w2[0] != dInt || w2[1] != dModel) {
            throw new IllegalArgumentException(
                    ctx + ": W2 must be [d_intermediate, d_model] with d_intermediate="
                            + dInt
                            + ", got "
                            + Arrays.toString(w2));
        }
    }

    public static float sum(Tensor a) {
        float[] data = a.internalBuffer();
        if (VECTOR_SUPPORTED && data.length >= VECTOR_LANE_COUNT) {
            FloatVector acc = FloatVector.zero(VECTOR_SPECIES);
            int i = 0;
            int bound = VECTOR_SPECIES.loopBound(data.length);
            for (; i < bound; i += VECTOR_LANE_COUNT) {
                acc = acc.add(FloatVector.fromArray(VECTOR_SPECIES, data, i));
            }
            float s = acc.reduceLanes(VectorOperators.ADD);
            for (; i < data.length; i++) {
                s += data[i];
            }
            return s;
        }
        float s = 0;
        for (float v : data) {
            s += v;
        }
        return s;
    }

    public static float mean(Tensor a) {
        float[] data = a.internalBuffer();
        return data.length == 0 ? 0f : sum(a) / data.length;
    }

    /** Случайная инициализация U(-scale, scale) для весов. */
    public static Tensor randomTensor(int[] shape, float scale) {
        Tensor t = new Tensor(shape);
        float[] data = t.internalBuffer();
        var rnd = ThreadLocalRandom.current();
        for (int i = 0; i < data.length; i++) {
            data[i] = (rnd.nextFloat() * 2f - 1f) * scale;
        }
        return t;
    }

    /** Все элементы {@code 1.0f} — типичная инициализация γ для RMSNorm/LayerNorm. */
    public static Tensor onesTensor(int[] shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.internalBuffer(), 1.0f);
        return t;
    }

    /**
     * LayerNorm: нормализация по последней оси.
     * {@code y = gamma * (x - mean) / sqrt(var + eps) + beta}, где mean/var — по последней размерности.
     */
    public static Tensor layerNorm(Tensor x, Tensor gamma, Tensor beta, float eps) {
        int[] shape = x.getShape();
        if (shape.length < 1) {
            throw new IllegalArgumentException("layerNorm requires at least 1D tensor");
        }
        int lastDim = shape[shape.length - 1];
        int outerSize = 1;
        for (int i = 0; i < shape.length - 1; i++) {
            outerSize *= shape[i];
        }
        layerNormValidateParams(gamma, beta, lastDim);

        Tensor result = new Tensor(shape);
        float[] src = x.internalBuffer();
        float[] dst = result.internalBuffer();
        float[] g = gamma.internalBuffer();
        float[] b = beta.internalBuffer();

        int n = outerSize * lastDim;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.layerNorm");
        TensorOpsGPU.layerNormGPU(src, g, b, dst, outerSize, lastDim, eps);
        return result;
    }

    /** Проверка форм gamma/beta для LayerNorm (тот же пакет, backward). */
    static void layerNormValidateParams(Tensor gamma, Tensor beta, int lastDim) {
        int[] gShape = gamma.getShape();
        int[] bShape = beta.getShape();
        if (gShape.length != 1 || gShape[0] != lastDim) {
            throw new IllegalArgumentException(
                    "gamma must have shape [" + lastDim + "], got " + Arrays.toString(gShape));
        }
        if (bShape.length != 1 || bShape[0] != lastDim) {
            throw new IllegalArgumentException(
                    "beta must have shape [" + lastDim + "], got " + Arrays.toString(bShape));
        }
    }

    /**
     * RMSNorm: {@code y_i = (x_i / RMS(x)) * gamma_i}, {@code RMS = sqrt(mean(x²) + eps)}.
     * Нормализация по последней оси (как в Llama).
     *
     * @param x     произвольная размерность, последняя — {@code d_model}
     * @param gamma [d_model]
     * @param eps   стабильность (в модели — обычно {@link TensorOpsGPU#rmsNormEps()})
     */
    public static Tensor rmsNorm(Tensor x, Tensor gamma, float eps) {
        int[] shape = x.getShape();
        if (shape.length < 1) {
            throw new IllegalArgumentException("rmsNorm requires at least 1D tensor");
        }
        int lastDim = shape[shape.length - 1];
        int outerSize = 1;
        for (int i = 0; i < shape.length - 1; i++) {
            outerSize *= shape[i];
        }
        int[] gShape = gamma.getShape();
        if (gShape.length != 1 || gShape[0] != lastDim) {
            throw new IllegalArgumentException(
                    "gamma must have shape [" + lastDim + "], got " + Arrays.toString(gShape));
        }

        Tensor result = new Tensor(shape);
        float[] src = x.internalBuffer();
        float[] dst = result.internalBuffer();
        float[] g = gamma.internalBuffer();

        int n = outerSize * lastDim;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.rmsNorm");
        TensorOpsGPU.rmsNormGPU(src, g, dst, outerSize, lastDim, eps, TensorOpsGPU.useFp16Matmul());
        return result;
    }

    /**
     * Один decoder-блок (pre-norm): RMSNorm → MHA (+опц. RoPE) → residual → RMSNorm → SwiGLU → residual.
     *
     * @param norm1, norm2 веса RMSNorm [d_model]
     */
    public static Tensor transformerBlock(
            Tensor x,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            Tensor W1,
            Tensor W2,
            Tensor W3,
            Tensor norm1,
            Tensor norm2,
            int numHeads,
            Tensor mask,
            boolean useRoPE) {
        int[] xShape = x.getShape();
        if (xShape.length != 3) {
            throw new IllegalArgumentException("x must be 3D [batch, seq_len, d_model]");
        }

        final float eps = TensorOpsGPU.rmsNormEps();

        Tensor xNorm1 = rmsNorm(x, norm1, eps);

        Tensor attnOut =
                useRoPE
                        ? multiHeadAttentionWithRoPE(xNorm1, Wq, Wk, Wv, Wo, numHeads, mask, true)
                        : multiHeadAttention(xNorm1, Wq, Wk, Wv, Wo, numHeads, mask);

        Tensor xResidual1 = add(x, attnOut);

        Tensor xNorm2 = rmsNorm(xResidual1, norm2, eps);

        Tensor ffnOut = feedForwardSwiGLU(xNorm2, W1, W2, W3);

        return add(xResidual1, ffnOut);
    }

    /**
     * Causal mask (верхний треугольник без диагонали = {@code -∞}): строка {@code i} не смотрит на
     * позиции {@code j > i}. Матрица [seqLen, seqLen].
     *
     * <p>Пример для seqLen=4: нижний треугольник и диагональ — 0, строго выше диагонали — -inf.
     *
     * <p>Один и тот же тензор может возвращаться при повторном запросе того же {@code seqLen}; не
     * изменять данные маски на месте.
     */
    public static Tensor createCausalMask(int seqLen) {
        if (seqLen <= 0) {
            throw new IllegalArgumentException("seqLen must be positive, got " + seqLen);
        }
        return CAUSAL_MASK_CACHE.computeIfAbsent(
                seqLen,
                k -> {
                    Tensor mask = new Tensor(new int[]{k, k});
                    float[] data = mask.internalBuffer();
                    Arrays.fill(data, 0.0f);
                    for (int i = 0; i < k; i++) {
                        for (int j = i + 1; j < k; j++) {
                            data[i * k + j] = Float.NEGATIVE_INFINITY;
                        }
                    }
                    return mask;
                });
    }

    /**
     * Scaled Dot-Product Attention с опциональной causal-маской.
     *
     * @param Q     [batch, seq_len, d_k]
     * @param K     [batch, seq_len, d_k]
     * @param V     [batch, seq_len, d_v]
     * @param mask  [seq_len, seq_len] или {@code null}
     * @param scale обычно 1/√d_k
     * @return [batch, seq_len, d_v]
     */
    public static Tensor scaledDotProductAttention(
            Tensor Q, Tensor K, Tensor V, Tensor mask, float scale) {
        return scaledDotProductAttentionWithWeights(Q, K, V, mask, scale).output;
    }

    static AttentionForwardResult scaledDotProductAttentionWithWeights(
            Tensor Q, Tensor K, Tensor V, Tensor mask, float scale) {
        int[] qShape = Q.getShape();
        if (qShape.length != 3) {
            throw new IllegalArgumentException("Q must be 3D [batch, seq_len, d_k]");
        }
        int batch = qShape[0];
        int seqLen = qShape[1];
        int dK = qShape[2];

        int[] kShape = K.getShape();
        int[] vShape = V.getShape();
        if (kShape.length != 3 || kShape[0] != batch || kShape[1] != seqLen || kShape[2] != dK) {
            throw new IllegalArgumentException(
                    "K must be [batch, seq_len, d_k] matching Q, got " + Arrays.toString(kShape));
        }
        if (vShape.length != 3 || vShape[0] != batch || vShape[1] != seqLen) {
            throw new IllegalArgumentException(
                    "V must be [batch, seq_len, d_v], got " + Arrays.toString(vShape));
        }
        int dV = vShape[2];

        if (mask != null) {
            int[] mShape = mask.getShape();
            if (mShape.length != 2 || mShape[0] != seqLen || mShape[1] != seqLen) {
                throw new IllegalArgumentException(
                        "mask must be [seq_len, seq_len], got " + Arrays.toString(mShape));
            }
        }

        TensorOpsGPU.requireCuda("TensorOps.scaledDotProductAttentionWithWeights");
        if (!TensorOpsGPU.shouldUseGpuMatmulBatched(batch, seqLen, seqLen, Math.max(dK, dV))) {
            throw new IllegalArgumentException(
                    "scaledDotProductAttention: ожидаются положительные batch, seqLen, dK, dV");
        }
        Tensor attentionWeights = new Tensor(new int[]{batch, seqLen, seqLen});
        Tensor output = new Tensor(new int[]{batch, seqLen, dV});
        TensorOpsGPU.scaledDotProductAttentionForwardFromHost(
                Q.internalBuffer(),
                K.internalBuffer(),
                V.internalBuffer(),
                mask != null ? mask.internalBuffer() : null,
                output.internalBuffer(),
                attentionWeights.internalBuffer(),
                batch,
                seqLen,
                dK,
                dV,
                scale,
                TensorOpsGPU.useFp16Matmul());
        return new AttentionForwardResult(output, attentionWeights);
    }

    /**
     * Scaled Dot-Product Attention без маски.
     * {@code Attention(Q, K, V) = softmax(Q @ Kᵀ / scale) @ V} (здесь {@code scale} обычно {@code 1/√d_k}).
     *
     * @param Q     [batch, seq_len, d_k]
     * @param K     [batch, seq_len, d_k]
     * @param V     [batch, seq_len, d_v]
     * @param scale коэффициент масштабирования скоров (часто 1/√d_k)
     * @return [batch, seq_len, d_v]
     */
    public static Tensor scaledDotProductAttention(Tensor Q, Tensor K, Tensor V, float scale) {
        return scaledDotProductAttention(Q, K, V, null, scale);
    }

    static Tensor matmulBatched3D(Tensor A, Tensor B) {
        int[] aShape = A.getShape();
        int[] bShape = B.getShape();
        if (aShape.length != 3 || bShape.length != 3) {
            throw new IllegalArgumentException("matmulBatched3D requires 3D tensors");
        }
        int batch = aShape[0];
        int m = aShape[1];
        int k = aShape[2];
        if (bShape[0] != batch || bShape[1] != k) {
            throw new IllegalArgumentException(
                    "matmulBatched3D incompatible shapes: "
                            + Arrays.toString(aShape)
                            + " x "
                            + Arrays.toString(bShape));
        }
        int n = bShape[2];
        Tensor result = new Tensor(new int[]{batch, m, n});
        TensorOpsGPU.requireCuda("TensorOps.matmulBatched3D");
        if (!TensorOpsGPU.shouldUseGpuMatmulBatched(batch, m, k, n)) {
            throw new IllegalArgumentException(
                    "matmulBatched3D: ожидаются положительные batch, m, k, n и доступная CUDA");
        }
        TensorOpsGPU.matmulBatchedGPUMaybeFp16(
                A.internalBuffer(), B.internalBuffer(), result.internalBuffer(), m, k, n, batch);
        return result;
    }

    /**
     * Один query-токен: Q [batch, 1, d_k], K,V [batch, seq_len, …] — softmax по полной длине K (KV-cache decode).
     */
    public static Tensor scaledDotProductAttentionQueryOne(
            Tensor Q, Tensor K, Tensor V, float scale) {
        int[] qShape = Q.getShape();
        if (qShape.length != 3 || qShape[1] != 1) {
            throw new IllegalArgumentException("Q must be [batch, 1, d_k]");
        }
        int batch = qShape[0];
        int dK = qShape[2];
        int[] kShape = K.getShape();
        int[] vShape = V.getShape();
        if (kShape.length != 3 || kShape[0] != batch || kShape[2] != dK) {
            throw new IllegalArgumentException("K must be [batch, seq_len, d_k] matching Q");
        }
        int seqLen = kShape[1];
        int dV = vShape[2];
        if (vShape.length != 3 || vShape[0] != batch || vShape[1] != seqLen) {
            throw new IllegalArgumentException("V must be [batch, seq_len, d_v] matching K");
        }

        Tensor kT = transpose2DLast(K);
        Tensor scores;
        if (batch == 1) {
            Tensor q2d = Tensor.wrap(Q.internalBuffer(), new int[]{1, dK});
            Tensor k2d = Tensor.wrap(kT.internalBuffer(), new int[]{dK, seqLen});
            Tensor score2d = matmul(q2d, k2d);
            scores = Tensor.wrap(score2d.internalBuffer(), new int[]{1, 1, seqLen});
        } else {
            scores = new Tensor(new int[]{batch, 1, seqLen});
            for (int b = 0; b < batch; b++) {
                Tensor qSlice = sliceBatch(Q, b);
                Tensor kSlice = sliceBatch(kT, b);
                Tensor scoreSlice = matmul(qSlice, kSlice);
                copyInto(scores, scoreSlice, b);
            }
        }
        Tensor scaled = multiplyScalar(scores, scale);
        Tensor attentionWeights = softmaxLastDim(scaled);
        Tensor output = new Tensor(new int[]{batch, 1, dV});
        if (batch == 1) {
            Tensor aw2d = Tensor.wrap(attentionWeights.internalBuffer(), new int[]{1, seqLen});
            Tensor v2d = Tensor.wrap(V.internalBuffer(), new int[]{seqLen, dV});
            Tensor out2d = matmul(aw2d, v2d);
            System.arraycopy(out2d.internalBuffer(), 0, output.internalBuffer(), 0, dV);
        } else {
            for (int b = 0; b < batch; b++) {
                Tensor awSlice = sliceBatch(attentionWeights, b);
                Tensor vSlice = sliceBatch(V, b);
                Tensor outSlice = matmul(awSlice, vSlice);
                copyInto(output, outSlice, b);
            }
        }
        return output;
    }

    /**
     * Multi-Head Attention.
     *
     * @param x        [batch, seq_len, d_model]
     * @param Wq, Wk, Wv, Wo проекции [d_model, d_model]
     * @param numHeads число голов
     * @param mask     causal [seq_len, seq_len] или {@code null}
     * @return [batch, seq_len, d_model]
     */
    public static Tensor multiHeadAttention(
            Tensor x,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            Tensor mask) {
        int[] xShape = x.getShape();
        if (xShape.length != 3) {
            throw new IllegalArgumentException("x must be 3D [batch, seq_len, d_model]");
        }

        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("d_model must be divisible by num_heads");
        }

        validateSquareProjection(Wq, dModel, "Wq");
        validateSquareProjection(Wk, dModel, "Wk");
        validateSquareProjection(Wv, dModel, "Wv");
        validateSquareProjection(Wo, dModel, "Wo");

        int dHead = dModel / numHeads;
        float scale = 1.0f / (float) Math.sqrt(dHead);

        Tensor xFlat = Tensor.wrap(x.internalBuffer(), new int[]{batch * seqLen, dModel});
        Tensor Q = Tensor.wrap(matmul(xFlat, Wq).internalBuffer(), xShape);
        Tensor K = Tensor.wrap(matmul(xFlat, Wk).internalBuffer(), xShape);
        Tensor V = Tensor.wrap(matmul(xFlat, Wv).internalBuffer(), xShape);

        Tensor qHeads = splitHeads(Q, numHeads);
        Tensor kHeads = splitHeads(K, numHeads);
        Tensor vHeads = splitHeads(V, numHeads);
        Tensor qFlat3 = Tensor.wrap(qHeads.internalBuffer(), new int[]{batch * numHeads, seqLen, dHead});
        Tensor kFlat3 = Tensor.wrap(kHeads.internalBuffer(), new int[]{batch * numHeads, seqLen, dHead});
        Tensor vFlat3 = Tensor.wrap(vHeads.internalBuffer(), new int[]{batch * numHeads, seqLen, dHead});

        Tensor attnFlat3 = scaledDotProductAttention(qFlat3, kFlat3, vFlat3, mask, scale);
        Tensor outputHeads = Tensor.wrap(attnFlat3.internalBuffer(), new int[]{batch, numHeads, seqLen, dHead});

        Tensor concat = concatHeads(outputHeads, numHeads);
        Tensor concatFlat = Tensor.wrap(concat.internalBuffer(), new int[]{batch * seqLen, dModel});
        return Tensor.wrap(matmul(concatFlat, Wo).internalBuffer(), xShape);
    }

    /**
     * RoPE (Rotary Position Embedding): поворот пар компонентов вдоль последней оси.
     * Форма как у {@link #splitHeads}: {@code [batch, num_heads, seq_len, d_head]}.
     *
     * @param x         тензор голов Q или K
     * @param positions позиции по длине последовательности ({@code seq_len} значений) или {@code null}
     *                  (тогда {@code 0, 1, …, seq_len-1})
     */
    public static Tensor applyRoPE(Tensor x, int[] positions) {
        int[] shape = x.getShape();
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                    "x must be 4D [batch, num_heads, seq_len, d_head]");
        }

        int batch = shape[0];
        int numHeads = shape[1];
        int seqLen = shape[2];
        int dHead = shape[3];

        if (dHead % 2 != 0) {
            throw new IllegalArgumentException("d_head must be even for RoPE");
        }

        if (positions != null && positions.length < seqLen) {
            throw new IllegalArgumentException(
                    "positions length " + positions.length + " < seq_len " + seqLen);
        }

        int n = batch * numHeads * seqLen * dHead;
        if (n <= 0) {
            return new Tensor(shape);
        }
        TensorOpsGPU.requireCuda("TensorOps.applyRoPE");
        Tensor result = new Tensor(shape);
        TensorOpsGPU.applyRoPE4DGPU(
                x.internalBuffer(),
                result.internalBuffer(),
                batch,
                numHeads,
                seqLen,
                dHead,
                positions,
                0);
        return result;
    }

    /**
     * Multi-Head Attention с опциональным RoPE на Q и K (после разбиения на головы).
     *
     * @param useRoPE если {@code false} — эквивалентно {@link #multiHeadAttention(Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor)}
     */
    public static Tensor multiHeadAttentionWithRoPE(
            Tensor x,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            Tensor mask,
            boolean useRoPE) {
        return multiHeadAttentionWithRoPE(x, Wq, Wk, Wv, Wo, numHeads, mask, useRoPE, null);
    }

    public static Tensor multiHeadAttentionWithRoPE(
            Tensor x,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            Tensor mask,
            boolean useRoPE,
            BlockActivationCache cache) {
        if (!useRoPE) {
            return multiHeadAttention(x, Wq, Wk, Wv, Wo, numHeads, mask);
        }

        int[] xShape = x.getShape();
        if (xShape.length != 3) {
            throw new IllegalArgumentException("x must be 3D [batch, seq_len, d_model]");
        }

        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("d_model must be divisible by num_heads");
        }

        validateSquareProjection(Wq, dModel, "Wq");
        validateSquareProjection(Wk, dModel, "Wk");
        validateSquareProjection(Wv, dModel, "Wv");
        validateSquareProjection(Wo, dModel, "Wo");

        int dHead = dModel / numHeads;
        float scale = 1.0f / (float) Math.sqrt(dHead);

        Tensor xFlat = Tensor.wrap(x.internalBuffer(), new int[]{batch * seqLen, dModel});
        Tensor Q = Tensor.wrap(matmul(xFlat, Wq).internalBuffer(), xShape);
        Tensor K = Tensor.wrap(matmul(xFlat, Wk).internalBuffer(), xShape);
        Tensor V = Tensor.wrap(matmul(xFlat, Wv).internalBuffer(), xShape);

        Tensor qHeads = applyRoPE(splitHeads(Q, numHeads), null);
        Tensor kHeads = applyRoPE(splitHeads(K, numHeads), null);
        Tensor vHeads = splitHeads(V, numHeads);
        Tensor qFlat3 = Tensor.wrap(qHeads.internalBuffer(), new int[]{batch * numHeads, seqLen, dHead});
        Tensor kFlat3 = Tensor.wrap(kHeads.internalBuffer(), new int[]{batch * numHeads, seqLen, dHead});
        Tensor vFlat3 = Tensor.wrap(vHeads.internalBuffer(), new int[]{batch * numHeads, seqLen, dHead});

        AttentionForwardResult attn = scaledDotProductAttentionWithWeights(qFlat3, kFlat3, vFlat3, mask, scale);
        Tensor attnFlat3 = attn.output;
        Tensor outputHeads = Tensor.wrap(attnFlat3.internalBuffer(), new int[]{batch, numHeads, seqLen, dHead});

        Tensor concat = concatHeads(outputHeads, numHeads);
        if (cache != null) {
            boolean fp16 = cache.fp16ForFusedGpuBackwardConsumptionSlots();
            cache.attnQHeads.store(qHeads, fp16);
            cache.attnKHeads.store(kHeads, fp16);
            cache.attnVHeads.store(vHeads, fp16);
            cache.attnProbs.store(attn.attentionWeights, fp16);
            cache.attnConcat.store(concat, fp16);
        }
        Tensor concatFlat = Tensor.wrap(concat.internalBuffer(), new int[]{batch * seqLen, dModel});
        return Tensor.wrap(matmul(concatFlat, Wo).internalBuffer(), xShape);
    }

    /**
     * MHA с RoPE (или без): RMSNorm и проекции Q/K/V на VRAM без H2D весов; split голов и (при {@code useRoPE}) RoPE —
     * на GPU; attention forward на CUDA по указателям (без лишнего H2D Q/K/V). Головы на CPU — только при {@code
     * cache != null} (для {@link BlockActivationCache}). Concat и {@code Wo} на device; итог — host-тензор.
     * При {@code cache != null} заполняет те же поля кэша, что и
     * {@link #multiHeadAttentionWithRoPE(Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor, boolean, BlockActivationCache)}.
     *
     * @return {@code null}, если CUDA недоступна или размеры не подходят
     */
    public static AttnGpuResidentResult tryMultiHeadAttentionWithRoPEGpuResident(
            Tensor xIn,
            float eps,
            GpuAttnResidentBuffers resident,
            int numHeads,
            Tensor mask,
            boolean useRoPE,
            BlockActivationCache cache) {
        return tryMultiHeadAttentionWithRoPEGpuResident(xIn, eps, resident, numHeads, mask, useRoPE, cache, null);
    }

    /**
     * Как {@link #tryMultiHeadAttentionWithRoPEGpuResident(Tensor, float, GpuAttnResidentBuffers, int, Tensor, boolean, BlockActivationCache)}:
     * при {@code devCache != null} активации attention пишутся D2D в VRAM без host-слотов Q/K/V/concat/probs.
     */
    public static AttnGpuResidentResult tryMultiHeadAttentionWithRoPEGpuResident(
            Tensor xIn,
            float eps,
            GpuAttnResidentBuffers resident,
            int numHeads,
            Tensor mask,
            boolean useRoPE,
            BlockActivationCache cache,
            BlockActivationCacheDevice devCache) {
        Objects.requireNonNull(resident, "resident");
        if (cache != null && devCache != null) {
            throw new IllegalArgumentException("pass either BlockActivationCache or BlockActivationCacheDevice, not both");
        }
        int[] xShape = xIn.getShape();
        if (xShape.length != 3) {
            return null;
        }
        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];
        if (dModel % numHeads != 0) {
            return null;
        }
        GpuFloatBuffer normG = resident.normGamma();
        GpuFloatBuffer wqB = resident.wq();
        GpuFloatBuffer wkB = resident.wk();
        GpuFloatBuffer wvB = resident.wv();
        GpuFloatBuffer woB = resident.wo();
        long d2 = (long) dModel * dModel;
        if (normG.numFloats() < dModel
                || wqB.numFloats() < d2
                || wkB.numFloats() < d2
                || wvB.numFloats() < d2
                || woB.numFloats() < d2) {
            return null;
        }
        if (mask != null) {
            int[] mShape = mask.getShape();
            if (mShape.length != 2 || mShape[0] != seqLen || mShape[1] != seqLen) {
                return null;
            }
        }
        int dHead = dModel / numHeads;
        float scale = 1.0f / (float) Math.sqrt(dHead);
        int rows = batch * seqLen;
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dModel)) {
            return null;
        }

        GpuAttentionResidentWorkspace ws = GpuAttentionResidentWorkspace.local();
        boolean fillHost = cache != null;
        boolean fillDev = devCache != null;
        boolean fillCache = fillHost || fillDev;
        boolean copyHeadsToHost = fillHost;
        int headFloats = batch * numHeads * seqLen * dHead;
        int probFloats = batch * numHeads * seqLen * seqLen;
        Tensor qHeads = copyHeadsToHost ? new Tensor(new int[] {batch, numHeads, seqLen, dHead}) : null;
        Tensor kHeads = copyHeadsToHost ? new Tensor(new int[] {batch, numHeads, seqLen, dHead}) : null;
        Tensor vHeads = copyHeadsToHost ? new Tensor(new int[] {batch, numHeads, seqLen, dHead}) : null;
        Tensor attnProb = fillCache ? new Tensor(new int[] {batch * numHeads, seqLen, seqLen}) : null;
        Tensor xNorm1 = null;
        Tensor concatForCache = null;
        Tensor out;
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            ws.getXIn().copyFrom(xIn.internalBuffer(), 0, rows * dModel);
            TensorOpsGPU.rmsNormGpuDevice(ws.getXIn(), normG, eps, ws.getXNorm(), rows, dModel);
            TensorOpsGPU.matmulGpuDeviceQkvProjections(
                    ws.getXNorm(), wqB, wkB, wvB, ws.getQ(), ws.getK(), ws.getV(), rows, dModel);
            if (fillHost) {
                xNorm1 = new Tensor(xShape);
                ws.getXNorm().copyTo(xNorm1.internalBuffer(), 0, rows * dModel);
            }
            downloadQkvHeadsAfterSplitRoPeGpu(
                    ws,
                    batch,
                    seqLen,
                    dModel,
                    numHeads,
                    dHead,
                    useRoPE,
                    0,
                    qHeads,
                    kHeads,
                    vHeads,
                    null,
                    null,
                    0,
                    0,
                    copyHeadsToHost);
            if (fillDev) {
                int plane = rows * dModel;
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_Q_HEADS, ws.getConcatFlat(), headFloats);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_K_HEADS, ws.getAttnOut(), headFloats);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_V_HEADS, ws.getQ(), headFloats);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.X_NORM1, ws.getXNorm(), plane);
            }
            int bAttn = batch * numHeads;
            TensorOpsGPU.scaledDotProductAttentionForwardGpuDevice(
                    ws.getConcatFlat(),
                    ws.getAttnOut(),
                    ws.getQ(),
                    mask != null ? mask.internalBuffer() : null,
                    ws.getK(),
                    attnProb != null ? attnProb.internalBuffer() : null,
                    bAttn,
                    seqLen,
                    dHead,
                    dHead,
                    scale,
                    TensorOpsGPU.useFp16Matmul());
            if (fillDev) {
                devCache.copySlotFromHostFloat(
                        BlockActivationCacheDevice.SlotId.ATTN_PROBS, attnProb.internalBuffer(), 0, probFloats);
            }
            TensorOpsGPU.concatHeadsGpuDevice(ws.getK(), ws.getConcatFlat(), batch, numHeads, seqLen, dHead);
            if (fillHost) {
                concatForCache = new Tensor(new int[] {batch, seqLen, dModel});
                ws.getConcatFlat().copyTo(concatForCache.internalBuffer(), 0, rows * dModel);
            } else if (fillDev) {
                devCache.copySlotFromDeviceFloat(
                        BlockActivationCacheDevice.SlotId.ATTN_CONCAT, ws.getConcatFlat(), rows * dModel);
            }
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getConcatFlat(), woB, ws.getAttnOut(), rows, dModel, dModel, false, false);
            if (fillDev) {
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_OUT, ws.getAttnOut(), rows * dModel);
            }
            ws.getAttnOut().copyTo(ws.getHostOut(), 0, rows * dModel);
            out = Tensor.wrap(Arrays.copyOf(ws.getHostOut(), rows * dModel), xShape);
        }
        if (fillHost) {
            boolean fp16 = cache.fp16ForFusedGpuBackwardConsumptionSlots();
            cache.attnQHeads.store(Objects.requireNonNull(qHeads), fp16);
            cache.attnKHeads.store(Objects.requireNonNull(kHeads), fp16);
            cache.attnVHeads.store(Objects.requireNonNull(vHeads), fp16);
            cache.attnProbs.store(Objects.requireNonNull(attnProb), fp16);
            cache.attnConcat.store(Objects.requireNonNull(concatForCache), fp16);
        }
        return new AttnGpuResidentResult(out, xNorm1);
    }

    /**
     * Resident MHA + остаток: входной planar-буфер на GPU {@code [batch*seqLen*dModel]} → {@code xRes1Out} = x + Wo·MHA
     * на GPU без D2H результата. При {@code devCache != null} пишет активации для backward; при {@code null} —
     * только инференс (без промежуточного сохранения Q/K/V/softmax и без H2D probs).
     *
     * @return {@code false} если CUDA/размеры не подходят
     */
    public static boolean multiHeadAttentionResidentDeviceToDevice(
            GpuFloatBuffer xInDevice,
            GpuFloatBuffer xRes1Out,
            float eps,
            GpuAttnResidentBuffers resident,
            int batch,
            int seqLen,
            int dModel,
            int numHeads,
            Tensor mask,
            boolean useRoPE,
            BlockActivationCacheDevice devCache) {
        Objects.requireNonNull(xInDevice, "xInDevice");
        Objects.requireNonNull(xRes1Out, "xRes1Out");
        Objects.requireNonNull(resident, "resident");
        if (dModel % numHeads != 0) {
            return false;
        }
        int rows = Math.multiplyExact(batch, seqLen);
        int plane = Math.multiplyExact(rows, dModel);
        if (xInDevice.numFloats() < plane || xRes1Out.numFloats() < plane) {
            return false;
        }
        GpuFloatBuffer normG = resident.normGamma();
        GpuFloatBuffer wqB = resident.wq();
        GpuFloatBuffer wkB = resident.wk();
        GpuFloatBuffer wvB = resident.wv();
        GpuFloatBuffer woB = resident.wo();
        long d2 = (long) dModel * dModel;
        if (normG.numFloats() < dModel
                || wqB.numFloats() < d2
                || wkB.numFloats() < d2
                || wvB.numFloats() < d2
                || woB.numFloats() < d2) {
            return false;
        }
        if (mask != null) {
            int[] mShape = mask.getShape();
            if (mShape.length != 2 || mShape[0] != seqLen || mShape[1] != seqLen) {
                return false;
            }
        }
        int dHead = dModel / numHeads;
        float scale = 1.0f / (float) Math.sqrt(dHead);
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dModel)) {
            return false;
        }
        int headFloats = Math.multiplyExact(Math.multiplyExact(Math.multiplyExact(batch, numHeads), seqLen), dHead);
        boolean useFlash = TensorOpsGPU.FLASH_ATTENTION && dHead == 16;
        int probFloats = useFlash ? 0 : Math.multiplyExact(Math.multiplyExact(batch * numHeads, seqLen), seqLen);
        GpuFloatBuffer probsWrite = (!useFlash && devCache != null) ? devCache.attnProbsWriteBufferAsFloat(probFloats) : null;
        GpuAttentionResidentWorkspace ws = GpuAttentionResidentWorkspace.local();
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            ws.getXIn().copyFromDevice(xInDevice, plane);
            TensorOpsGPU.rmsNormGpuDevice(ws.getXIn(), normG, eps, ws.getXNorm(), rows, dModel);
            TensorOpsGPU.matmulGpuDeviceQkvProjections(
                    ws.getXNorm(), wqB, wkB, wvB, ws.getQ(), ws.getK(), ws.getV(), rows, dModel);
            downloadQkvHeadsAfterSplitRoPeGpu(
                    ws,
                    batch,
                    seqLen,
                    dModel,
                    numHeads,
                    dHead,
                    useRoPE,
                    0,
                    null,
                    null,
                    null,
                    null,
                    null,
                    0,
                    0,
                    false);
            if (devCache != null) {
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.X_IN, xInDevice, plane);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_Q_HEADS, ws.getConcatFlat(), headFloats);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_K_HEADS, ws.getAttnOut(), headFloats);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_V_HEADS, ws.getQ(), headFloats);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.X_NORM1, ws.getXNorm(), plane);
            }
            int bAttn = batch * numHeads;
            if (useFlash) {
                // Q=[bAttn,S,Dh]=getConcatFlat, K=getAttnOut, V=getQ, O=getK (reuse)
                GpuFloatBuffer lseDev = devCache != null ? devCache.attnLseBuffer() : null;
                if (lseDev == null) {
                    // Inference path: no cache → allocate temporary LSE buffer
                    lseDev = GpuFloatBuffer.allocate((long) bAttn * seqLen);
                }
                TensorOpsGPU.flashAttentionForwardGpuDeviceResident(
                        ws.getConcatFlat(), ws.getAttnOut(), ws.getQ(),
                        ws.getK(),  // O output (head-wise) → stored here temporarily
                        lseDev,
                        bAttn, seqLen, dHead, scale);
                if (devCache != null) {
                    // Save O_heads (before concatHeads overwrites ws.getK()) for backward D computation
                    devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_OUT_HEADS, ws.getK(), headFloats);
                }
            } else {
                GpuFloatBuffer maskDev = null;
                if (mask != null) {
                    ws.ensureMask(seqLen);
                    maskDev = ws.getMaskDev();
                    if (!isDecoderGraphCaptureSkipAttentionMaskHostUpload()) {
                        maskDev.copyFrom(mask.internalBuffer(), 0, Math.multiplyExact(seqLen, seqLen));
                    }
                }
                TensorOpsGPU.scaledDotProductAttentionForwardGpuDeviceResident(
                        ws.getConcatFlat(),
                        ws.getAttnOut(),
                        ws.getQ(),
                        ws.getK(),
                        maskDev,
                        probsWrite,
                        bAttn,
                        seqLen,
                        dHead,
                        dHead,
                        scale,
                        TensorOpsGPU.useFp16Matmul());
                if (devCache != null) {
                    devCache.storeAttnProbsFromFloatStagingIfFp16(probsWrite, probFloats);
                }
            }
            TensorOpsGPU.concatHeadsGpuDevice(ws.getK(), ws.getConcatFlat(), batch, numHeads, seqLen, dHead);
            if (devCache != null) {
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_CONCAT, ws.getConcatFlat(), plane);
            }
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getConcatFlat(), woB, ws.getAttnOut(), rows, dModel, dModel, false, false);
            if (devCache != null) {
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_OUT, ws.getAttnOut(), plane);
            }
            xRes1Out.copyFromDevice(xInDevice, plane);
            TensorOpsGPU.accumulateAddGpuDevice(xRes1Out, ws.getAttnOut(), plane);
            if (devCache != null) {
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.X_RES1, xRes1Out, plane);
            }
        }
        return true;
    }

    /**
     * Fused второй RMSNorm + SwiGLU FFN + residual на GPU: {@code xRes1Device} → {@code blockOutDevice}, без D2H
     * активаций. При {@code devCache != null} пишет слоты для backward; при {@code null} — только инференс.
     *
     * @return {@code false} если размеры/CUDA не подходят
     */
    public static boolean fusedNormResidualSwiGLUResidentDeviceToDevice(
            GpuFloatBuffer xRes1Device,
            GpuFloatBuffer blockOutDevice,
            GpuFfnResidentBuffers resident,
            int batch,
            int seqLen,
            int dModel,
            BlockActivationCacheDevice devCache) {
        Objects.requireNonNull(xRes1Device, "xRes1Device");
        Objects.requireNonNull(blockOutDevice, "blockOutDevice");
        Objects.requireNonNull(resident, "resident");
        int rows = Math.multiplyExact(batch, seqLen);
        int plane = Math.multiplyExact(rows, dModel);
        if (xRes1Device.numFloats() < plane || blockOutDevice.numFloats() < plane) {
            return false;
        }
        GpuFloatBuffer normGammaGpu = resident.normGamma();
        GpuFloatBuffer w1Gpu = resident.w1();
        GpuFloatBuffer w2Gpu = resident.w2();
        GpuFloatBuffer w3Gpu = resident.w3();
        if (normGammaGpu.numFloats() < dModel) {
            return false;
        }
        if (w1Gpu.numFloats() % dModel != 0 || w1Gpu.numFloats() <= 0) {
            return false;
        }
        int dInt = (int) (w1Gpu.numFloats() / dModel);
        if (w2Gpu.numFloats() < (long) dInt * dModel || w3Gpu.numFloats() < (long) dModel * dInt) {
            return false;
        }
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dInt)
                || !TensorOpsGPU.shouldUseGpuElementwise(rows * dInt)
                || !TensorOpsGPU.shouldUseGpuElementwise(rows * dModel)) {
            return false;
        }
        int ffnMid = Math.multiplyExact(rows, dInt);
        GpuForwardBlockWorkspace ws = GpuForwardBlockWorkspace.local();
        synchronized (ws.exclusiveUseLock()) {
            ws.ensureFfnNormResidual(rows, dModel, dInt);
            ws.getXRes1().copyFromDevice(xRes1Device, plane);
            rmsNormThenFfnW1W3ProjectionsGpu(
                    ws.getXRes1(),
                    normGammaGpu,
                    TensorOpsGPU.rmsNormEps(),
                    ws.getXNorm2(),
                    w1Gpu,
                    w3Gpu,
                    ws.getH1(),
                    ws.getGate(),
                    rows,
                    dModel,
                    dInt);
            TensorOpsGPU.sigmoidGpuDevice(ws.getGate(), ws.getSig(), rows * dInt);
            TensorOpsGPU.multiplyGpuDevice(ws.getGate(), ws.getSig(), ws.getGateSwish(), rows * dInt);
            TensorOpsGPU.multiplyGpuDevice(ws.getH1(), ws.getGateSwish(), ws.getHAct(), rows * dInt);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getHAct(), w2Gpu, ws.getFfnOut(), rows, dInt, dModel, false, false);
            ws.getOut().copyFromDevice(ws.getXRes1(), plane);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getOut(), ws.getFfnOut(), plane);
            if (devCache != null) {
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.X_NORM2, ws.getXNorm2(), plane);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.FFN_OUT, ws.getFfnOut(), plane);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.FFN_H1, ws.getH1(), ffnMid);
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.FFN_GATE, ws.getGate(), ffnMid);
            }
            blockOutDevice.copyFromDevice(ws.getOut(), plane);
            if (devCache != null) {
                devCache.copySlotFromDeviceFloat(BlockActivationCacheDevice.SlotId.X_OUT, blockOutDevice, plane);
            }
        }
        return true;
    }

    /**
     * Prefill с KV-cache: как {@link #multiHeadAttentionWithRoPEPrefill}, но pre-norm и Q/K/V (и выходной Wo) без
     * H2D весов; вход — сырой {@code x} до первого RMSNorm (как в {@link #tryMultiHeadAttentionWithRoPEGpuResident}).
     *
     * @return {@code null}, если CUDA недоступна или размеры не подходят
     */
    public static Tensor tryMultiHeadAttentionWithRoPEPrefillGpuResident(
            Tensor xIn,
            float eps,
            GpuAttnResidentBuffers resident,
            int numHeads,
            Tensor mask,
            Tensor kCacheLayer,
            Tensor vCacheLayer,
            int ropeOffset) {
        Objects.requireNonNull(resident, "resident");
        int[] xShape = xIn.getShape();
        if (xShape.length != 3) {
            return null;
        }
        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];
        if (dModel % numHeads != 0) {
            return null;
        }
        GpuFloatBuffer normG = resident.normGamma();
        GpuFloatBuffer wqB = resident.wq();
        GpuFloatBuffer wkB = resident.wk();
        GpuFloatBuffer wvB = resident.wv();
        GpuFloatBuffer woB = resident.wo();
        long d2 = (long) dModel * dModel;
        if (normG.numFloats() < dModel
                || wqB.numFloats() < d2
                || wkB.numFloats() < d2
                || wvB.numFloats() < d2
                || woB.numFloats() < d2) {
            return null;
        }
        if (mask != null) {
            int[] mShape = mask.getShape();
            if (mShape.length != 2 || mShape[0] != seqLen || mShape[1] != seqLen) {
                return null;
            }
        }
        int dHead = dModel / numHeads;
        float scale = 1.0f / (float) Math.sqrt(dHead);
        int rows = batch * seqLen;
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dModel)) {
            return null;
        }

        GpuAttentionResidentWorkspace ws = GpuAttentionResidentWorkspace.local();
        Tensor qHeads = new Tensor(new int[] {batch, numHeads, seqLen, dHead});
        Tensor kHeads = new Tensor(new int[] {batch, numHeads, seqLen, dHead});
        Tensor vHeads = new Tensor(new int[] {batch, numHeads, seqLen, dHead});
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            ws.getXIn().copyFrom(xIn.internalBuffer(), 0, rows * dModel);
            TensorOpsGPU.rmsNormGpuDevice(ws.getXIn(), normG, eps, ws.getXNorm(), rows, dModel);
            TensorOpsGPU.matmulGpuDeviceQkvProjections(
                    ws.getXNorm(), wqB, wkB, wvB, ws.getQ(), ws.getK(), ws.getV(), rows, dModel);
            downloadQkvHeadsAfterSplitRoPeGpu(
                    ws,
                    batch,
                    seqLen,
                    dModel,
                    numHeads,
                    dHead,
                    true,
                    ropeOffset,
                    qHeads,
                    kHeads,
                    vHeads,
                    null,
                    null,
                    0,
                    0,
                    true);
        }

        copyKvHeadsIntoCache(kHeads, vHeads, kCacheLayer, vCacheLayer, seqLen);

        Tensor output = new Tensor(xShape);
        int bAttn = batch * numHeads;
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            TensorOpsGPU.scaledDotProductAttentionForwardGpuDevice(
                    ws.getConcatFlat(),
                    ws.getAttnOut(),
                    ws.getQ(),
                    mask != null ? mask.internalBuffer() : null,
                    ws.getK(),
                    null,
                    bAttn,
                    seqLen,
                    dHead,
                    dHead,
                    scale,
                    TensorOpsGPU.useFp16Matmul());
            TensorOpsGPU.concatHeadsGpuDevice(ws.getK(), ws.getConcatFlat(), batch, numHeads, seqLen, dHead);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getConcatFlat(), woB, ws.getAttnOut(), rows, dModel, dModel, false, false);
            ws.getAttnOut().copyTo(ws.getHostOut(), 0, rows * dModel);
            System.arraycopy(ws.getHostOut(), 0, output.internalBuffer(), 0, rows * dModel);
        }
        return output;
    }

    /**
     * Prefill с записью K/V в device-буферы (без host-{@link com.veles.llm.jgpt.core.Tensor} для кэша на горячем
     * пути). Расклад: {@code head * maxSeqLen * dHead + pos * dHead}.
     *
     * @return {@code null}, если CUDA недоступна или размеры не подходят
     */
    public static Tensor tryMultiHeadAttentionWithRoPEPrefillGpuResident(
            Tensor xIn,
            float eps,
            GpuAttnResidentBuffers resident,
            int numHeads,
            Tensor mask,
            GpuFloatBuffer kGpu,
            GpuFloatBuffer vGpu,
            int maxSeqLen,
            int ropeOffset) {
        Objects.requireNonNull(resident, "resident");
        Objects.requireNonNull(kGpu, "kGpu");
        Objects.requireNonNull(vGpu, "vGpu");
        int[] xShape = xIn.getShape();
        if (xShape.length != 3) {
            return null;
        }
        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];
        if (dModel % numHeads != 0) {
            return null;
        }
        int dHead = dModel / numHeads;
        long need = (long) numHeads * maxSeqLen * dHead;
        if (kGpu.numFloats() < need || vGpu.numFloats() < need || seqLen > maxSeqLen) {
            return null;
        }
        GpuFloatBuffer normG = resident.normGamma();
        GpuFloatBuffer wqB = resident.wq();
        GpuFloatBuffer wkB = resident.wk();
        GpuFloatBuffer wvB = resident.wv();
        GpuFloatBuffer woB = resident.wo();
        long d2 = (long) dModel * dModel;
        if (normG.numFloats() < dModel
                || wqB.numFloats() < d2
                || wkB.numFloats() < d2
                || wvB.numFloats() < d2
                || woB.numFloats() < d2) {
            return null;
        }
        if (mask != null) {
            int[] mShape = mask.getShape();
            if (mShape.length != 2 || mShape[0] != seqLen || mShape[1] != seqLen) {
                return null;
            }
        }
        float scale = 1.0f / (float) Math.sqrt(dHead);
        int rows = batch * seqLen;
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dModel)) {
            return null;
        }

        GpuAttentionResidentWorkspace ws = GpuAttentionResidentWorkspace.local();
        Tensor output = new Tensor(xShape);
        int bAttn = batch * numHeads;
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            ws.getXIn().copyFrom(xIn.internalBuffer(), 0, rows * dModel);
            TensorOpsGPU.rmsNormGpuDevice(ws.getXIn(), normG, eps, ws.getXNorm(), rows, dModel);
            TensorOpsGPU.matmulGpuDeviceQkvProjections(
                    ws.getXNorm(), wqB, wkB, wvB, ws.getQ(), ws.getK(), ws.getV(), rows, dModel);
            downloadQkvHeadsAfterSplitRoPeGpu(
                    ws,
                    batch,
                    seqLen,
                    dModel,
                    numHeads,
                    dHead,
                    true,
                    ropeOffset,
                    null,
                    null,
                    null,
                    kGpu,
                    vGpu,
                    maxSeqLen,
                    0,
                    false);
            TensorOpsGPU.scaledDotProductAttentionForwardGpuDevice(
                    ws.getConcatFlat(),
                    ws.getAttnOut(),
                    ws.getQ(),
                    mask != null ? mask.internalBuffer() : null,
                    ws.getK(),
                    null,
                    bAttn,
                    seqLen,
                    dHead,
                    dHead,
                    scale,
                    TensorOpsGPU.useFp16Matmul());
            TensorOpsGPU.concatHeadsGpuDevice(ws.getK(), ws.getConcatFlat(), batch, numHeads, seqLen, dHead);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getConcatFlat(), woB, ws.getAttnOut(), rows, dModel, dModel, false, false);
            ws.getAttnOut().copyTo(ws.getHostOut(), 0, rows * dModel);
            System.arraycopy(ws.getHostOut(), 0, output.internalBuffer(), 0, rows * dModel);
        }
        return output;
    }

    /**
     * Один токен декодирования с KV-cache: как {@link #multiHeadAttentionWithRoPEDecode}, но проекции и Wo без H2D
     * весов. Вход — сырой {@code x} до первого RMSNorm.
     *
     * @return {@code null}, если CUDA недоступна или форма не {@code [1,1,d_model]}
     */
    public static Tensor tryMultiHeadAttentionWithRoPEDecodeGpuResident(
            Tensor xIn,
            float eps,
            GpuAttnResidentBuffers resident,
            int numHeads,
            Tensor kCacheLayer,
            Tensor vCacheLayer,
            int cacheLenBefore,
            int ropePosition) {
        Objects.requireNonNull(resident, "resident");
        int[] xShape = xIn.getShape();
        if (xShape.length != 3 || xShape[0] != 1 || xShape[1] != 1) {
            return null;
        }
        int dModel = xShape[2];
        if (dModel % numHeads != 0) {
            return null;
        }
        GpuFloatBuffer normG = resident.normGamma();
        GpuFloatBuffer wqB = resident.wq();
        GpuFloatBuffer wkB = resident.wk();
        GpuFloatBuffer wvB = resident.wv();
        GpuFloatBuffer woB = resident.wo();
        long d2 = (long) dModel * dModel;
        if (normG.numFloats() < dModel
                || wqB.numFloats() < d2
                || wkB.numFloats() < d2
                || wvB.numFloats() < d2
                || woB.numFloats() < d2) {
            return null;
        }
        int dHead = dModel / numHeads;
        float scale = 1.0f / (float) Math.sqrt(dHead);
        int rows = 1;
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dModel)) {
            return null;
        }

        GpuAttentionResidentWorkspace ws = GpuAttentionResidentWorkspace.local();
        Tensor Q3 = new Tensor(xShape);
        Tensor K3 = new Tensor(xShape);
        Tensor V3 = new Tensor(xShape);
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            ws.getXIn().copyFrom(xIn.internalBuffer(), 0, dModel);
            TensorOpsGPU.rmsNormGpuDevice(ws.getXIn(), normG, eps, ws.getXNorm(), rows, dModel);
            TensorOpsGPU.matmulGpuDeviceQkvProjections(
                    ws.getXNorm(), wqB, wkB, wvB, ws.getQ(), ws.getK(), ws.getV(), rows, dModel);
            ws.getQ().copyTo(Q3.internalBuffer(), 0, dModel);
            ws.getK().copyTo(K3.internalBuffer(), 0, dModel);
            ws.getV().copyTo(V3.internalBuffer(), 0, dModel);
        }

        int[] ropePos = new int[] {ropePosition};
        Tensor qHeads = applyRoPE(splitHeads(Q3, numHeads), ropePos);
        Tensor kNewHeads = applyRoPE(splitHeads(K3, numHeads), ropePos);
        Tensor vNewHeads = splitHeads(V3, numHeads);

        copyKvOneTokenIntoCache(kNewHeads, vNewHeads, kCacheLayer, vCacheLayer, cacheLenBefore);

        int headLen = cacheLenBefore + 1;
        Tensor outputHeads = new Tensor(new int[]{1, numHeads, 1, dHead});

        for (int h = 0; h < numHeads; h++) {
            Tensor qHead = sliceHead(qHeads, 0, h);
            Tensor kFull = sliceKvHeadPrefix(kCacheLayer, h, headLen);
            Tensor vFull = sliceKvHeadPrefix(vCacheLayer, h, headLen);

            Tensor attn3 =
                    scaledDotProductAttentionQueryOne(
                            batch3DFrom2D(qHead),
                            kFull,
                            vFull,
                            scale);
            Tensor attnOut = sliceBatch(attn3, 0);
            copyInto(outputHeads, attnOut, 0, h);
        }

        Tensor concat = concatHeads(outputHeads, numHeads);
        Tensor output = new Tensor(xShape);
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            ws.getConcatFlat().copyFrom(concat.internalBuffer(), 0, dModel);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getConcatFlat(), woB, ws.getAttnOut(), rows, dModel, dModel, false, false);
            ws.getAttnOut().copyTo(ws.getHostOut(), 0, dModel);
            System.arraycopy(ws.getHostOut(), 0, output.internalBuffer(), 0, dModel);
        }
        return output;
    }

    /**
     * Decode с K/V на device (см. {@link #tryMultiHeadAttentionWithRoPEPrefillGpuResident(Tensor, float,
     * GpuAttnResidentBuffers, int, Tensor, GpuFloatBuffer, GpuFloatBuffer, int, int)}).
     *
     * @return {@code null}, если CUDA недоступна или форма не {@code [1,1,d_model]}
     */
    public static Tensor tryMultiHeadAttentionWithRoPEDecodeGpuResident(
            Tensor xIn,
            float eps,
            GpuAttnResidentBuffers resident,
            int numHeads,
            GpuFloatBuffer kGpu,
            GpuFloatBuffer vGpu,
            int maxSeqLen,
            int cacheLenBefore,
            int ropePosition) {
        Objects.requireNonNull(resident, "resident");
        Objects.requireNonNull(kGpu, "kGpu");
        Objects.requireNonNull(vGpu, "vGpu");
        int[] xShape = xIn.getShape();
        if (xShape.length != 3 || xShape[0] != 1 || xShape[1] != 1) {
            return null;
        }
        int dModel = xShape[2];
        if (dModel % numHeads != 0) {
            return null;
        }
        int dHead = dModel / numHeads;
        long need = (long) numHeads * maxSeqLen * dHead;
        if (kGpu.numFloats() < need || vGpu.numFloats() < need) {
            return null;
        }
        int headLen = cacheLenBefore + 1;
        if (headLen > maxSeqLen) {
            return null;
        }
        GpuFloatBuffer normG = resident.normGamma();
        GpuFloatBuffer wqB = resident.wq();
        GpuFloatBuffer wkB = resident.wk();
        GpuFloatBuffer wvB = resident.wv();
        GpuFloatBuffer woB = resident.wo();
        long d2 = (long) dModel * dModel;
        if (normG.numFloats() < dModel
                || wqB.numFloats() < d2
                || wkB.numFloats() < d2
                || wvB.numFloats() < d2
                || woB.numFloats() < d2) {
            return null;
        }
        float scale = 1.0f / (float) Math.sqrt(dHead);
        int rows = 1;
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dModel)) {
            return null;
        }

        GpuAttentionResidentWorkspace ws = GpuAttentionResidentWorkspace.local();
        Tensor Q3 = new Tensor(xShape);
        Tensor K3 = new Tensor(xShape);
        Tensor V3 = new Tensor(xShape);
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            ws.getXIn().copyFrom(xIn.internalBuffer(), 0, dModel);
            TensorOpsGPU.rmsNormGpuDevice(ws.getXIn(), normG, eps, ws.getXNorm(), rows, dModel);
            TensorOpsGPU.matmulGpuDeviceQkvProjections(
                    ws.getXNorm(), wqB, wkB, wvB, ws.getQ(), ws.getK(), ws.getV(), rows, dModel);
            ws.getQ().copyTo(Q3.internalBuffer(), 0, dModel);
            ws.getK().copyTo(K3.internalBuffer(), 0, dModel);
            ws.getV().copyTo(V3.internalBuffer(), 0, dModel);
        }

        int[] ropePos = new int[] {ropePosition};
        Tensor qHeads = applyRoPE(splitHeads(Q3, numHeads), ropePos);
        Tensor kNewHeads = applyRoPE(splitHeads(K3, numHeads), ropePos);
        Tensor vNewHeads = splitHeads(V3, numHeads);

        copyKvOneTokenIntoCacheGpu(kNewHeads, vNewHeads, kGpu, vGpu, maxSeqLen, cacheLenBefore);

        Tensor outputHeads = new Tensor(new int[] {1, numHeads, 1, dHead});

        for (int h = 0; h < numHeads; h++) {
            Tensor qHead = sliceHead(qHeads, 0, h);
            Tensor kFull = sliceKvHeadPrefixFromGpu(kGpu, h, headLen, maxSeqLen, dHead);
            Tensor vFull = sliceKvHeadPrefixFromGpu(vGpu, h, headLen, maxSeqLen, dHead);

            Tensor attn3 =
                    scaledDotProductAttentionQueryOne(
                            batch3DFrom2D(qHead),
                            kFull,
                            vFull,
                            scale);
            Tensor attnOut = sliceBatch(attn3, 0);
            copyInto(outputHeads, attnOut, 0, h);
        }

        Tensor concat = concatHeads(outputHeads, numHeads);
        Tensor output = new Tensor(xShape);
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(rows, dModel);
            ws.getConcatFlat().copyFrom(concat.internalBuffer(), 0, dModel);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getConcatFlat(), woB, ws.getAttnOut(), rows, dModel, dModel, false, false);
            ws.getAttnOut().copyTo(ws.getHostOut(), 0, dModel);
            System.arraycopy(ws.getHostOut(), 0, output.internalBuffer(), 0, dModel);
        }
        return output;
    }

    /**
     * Prefill с записью K/V после RoPE в кэш слоя (для последующего {@link #multiHeadAttentionWithRoPEDecode}).
     *
     * @param ropeOffset абсолютный сдвиг позиций RoPE (0 = как обычный forward; для окна — индекс начала)
     */
    public static Tensor multiHeadAttentionWithRoPEPrefill(
            Tensor x,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            Tensor mask,
            Tensor kCacheLayer,
            Tensor vCacheLayer,
            int ropeOffset) {
        int[] xShape = x.getShape();
        if (xShape.length != 3) {
            throw new IllegalArgumentException("x must be 3D [batch, seq_len, d_model]");
        }

        int batch = xShape[0];
        int seqLen = xShape[1];
        int dModel = xShape[2];

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("d_model must be divisible by num_heads");
        }

        validateSquareProjection(Wq, dModel, "Wq");
        validateSquareProjection(Wk, dModel, "Wk");
        validateSquareProjection(Wv, dModel, "Wv");
        validateSquareProjection(Wo, dModel, "Wo");

        int dHead = dModel / numHeads;
        float scale = 1.0f / (float) Math.sqrt(dHead);

        Tensor Q = new Tensor(xShape);
        Tensor K = new Tensor(xShape);
        Tensor V = new Tensor(xShape);

        for (int b = 0; b < batch; b++) {
            Tensor xSlice = sliceBatch(x, b);
            copyInto(Q, matmul(xSlice, Wq), b);
            copyInto(K, matmul(xSlice, Wk), b);
            copyInto(V, matmul(xSlice, Wv), b);
        }

        int[] ropePos = null;
        if (ropeOffset != 0) {
            ropePos = new int[seqLen];
            for (int i = 0; i < seqLen; i++) {
                ropePos[i] = ropeOffset + i;
            }
        }

        Tensor qHeads = applyRoPE(splitHeads(Q, numHeads), ropePos);
        Tensor kHeads = applyRoPE(splitHeads(K, numHeads), ropePos);
        Tensor vHeads = splitHeads(V, numHeads);

        copyKvHeadsIntoCache(kHeads, vHeads, kCacheLayer, vCacheLayer, seqLen);

        Tensor outputHeads = new Tensor(new int[]{batch, numHeads, seqLen, dHead});

        for (int h = 0; h < numHeads; h++) {
            for (int b = 0; b < batch; b++) {
                Tensor qHead = sliceHead(qHeads, b, h);
                Tensor kHead = sliceHead(kHeads, b, h);
                Tensor vHead = sliceHead(vHeads, b, h);

                Tensor attn3 =
                        scaledDotProductAttention(
                                batch3DFrom2D(qHead),
                                batch3DFrom2D(kHead),
                                batch3DFrom2D(vHead),
                                mask,
                                scale);
                Tensor attnOut = sliceBatch(attn3, 0);
                copyInto(outputHeads, attnOut, b, h);
            }
        }

        Tensor concat = concatHeads(outputHeads, numHeads);

        Tensor output = new Tensor(xShape);
        for (int b = 0; b < batch; b++) {
            Tensor concatSlice = sliceBatch(concat, b);
            copyInto(output, matmul(concatSlice, Wo), b);
        }

        return output;
    }

    /**
     * Один новый токен: дописывает K/V в кэш и считает attention только для последней позиции.
     */
    public static Tensor multiHeadAttentionWithRoPEDecode(
            Tensor x,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            Tensor kCacheLayer,
            Tensor vCacheLayer,
            int cacheLenBefore,
            int ropePosition) {
        int[] xShape = x.getShape();
        if (xShape[0] != 1 || xShape[1] != 1) {
            throw new IllegalArgumentException("decode expects x [1, 1, d_model]");
        }
        int dModel = xShape[2];

        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("d_model must be divisible by num_heads");
        }

        validateSquareProjection(Wq, dModel, "Wq");
        validateSquareProjection(Wk, dModel, "Wk");
        validateSquareProjection(Wv, dModel, "Wv");
        validateSquareProjection(Wo, dModel, "Wo");

        int dHead = dModel / numHeads;
        float scale = 1.0f / (float) Math.sqrt(dHead);

        Tensor xSlice = sliceBatch(x, 0);
        Tensor q2 = matmul(xSlice, Wq);
        Tensor k2 = matmul(xSlice, Wk);
        Tensor v2 = matmul(xSlice, Wv);

        Tensor Q3 = new Tensor(xShape);
        Tensor K3 = new Tensor(xShape);
        Tensor V3 = new Tensor(xShape);
        copyInto(Q3, q2, 0);
        copyInto(K3, k2, 0);
        copyInto(V3, v2, 0);

        int[] ropePos = new int[]{ropePosition};
        Tensor qHeads = applyRoPE(splitHeads(Q3, numHeads), ropePos);
        Tensor kNewHeads = applyRoPE(splitHeads(K3, numHeads), ropePos);
        Tensor vNewHeads = splitHeads(V3, numHeads);

        copyKvOneTokenIntoCache(kNewHeads, vNewHeads, kCacheLayer, vCacheLayer, cacheLenBefore);

        int headLen = cacheLenBefore + 1;
        Tensor outputHeads = new Tensor(new int[]{1, numHeads, 1, dHead});

        for (int h = 0; h < numHeads; h++) {
            Tensor qHead = sliceHead(qHeads, 0, h);
            Tensor kFull = sliceKvHeadPrefix(kCacheLayer, h, headLen);
            Tensor vFull = sliceKvHeadPrefix(vCacheLayer, h, headLen);

            Tensor attn3 =
                    scaledDotProductAttentionQueryOne(
                            batch3DFrom2D(qHead),
                            kFull,
                            vFull,
                            scale);
            Tensor attnOut = sliceBatch(attn3, 0);
            copyInto(outputHeads, attnOut, 0, h);
        }

        Tensor concat = concatHeads(outputHeads, numHeads);
        Tensor output = new Tensor(xShape);
        copyInto(output, matmul(sliceBatch(concat, 0), Wo), 0);
        return output;
    }

    /** Срез кэша [1,H,0:prefixLen,:] → [1, prefixLen, d_head]. */
    public static Tensor sliceKvHeadPrefix(Tensor cache4d, int headIdx, int prefixLen) {
        int[] sh = cache4d.getShape();
        int dHead = sh[3];
        Tensor out = new Tensor(new int[]{1, prefixLen, dHead});
        float[] src = cache4d.internalBuffer();
        float[] dst = out.internalBuffer();
        int[] s = cache4d.stridesInternal();
        int b = 0;
        for (int i = 0; i < prefixLen; i++) {
            int srcBase = b * s[0] + headIdx * s[1] + i * s[2];
            for (int j = 0; j < dHead; j++) {
                dst[i * dHead + j] = src[srcBase + j * s[3]];
            }
        }
        return out;
    }

    /**
     * Срез K-кэша на VRAM: форма как у {@link #sliceKvHeadPrefix(Tensor, int, int)} — {@code [1, prefixLen, d_head]}.
     * Расклад плоского буфера: {@code headIdx * maxSeqLen * dHead + pos * dHead}.
     */
    public static Tensor sliceKvHeadPrefixFromGpu(
            GpuFloatBuffer cache4d, int headIdx, int prefixLen, int maxSeqLen, int dHead) {
        Tensor out = new Tensor(new int[] {1, prefixLen, dHead});
        int deviceOff = headIdx * maxSeqLen * dHead;
        cache4d.copyTo(out.internalBuffer(), 0, prefixLen * dHead, deviceOff);
        return out;
    }

    /**
     * После matmul Q/K/V в {@link GpuAttentionResidentWorkspace}: split голов, опционально RoPE на GPU для Q и K,
     * split для V; при непустых {@code kvCacheGpuK/V} — запись K/V в device-кэш без CPU-упаковки; при {@code
     * copyHeadsToHost} — D2H в {@code qHeads}/{@code kHeads}/{@code vHeads} (иначе головы остаются в {@code
     * ws}: Q → {@link GpuAttentionResidentWorkspace#getConcatFlat()}, K → {@link
     * GpuAttentionResidentWorkspace#getAttnOut()}, V → {@link GpuAttentionResidentWorkspace#getQ()}).
     */
    private static void downloadQkvHeadsAfterSplitRoPeGpu(
            GpuAttentionResidentWorkspace ws,
            int batch,
            int seqLen,
            int dModel,
            int numHeads,
            int dHead,
            boolean useRoPE,
            int ropeOffset,
            Tensor qHeads,
            Tensor kHeads,
            Tensor vHeads,
            GpuFloatBuffer kvCacheGpuK,
            GpuFloatBuffer kvCacheGpuV,
            int maxSeqLenForKvCache,
            int kvCacheBatchIdx,
            boolean copyHeadsToHost) {
        int headFloats = batch * numHeads * seqLen * dHead;
        TensorOpsGPU.splitHeadsGpuDevice(ws.getQ(), ws.getConcatFlat(), batch, seqLen, dModel, numHeads);
        if (useRoPE) {
            TensorOpsGPU.applyRoPE4dGpuDevice(
                    ws.getConcatFlat(), ws.getConcatFlat(), batch, numHeads, seqLen, dHead, null, ropeOffset);
        }
        TensorOpsGPU.splitHeadsGpuDevice(ws.getK(), ws.getAttnOut(), batch, seqLen, dModel, numHeads);
        if (useRoPE) {
            TensorOpsGPU.applyRoPE4dGpuDevice(
                    ws.getAttnOut(), ws.getAttnOut(), batch, numHeads, seqLen, dHead, null, ropeOffset);
        }
        TensorOpsGPU.splitHeadsGpuDevice(ws.getV(), ws.getQ(), batch, seqLen, dModel, numHeads);
        if (kvCacheGpuK != null) {
            Objects.requireNonNull(kvCacheGpuV, "kvCacheGpuV");
            if (maxSeqLenForKvCache <= 0 || kvCacheBatchIdx < 0 || kvCacheBatchIdx >= batch) {
                throw new IllegalArgumentException(
                        "kv cache: maxSeqLen=" + maxSeqLenForKvCache + ", batchIdx=" + kvCacheBatchIdx + ", batch=" + batch);
            }
            TensorOpsGPU.copyKvHeads4dToCacheGpuDevice(
                    ws.getAttnOut(),
                    kvCacheGpuK,
                    numHeads,
                    seqLen,
                    maxSeqLenForKvCache,
                    dHead,
                    kvCacheBatchIdx,
                    batch);
            TensorOpsGPU.copyKvHeads4dToCacheGpuDevice(
                    ws.getQ(), kvCacheGpuV, numHeads, seqLen, maxSeqLenForKvCache, dHead, kvCacheBatchIdx, batch);
        }
        if (copyHeadsToHost) {
            Objects.requireNonNull(qHeads, "qHeads");
            Objects.requireNonNull(kHeads, "kHeads");
            Objects.requireNonNull(vHeads, "vHeads");
            ws.getConcatFlat().copyTo(qHeads.internalBuffer(), 0, headFloats);
            ws.getAttnOut().copyTo(kHeads.internalBuffer(), 0, headFloats);
            ws.getQ().copyTo(vHeads.internalBuffer(), 0, headFloats);
        }
    }

    /**
     * Thread-local хост-буфер для H2D KV (избегает аллокаций и уменьшает число JNI/cudaMemcpy на prefill).
     */
    private static final class KvGpuH2dScratch {
        private static final ThreadLocal<float[]> BUF = new ThreadLocal<>();

        static float[] ensure(int minFloats) {
            float[] a = BUF.get();
            if (a == null || a.length < minFloats) {
                a = new float[minFloats];
                BUF.set(a);
            }
            return a;
        }
    }

    /**
     * Запись голов K/V prefill в плоский device-буфер кэша (тот же расклад, что у
     * {@link com.veles.llm.jgpt.model.KvCache}).
     * <p>
     * На голову — одно H2D с непрерывным сегментом {@code seqLen * d_head} (вместо {@code seqLen} вызовов по
     * одному токену).
     */
    /** K/V из 4D device-буферов голов в плоский device-кэш (ядро D2D, без CPU). */
    public static void copyKvHeadsIntoCacheGpuDevice(
            GpuFloatBuffer srcKHeads4d,
            GpuFloatBuffer srcVHeads4d,
            GpuFloatBuffer kBuf,
            GpuFloatBuffer vBuf,
            int numHeads,
            int seqLen,
            int maxSeqLen,
            int dHead,
            int batchIdx,
            int batch) {
        TensorOpsGPU.copyKvHeads4dToCacheGpuDevice(
                srcKHeads4d, kBuf, numHeads, seqLen, maxSeqLen, dHead, batchIdx, batch);
        TensorOpsGPU.copyKvHeads4dToCacheGpuDevice(
                srcVHeads4d, vBuf, numHeads, seqLen, maxSeqLen, dHead, batchIdx, batch);
    }

    public static void copyKvHeadsIntoCacheGpu(
            Tensor kHeads,
            Tensor vHeads,
            GpuFloatBuffer kBuf,
            GpuFloatBuffer vBuf,
            int maxSeqLen,
            int seqLen) {
        int[] sh = kHeads.getShape();
        int h = sh[1];
        int dh = sh[3];
        float[] ks = kHeads.internalBuffer();
        float[] vs = vHeads.internalBuffer();
        int[] sk = kHeads.stridesInternal();
        int b = 0;
        int segLen = seqLen * dh;
        float[] seg = KvGpuH2dScratch.ensure(segLen);
        for (int hi = 0; hi < h; hi++) {
            for (int s = 0; s < seqLen; s++) {
                int srcBase = b * sk[0] + hi * sk[1] + s * sk[2];
                int dst = s * dh;
                for (int j = 0; j < dh; j++) {
                    seg[dst + j] = ks[srcBase + j * sk[3]];
                }
            }
            int dstOff = hi * maxSeqLen * dh;
            kBuf.copyFrom(seg, 0, segLen, dstOff);
            for (int s = 0; s < seqLen; s++) {
                int srcBase = b * sk[0] + hi * sk[1] + s * sk[2];
                int dst = s * dh;
                for (int j = 0; j < dh; j++) {
                    seg[dst + j] = vs[srcBase + j * sk[3]];
                }
            }
            vBuf.copyFrom(seg, 0, segLen, dstOff);
        }
    }

    /** Запись одного токена decode в device-кэш (по голове — один H2D на K и на V). */
    public static void copyKvOneTokenIntoCacheGpu(
            Tensor kHeadsOne,
            Tensor vHeadsOne,
            GpuFloatBuffer kBuf,
            GpuFloatBuffer vBuf,
            int maxSeqLen,
            int seqIdx) {
        int[] sh = kHeadsOne.getShape();
        int heads = sh[1];
        int dh = sh[3];
        float[] ks = kHeadsOne.internalBuffer();
        float[] vs = vHeadsOne.internalBuffer();
        int[] sk = kHeadsOne.stridesInternal();
        int b = 0;
        float[] row = KvGpuH2dScratch.ensure(dh);
        for (int hi = 0; hi < heads; hi++) {
            int srcBase = b * sk[0] + hi * sk[1];
            int dstOff = hi * maxSeqLen * dh + seqIdx * dh;
            for (int j = 0; j < dh; j++) {
                row[j] = ks[srcBase + j * sk[3]];
            }
            kBuf.copyFrom(row, 0, dh, dstOff);
            for (int j = 0; j < dh; j++) {
                row[j] = vs[srcBase + j * sk[3]];
            }
            vBuf.copyFrom(row, 0, dh, dstOff);
        }
    }

    private static void copyKvHeadsIntoCache(
            Tensor kHeads, Tensor vHeads, Tensor kCache, Tensor vCache, int seqLen) {
        int[] sh = kHeads.getShape();
        int H = sh[1];
        int Dh = sh[3];
        float[] ks = kHeads.internalBuffer();
        float[] kd = kCache.internalBuffer();
        float[] vs = vHeads.internalBuffer();
        float[] vd = vCache.internalBuffer();
        int[] sk = kHeads.stridesInternal();
        int[] dk = kCache.stridesInternal();
        int b = 0;
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < seqLen; s++) {
                int srcBase = b * sk[0] + h * sk[1] + s * sk[2];
                int dstBase = b * dk[0] + h * dk[1] + s * dk[2];
                for (int j = 0; j < Dh; j++) {
                    kd[dstBase + j * dk[3]] = ks[srcBase + j * sk[3]];
                    vd[dstBase + j * dk[3]] = vs[srcBase + j * sk[3]];
                }
            }
        }
    }

    private static void copyKvOneTokenIntoCache(
            Tensor kHeadsOne, Tensor vHeadsOne, Tensor kCache, Tensor vCache, int seqIdx) {
        int[] sh = kHeadsOne.getShape();
        int H = sh[1];
        int Dh = sh[3];
        float[] ks = kHeadsOne.internalBuffer();
        float[] kd = kCache.internalBuffer();
        float[] vs = vHeadsOne.internalBuffer();
        float[] vd = vCache.internalBuffer();
        int[] sk = kHeadsOne.stridesInternal();
        int[] dk = kCache.stridesInternal();
        int b = 0;
        for (int h = 0; h < H; h++) {
            int srcBase = b * sk[0] + h * sk[1];
            int dstBase = b * dk[0] + h * dk[1] + seqIdx * dk[2];
            for (int j = 0; j < Dh; j++) {
                kd[dstBase + j * dk[3]] = ks[srcBase + j * sk[3]];
                vd[dstBase + j * dk[3]] = vs[srcBase + j * sk[3]];
            }
        }
    }

    private static void validateSquareProjection(Tensor w, int dModel, String name) {
        int[] sh = w.getShape();
        if (sh.length != 2 || sh[0] != dModel || sh[1] != dModel) {
            throw new IllegalArgumentException(
                    name + " must be [d_model, d_model] = ["
                            + dModel
                            + ", "
                            + dModel
                            + "], got "
                            + Arrays.toString(sh));
        }
    }

    /** [rows, cols] → [1, rows, cols] для attention с batch=1. */
    public static Tensor batch3DFrom2D(Tensor t2d) {
        int[] sh = t2d.getShape();
        if (sh.length != 2) {
            throw new IllegalArgumentException("batch3DFrom2D requires 2D tensor");
        }
        Tensor out = new Tensor(new int[]{1, sh[0], sh[1]});
        System.arraycopy(t2d.internalBuffer(), 0, out.internalBuffer(), 0, sh[0] * sh[1]);
        return out;
    }

    /** [batch, seq_len, d_model] → [batch, num_heads, seq_len, d_head]. */
    public static Tensor splitHeads(Tensor x, int numHeads) {
        int[] shape = x.getShape();
        int batch = shape[0];
        int seqLen = shape[1];
        int dModel = shape[2];
        int dHead = dModel / numHeads;

        Tensor result = new Tensor(new int[]{batch, numHeads, seqLen, dHead});
        float[] src = x.internalBuffer();
        float[] dst = result.internalBuffer();

        int n = batch * seqLen * dModel;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.splitHeads");
        TensorOpsGPU.splitHeadsFromHost(src, dst, batch, seqLen, dModel, numHeads);
        return result;
    }

    /** [batch, num_heads, seq_len, d_head] → [batch, seq_len, d_model]. */
    public static Tensor concatHeads(Tensor x, int numHeads) {
        int[] shape = x.getShape();
        if (shape.length != 4 || shape[1] != numHeads) {
            throw new IllegalArgumentException(
                    "expected [batch, num_heads, seq, d_head] with num_heads="
                            + numHeads
                            + ", got "
                            + Arrays.toString(shape));
        }
        int batch = shape[0];
        int seqLen = shape[2];
        int dHead = shape[3];
        int dModel = numHeads * dHead;

        Tensor result = new Tensor(new int[]{batch, seqLen, dModel});
        float[] src = x.internalBuffer();
        float[] dst = result.internalBuffer();

        int n = batch * seqLen * dModel;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.concatHeads");
        TensorOpsGPU.concatHeadsFromHost(src, dst, batch, numHeads, seqLen, dHead);
        return result;
    }

    /** Срез головы: [batch, heads, seq, d_head] → [seq, d_head]. */
    public static Tensor sliceHead(Tensor t, int batchIdx, int headIdx) {
        int[] shape = t.getShape();
        if (shape.length != 4) {
            throw new IllegalArgumentException("sliceHead requires 4D tensor");
        }
        int seqLen = shape[2];
        int dHead = shape[3];
        Tensor slice = new Tensor(new int[]{seqLen, dHead});
        float[] src = t.internalBuffer();
        float[] dst = slice.internalBuffer();
        int[] s = t.stridesInternal();

        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < dHead; j++) {
                int srcIdx = batchIdx * s[0] + headIdx * s[1] + i * s[2] + j * s[3];
                dst[i * dHead + j] = src[srcIdx];
            }
        }
        return slice;
    }

    /** Копирование [seq_len, d_head] в [batch, heads, seq_len, d_head]. */
    public static void copyInto(Tensor dest, Tensor src, int batchIdx, int headIdx) {
        int[] dShape = dest.getShape();
        int[] sShape = src.getShape();
        if (dShape.length != 4) {
            throw new IllegalArgumentException("dest must be 4D");
        }
        if (sShape.length != 2 || sShape[0] != dShape[2] || sShape[1] != dShape[3]) {
            throw new IllegalArgumentException(
                    "src must be [seq_len, d_head], dest " + Arrays.toString(dShape));
        }
        float[] d = dest.internalBuffer();
        float[] sbuf = src.internalBuffer();
        int[] ds = dest.stridesInternal();
        int[] ss = src.stridesInternal();

        int destBase = batchIdx * ds[0] + headIdx * ds[1];
        for (int i = 0; i < sShape[0]; i++) {
            for (int j = 0; j < sShape[1]; j++) {
                d[destBase + i * ds[2] + j * ds[3]] = sbuf[i * ss[0] + j * ss[1]];
            }
        }
    }

    private static void copyFrom(Tensor src, Tensor dest) {
        if (!Arrays.equals(src.getShape(), dest.getShape())) {
            throw new IllegalArgumentException(
                    "copyFrom shape mismatch: "
                            + Arrays.toString(src.getShape())
                            + " vs "
                            + Arrays.toString(dest.getShape()));
        }
        float[] s = src.internalBuffer();
        float[] d = dest.internalBuffer();
        System.arraycopy(s, 0, d, 0, s.length);
    }

    /** Добавляет к scores[b,i,j] значение mask[i,j] (типично 0 или -inf). */
    public static Tensor applyCausalMask(Tensor scores, Tensor mask) {
        int[] sShape = scores.getShape();
        int[] mShape = mask.getShape();
        if (sShape.length != 3) {
            throw new IllegalArgumentException("scores must be 3D");
        }
        if (mShape.length != 2 || sShape[1] != mShape[0] || sShape[2] != mShape[1]) {
            throw new IllegalArgumentException(
                    "mask shape mismatch: scores " + Arrays.toString(sShape) + ", mask "
                            + Arrays.toString(mShape));
        }

        Tensor result = new Tensor(sShape);
        float[] s = scores.internalBuffer();
        float[] m = mask.internalBuffer();
        float[] r = result.internalBuffer();

        int seqLen = sShape[1];
        int batch = sShape[0];
        int n = batch * seqLen * seqLen;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.applyCausalMask");
        TensorOpsGPU.applyCausalMask3DGPU(s, m, r, batch, seqLen);
        return result;
    }

    /**
     * Softmax по последней оси для 3D тензора (для каждого среза [*, *, j] по j).
     */
    public static Tensor softmaxLastDim(Tensor x) {
        int[] shape = x.getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("softmaxLastDim requires 3D tensor");
        }
        int batch = shape[0];
        int mid = shape[1];
        int inner = shape[2];
        Tensor result = new Tensor(shape);
        float[] src = x.internalBuffer();
        float[] dst = result.internalBuffer();

        int n = batch * mid * inner;
        if (n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.softmaxLastDim");
        TensorOpsGPU.softmaxLastDimGPU(src, dst, batch, mid, inner, TensorOpsGPU.useFp16Matmul());
        return result;
    }

    /** Транспонирование последних двух осей: [a,b,c] → [a,c,b]. */
    public static Tensor transpose2DLast(Tensor a) {
        int[] shape = a.getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("transpose2DLast requires 3D tensor");
        }
        Tensor result = new Tensor(new int[]{shape[0], shape[2], shape[1]});
        float[] src = a.internalBuffer();
        float[] dst = result.internalBuffer();
        int vol = shape[0] * shape[1] * shape[2];
        if (vol <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.transpose2DLast");
        TensorOpsGPU.transpose2DLastGPU(src, dst, shape[0], shape[1], shape[2]);
        return result;
    }

    /** Срез по батчу: [batch, r, c] → [r, c]. */
    public static Tensor sliceBatch(Tensor t, int batchIdx) {
        int[] shape = t.getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("sliceBatch requires 3D tensor");
        }
        Tensor slice = new Tensor(new int[]{shape[1], shape[2]});
        float[] src = t.internalBuffer();
        float[] dst = slice.internalBuffer();
        int[] s = t.stridesInternal();

        int srcBase = batchIdx * s[0];
        for (int i = 0; i < shape[1]; i++) {
            for (int j = 0; j < shape[2]; j++) {
                dst[i * shape[2] + j] = src[srcBase + i * s[1] + j * s[2]];
            }
        }
        return slice;
    }

    /** Копирование 2D-плоскости в 3D-тензор по индексу батча. */
    public static void copyInto(Tensor dest, Tensor src, int batchIdx) {
        int[] dShape = dest.getShape();
        int[] sShape = src.getShape();
        if (dShape.length != 3) {
            throw new IllegalArgumentException("dest must be 3D");
        }
        if (sShape[0] != dShape[1] || sShape[1] != dShape[2]) {
            throw new IllegalArgumentException(
                    "shape mismatch: dest [," + dShape[1] + "," + dShape[2] + "] vs src "
                            + Arrays.toString(sShape));
        }
        float[] d = dest.internalBuffer();
        float[] sbuf = src.internalBuffer();
        int[] ds = dest.stridesInternal();
        int[] ss = src.stridesInternal();

        int destBase = batchIdx * ds[0];
        for (int i = 0; i < sShape[0]; i++) {
            for (int j = 0; j < sShape[1]; j++) {
                d[destBase + i * ds[1] + j * ds[2]] = sbuf[i * ss[0] + j * ss[1]];
            }
        }
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        int[] aShape = a.getShape();
        int[] bShape = b.getShape();

        if (aShape.length != 2 || bShape.length != 2) {
            throw new IllegalArgumentException("matmul требует 2D тензоры");
        }
        if (aShape[1] != bShape[0]) {
            throw new IllegalArgumentException(
                    "Несовместимые размерности: " + aShape[1] + " != " + bShape[0]);
        }

        int m = aShape[0];
        int k = aShape[1];
        int n = bShape[1];
        Tensor result = new Tensor(new int[]{m, n});
        if (m <= 0 || k <= 0 || n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.matmul");
        TensorOpsGPU.matmulGPUMaybeFp16(
                a.internalBuffer(),
                b.internalBuffer(),
                result.internalBuffer(),
                m,
                k,
                n);
        return result;
    }

    /**
     * Matmul после разворачивания int8 → float32 (деquantize в {@link QuantizedTensor#toTensor()}).
     * Буферы весов остаются компактными; пиковая память при вычислении — float-копии A и B.
     */
    public static Tensor matmul(QuantizedTensor a, QuantizedTensor b) {
        return matmul(a.toTensor(), b.toTensor());
    }

    /**
     * Matmul + bias + ReLU за два прохода по данным: GEMM, затем слитое {@code max(0, x + bias[j])}
     * по всем выходам (без отдельных тензорных add/relu).
     */
    public static Tensor matmulAddRelu(Tensor a, Tensor b, Tensor bias) {
        int[] aShape = a.getShape();
        int[] bShape = b.getShape();
        int[] biasShape = bias.getShape();

        if (aShape.length != 2 || bShape.length != 2) {
            throw new IllegalArgumentException("matmulAddRelu требует 2D тензоры для A и B");
        }
        if (aShape[1] != bShape[0]) {
            throw new IllegalArgumentException(
                    "Несовместимые размерности: " + aShape[1] + " != " + bShape[0]);
        }
        int n = bShape[1];
        if (biasShape.length != 1 || biasShape[0] != n) {
            throw new IllegalArgumentException(
                    "bias должен иметь форму [n], n=" + n + ", получено " + Arrays.toString(biasShape));
        }

        int m = aShape[0];
        int k = aShape[1];
        Tensor result = new Tensor(new int[]{m, n});
        if (m <= 0 || k <= 0 || n <= 0) {
            return result;
        }
        TensorOpsGPU.requireCuda("TensorOps.matmulAddRelu");
        if (!TensorOpsGPU.shouldUseGpuMatmul(m, k, n)) {
            throw new IllegalArgumentException("matmulAddRelu: ожидаются положительные m, k, n");
        }
        TensorOpsGPU.matmulAddReluGPUMaybeFp16(
                a.internalBuffer(),
                b.internalBuffer(),
                bias.internalBuffer(),
                result.internalBuffer(),
                m,
                k,
                n);
        return result;
    }

    /** GEMM в {@code result} на GPU; package-private для хостовых путей без лишнего аллока. */
    static void matmulInto(Tensor a, Tensor b, Tensor result) {
        int[] aShape = a.getShape();
        int[] bShape = b.getShape();
        int m = aShape[0];
        int k = aShape[1];
        int n = bShape[1];
        if (m <= 0 || k <= 0 || n <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOps.matmulInto");
        TensorOpsGPU.matmulGPUMaybeFp16(
                a.internalBuffer(),
                b.internalBuffer(),
                result.internalBuffer(),
                m,
                k,
                n);
    }

    private static void validateSameShape(Tensor a, Tensor b) {
        Objects.requireNonNull(a, "a cannot be null");
        Objects.requireNonNull(b, "b cannot be null");
        if (!Arrays.equals(a.getShape(), b.getShape())) {
            throw new IllegalArgumentException(
                    "Формы не совпадают: " + Arrays.toString(a.getShape())
                            + " vs " + Arrays.toString(b.getShape()));
        }
    }
}
