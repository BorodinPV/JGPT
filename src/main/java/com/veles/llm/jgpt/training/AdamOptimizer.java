package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * AdamW: адаптивные моменты и decoupled weight decay.
 *
 * <p><b>Потокобезопасность:</b> не thread-safe; все вызовы должны идти из одного потока обучения (как
 * в {@link LLMTrainer}). Параллельный доступ из нескольких потоков к одному экземпляру запрещён.
 * Типичный сценарий: {@link #beginStep()} один раз, затем {@link #stepWithParamGrad(Tensor)} по каждому
 * параметру либо {@link #stepAllWithParamGrad(List)} (батч JNI для крупных тензоров на GPU).
 *
 * <p><b>Где «GPU»:</b> (1) классический путь — {@link TensorOpsGPU#adamWStepGPU} с H2D/D2H хостовых
 * {@code float[]}; (2) полностью на device — {@link #stepGpu(GpuTensor, GpuTensor, GpuTensor)} /
 * {@link #stepAllGpu(java.util.List, java.util.List, java.util.List)}: параметр, градиент и моменты m/v в
 * {@link GpuTensor} без копирования на хост; несколько тензоров — {@link #stepAllGpu}. Накопленные на GPU куски градиента для CPU-пути сливаются
 * через {@link com.veles.llm.jgpt.cuda.GpuPendingGradients#flushAllToHost()} до clip/шага.
 */
public final class AdamOptimizer {

    private static final Logger log = LoggerFactory.getLogger(AdamOptimizer.class);

    /** Склейка нескольких GPU-параметров в один вызов {@link TensorOpsGPU#adamWStepFusedGPU}. */
    private static final class PackWorkspace {
        float[] p;
        float[] g;
        float[] m;
        float[] v;
    }

    private static final ThreadLocal<PackWorkspace> TL_ADAM_PACK = ThreadLocal.withInitial(PackWorkspace::new);

    private static float[] ensureFloatCapacity(float[] cur, int need) {
        if (cur == null || cur.length < need) {
            return new float[need];
        }
        return cur;
    }

    private float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;

    private final Map<Tensor, Tensor> m;
    private final Map<Tensor, Tensor> v;
    private int step;
    /** 1 / (1 - β₁^t); при {@code step == 0} — 0 до первого {@link #beginStep()}. */
    private float invBias1;
    /** 1 / (1 - β₂^t). */
    private float invBias2;

    public AdamOptimizer(
            float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.m = new IdentityHashMap<>();
        this.v = new IdentityHashMap<>();
        this.step = 0;
        this.invBias1 = 0f;
        this.invBias2 = 0f;
    }

    public static AdamOptimizer defaultLLM() {
        return new AdamOptimizer(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    public static AdamOptimizer fromConfig(TrainingConfig config) {
        return new AdamOptimizer(
                config.learningRate, 0.9f, 0.999f, 1e-8f, config.weightDecay);
    }

    /**
     * Для юнит-тестов: {@code lr = 0.1}, {@code weightDecay = 0} (например минимизация x²).
     */
    public static AdamOptimizer forTesting() {
        return new AdamOptimizer(0.1f, 0.9f, 0.999f, 1e-8f, 0f);
    }

    /** Эффективный шаг Adam на текущую итерацию (scheduler в {@link LLMTrainer}). */
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getLearningRate() {
        return learningRate;
    }

    /**
     * Один шаг по глобальной итерации: вызвать ровно один раз перед серией
     * {@link #stepWithParamGrad(Tensor)} по всем параметрам (см. {@link LLMTrainer}).
     */
    public void beginStep() {
        step++;
        recomputeBiasCorrection();
    }

    /**
     * Один полный шаг для пары (параметр, градиент): внутри вызывает {@link #beginStep()}. Для
     * нескольких параметров за ту же итерацию используйте {@link #beginStep()} + серию
     * {@link #stepWithParamGrad(Tensor)}.
     *
     * @param grad градиенты в {@link Tensor#gradBuffer()} либо в {@link Tensor#internalBuffer()} (см.
     *     {@link #adamGradientBuffer(Tensor)})
     */
    public void step(Tensor param, Tensor grad) {
        Objects.requireNonNull(param, "param");
        Objects.requireNonNull(grad, "grad");
        beginStep();
        stepInternal(param, grad);
    }

    private void stepInternal(Tensor param, Tensor grad) {
        if (step <= 0) {
            throw new IllegalStateException(
                    "step counter is 0 — call beginStep() before stepWithParamGrad, or use step(param, grad)");
        }
        if (!Arrays.equals(param.getShape(), grad.getShape())) {
            throw new IllegalArgumentException(
                    "param and grad shape mismatch: "
                            + Arrays.toString(param.getShape())
                            + " vs "
                            + Arrays.toString(grad.getShape()));
        }

        float[] p = param.internalBuffer();
        float[] g = adamGradientBuffer(grad);

        Tensor mT = m.get(param);
        Tensor vT = v.get(param);
        if (mT == null) {
            mT = new Tensor(param.getShape());
            vT = new Tensor(param.getShape());
            m.put(param, mT);
            v.put(param, vT);
        }
        float[] mVal = mT.internalBuffer();
        float[] vVal = vT.internalBuffer();

        if (p.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("AdamOptimizer.stepInternal");
        TensorOpsGPU.adamWStepGPU(
                p,
                g,
                mVal,
                vVal,
                learningRate,
                beta1,
                beta2,
                epsilon,
                weightDecay,
                invBias1,
                invBias2,
                p.length);
    }

    /**
     * Обновление весов из {@code param.gradBuffer()}; перед этим в той же итерации должен быть вызван
     * {@link #beginStep()}.
     */
    public void stepWithParamGrad(Tensor param) {
        Objects.requireNonNull(param, "param");
        if (!param.hasGrad()) {
            throw new IllegalStateException("param has no grad buffer — call zeroGrad() and backward first");
        }
        stepInternal(param, param);
    }

    /**
     * Один шаг AdamW по всем параметрам из списка (с градиентом). При доступной CUDA все параметры склеиваются в
     * один буфер и обновляются одним вызовом {@link TensorOpsGPU#adamWStepFusedGPU}. Перед вызовом — ровно один {@link #beginStep()}.
     */
    public void stepAllWithParamGrad(List<Tensor> parameters) {
        Objects.requireNonNull(parameters, "parameters");
        if (step <= 0) {
            throw new IllegalStateException(
                    "step counter is 0 — call beginStep() before stepAllWithParamGrad");
        }
        List<Tensor> withGrad = new ArrayList<>();
        for (Tensor param : parameters) {
            if (param == null || !param.hasGrad()) {
                continue;
            }
            withGrad.add(param);
        }
        if (withGrad.isEmpty()) {
            return;
        }
        int total = 0;
        for (Tensor p : withGrad) {
            total += p.size();
        }
        if (total > 0) {
            stepInternalGpuBatched(withGrad);
            return;
        }
        for (Tensor p : withGrad) {
            stepInternal(p, p);
        }
    }

    private void stepInternalGpuBatched(List<Tensor> params) {
        int total = 0;
        for (Tensor p : params) {
            total += p.size();
        }
        PackWorkspace ws = TL_ADAM_PACK.get();
        ws.p = ensureFloatCapacity(ws.p, total);
        ws.g = ensureFloatCapacity(ws.g, total);
        ws.m = ensureFloatCapacity(ws.m, total);
        ws.v = ensureFloatCapacity(ws.v, total);

        int off = 0;
        for (Tensor param : params) {
            Tensor mT = m.get(param);
            Tensor vT = v.get(param);
            if (mT == null) {
                mT = new Tensor(param.getShape());
                vT = new Tensor(param.getShape());
                m.put(param, mT);
                v.put(param, vT);
            }
            float[] pBuf = param.internalBuffer();
            float[] gBuf = adamGradientBuffer(param);
            float[] mVal = mT.internalBuffer();
            float[] vVal = vT.internalBuffer();
            int len = param.size();
            System.arraycopy(pBuf, 0, ws.p, off, len);
            System.arraycopy(gBuf, 0, ws.g, off, len);
            System.arraycopy(mVal, 0, ws.m, off, len);
            System.arraycopy(vVal, 0, ws.v, off, len);
            off += len;
        }

        TensorOpsGPU.adamWStepFusedGPU(
                ws.p,
                ws.g,
                ws.m,
                ws.v,
                learningRate,
                beta1,
                beta2,
                epsilon,
                weightDecay,
                invBias1,
                invBias2,
                total);

        off = 0;
        for (Tensor param : params) {
            Tensor mT = m.get(param);
            Tensor vT = v.get(param);
            float[] pBuf = param.internalBuffer();
            float[] mVal = mT.internalBuffer();
            float[] vVal = vT.internalBuffer();
            int len = param.size();
            System.arraycopy(ws.p, off, pBuf, 0, len);
            System.arraycopy(ws.m, off, mVal, 0, len);
            System.arraycopy(ws.v, off, vVal, 0, len);
            off += len;
        }
    }

    /**
     * Один шаг AdamW на GPU: веса, градиент (после backward), первый и второй моменты — в {@link GpuTensor};
     * без H2D/D2H. Моменты должны быть инициализированы нулями до первого шага. Перед вызовом —
     * {@link #beginStep()}.
     */
    public void stepGpu(GpuTensor param, GpuTensor mState, GpuTensor vState) {
        Objects.requireNonNull(param, "param");
        Objects.requireNonNull(mState, "mState");
        Objects.requireNonNull(vState, "vState");
        if (step <= 0) {
            throw new IllegalStateException("call beginStep() before stepGpu");
        }
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("stepGpu requires CUDA");
        }
        if (!param.hasGradBuffer()) {
            throw new IllegalStateException("param needs grad buffer — zeroGrad() and backward on GPU first");
        }
        int n = param.size();
        if (mState.size() != n || vState.size() != n) {
            throw new IllegalArgumentException(
                    "mState/vState size must match param: " + n + " vs " + mState.size() + " / " + vState.size());
        }
        TensorOpsGPU.adamWStepGpuDevice(
                param.dataBuffer(),
                param.gradBuffer(),
                mState.dataBuffer(),
                vState.dataBuffer(),
                learningRate,
                beta1,
                beta2,
                epsilon,
                weightDecay,
                invBias1,
                invBias2,
                n);
    }

    /**
     * Несколько параметров на device: при {@code size > 1} — один JNI и одно ядро {@link
     * TensorOpsGPU#adamWStepGPUDeviceFused}; при одном элементе — {@link #stepGpu(GpuTensor, GpuTensor,
     * GpuTensor)}.
     */
    public void stepAllGpu(List<GpuTensor> params, List<GpuTensor> mStates, List<GpuTensor> vStates) {
        Objects.requireNonNull(params, "params");
        Objects.requireNonNull(mStates, "mStates");
        Objects.requireNonNull(vStates, "vStates");
        if (params.size() != mStates.size() || params.size() != vStates.size()) {
            throw new IllegalArgumentException("params, mStates and vStates must have the same size");
        }
        if (params.isEmpty()) {
            return;
        }
        if (step <= 0) {
            throw new IllegalStateException("call beginStep() before stepAllGpu");
        }
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("stepAllGpu requires CUDA");
        }
        int n = params.size();
        if (n == 1) {
            stepGpu(params.get(0), mStates.get(0), vStates.get(0));
            return;
        }
        long[] pp = new long[n];
        long[] gp = new long[n];
        long[] mp = new long[n];
        long[] vp = new long[n];
        int[] lens = new int[n];
        for (int i = 0; i < n; i++) {
            GpuTensor p = params.get(i);
            GpuTensor mT = mStates.get(i);
            GpuTensor vT = vStates.get(i);
            int len = p.size();
            if (mT.size() != len || vT.size() != len) {
                throw new IllegalArgumentException(
                        "mState/vState size must match param at index " + i + ": " + len);
            }
            if (!p.hasGradBuffer()) {
                throw new IllegalStateException("param needs grad buffer — zeroGrad() and backward on GPU first");
            }
            lens[i] = len;
            pp[i] = p.dataBuffer().devicePointer();
            gp[i] = p.gradBuffer().devicePointer();
            mp[i] = mT.dataBuffer().devicePointer();
            vp[i] = vT.dataBuffer().devicePointer();
        }
        TensorOpsGPU.adamWStepGPUDeviceFused(
                pp,
                gp,
                mp,
                vp,
                lens,
                learningRate,
                beta1,
                beta2,
                epsilon,
                weightDecay,
                invBias1,
                invBias2);
    }

    /**
     * Буфер с компонентами градиента для шага Adam.
     *
     * <p>Если у тензора есть {@link Tensor#gradBuffer()} (после backward) — используется он. Иначе
     * берётся {@link Tensor#internalBuffer()} — для отдельного тензора градиентов, заданного в данных.
     */
    private static float[] adamGradientBuffer(Tensor grad) {
        if (grad.hasGrad()) {
            return grad.gradBuffer();
        }
        return grad.internalBuffer();
    }

    private void recomputeBiasCorrection() {
        if (step <= 0) {
            invBias1 = 0f;
            invBias2 = 0f;
            return;
        }
        float b1t = (float) Math.pow(beta1, step);
        float b2t = (float) Math.pow(beta2, step);
        invBias1 = 1f / (1f - b1t);
        invBias2 = 1f / (1f - b2t);
    }

    /**
     * Глобальный клиппинг градиентов по L2-норме; не требует экземпляра оптимизатора (только те же
     * правила, что у {@link #clipGradients(List, float)}).
     */
    public static float clipGradientsGlobal(List<Tensor> tensors, float maxNorm) {
        Objects.requireNonNull(tensors, "tensors");
        double totalNormSq = 0.0;

        for (Tensor t : tensors) {
            if (t == null || !t.hasGrad()) {
                continue;
            }
            float[] g = t.gradBuffer();
            if (g.length > 0) {
                totalNormSq += TensorOpsGPU.sumSquaresGPU(g, g.length);
            }
        }

        float totalNorm = (float) Math.sqrt(totalNormSq);

        if (totalNorm > maxNorm && totalNorm > 0f) {
            float scale = maxNorm / totalNorm;
            for (Tensor t : tensors) {
                if (t == null || !t.hasGrad()) {
                    continue;
                }
                float[] g = t.gradBuffer();
                if (g.length > 0) {
                    TensorOpsGPU.scaleInPlaceGPU(g, g.length, scale);
                }
            }
            log.info(
                    "Клиппинг градиента (глобальная норма): {} → {}",
                    String.format("%.2f", totalNorm),
                    String.format("%.2f", maxNorm));
        }

        return totalNorm;
    }

    public float clipGradients(List<Tensor> tensors, float maxNorm) {
        return clipGradientsGlobal(tensors, maxNorm);
    }

    /**
     * Сброс моментов и счётчика шага. После вызова снова нужны {@link #beginStep()} перед
     * {@link #stepWithParamGrad(Tensor)}.
     */
    public void reset() {
        m.clear();
        v.clear();
        step = 0;
        recomputeBiasCorrection();
    }

    /**
     * Удаляет записи m/v для параметров, чьих ссылок нет в {@code activeParameters} (сравнение по
     * ссылке, как в {@link IdentityHashMap}).
     */
    public void retainMomentumOnlyFor(List<Tensor> activeParameters) {
        Objects.requireNonNull(activeParameters, "activeParameters");
        IdentityHashMap<Tensor, Boolean> keep = new IdentityHashMap<>();
        for (Tensor t : activeParameters) {
            if (t != null) {
                keep.put(t, Boolean.TRUE);
            }
        }
        m.keySet().removeIf(t -> !keep.containsKey(t));
        v.keySet().removeIf(t -> !keep.containsKey(t));
    }

    /**
     * Синхронизация с {@link LLMTrainer#loadCheckpoint} для старых чекпоинтов без m/v (только номер шага).
     */
    public void setStep(int step) {
        if (step < 0) {
            throw new IllegalArgumentException("step must be >= 0");
        }
        this.step = step;
        recomputeBiasCorrection();
    }

    public int getStep() {
        return step;
    }

    /**
     * Сохраняет первые и вторые моменты Adam для параметров в том же порядке, что {@code parameters}.
     */
    public void writeMomentBuffers(DataOutputStream out, List<Tensor> parameters) throws IOException {
        Objects.requireNonNull(out, "out");
        Objects.requireNonNull(parameters, "parameters");
        syncMomentBuffersFromGpu();
        out.writeInt(parameters.size());
        for (Tensor param : parameters) {
            int[] shape = param.getShape();
            out.writeInt(shape.length);
            for (int d : shape) {
                out.writeInt(d);
            }
            int len = param.internalBuffer().length;
            Tensor mT = m.get(param);
            Tensor vT = v.get(param);
            float[] mBuf = mT != null ? mT.internalBuffer() : new float[len];
            float[] vBuf = vT != null ? vT.internalBuffer() : new float[len];
            for (int i = 0; i < len; i++) {
                out.writeFloat(mBuf[i]);
            }
            for (int i = 0; i < len; i++) {
                out.writeFloat(vBuf[i]);
            }
        }
    }

    /**
     * Восстанавливает m/v; перед вызовом нужен {@link #setStep(int)} из заголовка чекпоинта.
     */
    public void readMomentBuffers(DataInputStream in, List<Tensor> parameters) throws IOException {
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(parameters, "parameters");
        m.clear();
        v.clear();
        closeGpuMomentBuffers();
        int n = in.readInt();
        if (n != parameters.size()) {
            throw new IOException(
                    "Adam state: parameter count mismatch, checkpoint has "
                            + n
                            + ", model has "
                            + parameters.size());
        }
        for (Tensor param : parameters) {
            int rank = in.readInt();
            int[] shape = new int[rank];
            for (int i = 0; i < rank; i++) {
                shape[i] = in.readInt();
            }
            if (!Arrays.equals(shape, param.getShape())) {
                throw new IOException(
                        "Adam state: shape mismatch, expected " + Arrays.toString(param.getShape()));
            }
            int len = param.internalBuffer().length;
            Tensor mT = new Tensor(shape);
            Tensor vT = new Tensor(shape);
            float[] mBuf = mT.internalBuffer();
            float[] vBuf = vT.internalBuffer();
            for (int i = 0; i < len; i++) {
                mBuf[i] = in.readFloat();
            }
            for (int i = 0; i < len; i++) {
                vBuf[i] = in.readFloat();
            }
            m.put(param, mT);
            v.put(param, vT);
        }
    }

    private final Map<Tensor, GpuTensor> gpuM = new IdentityHashMap<>();
    private final Map<Tensor, GpuTensor> gpuV = new IdentityHashMap<>();

    /** Grow-only scratch lists, reused каждый шаг в {@link #stepAllGpuDevice}. */
    private final List<GpuTensor> stepScratchParams  = new ArrayList<>();
    private final List<GpuTensor> stepScratchMStates = new ArrayList<>();
    private final List<GpuTensor> stepScratchVStates = new ArrayList<>();

    public void syncMomentBuffersFromGpu() {
        for (Map.Entry<Tensor, GpuTensor> e : gpuM.entrySet()) {
            Tensor cpuParam = e.getKey();
            GpuTensor mGpu = e.getValue();
            if (mGpu == null || mGpu.isClosed()) {
                continue;
            }
            Tensor cpuMT = m.get(cpuParam);
            if (cpuMT == null) {
                cpuMT = new Tensor(cpuParam.getShape());
                m.put(cpuParam, cpuMT);
            }
            mGpu.downloadTo(cpuMT.internalBuffer(), 0, cpuMT.size());
        }
        for (Map.Entry<Tensor, GpuTensor> e : gpuV.entrySet()) {
            Tensor cpuParam = e.getKey();
            GpuTensor vGpu = e.getValue();
            if (vGpu == null || vGpu.isClosed()) {
                continue;
            }
            Tensor cpuVT = v.get(cpuParam);
            if (cpuVT == null) {
                cpuVT = new Tensor(cpuParam.getShape());
                v.put(cpuParam, cpuVT);
            }
            vGpu.downloadTo(cpuVT.internalBuffer(), 0, cpuVT.size());
        }
    }

    private void closeGpuMomentBuffers() {
        for (GpuTensor t : gpuM.values()) {
            if (t != null && !t.isClosed()) {
                t.close();
            }
        }
        for (GpuTensor t : gpuV.values()) {
            if (t != null && !t.isClosed()) {
                t.close();
            }
        }
        gpuM.clear();
        gpuV.clear();
    }

    /** Освобождает VRAM для m/v Adam на GPU; хостовые карты моментов сохраняются. Перед сменой книги/модели в том же JVM. */
    public void releaseGpuMomentBuffers() {
        closeGpuMomentBuffers();
    }

    /**
     * Full-GPU Adam step for all parameters in paramMap. Lazily creates GPU-side m/v states.
     */
    public void stepAllGpuDevice(Map<Tensor, GpuTensor> paramMap) {
        if (step <= 0) {
            throw new IllegalStateException("call beginStep() before stepAllGpuDevice");
        }
        stepScratchParams.clear();
        stepScratchMStates.clear();
        stepScratchVStates.clear();
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            Tensor cpuParam = e.getKey();
            GpuTensor gt = e.getValue();
            if (!gt.hasGradBuffer()) continue;
            GpuTensor mGpu = gpuM.get(cpuParam);
            GpuTensor vGpu = gpuV.get(cpuParam);
            if (mGpu == null || mGpu.isClosed()) {
                Tensor cpuMT = m.get(cpuParam);
                mGpu = cpuMT == null
                        ? GpuTensor.allocate(cpuParam.getShape())
                        : GpuTensor.fromHostTensor(cpuMT);
                gpuM.put(cpuParam, mGpu);
            }
            if (vGpu == null || vGpu.isClosed()) {
                Tensor cpuVT = v.get(cpuParam);
                vGpu = cpuVT == null
                        ? GpuTensor.allocate(cpuParam.getShape())
                        : GpuTensor.fromHostTensor(cpuVT);
                gpuV.put(cpuParam, vGpu);
            }
            stepScratchParams.add(gt);
            stepScratchMStates.add(mGpu);
            stepScratchVStates.add(vGpu);
        }
        if (!stepScratchParams.isEmpty()) {
            stepAllGpu(stepScratchParams, stepScratchMStates, stepScratchVStates);
        }
    }
}
