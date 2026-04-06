package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.GpuIntBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.model.BlockActivationCache;
import com.veles.llm.jgpt.model.BlockActivationCacheDevice;
import com.veles.llm.jgpt.util.DebugGpuTrain;

import java.util.Objects;

/**
 * Backward pass for transformer block (attention + SwiGLU + residual + RMSNorm).
 *
 * <p><b>Thread-safety:</b> All methods are stateless and thread-safe.
 *
 * <p><b>Memory model:</b> Intermediate tensors are allocated per call; consider reusing workspace
 * objects for long sequences.
 *
 * @implNote Вычисления через {@link TensorOpsGPU} (CUDA).
 */
public final class TransformerBackward {

    private static void agentLogB39372RmsTargets(
            float dx0, float dx1, float ng0, float ng1, int rows, int dModel) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H_rms_accum_tgt\",\"location\":\"TransformerBackward.transformerBlockBackwardGpuDevice\",\"message\":\"pre_clear_rmsnorm1_out_buffers\",\"data\":{\"rows\":"
                        + rows
                        + ",\"dModel\":"
                        + dModel
                        + ",\"dx0\":"
                        + dx0
                        + ",\"dx1\":"
                        + dx1
                        + ",\"ng0\":"
                        + ng0
                        + ",\"ng1\":"
                        + ng1
                        + "},\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }

    private TransformerBackward() {}

    private static boolean canUseAttnResidentWeights(
            TensorOps.GpuAttnResidentBuffers r,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int dModelSq) {
        if (r == null) {
            return false;
        }
        if (Wq.size() != dModelSq
                || Wk.size() != dModelSq
                || Wv.size() != dModelSq
                || Wo.size() != dModelSq) {
            return false;
        }
        return r.wq().numFloats() >= dModelSq
                && r.wk().numFloats() >= dModelSq
                && r.wv().numFloats() >= dModelSq
                && r.wo().numFloats() >= dModelSq;
    }

    private static boolean canUseFfnResidentWeights(
            TensorOps.GpuFfnResidentBuffers r,
            Tensor W1,
            Tensor W2,
            Tensor W3,
            Tensor norm2,
            int dModel,
            int dInt) {
        if (r == null) {
            return false;
        }
        int w1El = dModel * dInt;
        int w2El = dInt * dModel;
        int w3El = dModel * dInt;
        if (W1.size() != w1El || W2.size() != w2El || W3.size() != w3El || norm2.size() != dModel) {
            return false;
        }
        return r.normGamma().numFloats() >= dModel
                && r.w1().numFloats() >= w1El
                && r.w2().numFloats() >= w2El
                && r.w3().numFloats() >= w3El;
    }

    /**
     * Backward RoPE: ∂L/∂x = Rᵀ ∂L/∂y.
     *
     * @param gradY gradient w.r.t. rotated output [B,H,S,D]
     * @param gradX tensor to accumulate gradient w.r.t. input
     * @param positions optional position indices ({@code null} = 0..S-1)
     */
    public static void applyRoPEBackward(Tensor gradY, Tensor gradX, int[] positions) {
        Objects.requireNonNull(gradY, "gradY cannot be null");
        Objects.requireNonNull(gradX, "gradX cannot be null");
        int[] shape = gradY.getShape();
        int batch = shape[0];
        int numHeads = shape[1];
        int seqLen = shape[2];
        int dHead = shape[3];
        if (dHead % 2 != 0) {
            throw new IllegalArgumentException("d_head must be even for RoPE");
        }

        if (positions != null && positions.length < seqLen) {
            throw new IllegalArgumentException("positions length " + positions.length + " < seq_len " + seqLen);
        }

        if (!gradX.hasGrad()) {
            gradX.zeroGrad();
        }
        float[] gy = gradY.hasGrad() ? gradY.gradBuffer() : gradY.internalBuffer();
        float[] gx = gradX.gradBuffer();
        int n = batch * numHeads * seqLen * dHead;
        if (n <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TransformerBackward.applyRoPEBackward");
        TensorOpsGPU.applyRoPEBackward4DGPU(gy, gx, batch, numHeads, seqLen, dHead, positions, 0);
    }

    /**
     * Scaled dot-product attention backward.
     *
     * @param gradOut ∂L/∂O [B,S,Dv] with grad buffer allocated
     * @param mask optional causal mask
     * @param probsCached optional cached attention weights (avoids recomputation)
     */
    public static void scaledDotProductAttentionBackward(
            Tensor gradOut,
            Tensor Q,
            Tensor K,
            Tensor V,
            Tensor mask,
            float scale,
            Tensor probsCached,
            Tensor gradQ,
            Tensor gradK,
            Tensor gradV) {
        Objects.requireNonNull(gradOut, "gradOut cannot be null");
        Objects.requireNonNull(Q, "Q cannot be null");
        Objects.requireNonNull(K, "K cannot be null");
        Objects.requireNonNull(V, "V cannot be null");
        Objects.requireNonNull(gradQ, "gradQ cannot be null");
        Objects.requireNonNull(gradK, "gradK cannot be null");
        Objects.requireNonNull(gradV, "gradV cannot be null");
        int[] qShape = Q.getShape();
        int batch = qShape[0];
        int seqLen = qShape[1];
        int dK = qShape[2];
        int dV = V.getShape()[2];

        if (!gradQ.hasGrad()) {
            gradQ.zeroGrad();
        }
        if (!gradK.hasGrad()) {
            gradK.zeroGrad();
        }
        if (!gradV.hasGrad()) {
            gradV.zeroGrad();
        }

        Tensor gradOutData = Tensor.wrap(gradOut.gradBuffer(), gradOut.getShape());
        Tensor probs = probsCached;
        if (probs == null) {
            Tensor kT = TensorOps.transpose2DLast(K);
            Tensor scores = TensorOps.matmulBatched3D(Q, kT);
            Tensor scaled = TensorOps.multiplyScalar(scores, scale);
            if (mask != null) {
                scaled = TensorOps.applyCausalMask(scaled, mask);
            }
            probs = TensorOps.softmaxLastDim(scaled);
        }

        TensorOpsGPU.requireCuda("TransformerBackward.scaledDotProductAttentionBackward");
        if (!TensorOpsGPU.shouldUseGpuMatmulBatched(batch, seqLen, seqLen, Math.max(dK, dV))) {
            throw new IllegalArgumentException(
                    "scaledDotProductAttentionBackward: ожидаются положительные batch, seqLen, dK, dV");
        }
        TensorOpsGPU.scaledDotProductAttentionBackwardGPU(
                gradOutData.internalBuffer(),
                probs.internalBuffer(),
                Q.internalBuffer(),
                K.internalBuffer(),
                V.internalBuffer(),
                gradQ.gradBuffer(),
                gradK.gradBuffer(),
                gradV.gradBuffer(),
                batch,
                seqLen,
                dK,
                dV,
                scale,
                mask != null ? mask.internalBuffer() : null,
                TensorOpsGPU.useFp16Matmul());
    }

    /**
     * SwiGLU: (xW₁ ⊙ SiLU(xW₃)) W₂.
     */
    public static void feedForwardSwiGLUBackward(
            Tensor gradOut3d,
            Tensor x,
            Tensor W1,
            Tensor W2,
            Tensor W3,
            Tensor gradX,
            Tensor gradW1,
            Tensor gradW2,
            Tensor gradW3) {
        Objects.requireNonNull(gradOut3d, "gradOut3d cannot be null");
        Objects.requireNonNull(x, "x cannot be null");
        Objects.requireNonNull(W1, "W1 cannot be null");
        Objects.requireNonNull(W2, "W2 cannot be null");
        Objects.requireNonNull(W3, "W3 cannot be null");
        Objects.requireNonNull(gradX, "gradX cannot be null");
        Objects.requireNonNull(gradW1, "gradW1 cannot be null");
        Objects.requireNonNull(gradW2, "gradW2 cannot be null");
        Objects.requireNonNull(gradW3, "gradW3 cannot be null");
        feedForwardSwiGLUBackward(gradOut3d, x, W1, W2, W3, gradX, gradW1, gradW2, gradW3, null, null, null, null, null);
    }

    /**
     * То же, что {@link #feedForwardSwiGLUBackward(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)};
     * при непустом кэше из forward (все поля из {@link BlockActivationCache}) не пересчитываются matmul для h1/gate/σ.
     *
     * <p>Intermediate {@link Tensor}s are short-lived; memory is reclaimed by the GC after return (no
     * manual free). For very large {@code batch×seq×d} consider a reusable workspace API later.
     */
    public static void feedForwardSwiGLUBackward(
            Tensor gradOut3d,
            Tensor x,
            Tensor W1,
            Tensor W2,
            Tensor W3,
            Tensor gradX,
            Tensor gradW1,
            Tensor gradW2,
            Tensor gradW3,
            Tensor ffnH1,
            Tensor ffnGate,
            Tensor ffnSig,
            Tensor ffnGateSwish,
            Tensor ffnHActivated) {
        Objects.requireNonNull(gradOut3d, "gradOut3d cannot be null");
        Objects.requireNonNull(x, "x cannot be null");
        Objects.requireNonNull(W1, "W1 cannot be null");
        Objects.requireNonNull(W2, "W2 cannot be null");
        Objects.requireNonNull(W3, "W3 cannot be null");
        Objects.requireNonNull(gradX, "gradX cannot be null");
        Objects.requireNonNull(gradW1, "gradW1 cannot be null");
        Objects.requireNonNull(gradW2, "gradW2 cannot be null");
        Objects.requireNonNull(gradW3, "gradW3 cannot be null");
        int[] xs = x.getShape();
        int batch = xs[0];
        int seqLen = xs[1];
        int dModel = xs[2];
        int dInt = W1.getShape()[1];

        if (!gradX.hasGrad()) {
            gradX.zeroGrad();
        }
        if (!gradW1.hasGrad()) {
            gradW1.zeroGrad();
        }
        if (!gradW2.hasGrad()) {
            gradW2.zeroGrad();
        }
        if (!gradW3.hasGrad()) {
            gradW3.zeroGrad();
        }

        boolean useFfnCache = ffnH1 != null && ffnGate != null;
        int rows = batch * seqLen;
        Tensor xFlat = Tensor.wrap(x.internalBuffer(), new int[]{rows, dModel});
        Tensor goFlat = Tensor.wrap(gradOut3d.gradBuffer(), new int[]{rows, dModel});

        Tensor h1Flat;
        Tensor gateFlat;
        Tensor sigFlat;
        Tensor gateSwishFlat;
        Tensor hActFlat;
        if (useFfnCache) {
            h1Flat = Tensor.wrap(ffnH1.internalBuffer(), new int[]{rows, dInt});
            gateFlat = Tensor.wrap(ffnGate.internalBuffer(), new int[]{rows, dInt});
            sigFlat = ffnSig != null ? Tensor.wrap(ffnSig.internalBuffer(), new int[]{rows, dInt}) : TensorOps.sigmoid(gateFlat);
            gateSwishFlat =
                    ffnGateSwish != null
                            ? Tensor.wrap(ffnGateSwish.internalBuffer(), new int[]{rows, dInt})
                            : TensorOps.multiply(gateFlat, sigFlat);
            hActFlat =
                    ffnHActivated != null
                            ? Tensor.wrap(ffnHActivated.internalBuffer(), new int[]{rows, dInt})
                            : TensorOps.multiply(h1Flat, gateSwishFlat);
        } else {
            h1Flat = TensorOps.matmul(xFlat, W1);
            gateFlat = TensorOps.matmul(xFlat, W3);
            sigFlat = TensorOps.sigmoid(gateFlat);
            gateSwishFlat = TensorOps.multiply(gateFlat, sigFlat);
            hActFlat = TensorOps.multiply(h1Flat, gateSwishFlat);
        }

        Tensor w2T = TensorOpsBackward.transpose(W2);
        Tensor w1T = TensorOpsBackward.transpose(W1);
        Tensor w3T = TensorOpsBackward.transpose(W3);

        Tensor dHact = TensorOps.matmul(goFlat, w2T);
        TensorOpsBackward.accumulateGradientInto(gradW2, TensorOps.matmul(TensorOpsBackward.transpose(hActFlat), goFlat));

        Tensor dHactGrad = new Tensor(dHact.getShape());
        dHactGrad.zeroGrad();
        System.arraycopy(dHact.internalBuffer(), 0, dHactGrad.gradBuffer(), 0, dHact.internalBuffer().length);

        Tensor dH1 = new Tensor(new int[]{rows, dInt});
        Tensor dGateSwish = new Tensor(new int[]{rows, dInt});
        dH1.zeroGrad();
        dGateSwish.zeroGrad();
        TensorOpsBackward.multiplyBackward(dHactGrad, h1Flat, gateSwishFlat, dH1, dGateSwish);

        Tensor dGate = new Tensor(new int[]{rows, dInt});
        Tensor dSig = new Tensor(new int[]{rows, dInt});
        dGate.zeroGrad();
        dSig.zeroGrad();
        TensorOpsBackward.multiplyBackward(dGateSwish, gateFlat, sigFlat, dGate, dSig);
        TensorOpsBackward.sigmoidBackward(dSig, gateFlat, dGate);

        Tensor dXh = TensorOps.matmul(Tensor.wrap(dH1.gradBuffer(), new int[]{rows, dInt}), w1T);
        Tensor dXg = TensorOps.matmul(Tensor.wrap(dGate.gradBuffer(), new int[]{rows, dInt}), w3T);
        float[] gx = gradX.gradBuffer();
        float[] dxh = dXh.internalBuffer();
        float[] dxg = dXg.internalBuffer();
        for (int i = 0; i < gx.length; i++) {
            gx[i] += dxh[i] + dxg[i];
        }

        TensorOpsBackward.accumulateGradientInto(
                gradW1,
                TensorOps.matmul(TensorOpsBackward.transpose(xFlat), Tensor.wrap(dH1.gradBuffer(), new int[]{rows, dInt})));
        TensorOpsBackward.accumulateGradientInto(
                gradW3,
                TensorOps.matmul(TensorOpsBackward.transpose(xFlat), Tensor.wrap(dGate.gradBuffer(), new int[]{rows, dInt})));
    }

    /**
     * MHA с RoPE на Q/K после split heads.
     */
    public static void multiHeadAttentionWithRoPEBackward(
            Tensor gradOut,
            Tensor xNorm,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            Tensor mask,
            BlockActivationCache cache,
            Tensor gradX,
            Tensor gradWq,
            Tensor gradWk,
            Tensor gradWv,
            Tensor gradWo) {
        multiHeadAttentionWithRoPEBackward(
                gradOut,
                xNorm,
                Wq,
                Wk,
                Wv,
                Wo,
                numHeads,
                mask,
                cache,
                gradX,
                gradWq,
                gradWk,
                gradWv,
                gradWo,
                null);
    }

    /**
     * @param mask зарезервирован ( softmax/mask уже учтены в {@code attnProbs} из кэша).
     * @param attnResident при ненулевом значении и совпадении размеров — fused GPU backward использует VRAM-веса без
     *     H2D копий в workspace (см. {@link TensorOps.GpuAttnResidentBuffers}).
     */
    @SuppressWarnings("unused")
    public static void multiHeadAttentionWithRoPEBackward(
            Tensor gradOut,
            Tensor xNorm,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            Tensor mask,
            BlockActivationCache cache,
            Tensor gradX,
            Tensor gradWq,
            Tensor gradWk,
            Tensor gradWv,
            Tensor gradWo,
            TensorOps.GpuAttnResidentBuffers attnResident) {
        Objects.requireNonNull(gradOut, "gradOut cannot be null");
        Objects.requireNonNull(xNorm, "xNorm cannot be null");
        Objects.requireNonNull(Wq, "Wq cannot be null");
        Objects.requireNonNull(Wk, "Wk cannot be null");
        Objects.requireNonNull(Wv, "Wv cannot be null");
        Objects.requireNonNull(Wo, "Wo cannot be null");
        Objects.requireNonNull(gradX, "gradX cannot be null");
        Objects.requireNonNull(gradWq, "gradWq cannot be null");
        Objects.requireNonNull(gradWk, "gradWk cannot be null");
        Objects.requireNonNull(gradWv, "gradWv cannot be null");
        Objects.requireNonNull(gradWo, "gradWo cannot be null");
        int[] xs = xNorm.getShape();
        int batch = xs[0];
        int seqLen = xs[1];
        int dModel = xs[2];

        if (!gradOut.hasGrad()) {
            throw new IllegalStateException("gradOut must have grad buffer");
        }

        int work = batch * seqLen * dModel;
        if (work <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TransformerBackward.multiHeadAttentionWithRoPEBackward");

        if (!tryFusedAttentionBackwardGpu(
                gradOut,
                xNorm,
                Wq,
                Wk,
                Wv,
                Wo,
                numHeads,
                cache,
                gradX,
                gradWq,
                gradWk,
                gradWv,
                gradWo,
                attnResident)) {
            throw new IllegalStateException(
                    "multiHeadAttentionWithRoPEBackward требует полный BlockActivationCache от forward при обучении: "
                            + "attnQHeads, attnKHeads, attnVHeads, attnProbs, attnConcat (см. TensorOps MHA forward).");
        }
    }

    /**
     * Fused GPU backward for MHA+RoPE when cache holds Q/K/V/probs/concat.
     *
     * <p>Gradients are queued via {@link GpuPendingGradients}; the trainer calls {@link
     * GpuPendingGradients#flushAllToHost()} before the optimizer step.
     */
    static boolean tryFusedAttentionBackwardGpu(
            Tensor gradOut,
            Tensor xNorm,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            BlockActivationCache cache,
            Tensor gradX,
            Tensor gradWq,
            Tensor gradWk,
            Tensor gradWv,
            Tensor gradWo,
            TensorOps.GpuAttnResidentBuffers attnResident) {
        if (cache == null
                || cache.attnQHeads.isEmpty()
                || cache.attnKHeads.isEmpty()
                || cache.attnVHeads.isEmpty()
                || cache.attnProbs.isEmpty()
                || cache.attnConcat.isEmpty()) {
            return false;
        }

        int[] xs = xNorm.getShape();
        int batch = xs[0];
        int seqLen = xs[1];
        int dModel = xs[2];
        int dHead = dModel / numHeads;
        int rows = batch * seqLen;
        int batchHeads = batch * numHeads;
        float attScale = 1.0f / (float) Math.sqrt(dHead);

        int dModelSq = dModel * dModel;
        boolean useResidentW = canUseAttnResidentWeights(attnResident, Wq, Wk, Wv, Wo, dModelSq);

        GpuAttentionBackwardWorkspace ws = GpuAttentionBackwardWorkspace.local();
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(batch, numHeads, seqLen, dModel);
            ws.getXFlat().copyFrom(xNorm.internalBuffer(), 0, rows * dModel);
            ws.getGradOutFlat().copyFrom(gradOut.gradBuffer(), 0, rows * dModel);
            if (!useResidentW) {
                ws.getWq().copyFrom(Wq.internalBuffer(), 0, dModelSq);
                ws.getWk().copyFrom(Wk.internalBuffer(), 0, dModelSq);
                ws.getWv().copyFrom(Wv.internalBuffer(), 0, dModelSq);
                ws.getWo().copyFrom(Wo.internalBuffer(), 0, dModelSq);
            }
            ws.getQHeads().copyFrom(cache.attnQHeads.getTensor().internalBuffer(), 0, rows * dModel);
            ws.getKHeads().copyFrom(cache.attnKHeads.getTensor().internalBuffer(), 0, rows * dModel);
            ws.getVHeads().copyFrom(cache.attnVHeads.getTensor().internalBuffer(), 0, rows * dModel);
            ws.getProbs().copyFrom(cache.attnProbs.getTensor().internalBuffer(), 0, batchHeads * seqLen * seqLen);
            ws.getConcat().copyFrom(cache.attnConcat.getTensor().internalBuffer(), 0, rows * dModel);

            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getGradOutFlat(),
                    useResidentW ? attnResident.wo() : ws.getWo(),
                    ws.getGradConcat(),
                    rows,
                    dModel,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.splitHeadsGpuDevice(
                    ws.getGradConcat(), ws.getDHeads(), batch, seqLen, dModel, numHeads);
            /* Кэшированные probs: FP32 softmax-bwd от p стабильнее fp16-recompute из QK. */
            TensorOpsGPU.scaledDotProductAttentionBackwardGpuDevice(
                    ws.getDHeads(),
                    ws.getProbs(),
                    ws.getQHeads(),
                    ws.getKHeads(),
                    ws.getVHeads(),
                    ws.getGradQh(),
                    ws.getGradKh(),
                    ws.getGradVh(),
                    batchHeads,
                    seqLen,
                    dHead,
                    dHead,
                    attScale,
                    0L,
                    false);

            ws.getQHeads().clear();
            ws.getKHeads().clear();
            TensorOpsGPU.applyRoPEBackwardGpuDevice(
                    ws.getGradQh(), ws.getQHeads(), batch, numHeads, seqLen, dHead);
            TensorOpsGPU.applyRoPEBackwardGpuDevice(
                    ws.getGradKh(), ws.getKHeads(), batch, numHeads, seqLen, dHead);

            TensorOpsGPU.concatHeadsGpuDevice(ws.getQHeads(), ws.getDQ(), batch, numHeads, seqLen, dHead);
            TensorOpsGPU.concatHeadsGpuDevice(ws.getKHeads(), ws.getDK(), batch, numHeads, seqLen, dHead);
            TensorOpsGPU.concatHeadsGpuDevice(ws.getGradVh(), ws.getDV(), batch, numHeads, seqLen, dHead);

            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDQ(),
                    useResidentW ? attnResident.wq() : ws.getWq(),
                    ws.getGradX(),
                    rows,
                    dModel,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDK(),
                    useResidentW ? attnResident.wk() : ws.getWk(),
                    ws.getGradTmp(),
                    rows,
                    dModel,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getGradX(), ws.getGradTmp(), rows * dModel);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDV(),
                    useResidentW ? attnResident.wv() : ws.getWv(),
                    ws.getGradTmp(),
                    rows,
                    dModel,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getGradX(), ws.getGradTmp(), rows * dModel);

            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDQ(), ws.getGradWq(), dModel, rows, dModel, true, false);
            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDK(), ws.getGradWk(), dModel, rows, dModel, true, false);
            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDV(), ws.getGradWv(), dModel, rows, dModel, true, false);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getConcat(), ws.getGradOutFlat(), ws.getGradWo(), dModel, rows, dModel, true, false);

            if (!gradX.hasGrad()) {
                gradX.zeroGrad();
            }
            ws.getGradX().copyTo(gradX.gradBuffer(), 0, rows * dModel);
            GpuPendingGradients.accumulate(gradWq, ws.getGradWq(), dModel * dModel);
            GpuPendingGradients.accumulate(gradWk, ws.getGradWk(), dModel * dModel);
            GpuPendingGradients.accumulate(gradWv, ws.getGradWv(), dModel * dModel);
            GpuPendingGradients.accumulate(gradWo, ws.getGradWo(), dModel * dModel);
        }

        return true;
    }

    /**
     * Decoder-блок (pre-norm): RMSNorm → MHA+RoPE → + → RMSNorm → SwiGLU → +.
     */
    public static void transformerBlockBackward(
            Tensor gradOut,
            BlockActivationCache cache,
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
            boolean useRoPE,
            Tensor gradXIn,
            Tensor gradWq,
            Tensor gradWk,
            Tensor gradWv,
            Tensor gradWo,
            Tensor gradW1,
            Tensor gradW2,
            Tensor gradW3,
            Tensor gradNorm1,
            Tensor gradNorm2) {
        transformerBlockBackward(
                gradOut,
                cache,
                Wq,
                Wk,
                Wv,
                Wo,
                W1,
                W2,
                W3,
                norm1,
                norm2,
                numHeads,
                mask,
                useRoPE,
                gradXIn,
                gradWq,
                gradWk,
                gradWv,
                gradWo,
                gradW1,
                gradW2,
                gradW3,
                gradNorm1,
                gradNorm2,
                null,
                null);
    }

    /**
     * @param attnResident при ненулевом значении — fused attention backward без H2D копий Q/K/V/O весов.
     * @param ffnResident при ненулевом значении — fused FFN backward без H2D копий SwiGLU и γ второго norm.
     */
    public static void transformerBlockBackward(
            Tensor gradOut,
            BlockActivationCache cache,
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
            boolean useRoPE,
            Tensor gradXIn,
            Tensor gradWq,
            Tensor gradWk,
            Tensor gradWv,
            Tensor gradWo,
            Tensor gradW1,
            Tensor gradW2,
            Tensor gradW3,
            Tensor gradNorm1,
            Tensor gradNorm2,
            TensorOps.GpuAttnResidentBuffers attnResident,
            TensorOps.GpuFfnResidentBuffers ffnResident) {
        Objects.requireNonNull(gradOut, "gradOut cannot be null");
        Objects.requireNonNull(cache, "cache cannot be null");
        Objects.requireNonNull(Wq, "Wq cannot be null");
        Objects.requireNonNull(Wk, "Wk cannot be null");
        Objects.requireNonNull(Wv, "Wv cannot be null");
        Objects.requireNonNull(Wo, "Wo cannot be null");
        Objects.requireNonNull(W1, "W1 cannot be null");
        Objects.requireNonNull(W2, "W2 cannot be null");
        Objects.requireNonNull(W3, "W3 cannot be null");
        Objects.requireNonNull(norm1, "norm1 cannot be null");
        Objects.requireNonNull(norm2, "norm2 cannot be null");
        Objects.requireNonNull(gradXIn, "gradXIn cannot be null");
        Objects.requireNonNull(gradWq, "gradWq cannot be null");
        Objects.requireNonNull(gradWk, "gradWk cannot be null");
        Objects.requireNonNull(gradWv, "gradWv cannot be null");
        Objects.requireNonNull(gradWo, "gradWo cannot be null");
        Objects.requireNonNull(gradW1, "gradW1 cannot be null");
        Objects.requireNonNull(gradW2, "gradW2 cannot be null");
        Objects.requireNonNull(gradW3, "gradW3 cannot be null");
        Objects.requireNonNull(gradNorm1, "gradNorm1 cannot be null");
        Objects.requireNonNull(gradNorm2, "gradNorm2 cannot be null");
        Tensor xIn = cache.xIn.getTensor();
        Tensor xNorm1 = cache.xNorm1.getTensor();
        Tensor attnOut = cache.attnOut.getTensor();
        Tensor xRes1 = cache.xRes1.getTensor();
        Tensor xNorm2 = cache.xNorm2.getTensor();

        float[] gOut = gradOut.gradBuffer();
        int[] xis = xIn.getShape();
        int blockWork = xis[0] * xis[1] * xis[2];
        if (blockWork > 0) {
            TensorOpsGPU.requireCuda("TransformerBackward.transformerBlockBackward");
        }

        Tensor gradXRes1 = new Tensor(xRes1.getShape());
        gradXRes1.zeroGrad();
        if (!tryFusedFfnNormResidualBackwardGpu(
                gradOut,
                cache,
                xRes1,
                xNorm2,
                W1,
                W2,
                W3,
                norm2,
                gradXRes1,
                gradW1,
                gradW2,
                gradW3,
                gradNorm2,
                ffnResident)) {
            throw new IllegalStateException(
                    "transformerBlockBackward требует активации SwiGLU в кэше: ffnH1, ffnGate (и при необходимости "
                            + "остальные слоты ffn*), как при forward с training=true.");
        }

        Tensor gradAttn = new Tensor(attnOut.getShape());
        gradAttn.zeroGrad();
        System.arraycopy(gradXRes1.gradBuffer(), 0, gradAttn.gradBuffer(), 0, gOut.length);

        Tensor gradXNorm1Path = new Tensor(xNorm1.getShape());
        gradXNorm1Path.zeroGrad();

        if (!useRoPE) {
            throw new UnsupportedOperationException("MHA without RoPE backward not implemented");
        }
        multiHeadAttentionWithRoPEBackward(
                gradAttn,
                xNorm1,
                Wq,
                Wk,
                Wv,
                Wo,
                numHeads,
                mask,
                cache,
                gradXNorm1Path,
                gradWq,
                gradWk,
                gradWv,
                gradWo,
                attnResident);

        TensorOpsBackward.rmsNormBackward(
                gradXNorm1Path, xIn, norm1, TensorOpsGPU.rmsNormEps(), gradXIn, gradNorm1);

        float[] gxin = gradXIn.gradBuffer();
        float[] gxres = gradXRes1.gradBuffer();
        if (gxin.length > 0) {
            TensorOpsGPU.requireCuda("TransformerBackward.transformerBlockBackward/residual");
            TensorOpsGPU.accumulateAddGPU(gxin, gxres, gxin.length);
        }
    }

    /**
     * Fused GPU backward for FFN + second RMSNorm + residual branch.
     *
     * <p>Same {@link GpuPendingGradients} contract as {@link #tryFusedAttentionBackwardGpu}.
     */
    static boolean tryFusedFfnNormResidualBackwardGpu(
            Tensor gradOut,
            BlockActivationCache cache,
            Tensor xRes1,
            Tensor xNorm2,
            Tensor W1,
            Tensor W2,
            Tensor W3,
            Tensor norm2,
            Tensor gradXRes1,
            Tensor gradW1,
            Tensor gradW2,
            Tensor gradW3,
            Tensor gradNorm2,
            TensorOps.GpuFfnResidentBuffers ffnResident) {
        if (cache == null
                || cache.ffnH1.isEmpty()
                || cache.ffnGate.isEmpty()) {
            return false;
        }

        int[] xs = xNorm2.getShape();
        int rows = xs[0] * xs[1];
        int dModel = xs[2];
        int dInt = W1.getShape()[1];

        boolean useResidentW = canUseFfnResidentWeights(ffnResident, W1, W2, W3, norm2, dModel, dInt);

        GpuBlockWorkspace ws = GpuBlockWorkspace.local();
        synchronized (ws.exclusiveUseLock()) {
            ws.ensureFfnNorm2(rows, dModel, dInt);
            ws.getXFlat().copyFrom(xNorm2.internalBuffer(), 0, rows * dModel);
            ws.getGradOut().copyFrom(gradOut.gradBuffer(), 0, rows * dModel);
            ws.getXRes1().copyFrom(xRes1.internalBuffer(), 0, rows * dModel);
            if (!useResidentW) {
                ws.getNormGamma().copyFrom(norm2.internalBuffer(), 0, dModel);
                ws.getW1().copyFrom(W1.internalBuffer(), 0, dModel * dInt);
                ws.getW2().copyFrom(W2.internalBuffer(), 0, dInt * dModel);
                ws.getW3().copyFrom(W3.internalBuffer(), 0, dModel * dInt);
            }
            ws.getH1().copyFrom(cache.ffnH1.getTensor().internalBuffer(), 0, rows * dInt);
            ws.getGate().copyFrom(cache.ffnGate.getTensor().internalBuffer(), 0, rows * dInt);
            if (!cache.ffnSig.isEmpty()) {
                ws.getSig().copyFrom(cache.ffnSig.getTensor().internalBuffer(), 0, rows * dInt);
            } else {
                TensorOpsGPU.sigmoidGpuDevice(ws.getGate(), ws.getSig(), rows * dInt);
            }
            if (!cache.ffnGateSwish.isEmpty()) {
                ws.getGateSwish().copyFrom(cache.ffnGateSwish.getTensor().internalBuffer(), 0, rows * dInt);
            } else {
                TensorOpsGPU.multiplyGpuDevice(ws.getGate(), ws.getSig(), ws.getGateSwish(), rows * dInt);
            }
            if (!cache.ffnHActivated.isEmpty()) {
                ws.getHAct().copyFrom(cache.ffnHActivated.getTensor().internalBuffer(), 0, rows * dInt);
            } else {
                TensorOpsGPU.multiplyGpuDevice(ws.getH1(), ws.getGateSwish(), ws.getHAct(), rows * dInt);
            }

            ws.getDHact().clear();
            ws.getDH1().clear();
            ws.getDGateSwish().clear();
            ws.getDGate().clear();
            ws.getDSig().clear();
            ws.getDXNorm2().clear();
            ws.getDXTmp().clear();
            ws.getDXRes1().clear();
            ws.getDNorm2().clear();
            ws.getDW1().clear();
            ws.getDW2().clear();
            ws.getDW3().clear();

            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getGradOut(),
                    useResidentW ? ffnResident.w2() : ws.getW2(),
                    ws.getDHact(),
                    rows,
                    dModel,
                    dInt,
                    false,
                    true);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getHAct(), ws.getGradOut(), ws.getDW2(), dInt, rows, dModel, true, false);
            TensorOpsGPU.multiplyBackwardGpuDevice(
                    ws.getDHact(), ws.getH1(), ws.getGateSwish(), ws.getDH1(), ws.getDGateSwish(), rows * dInt);
            TensorOpsGPU.multiplyBackwardGpuDevice(
                    ws.getDGateSwish(), ws.getGate(), ws.getSig(), ws.getDGate(), ws.getDSig(), rows * dInt);
            TensorOpsGPU.sigmoidBackwardGpuDevice(ws.getDSig(), ws.getGate(), ws.getDGate(), rows * dInt);

            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDH1(),
                    useResidentW ? ffnResident.w1() : ws.getW1(),
                    ws.getDXNorm2(),
                    rows,
                    dInt,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDGate(),
                    useResidentW ? ffnResident.w3() : ws.getW3(),
                    ws.getDXTmp(),
                    rows,
                    dInt,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getDXNorm2(), ws.getDXTmp(), rows * dModel);

            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDH1(), ws.getDW1(), dModel, rows, dInt, true, false);
            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDGate(), ws.getDW3(), dModel, rows, dInt, true, false);

            TensorOpsGPU.rmsNormBackwardGpuDevice(
                    ws.getDXNorm2(),
                    ws.getXRes1(),
                    useResidentW ? ffnResident.normGamma() : ws.getNormGamma(),
                    TensorOpsGPU.rmsNormEps(),
                    ws.getDXRes1(),
                    ws.getDNorm2(),
                    rows,
                    dModel);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getDXRes1(), ws.getGradOut(), rows * dModel);

            ws.getDXRes1().copyTo(gradXRes1.gradBuffer(), 0, rows * dModel);
            GpuPendingGradients.accumulate(gradNorm2, ws.getDNorm2(), dModel);
            GpuPendingGradients.accumulate(gradW1, ws.getDW1(), dModel * dInt);
            GpuPendingGradients.accumulate(gradW2, ws.getDW2(), dInt * dModel);
            GpuPendingGradients.accumulate(gradW3, ws.getDW3(), dModel * dInt);
        }

        return true;
    }

    private static void fusedFfnNormResidualBackwardGpuDevice(
            GpuFloatBuffer dGradOut,
            BlockActivationCacheDevice cache,
            Tensor W1,
            Tensor W2,
            Tensor W3,
            Tensor norm2,
            GpuFloatBuffer dGradXRes1,
            Tensor gradW1,
            Tensor gradW2,
            Tensor gradW3,
            Tensor gradNorm2,
            TensorOps.GpuFfnResidentBuffers ffnResident) {
        int batch = cache.batch();
        int seqLen = cache.seqLen();
        int dModel = cache.dModel();
        int rows = batch * seqLen;
        int dInt = W1.getShape()[1];
        boolean useResidentW = canUseFfnResidentWeights(ffnResident, W1, W2, W3, norm2, dModel, dInt);

        GpuBlockWorkspace ws = GpuBlockWorkspace.local();
        synchronized (ws.exclusiveUseLock()) {
            ws.ensureFfnNorm2(rows, dModel, dInt);
            int plane = rows * dModel;
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.X_NORM2, ws.getXFlat(), plane);
            ws.getGradOut().copyFromDevice(dGradOut, plane);
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.X_RES1, ws.getXRes1(), plane);
            if (!useResidentW) {
                ws.getNormGamma().copyFrom(norm2.internalBuffer(), 0, dModel);
                ws.getW1().copyFrom(W1.internalBuffer(), 0, dModel * dInt);
                ws.getW2().copyFrom(W2.internalBuffer(), 0, dInt * dModel);
                ws.getW3().copyFrom(W3.internalBuffer(), 0, dModel * dInt);
            }
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.FFN_H1, ws.getH1(), rows * dInt);
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.FFN_GATE, ws.getGate(), rows * dInt);
            TensorOpsGPU.sigmoidGpuDevice(ws.getGate(), ws.getSig(), rows * dInt);
            TensorOpsGPU.multiplyGpuDevice(ws.getGate(), ws.getSig(), ws.getGateSwish(), rows * dInt);
            TensorOpsGPU.multiplyGpuDevice(ws.getH1(), ws.getGateSwish(), ws.getHAct(), rows * dInt);

            ws.getDHact().clear();
            ws.getDH1().clear();
            ws.getDGateSwish().clear();
            ws.getDGate().clear();
            ws.getDSig().clear();
            ws.getDXNorm2().clear();
            ws.getDXTmp().clear();
            ws.getDXRes1().clear();
            ws.getDNorm2().clear();
            ws.getDW1().clear();
            ws.getDW2().clear();
            ws.getDW3().clear();

            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getGradOut(),
                    useResidentW ? ffnResident.w2() : ws.getW2(),
                    ws.getDHact(),
                    rows,
                    dModel,
                    dInt,
                    false,
                    true);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getHAct(), ws.getGradOut(), ws.getDW2(), dInt, rows, dModel, true, false);
            TensorOpsGPU.multiplyBackwardGpuDevice(
                    ws.getDHact(), ws.getH1(), ws.getGateSwish(), ws.getDH1(), ws.getDGateSwish(), rows * dInt);
            TensorOpsGPU.multiplyBackwardGpuDevice(
                    ws.getDGateSwish(), ws.getGate(), ws.getSig(), ws.getDGate(), ws.getDSig(), rows * dInt);
            TensorOpsGPU.sigmoidBackwardGpuDevice(ws.getDSig(), ws.getGate(), ws.getDGate(), rows * dInt);

            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDH1(),
                    useResidentW ? ffnResident.w1() : ws.getW1(),
                    ws.getDXNorm2(),
                    rows,
                    dInt,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDGate(),
                    useResidentW ? ffnResident.w3() : ws.getW3(),
                    ws.getDXTmp(),
                    rows,
                    dInt,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getDXNorm2(), ws.getDXTmp(), rows * dModel);

            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDH1(), ws.getDW1(), dModel, rows, dInt, true, false);
            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDGate(), ws.getDW3(), dModel, rows, dInt, true, false);

            TensorOpsGPU.rmsNormBackwardGpuDevice(
                    ws.getDXNorm2(),
                    ws.getXRes1(),
                    useResidentW ? ffnResident.normGamma() : ws.getNormGamma(),
                    TensorOpsGPU.rmsNormEps(),
                    ws.getDXRes1(),
                    ws.getDNorm2(),
                    rows,
                    dModel);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getDXRes1(), ws.getGradOut(), rows * dModel);

            dGradXRes1.copyFromDevice(ws.getDXRes1(), rows * dModel);
            GpuPendingGradients.accumulate(gradNorm2, ws.getDNorm2(), dModel);
            GpuPendingGradients.accumulate(gradW1, ws.getDW1(), dModel * dInt);
            GpuPendingGradients.accumulate(gradW2, ws.getDW2(), dInt * dModel);
            GpuPendingGradients.accumulate(gradW3, ws.getDW3(), dModel * dInt);
        }
    }

    private static void fusedAttentionBackwardGpuDevice(
            GpuFloatBuffer dGradOut,
            BlockActivationCacheDevice cache,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            GpuFloatBuffer dGradX,
            Tensor gradWq,
            Tensor gradWk,
            Tensor gradWv,
            Tensor gradWo,
            TensorOps.GpuAttnResidentBuffers attnResident) {
        int batch = cache.batch();
        int seqLen = cache.seqLen();
        int dModel = cache.dModel();
        int dHead = dModel / numHeads;
        int rows = batch * seqLen;
        int batchHeads = batch * numHeads;
        float attScale = 1.0f / (float) Math.sqrt(dHead);
        int dModelSq = dModel * dModel;
        boolean useResidentW = canUseAttnResidentWeights(attnResident, Wq, Wk, Wv, Wo, dModelSq);

        GpuAttentionBackwardWorkspace ws = GpuAttentionBackwardWorkspace.local();
        synchronized (ws.exclusiveUseLock()) {
            ws.ensure(batch, numHeads, seqLen, dModel);
            int plane = rows * dModel;
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.X_NORM1, ws.getXFlat(), plane);
            ws.getGradOutFlat().copyFromDevice(dGradOut, plane);
            if (!useResidentW) {
                ws.getWq().copyFrom(Wq.internalBuffer(), 0, dModelSq);
                ws.getWk().copyFrom(Wk.internalBuffer(), 0, dModelSq);
                ws.getWv().copyFrom(Wv.internalBuffer(), 0, dModelSq);
                ws.getWo().copyFrom(Wo.internalBuffer(), 0, dModelSq);
            }
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_Q_HEADS, ws.getQHeads(), rows * dModel);
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_K_HEADS, ws.getKHeads(), rows * dModel);
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_V_HEADS, ws.getVHeads(), rows * dModel);
            boolean useFlash = TensorOpsGPU.FLASH_ATTENTION && dHead == 16;
            if (!useFlash) {
                cache.copySlotToDeviceFloat(
                        BlockActivationCacheDevice.SlotId.ATTN_PROBS, ws.getProbs(), batchHeads * seqLen * seqLen);
            }
            cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.ATTN_CONCAT, ws.getConcat(), plane);

            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getGradOutFlat(),
                    useResidentW ? attnResident.wo() : ws.getWo(),
                    ws.getGradConcat(),
                    rows,
                    dModel,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.splitHeadsGpuDevice(
                    ws.getGradConcat(), ws.getDHeads(), batch, seqLen, dModel, numHeads);
            if (useFlash) {
                /* FlashAttention-2 backward: recomputes attention from Q/K/V + LSE.
                 * O_heads (saved in forward) is needed for D = dot(dO, O) computation. */
                cache.copySlotToDeviceFloat(
                        BlockActivationCacheDevice.SlotId.ATTN_OUT_HEADS, ws.getProbs(), batchHeads * seqLen * dHead);
                TensorOpsGPU.flashAttentionBackwardGpuDeviceResident(
                        ws.getQHeads(),
                        ws.getKHeads(),
                        ws.getVHeads(),
                        ws.getProbs(),    // O_heads (borrowed from ws.getProbs() buffer for reuse)
                        ws.getDHeads(),   // dO
                        cache.attnLseBuffer(),
                        ws.getGradQh(),
                        ws.getGradKh(),
                        ws.getGradVh(),
                        batchHeads,
                        seqLen,
                        attScale);
            } else {
                /* Classic backward: use cached softmax probs (more numerically stable). */
                TensorOpsGPU.scaledDotProductAttentionBackwardGpuDevice(
                        ws.getDHeads(),
                        ws.getProbs(),
                        ws.getQHeads(),
                        ws.getKHeads(),
                        ws.getVHeads(),
                        ws.getGradQh(),
                        ws.getGradKh(),
                        ws.getGradVh(),
                        batchHeads,
                        seqLen,
                        dHead,
                        dHead,
                        attScale,
                        0L,
                        false);
            }

            ws.getQHeads().clear();
            ws.getKHeads().clear();
            TensorOpsGPU.applyRoPEBackwardGpuDevice(
                    ws.getGradQh(), ws.getQHeads(), batch, numHeads, seqLen, dHead);
            TensorOpsGPU.applyRoPEBackwardGpuDevice(
                    ws.getGradKh(), ws.getKHeads(), batch, numHeads, seqLen, dHead);

            TensorOpsGPU.concatHeadsGpuDevice(ws.getQHeads(), ws.getDQ(), batch, numHeads, seqLen, dHead);
            TensorOpsGPU.concatHeadsGpuDevice(ws.getKHeads(), ws.getDK(), batch, numHeads, seqLen, dHead);
            TensorOpsGPU.concatHeadsGpuDevice(ws.getGradVh(), ws.getDV(), batch, numHeads, seqLen, dHead);

            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDQ(),
                    useResidentW ? attnResident.wq() : ws.getWq(),
                    ws.getGradX(),
                    rows,
                    dModel,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDK(),
                    useResidentW ? attnResident.wk() : ws.getWk(),
                    ws.getGradTmp(),
                    rows,
                    dModel,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getGradX(), ws.getGradTmp(), rows * dModel);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getDV(),
                    useResidentW ? attnResident.wv() : ws.getWv(),
                    ws.getGradTmp(),
                    rows,
                    dModel,
                    dModel,
                    false,
                    true);
            TensorOpsGPU.accumulateAddGpuDevice(ws.getGradX(), ws.getGradTmp(), rows * dModel);

            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDQ(), ws.getGradWq(), dModel, rows, dModel, true, false);
            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDK(), ws.getGradWk(), dModel, rows, dModel, true, false);
            TensorOpsGPU.matmulGpuDeviceEx(ws.getXFlat(), ws.getDV(), ws.getGradWv(), dModel, rows, dModel, true, false);
            TensorOpsGPU.matmulGpuDeviceEx(
                    ws.getConcat(), ws.getGradOutFlat(), ws.getGradWo(), dModel, rows, dModel, true, false);

            dGradX.copyFromDevice(ws.getGradX(), rows * dModel);
            GpuPendingGradients.accumulate(gradWq, ws.getGradWq(), dModel * dModel);
            GpuPendingGradients.accumulate(gradWk, ws.getGradWk(), dModel * dModel);
            GpuPendingGradients.accumulate(gradWv, ws.getGradWv(), dModel * dModel);
            GpuPendingGradients.accumulate(gradWo, ws.getGradWo(), dModel * dModel);
        }
    }

    /**
     * Sampled LM head backward on GPU using thread-local workspace; eliminates per-step
     * {@code cudaMalloc} for {@code dHidden}, {@code dGradBeforeNorm}, {@code dGGamma}.
     *
     * <p><b>Ownership:</b> returns the workspace {@code dGradBeforeNorm} buffer — caller must
     * <b>NOT</b> close it. The buffer is valid until the next call to this method on the same
     * thread, or until {@link GpuWorkspaceCleanup#releaseAllGpuWorkspacesThreadLocal()}.
     *
     * @param lmHeadWeights    device weight matrix [dModel × vocabSize]
     * @param lmHeadGrad       device grad buffer for LM head [dModel × vocabSize]
     * @param normGamma        device gamma of the final RMSNorm [dModel]
     * @param normGammaGrad    device grad buffer for gamma [dModel]
     * @param normGammaSize    number of floats in gamma (== dModel)
     */
    public static GpuFloatBuffer backwardSampledLmHeadDevice(
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateGrad,
            GpuFloatBuffer dNormedHidden,
            GpuFloatBuffer dXBeforeNorm,
            GpuFloatBuffer lmHeadWeights,
            GpuFloatBuffer lmHeadGrad,
            GpuFloatBuffer normGamma,
            GpuFloatBuffer normGammaGrad,
            int normGammaSize,
            int rows,
            int dModel,
            int vocabSize,
            int candidateCount) {
        Objects.requireNonNull(candidateIds, "candidateIds");
        Objects.requireNonNull(candidateGrad, "candidateGrad");
        Objects.requireNonNull(dNormedHidden, "dNormedHidden");
        Objects.requireNonNull(dXBeforeNorm, "dXBeforeNorm");
        Objects.requireNonNull(lmHeadWeights, "lmHeadWeights");
        Objects.requireNonNull(lmHeadGrad, "lmHeadGrad");
        Objects.requireNonNull(normGamma, "normGamma");
        Objects.requireNonNull(normGammaGrad, "normGammaGrad");

        GpuSampledLmHeadBackwardWorkspace ws = GpuSampledLmHeadBackwardWorkspace.local();
        ws.ensure(rows, dModel);

        GpuFloatBuffer dHidden = ws.getDHidden();
        GpuFloatBuffer dGradBeforeNorm = ws.getDGradBeforeNorm();
        GpuFloatBuffer dGGamma = ws.getDGGamma();

        // dHidden fully overwritten by sampledLmHeadBackwardGpuDevice — no need to clear
        TensorOpsGPU.sampledLmHeadBackwardGpuDevice(
                candidateIds,
                candidateGrad,
                dNormedHidden,
                lmHeadWeights,
                dHidden,
                lmHeadGrad,
                rows,
                dModel,
                vocabSize,
                candidateCount);

        // rmsNormBackwardGpuDevice uses += on outputs; must zero before call
        dGradBeforeNorm.clear();
        dGGamma.clear();
        TensorOpsGPU.rmsNormBackwardGpuDevice(
                dHidden, dXBeforeNorm, normGamma,
                TensorOpsGPU.rmsNormEps(),
                dGradBeforeNorm, dGGamma,
                rows, dModel);
        TensorOpsGPU.accumulateAddGpuDevice(normGammaGrad, dGGamma, normGammaSize);

        return dGradBeforeNorm;
    }

    /**
     * Decoder block backward entirely on device: grad flow stays in {@link GpuFloatBuffer}.
     */
    public static void transformerBlockBackwardGpuDevice(
            GpuFloatBuffer dGradOut,
            BlockActivationCacheDevice cache,
            Tensor Wq, Tensor Wk, Tensor Wv, Tensor Wo,
            Tensor W1, Tensor W2, Tensor W3,
            Tensor norm1, Tensor norm2,
            int numHeads, Tensor mask, boolean useRoPE,
            GpuFloatBuffer dGradXIn,
            Tensor gradWq, Tensor gradWk, Tensor gradWv, Tensor gradWo,
            Tensor gradW1, Tensor gradW2, Tensor gradW3,
            Tensor gradNorm1, Tensor gradNorm2,
            TensorOps.GpuAttnResidentBuffers attnResident,
            TensorOps.GpuFfnResidentBuffers ffnResident) {
        Objects.requireNonNull(cache, "cache");
        if (!useRoPE) {
            throw new UnsupportedOperationException("MHA without RoPE backward not implemented");
        }
        int batch = cache.batch();
        int seqLen = cache.seqLen();
        int dModel = cache.dModel();
        int rows = batch * seqLen;
        int flat = rows * dModel;

        GpuTransformerOuterBackwardWorkspace outer = GpuTransformerOuterBackwardWorkspace.local();
        outer.ensure(rows, dModel);
        GpuFloatBuffer dGradXRes1 = outer.getDGradXRes1();
        GpuFloatBuffer dGradNorm1Path = outer.getDGradNorm1Path();
        GpuFloatBuffer dNorm1 = outer.getDNorm1();
        GpuFloatBuffer xInScratch = outer.getXInScratch();
        GpuFloatBuffer norm1GammaTmp = attnResident == null ? outer.norm1GammaStaging() : null;

        fusedFfnNormResidualBackwardGpuDevice(
                dGradOut,
                cache,
                W1,
                W2,
                W3,
                norm2,
                dGradXRes1,
                gradW1,
                gradW2,
                gradW3,
                gradNorm2,
                ffnResident);
        fusedAttentionBackwardGpuDevice(
                dGradXRes1,
                cache,
                Wq,
                Wk,
                Wv,
                Wo,
                numHeads,
                dGradNorm1Path,
                gradWq,
                gradWk,
                gradWv,
                gradWo,
                attnResident);

        GpuFloatBuffer norm1Gamma = attnResident != null ? attnResident.normGamma() : norm1GammaTmp;
        if (norm1GammaTmp != null) {
            norm1GammaTmp.copyFrom(norm1.internalBuffer(), 0, dModel);
        }
        cache.copySlotToDeviceFloat(BlockActivationCacheDevice.SlotId.X_IN, xInScratch, flat);
        if (DebugGpuTrain.isEnabled()) {
            TensorOpsGPU.synchronizeStream();
            float[] tgx = new float[2];
            dGradXIn.copyTo(tgx, 0, Math.min(2, flat));
            float[] tn1 = new float[2];
            dNorm1.copyTo(tn1, 0, Math.min(2, dModel));
            agentLogB39372RmsTargets(
                    tgx[0], tgx.length > 1 ? tgx[1] : 0f, tn1[0], tn1.length > 1 ? tn1[1] : 0f, rows, dModel);
        }
        // rms_norm_bwd uses += on gX; ping-pong / malloc content must be zero.
        dGradXIn.clear();
        dNorm1.clear();
        TensorOpsGPU.rmsNormBackwardGpuDevice(
                dGradNorm1Path,
                xInScratch,
                norm1Gamma,
                TensorOpsGPU.rmsNormEps(),
                dGradXIn,
                dNorm1,
                rows,
                dModel);
        TensorOpsGPU.accumulateAddGpuDevice(dGradXIn, dGradXRes1, flat);
        GpuPendingGradients.accumulate(gradNorm1, dNorm1, dModel);
    }

}
