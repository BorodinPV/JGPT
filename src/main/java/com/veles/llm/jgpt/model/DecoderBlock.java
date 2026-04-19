package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.List;

/**
 * Декодер-блок (backward на CPU). При {@link GPTModel#isGpuResident()}: аттеншн и FFN без H2D весов на шаг
 * (см. {@link TensorOps#tryMultiHeadAttentionWithRoPEGpuResident} и fused FFN).
 */
final class DecoderBlock {
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
    private float residualDropout = 0f;
    private float attentionDropout = 0f;
    private long dropoutSeed = 42L;

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


    void setDropout(float residual, float attention, long layerIdx) {
        this.residualDropout = Math.max(0f, Math.min(1f, residual));
        this.attentionDropout = Math.max(0f, Math.min(1f, attention));
        this.dropoutSeed = 42L + layerIdx * 1000L;
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
        Tensor attnAfterDropout;
        if (attentionDropout > 0f) {
            attnAfterDropout = com.veles.llm.jgpt.ops.TensorOps.dropout(attnOut, attentionDropout, dropoutSeed);
        } else {
            attnAfterDropout = attnOut;
        }
        Tensor xRes1 = TensorOps.add(x, attnAfterDropout);
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
            Tensor ffnAfterDropout;
            if (residualDropout > 0f) {
                ffnAfterDropout = com.veles.llm.jgpt.ops.TensorOps.dropout(ffnOut, residualDropout, dropoutSeed + 1L);
            } else {
                ffnAfterDropout = ffnOut;
            }
            out = TensorOps.add(xRes1, ffnAfterDropout);
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
        Tensor attnAfterDropout;
        if (attentionDropout > 0f) {
            attnAfterDropout = com.veles.llm.jgpt.ops.TensorOps.dropout(attnOut, attentionDropout, dropoutSeed);
        } else {
            attnAfterDropout = attnOut;
        }
        Tensor xRes1 = TensorOps.add(x, attnAfterDropout);
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
        Tensor attnAfterDropout;
        if (attentionDropout > 0f) {
            attnAfterDropout = com.veles.llm.jgpt.ops.TensorOps.dropout(attnOut, attentionDropout, dropoutSeed);
        } else {
            attnAfterDropout = attnOut;
        }
        Tensor xRes1 = TensorOps.add(x, attnAfterDropout);
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
        Tensor attnAfterDropout;
        if (attentionDropout > 0f) {
            attnAfterDropout = com.veles.llm.jgpt.ops.TensorOps.dropout(attnOut, attentionDropout, dropoutSeed);
        } else {
            attnAfterDropout = attnOut;
        }
        Tensor xRes1 = TensorOps.add(x, attnAfterDropout);
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
        Tensor attnAfterDropout;
        if (attentionDropout > 0f) {
            attnAfterDropout = com.veles.llm.jgpt.ops.TensorOps.dropout(attnOut, attentionDropout, dropoutSeed);
        } else {
            attnAfterDropout = attnOut;
        }
        Tensor xRes1 = TensorOps.add(x, attnAfterDropout);
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
