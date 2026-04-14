package com.veles.llm.jgpt;

import java.util.Objects;

/** Fused RMSNorm+GEMM, heads layout, RoPE, SDPA forward; JNI в {@link TensorOpsGPU}. */
final class TensorOpsGpuDeviceFusedLayout {

    private TensorOpsGpuDeviceFusedLayout() {}

    static void rmsNormMatmulLmHeadGpuDevice(
            GpuFloatBuffer x,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer normOut,
            GpuFloatBuffer w,
            GpuFloatBuffer logits,
            int rows,
            int dModel,
            int vocab,
            boolean fp16Rms) {
        if (rows <= 0 || dModel <= 0 || vocab <= 0) {
            throw new IllegalArgumentException("rows, dModel, vocab must be positive");
        }
        long plane = (long) rows * dModel;
        long logitsNeed = (long) rows * vocab;
        long wNeed = (long) dModel * vocab;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(x, "x"), plane, "x");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gamma, "gamma"), dModel, "gamma");
        if (normOut != null) {
            TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(normOut, "normOut"), plane, "normOut");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(w, "w"), wNeed, "w");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, "logits"), logitsNeed, "logits");
        long dNormOut = normOut != null ? normOut.devicePointer() : 0L;
        TensorOpsGPU.rmsNormMatmulLmHeadGPUDevice(
                x.devicePointer(),
                gamma.devicePointer(),
                eps,
                dNormOut,
                w.devicePointer(),
                logits.devicePointer(),
                rows,
                dModel,
                vocab,
                fp16Rms);
    }

    static void rmsNormMatmulFfnW1W3GpuDevice(
            GpuFloatBuffer xRes1,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer normOut,
            GpuFloatBuffer w1,
            GpuFloatBuffer w3,
            GpuFloatBuffer h1,
            GpuFloatBuffer gate,
            int rows,
            int dModel,
            int dIntermediate,
            boolean fp16Rms) {
        if (rows <= 0 || dModel <= 0 || dIntermediate <= 0) {
            throw new IllegalArgumentException("rows, dModel, dIntermediate must be positive");
        }
        long plane = (long) rows * dModel;
        long wNeed = (long) dModel * dIntermediate;
        long mid = (long) rows * dIntermediate;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(xRes1, "xRes1"), plane, "xRes1");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gamma, "gamma"), dModel, "gamma");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(normOut, "normOut"), plane, "normOut");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(w1, "w1"), wNeed, "w1");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(w3, "w3"), wNeed, "w3");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(h1, "h1"), mid, "h1");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gate, "gate"), mid, "gate");
        TensorOpsGPU.rmsNormMatmulFfnW1W3GPUDevice(
                xRes1.devicePointer(),
                gamma.devicePointer(),
                eps,
                normOut.devicePointer(),
                w1.devicePointer(),
                w3.devicePointer(),
                h1.devicePointer(),
                gate.devicePointer(),
                rows,
                dModel,
                dIntermediate,
                fp16Rms);
    }

    static void accumulateAddGpuFromHost(GpuFloatBuffer acc, float[] host, int off, int len) {
        Objects.requireNonNull(host, "host");
        if (len < 0 || off < 0 || (long) off + len > host.length) {
            throw new IllegalArgumentException("host range invalid");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(acc, "acc"), len, "acc");
        TensorOpsGPU.accumulateAddFromHostGPUDevice(acc.devicePointer(), host, off, len);
    }

    static void splitHeadsGpuDevice(
            GpuFloatBuffer src, GpuFloatBuffer dst, int batch, int seqLen, int dModel, int numHeads) {
        if (batch <= 0 || seqLen <= 0 || dModel <= 0 || numHeads <= 0) {
            throw new IllegalArgumentException("batch, seqLen, dModel, numHeads must be positive");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by numHeads");
        }
        int dHead = dModel / numHeads;
        long needSrc = (long) batch * seqLen * dModel;
        long needDst = (long) batch * numHeads * seqLen * dHead;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(src, "src"), needSrc, "src");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dst, "dst"), needDst, "dst");
        TensorOpsGPU.splitHeadsGPUDevice(src.devicePointer(), dst.devicePointer(), batch, seqLen, dModel, numHeads);
    }

    static void concatHeadsGpuDevice(
            GpuFloatBuffer src, GpuFloatBuffer dst, int batch, int numHeads, int seqLen, int dHead) {
        if (batch <= 0 || numHeads <= 0 || seqLen <= 0 || dHead <= 0) {
            throw new IllegalArgumentException("batch, numHeads, seqLen, dHead must be positive");
        }
        int dModel = numHeads * dHead;
        long needSrc = (long) batch * numHeads * seqLen * dHead;
        long needDst = (long) batch * seqLen * dModel;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(src, "src"), needSrc, "src");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dst, "dst"), needDst, "dst");
        TensorOpsGPU.concatHeadsGPUDevice(src.devicePointer(), dst.devicePointer(), batch, numHeads, seqLen, dHead);
    }

    static void copyKvHeads4dToCacheGpuDevice(
            GpuFloatBuffer srcHeads4d,
            GpuFloatBuffer dstCache,
            int numHeads,
            int seqLen,
            int maxSeqLen,
            int dHead,
            int batchIdx,
            int batch) {
        if (numHeads <= 0 || seqLen <= 0 || maxSeqLen <= 0 || dHead <= 0 || batch <= 0) {
            throw new IllegalArgumentException("dimensions must be positive");
        }
        if (batchIdx < 0 || batchIdx >= batch) {
            throw new IllegalArgumentException("batchIdx out of range");
        }
        TensorOpsGpuBufferChecks.requireGpu(srcHeads4d, "srcHeads4d");
        TensorOpsGpuBufferChecks.requireGpu(dstCache, "dstCache");
        TensorOpsGPU.copyKvHeads4dToCacheGPUDevice(
                srcHeads4d.devicePointer(),
                dstCache.devicePointer(),
                numHeads,
                seqLen,
                maxSeqLen,
                dHead,
                batchIdx,
                batch);
    }

    static void applyRoPEBackwardGpuDevice(
            GpuFloatBuffer gradY, GpuFloatBuffer gradX, int batch, int numHeads, int seqLen, int dHead) {
        if (batch <= 0 || numHeads <= 0 || seqLen <= 0 || dHead <= 0) {
            throw new IllegalArgumentException("batch, numHeads, seqLen, dHead must be positive");
        }
        long need = (long) batch * numHeads * seqLen * dHead;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gradY, "gradY"), need, "gradY");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gradX, "gradX"), need, "gradX");
        TensorOpsGPU.applyRoPEBackward4DGPUDevice(
                gradY.devicePointer(), gradX.devicePointer(), batch, numHeads, seqLen, dHead, 0);
    }

    static void applyRoPE4dGpuDevice(
            GpuFloatBuffer src,
            GpuFloatBuffer dst,
            int batch,
            int numHeads,
            int seqLen,
            int dHead,
            int[] positions,
            int posBaseOffset) {
        if (batch <= 0 || numHeads <= 0 || seqLen <= 0 || dHead <= 0) {
            throw new IllegalArgumentException("batch, numHeads, seqLen, dHead must be positive");
        }
        long need = (long) batch * numHeads * seqLen * dHead;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(src, "src"), need, "src");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dst, "dst"), need, "dst");
        TensorOpsGPU.applyRoPE4DGPUDevice(
                src.devicePointer(),
                dst.devicePointer(),
                batch,
                numHeads,
                seqLen,
                dHead,
                positions,
                posBaseOffset);
    }

    static void scaledDotProductAttentionForwardGpuDevice(
            GpuFloatBuffer dQ,
            GpuFloatBuffer dK,
            GpuFloatBuffer dV,
            float[] maskOrNull,
            GpuFloatBuffer dOut,
            float[] h_probsOrNull,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            boolean useFp16Softmax) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.scaledDotProductAttentionForwardGpuDevice");
        if (batch <= 0 || seqLen <= 0 || dKDim <= 0 || dVDim <= 0) {
            throw new IllegalArgumentException("batch, seqLen, dKDim, dVDim must be positive");
        }
        long qk = (long) batch * seqLen * dKDim;
        long vSz = (long) batch * seqLen * dVDim;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dQ, "dQ"), qk, "dQ");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dK, "dK"), qk, "dK");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dV, "dV"), vSz, "dV");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dOut, "dOut"), vSz, "dOut");
        if (h_probsOrNull != null) {
            int probNeed = Math.multiplyExact(Math.multiplyExact(batch, seqLen), seqLen);
            if (h_probsOrNull.length < probNeed) {
                throw new IllegalArgumentException(
                        "h_probs length " + h_probsOrNull.length + " < " + probNeed);
            }
        }
        TensorOpsGPU.scaledDotProductAttentionForwardGPUDevice(
                dQ.devicePointer(),
                dK.devicePointer(),
                dV.devicePointer(),
                maskOrNull,
                dOut.devicePointer(),
                h_probsOrNull,
                batch,
                seqLen,
                dKDim,
                dVDim,
                scale,
                useFp16Softmax);
    }

    static void scaledDotProductAttentionForwardGpuDeviceResident(
            GpuFloatBuffer dQ,
            GpuFloatBuffer dK,
            GpuFloatBuffer dV,
            GpuFloatBuffer dOut,
            GpuFloatBuffer dMaskOrNull,
            GpuFloatBuffer dProbsOutOrNull,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            boolean useFp16Softmax) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.scaledDotProductAttentionForwardGpuDeviceResident");
        if (batch <= 0 || seqLen <= 0 || dKDim <= 0 || dVDim <= 0) {
            throw new IllegalArgumentException("batch, seqLen, dKDim, dVDim must be positive");
        }
        long qk = (long) batch * seqLen * dKDim;
        long vSz = (long) batch * seqLen * dVDim;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dQ, "dQ"), qk, "dQ");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dK, "dK"), qk, "dK");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dV, "dV"), vSz, "dV");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dOut, "dOut"), vSz, "dOut");
        long dMask = 0L;
        if (dMaskOrNull != null) {
            int needMask = Math.multiplyExact(seqLen, seqLen);
            TensorOpsGpuBufferChecks.requireMinFloats(
                    TensorOpsGpuBufferChecks.requireGpu(dMaskOrNull, "dMask"), needMask, "dMask");
            dMask = dMaskOrNull.devicePointer();
        }
        long dProbs = 0L;
        if (dProbsOutOrNull != null) {
            int probNeed = Math.multiplyExact(Math.multiplyExact(batch, seqLen), seqLen);
            TensorOpsGpuBufferChecks.requireMinFloats(
                    TensorOpsGpuBufferChecks.requireGpu(dProbsOutOrNull, "dProbsOut"), probNeed, "dProbsOut");
            dProbs = dProbsOutOrNull.devicePointer();
        }
        TensorOpsGPU.scaledDotProductAttentionForwardGPUDeviceResident(
                dQ.devicePointer(),
                dK.devicePointer(),
                dV.devicePointer(),
                dOut.devicePointer(),
                dMask,
                dProbs,
                batch,
                seqLen,
                dKDim,
                dVDim,
                scale,
                useFp16Softmax);
    }

    static void scaledDotProductAttentionBackwardGpuDevice(
            GpuFloatBuffer gradOut,
            GpuFloatBuffer probs,
            GpuFloatBuffer q,
            GpuFloatBuffer k,
            GpuFloatBuffer v,
            GpuFloatBuffer gradQ,
            GpuFloatBuffer gradK,
            GpuFloatBuffer gradV,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            long dMask,
            boolean useFp16Softmax) {
        if (batch <= 0 || seqLen <= 0 || dKDim <= 0 || dVDim <= 0) {
            throw new IllegalArgumentException("batch, seqLen, dKDim, dVDim must be positive");
        }
        long qk = (long) batch * seqLen * dKDim;
        long probSz = (long) batch * seqLen * seqLen;
        long vSz = (long) batch * seqLen * dVDim;
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(gradOut, "gradOut"), vSz, "gradOut");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(probs, "probs"), probSz, "probs");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(q, "q"), qk, "q");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(k, "k"), qk, "k");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(v, "v"), vSz, "v");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gradQ, "gradQ"), qk, "gradQ");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gradK, "gradK"), qk, "gradK");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gradV, "gradV"), vSz, "gradV");
        TensorOpsGPU.scaledDotProductAttentionBackwardGPUDevice(
                gradOut.devicePointer(),
                probs.devicePointer(),
                q.devicePointer(),
                k.devicePointer(),
                v.devicePointer(),
                gradQ.devicePointer(),
                gradK.devicePointer(),
                gradV.devicePointer(),
                batch,
                seqLen,
                dKDim,
                dVDim,
                scale,
                dMask,
                useFp16Softmax);
    }

    static void rmsNormBackwardGpuDevice(
            GpuFloatBuffer gOut,
            GpuFloatBuffer x,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer gX,
            GpuFloatBuffer gGamma,
            int outer,
            int lastDim) {
        if (outer <= 0 || lastDim <= 0) {
            throw new IllegalArgumentException("outer and lastDim must be positive");
        }
        long plane = (long) outer * lastDim;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gOut, "gOut"), plane, "gOut");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(x, "x"), plane, "x");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gamma, "gamma"), lastDim, "gamma");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gX, "gX"), plane, "gX");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gGamma, "gGamma"), lastDim, "gGamma");
        TensorOpsGPU.rmsNormBackwardGPUDevice(
                gOut.devicePointer(),
                x.devicePointer(),
                gamma.devicePointer(),
                eps,
                gX.devicePointer(),
                gGamma.devicePointer(),
                outer,
                lastDim);
    }
}
