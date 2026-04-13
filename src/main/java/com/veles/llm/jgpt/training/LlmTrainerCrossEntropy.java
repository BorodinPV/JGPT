package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.GpuIntBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.util.DebugGpuTrain;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Objects;

/** CE / sampled train loss и eval CE по логитам. */
final class LlmTrainerCrossEntropy {

    private LlmTrainerCrossEntropy() {}

    static float applyTrainLossAndGrad(LLMTrainer t, Tensor logits, Tensor target, float gradScale) {
        if (t.config.usesSampledTrainLoss()) {
            return applySampledTrainLossAndGradDevice(t, logits, target, gradScale);
        }
        return applyCrossEntropyLossAndGrad(t, logits, target, gradScale);
    }

    static float applyCrossEntropyLossAndGrad(LLMTrainer t, Tensor logits, Tensor target, float gradScale) {
        t.model.clearSampledTrainLossGrad();
        if (t.config.deviceLogitsTrainStep && t.model.hasDeviceLogitsBuffers()) {
            return applyCrossEntropyLossAndGradDevice(t, logits, target, gradScale);
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

        if (!TensorOpsGPU.shouldUseGpuCrossEntropy(logits.size())) {
            return 0f;
        }
        float gradScaleOverTotal = ceFusedGradScaleOverTotal(t, totalTokens, gradScale);
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
                    t.fp16Matmul);
        }
        float[] logitData = logits.internalBuffer();
        float[] targetData = target.internalBuffer();
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpu(
                logitData, targetData, gradData, batch, seqLen, vocabSize, gradScaleOverTotal);
    }

    static float ceFusedGradScaleOverTotal(LLMTrainer t, int totalTokens, float microbatchGradScale) {
        return microbatchGradScale * t.lossScaleForForward() / (float) totalTokens;
    }

    static int ceTokenIdOrInvalid(float v, int vocabSize) {
        if (!Float.isFinite(v) || v < 0f || v >= (float) vocabSize) {
            return -1;
        }
        int ti = (int) v;
        if ((float) ti != v) {
            return -1;
        }
        return ti;
    }

    static void fillCeTargetsHostSanitized(LLMTrainer t, Tensor target, int nrows, int vocabSize) {
        if (t.ceHostTargetScratch == null || t.ceHostTargetScratch.length < nrows) {
            t.ceHostTargetScratch = new int[nrows];
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
                int ti = ceTokenIdOrInvalid(fb.get(), vocabSize);
                t.ceHostTargetScratch[i] = ti;
                if (ti < 0) {
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
                int ti = ceTokenIdOrInvalid(td[i], vocabSize);
                t.ceHostTargetScratch[i] = ti;
                if (ti < 0) {
                    invalid++;
                }
            }
        }
        if (invalid > 0 && DebugGpuTrain.isEnabled()) {
            LlmTrainerDebugLog.b39372(
                    "H_targets",
                    "LLMTrainer.fillCeTargetsDeviceSanitized",
                    "invalid_ce_targets",
                    "{\"globalStep\":"
                            + t.globalStep
                            + ",\"invalid\":"
                            + invalid
                            + ",\"nrows\":"
                            + nrows
                            + ",\"vocabSize\":"
                            + vocabSize
                            + "}");
        }
    }

    static void fillCeTargetsDeviceSanitized(LLMTrainer t, Tensor target, int nrows, int vocabSize) {
        fillCeTargetsHostSanitized(t, target, nrows, vocabSize);
        t.ceTargetsDevice.copyFrom(t.ceHostTargetScratch, 0, nrows);
    }

    static float applyCrossEntropyLossAndGradDevice(LLMTrainer t, Tensor logits, Tensor target, float gradScale) {
        t.model.clearSampledTrainLossGrad();
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];
        int totalTokens = batch * seqLen;
        float gradScaleOverTotal = ceFusedGradScaleOverTotal(t, totalTokens, gradScale);
        GpuFloatBuffer logitsGpu = t.model.deviceLogitsBuffer();
        GpuFloatBuffer gradGpu = t.model.ensureDeviceLogitsGradBuffer(batch * seqLen * vocabSize);
        gradGpu.clear();
        int nrows = totalTokens;
        if (t.ceTargetsDevice == null || t.ceTargetsCapRows < nrows) {
            if (t.ceTargetsDevice != null) {
                t.ceTargetsDevice.close();
            }
            t.ceTargetsDevice = GpuIntBuffer.allocate(nrows);
            t.ceTargetsCapRows = nrows;
        }
        fillCeTargetsDeviceSanitized(t, target, nrows, vocabSize);
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceTargetsDevice(
                logitsGpu,
                t.ceTargetsDevice,
                gradGpu,
                batch,
                seqLen,
                vocabSize,
                gradScaleOverTotal,
                t.fp16Matmul);
    }

    static void applyCrossEntropyLossAndGradDeviceAsync(LLMTrainer t, Tensor logits, Tensor target, float gradScale) {
        t.model.clearSampledTrainLossGrad();
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];
        int totalTokens = batch * seqLen;
        float gradScaleOverTotal = ceFusedGradScaleOverTotal(t, totalTokens, gradScale);
        GpuFloatBuffer logitsGpu = t.model.deviceLogitsBuffer();
        GpuFloatBuffer gradGpu = t.model.ensureDeviceLogitsGradBuffer(batch * seqLen * vocabSize);
        gradGpu.clear();
        int nrows = totalTokens;
        if (t.ceTargetsDevice == null || t.ceTargetsCapRows < nrows) {
            if (t.ceTargetsDevice != null) {
                t.ceTargetsDevice.close();
            }
            t.ceTargetsDevice = GpuIntBuffer.allocate(nrows);
            t.ceTargetsCapRows = nrows;
        }
        fillCeTargetsDeviceSanitized(t, target, nrows, vocabSize);
        TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceTargetsDeviceAsync(
                logitsGpu,
                t.ceTargetsDevice,
                gradGpu,
                batch,
                seqLen,
                vocabSize,
                gradScaleOverTotal,
                t.fp16Matmul);
    }

    static int effectiveSampledCandidateCount(LLMTrainer t, int vocabSize) {
        return Math.max(2, Math.min(t.config.sampledCeCandidates, vocabSize));
    }

    static boolean canDeviceSampledTrainForward(LLMTrainer t) {
        return t.config.usesSampledTrainLoss()
                && t.config.deviceLogitsTrainStep
                && t.config.deviceDecoderBackward
                && t.model.canFullGpuTrain()
                && t.model.isDeviceLogitsEnabled();
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

    static void ensureSampledCandidateDeviceBuffers(LLMTrainer t, int totalCandidateElems) {
        if (t.sampledCandidateIdsDevice == null || t.sampledCandidateIdsCapElems < totalCandidateElems) {
            if (t.sampledCandidateIdsDevice != null) {
                t.sampledCandidateIdsDevice.close();
            }
            t.sampledCandidateIdsDevice = GpuIntBuffer.allocate(totalCandidateElems);
            t.sampledCandidateIdsCapElems = totalCandidateElems;
        }
        if (t.sampledCandidateLogitsDevice == null || t.sampledCandidateFloatCapElems < totalCandidateElems) {
            if (t.sampledCandidateLogitsDevice != null) {
                t.sampledCandidateLogitsDevice.close();
            }
            if (t.sampledCandidateGradDevice != null) {
                t.sampledCandidateGradDevice.close();
            }
            t.sampledCandidateLogitsDevice = GpuFloatBuffer.allocate(totalCandidateElems);
            t.sampledCandidateGradDevice = GpuFloatBuffer.allocate(totalCandidateElems);
            t.sampledCandidateFloatCapElems = totalCandidateElems;
        }
    }

    static int prepareSampledCandidateIds(LLMTrainer t, Tensor target, int rows, int vocabSize, int candidates) {
        fillCeTargetsHostSanitized(t, target, rows, vocabSize);
        int total = rows * candidates;
        if (t.sampledCandidateIdsHostScratch == null || t.sampledCandidateIdsHostScratch.length < total) {
            t.sampledCandidateIdsHostScratch = new int[total];
        }
        int negativeCount = candidates - 1;
        if (t.sampledSharedNegativeScratch == null || t.sampledSharedNegativeScratch.length < negativeCount) {
            t.sampledSharedNegativeScratch = new int[negativeCount];
        }
        long sharedSeed =
                mix64((((long) t.globalStep + 1L) << 32) ^ ((long) rows << 8) ^ ((long) vocabSize << 1) ^ candidates);
        for (int j = 0; j < negativeCount; j++) {
            t.sampledSharedNegativeScratch[j] =
                    nextDistinctCandidate(
                            vocabSize,
                            -1,
                            t.sampledSharedNegativeScratch,
                            0,
                            j,
                            sharedSeed ^ ((long) (j + 1) * 0x9E3779B97F4A7C15L));
        }
        for (int row = 0; row < rows; row++) {
            int base = row * candidates;
            int targetId = t.ceHostTargetScratch[row];
            t.sampledCandidateIdsHostScratch[base] = targetId;
            if (targetId < 0) {
                for (int j = 1; j < candidates; j++) {
                    t.sampledCandidateIdsHostScratch[base + j] = -1;
                }
                continue;
            }
            int used = 1;
            for (int j = 0; j < negativeCount; j++) {
                int neg = t.sampledSharedNegativeScratch[j];
                if (neg == targetId || containsCandidate(t.sampledCandidateIdsHostScratch, base, used, neg)) {
                    neg =
                            nextDistinctCandidate(
                                    vocabSize,
                                    targetId,
                                    t.sampledCandidateIdsHostScratch,
                                    base,
                                    used,
                                    sharedSeed
                                            ^ ((long) (row + 1) * 0xD1B54A32D192ED03L)
                                            ^ ((long) (j + 1) * 0x94D049BB133111EBL));
                }
                t.sampledCandidateIdsHostScratch[base + used] = neg;
                used++;
            }
        }
        ensureSampledCandidateDeviceBuffers(t, total);
        t.sampledCandidateIdsDevice.copyFrom(t.sampledCandidateIdsHostScratch, 0, total);
        t.sampledTrainCandidatesPerRow = candidates;
        return candidates;
    }

    static float applySampledTrainLossAndGradDevice(LLMTrainer t, Tensor logits, Tensor target, float gradScale) {
        if (!t.config.deviceLogitsTrainStep || !t.model.hasDeviceSampledTrainLmHeadActivations()) {
            throw new IllegalStateException(
                    "sampled train loss requires sampled device forward (LM activations without full logits)");
        }
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int rows = batch * seqLen;
        int candidates = t.sampledTrainCandidatesPerRow;
        if (candidates <= 0) {
            throw new IllegalStateException("sampled train loss: candidates not prepared before forward");
        }
        Objects.requireNonNull(target, "target");
        if (t.sampledCandidateLogitsDevice == null
                || t.sampledCandidateGradDevice == null
                || t.sampledCandidateIdsDevice == null) {
            throw new IllegalStateException("sampled train buffers missing");
        }
        long need = (long) rows * candidates;
        if (t.sampledCandidateLogitsDevice.numFloats() < need
                || t.sampledCandidateGradDevice.numFloats() < need
                || t.sampledCandidateIdsDevice.numInts() < need) {
            throw new IllegalStateException("sampled train buffers too small for logits shape");
        }
        t.sampledCandidateGradDevice.clear();
        t.model.clearSampledTrainLossGrad();
        float loss =
                TensorOpsGPU.sampledCrossEntropyGradLossGpuDeviceFirstSlot(
                        t.sampledCandidateLogitsDevice,
                        t.sampledCandidateIdsDevice,
                        t.sampledCandidateGradDevice,
                        rows,
                        candidates,
                        ceFusedGradScaleOverTotal(t, rows, gradScale));
        t.model.setSampledTrainLossGrad(t.sampledCandidateIdsDevice, t.sampledCandidateGradDevice, candidates);
        return loss;
    }

    static float evaluateCrossEntropyLoss(LLMTrainer t, Tensor logits, Tensor target) {
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
        if (t.evalCeGradScratch == null || t.evalCeGradScratch.length < len) {
            t.evalCeGradScratch = new float[len];
        }
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpu(
                logitData, targetData, t.evalCeGradScratch, batch, seqLen, vocabSize, 0f);
    }

    static float evaluateCrossEntropyLossDevice(
            LLMTrainer t, Tensor target, GpuFloatBuffer logitsGpu, int batch, int seqLen, int vocabSize) {
        int len = batch * seqLen * vocabSize;
        if (!TensorOpsGPU.shouldUseGpuCrossEntropy(len)) {
            return 0f;
        }
        GpuFloatBuffer grad = t.model.ensureDeviceLogitsGradBuffer(len);
        grad.clear();
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpuDevice(
                logitsGpu,
                target.internalBuffer(),
                grad,
                batch,
                seqLen,
                vocabSize,
                0f,
                t.fp16Matmul);
    }
}
