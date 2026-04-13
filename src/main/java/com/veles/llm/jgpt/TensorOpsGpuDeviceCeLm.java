package com.veles.llm.jgpt;

import java.util.Objects;

/** CE, gather по id, LM-head по кандидатам; JNI в {@link TensorOpsGPU}. */
final class TensorOpsGpuDeviceCeLm {

    private TensorOpsGpuDeviceCeLm() {}

    static float crossEntropySoftmaxGradLossGpuDevice(
            GpuFloatBuffer logits,
            float[] targets,
            GpuFloatBuffer grad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean fp16) {
        if (batch <= 0 || seqLen <= 0 || vocab <= 0) {
            throw new IllegalArgumentException("batch, seqLen, vocab must be positive");
        }
        long rows = (long) batch * seqLen;
        long need = rows * vocab;
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("CE logits/grad", need);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("CE targets row count", rows);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, "logits"), need, "logits");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(grad, "grad"), need, "grad");
        Objects.requireNonNull(targets, "targets");
        if (targets.length < rows) {
            throw new IllegalArgumentException(
                    "targets too small: need " + rows + " floats, have " + targets.length);
        }
        return TensorOpsGPU.crossEntropySoftmaxGradLossGPUDevice(
                logits.devicePointer(), targets, grad.devicePointer(), batch, seqLen, vocab, gradScale, fp16);
    }

    static float crossEntropySoftmaxGradLossGpuDeviceTargetsDevice(
            GpuFloatBuffer logits,
            GpuIntBuffer targets,
            GpuFloatBuffer grad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean fp16) {
        if (batch <= 0 || seqLen <= 0 || vocab <= 0) {
            throw new IllegalArgumentException("batch, seqLen, vocab must be positive");
        }
        long rows = (long) batch * seqLen;
        long need = rows * vocab;
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("CE logits/grad", need);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("CE targets row count", rows);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, "logits"), need, "logits");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(grad, "grad"), need, "grad");
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(targets, "targets"), rows, "targets");
        return TensorOpsGPU.crossEntropySoftmaxGradLossGPUDeviceTargetsDevice(
                logits.devicePointer(),
                targets.devicePointer(),
                grad.devicePointer(),
                batch,
                seqLen,
                vocab,
                gradScale,
                fp16);
    }

    static void crossEntropySoftmaxGradLossGpuDeviceTargetsDeviceAsync(
            GpuFloatBuffer logits,
            GpuIntBuffer targets,
            GpuFloatBuffer grad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean fp16) {
        if (batch <= 0 || seqLen <= 0 || vocab <= 0) {
            throw new IllegalArgumentException("batch, seqLen, vocab must be positive");
        }
        long rows = (long) batch * seqLen;
        long need = rows * vocab;
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("CE logits/grad", need);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("CE targets row count", rows);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, "logits"), need, "logits");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(grad, "grad"), need, "grad");
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(targets, "targets"), rows, "targets");
        TensorOpsGPU.crossEntropySoftmaxGradLossGPUDeviceTargetsDeviceAsync(
                logits.devicePointer(),
                targets.devicePointer(),
                grad.devicePointer(),
                batch,
                seqLen,
                vocab,
                gradScale,
                fp16);
    }

    static void crossEntropySoftmaxGradLossGpuDeviceHostFloatTargetsAsync(
            GpuFloatBuffer logits,
            float[] targets,
            GpuFloatBuffer grad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean fp16) {
        if (batch <= 0 || seqLen <= 0 || vocab <= 0) {
            throw new IllegalArgumentException("batch, seqLen, vocab must be positive");
        }
        Objects.requireNonNull(targets, "targets");
        long rows = (long) batch * seqLen;
        long need = rows * vocab;
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("CE logits/grad", need);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("CE targets row count", rows);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, "logits"), need, "logits");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(grad, "grad"), need, "grad");
        if (targets.length < rows) {
            throw new IllegalArgumentException("targets too small: need " + rows + ", have " + targets.length);
        }
        TensorOpsGPU.crossEntropySoftmaxGradLossGPUDeviceHostFloatTargetsAsync(
                logits.devicePointer(), targets, grad.devicePointer(), batch, seqLen, vocab, gradScale, fp16);
    }

    static float crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost() {
        return TensorOpsGPU.crossEntropySoftmaxGradLossGPUDeviceReadPendingFromHost();
    }

    static void gatherLogitsByIdsGpuDevice(
            GpuFloatBuffer logits,
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateLogits,
            int rows,
            int vocab,
            int candidates) {
        if (rows <= 0 || vocab <= 0 || candidates <= 0) {
            throw new IllegalArgumentException("rows, vocab, candidates must be positive");
        }
        long logitNeed = (long) rows * vocab;
        long candidateNeed = (long) rows * candidates;
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("gather logits plane", logitNeed);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount("gather candidate plane", candidateNeed);
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(logits, "logits"), logitNeed, "logits");
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(candidateIds, "candidateIds"), candidateNeed, "candidateIds");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateLogits, "candidateLogits"), candidateNeed, "candidateLogits");
        TensorOpsGPU.gatherLogitsByIdsGPUDevice(
                logits.devicePointer(),
                candidateIds.devicePointer(),
                candidateLogits.devicePointer(),
                rows,
                vocab,
                candidates);
    }

    static void lmHeadCandidateLogitsGpuDevice(
            GpuFloatBuffer normedHidden,
            GpuFloatBuffer lmHeadWeights,
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateLogits,
            int rows,
            int dModel,
            int vocab,
            int candidates) {
        if (rows <= 0 || dModel <= 0 || vocab <= 0 || candidates <= 0) {
            throw new IllegalArgumentException("rows, dModel, vocab, candidates must be positive");
        }
        long hiddenNeed = (long) rows * dModel;
        long weightNeed = (long) dModel * vocab;
        long candidateNeed = (long) rows * candidates;
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(normedHidden, "normedHidden"), hiddenNeed, "normedHidden");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(lmHeadWeights, "lmHeadWeights"), weightNeed, "lmHeadWeights");
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(candidateIds, "candidateIds"), candidateNeed, "candidateIds");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateLogits, "candidateLogits"), candidateNeed, "candidateLogits");
        TensorOpsGPU.lmHeadCandidateLogitsGPUDevice(
                normedHidden.devicePointer(),
                lmHeadWeights.devicePointer(),
                candidateIds.devicePointer(),
                candidateLogits.devicePointer(),
                rows,
                dModel,
                vocab,
                candidates);
    }

    static float sampledCrossEntropyGradLossGpuDeviceFirstSlot(
            GpuFloatBuffer candidateLogits,
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateGrad,
            int rows,
            int candidates,
            float gradScale) {
        if (rows <= 0 || candidates <= 0) {
            throw new IllegalArgumentException("rows and candidates must be positive");
        }
        long need = (long) rows * candidates;
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateLogits, "candidateLogits"), need, "candidateLogits");
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(candidateIds, "candidateIds"), need, "candidateIds");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateGrad, "candidateGrad"), need, "candidateGrad");
        return TensorOpsGPU.sampledCrossEntropyGradLossGPUDeviceFirstSlot(
                candidateLogits.devicePointer(),
                candidateIds.devicePointer(),
                candidateGrad.devicePointer(),
                rows,
                candidates,
                gradScale);
    }

    static void sampledLmHeadBackwardGpuDevice(
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateGrad,
            GpuFloatBuffer normedHidden,
            GpuFloatBuffer lmHeadWeights,
            GpuFloatBuffer dHidden,
            GpuFloatBuffer dLmHead,
            int rows,
            int dModel,
            int vocab,
            int candidates) {
        if (rows <= 0 || dModel <= 0 || vocab <= 0 || candidates <= 0) {
            throw new IllegalArgumentException("rows, dModel, vocab, candidates must be positive");
        }
        long candNeed = (long) rows * candidates;
        long hiddenNeed = (long) rows * dModel;
        long weightNeed = (long) dModel * vocab;
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(candidateIds, "candidateIds"), candNeed, "candidateIds");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateGrad, "candidateGrad"), candNeed, "candidateGrad");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(normedHidden, "normedHidden"), hiddenNeed, "normedHidden");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(lmHeadWeights, "lmHeadWeights"), weightNeed, "lmHeadWeights");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(dHidden, "dHidden"), hiddenNeed, "dHidden");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(dLmHead, "dLmHead"), weightNeed, "dLmHead");
        TensorOpsGPU.sampledLmHeadBackwardGPUDevice(
                candidateIds.devicePointer(),
                candidateGrad.devicePointer(),
                normedHidden.devicePointer(),
                lmHeadWeights.devicePointer(),
                dHidden.devicePointer(),
                dLmHead.devicePointer(),
                rows,
                dModel,
                vocab,
                candidates);
    }
}
