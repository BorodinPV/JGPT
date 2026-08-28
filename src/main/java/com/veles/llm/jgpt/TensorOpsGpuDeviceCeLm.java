package com.veles.llm.jgpt;

import java.util.Objects;

/** CE, gather по id, LM-head по кандидатам; JNI в {@link TensorOpsGPU}. */
final class TensorOpsGpuDeviceCeLm {

    private static final String NAME_LOGITS = "logits";
    private static final String NAME_TARGETS = "targets";
    private static final String NAME_NORMED_HIDDEN = "normedHidden";
    private static final String NAME_LM_HEAD_WEIGHTS = "lmHeadWeights";
    private static final String NAME_CANDIDATE_LOGITS = "candidateLogits";
    private static final String NAME_CANDIDATE_GRAD = "candidateGrad";
    private static final String NAME_CANDIDATE_IDS = "candidateIds";
    private static final String MSG_CE_LOGITS_GRAD = "CE logits/grad";
    private static final String MSG_CE_TARGETS_ROWS = "CE targets row count";

    private TensorOpsGpuDeviceCeLm() {
        // utility class
    }

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
        TensorOpsGpuBufferChecks.requireJniFlatElementCount(MSG_CE_LOGITS_GRAD, need);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount(MSG_CE_TARGETS_ROWS, rows);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, NAME_LOGITS), need, NAME_LOGITS);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(grad, "grad"), need, "grad");
        Objects.requireNonNull(targets, NAME_TARGETS);
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
        TensorOpsGpuBufferChecks.requireJniFlatElementCount(MSG_CE_LOGITS_GRAD, need);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount(MSG_CE_TARGETS_ROWS, rows);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, NAME_LOGITS), need, NAME_LOGITS);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(grad, "grad"), need, "grad");
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(targets, NAME_TARGETS), rows, NAME_TARGETS);
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
        TensorOpsGpuBufferChecks.requireJniFlatElementCount(MSG_CE_LOGITS_GRAD, need);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount(MSG_CE_TARGETS_ROWS, rows);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, NAME_LOGITS), need, NAME_LOGITS);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(grad, "grad"), need, "grad");
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(targets, NAME_TARGETS), rows, NAME_TARGETS);
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
        Objects.requireNonNull(targets, NAME_TARGETS);
        long rows = (long) batch * seqLen;
        long need = rows * vocab;
        TensorOpsGpuBufferChecks.requireJniFlatElementCount(MSG_CE_LOGITS_GRAD, need);
        TensorOpsGpuBufferChecks.requireJniFlatElementCount(MSG_CE_TARGETS_ROWS, rows);
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(logits, NAME_LOGITS), need, NAME_LOGITS);
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
                TensorOpsGpuBufferChecks.requireGpu(logits, NAME_LOGITS), logitNeed, NAME_LOGITS);
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(candidateIds, NAME_CANDIDATE_IDS), candidateNeed, NAME_CANDIDATE_IDS);
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateLogits, NAME_CANDIDATE_LOGITS), candidateNeed, NAME_CANDIDATE_LOGITS);
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
                TensorOpsGpuBufferChecks.requireGpu(normedHidden, NAME_NORMED_HIDDEN), hiddenNeed, NAME_NORMED_HIDDEN);
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(lmHeadWeights, NAME_LM_HEAD_WEIGHTS), weightNeed, NAME_LM_HEAD_WEIGHTS);
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(candidateIds, NAME_CANDIDATE_IDS), candidateNeed, NAME_CANDIDATE_IDS);
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateLogits, NAME_CANDIDATE_LOGITS), candidateNeed, NAME_CANDIDATE_LOGITS);
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
                TensorOpsGpuBufferChecks.requireGpu(candidateLogits, NAME_CANDIDATE_LOGITS), need, NAME_CANDIDATE_LOGITS);
        TensorOpsGpuBufferChecks.requireMinInts(
                TensorOpsGpuBufferChecks.requireGpuInt(candidateIds, NAME_CANDIDATE_IDS), need, NAME_CANDIDATE_IDS);
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateGrad, NAME_CANDIDATE_GRAD), need, NAME_CANDIDATE_GRAD);
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
                TensorOpsGpuBufferChecks.requireGpuInt(candidateIds, NAME_CANDIDATE_IDS), candNeed, NAME_CANDIDATE_IDS);
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(candidateGrad, NAME_CANDIDATE_GRAD), candNeed, NAME_CANDIDATE_GRAD);
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(normedHidden, NAME_NORMED_HIDDEN), hiddenNeed, NAME_NORMED_HIDDEN);
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(lmHeadWeights, NAME_LM_HEAD_WEIGHTS), weightNeed, NAME_LM_HEAD_WEIGHTS);
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
