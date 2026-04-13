package com.veles.llm.jgpt;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * Host-путь fused CE + softmax + ∂logits: scratch для скаляра loss на поток и вызов package-private native из
 * {@link TensorOpsGPU}.
 */
final class TensorOpsGpuCrossEntropyHost {

    /**
     * Один scratch-float на поток для выхода mean CE из native ({@code lossOut[0]}). Перед JNI сбрасывается в {@code 0f}.
     */
    private static final ThreadLocal<float[]> CE_LOSS_OUT = ThreadLocal.withInitial(() -> new float[1]);

    private TensorOpsGpuCrossEntropyHost() {}

    static float crossEntropySoftmaxGradLossGpuEx(
            float[] logits,
            float[] targets,
            float[] gradOut,
            int batch,
            int seqLen,
            int vocab,
            float gradScaleOverTotalTokens,
            boolean useFp16Softmax) {
        float[] lossOut = CE_LOSS_OUT.get();
        lossOut[0] = 0f;
        TensorOpsGPU.crossEntropySoftmaxGradLossGPU(
                logits,
                targets,
                gradOut,
                batch,
                seqLen,
                vocab,
                gradScaleOverTotalTokens,
                lossOut,
                useFp16Softmax);
        return lossOut[0];
    }

    static float crossEntropySoftmaxGradLossGpuDirectEx(
            ByteBuffer logits,
            long logitsByteOffset,
            ByteBuffer targets,
            long targetsByteOffset,
            float[] gradOut,
            int batch,
            int seqLen,
            int vocab,
            float gradScaleOverTotalTokens,
            boolean useFp16Softmax) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(targets, "targets");
        Objects.requireNonNull(gradOut, "gradOut");
        if (!logits.isDirect() || !targets.isDirect()) {
            throw new IllegalArgumentException("logits and targets must be direct ByteBuffers");
        }
        int nrows = batch * seqLen;
        int logitElems = nrows * vocab;
        long needLogBytes = (long) logitElems * Integer.BYTES;
        long needTgtBytes = (long) nrows * Integer.BYTES;
        if (logitsByteOffset < 0 || logitsByteOffset + needLogBytes > logits.capacity()) {
            throw new IllegalArgumentException("logits buffer range invalid");
        }
        if (targetsByteOffset < 0 || targetsByteOffset + needTgtBytes > targets.capacity()) {
            throw new IllegalArgumentException("targets buffer range invalid");
        }
        if (gradOut.length < logitElems) {
            throw new IllegalArgumentException("gradOut too small");
        }
        float[] lossOut = CE_LOSS_OUT.get();
        lossOut[0] = 0f;
        TensorOpsGPU.crossEntropySoftmaxGradLossGPUDirect(
                logits,
                logitsByteOffset,
                targets,
                targetsByteOffset,
                gradOut,
                batch,
                seqLen,
                vocab,
                gradScaleOverTotalTokens,
                lossOut,
                useFp16Softmax);
        return lossOut[0];
    }
}
