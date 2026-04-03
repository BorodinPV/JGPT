package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Async CE с host float targets согласован с синхронным JNI после {@link TensorOpsGPU#synchronizeStream()}. */
class CeAsyncHostFloatParityTest {

    @Test
    void hostFloatAsyncMatchesSyncDeviceCe() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");

        int batch = 2;
        int seqLen = 3;
        int vocab = 11;
        int nrows = batch * seqLen;
        int logitN = nrows * vocab;
        float gradScale = 1f / (float) nrows;
        boolean fp16 = false;

        float[] targets = new float[nrows];
        for (int i = 0; i < nrows; i++) {
            targets[i] = (i % vocab);
        }

        try (GpuFloatBuffer logitsA = GpuFloatBuffer.allocate(logitN);
                GpuFloatBuffer logitsB = GpuFloatBuffer.allocate(logitN);
                GpuFloatBuffer gradA = GpuFloatBuffer.allocate(logitN);
                GpuFloatBuffer gradB = GpuFloatBuffer.allocate(logitN)) {
            float[] hLog = new float[logitN];
            for (int i = 0; i < logitN; i++) {
                hLog[i] = (i % 17) * 0.04f - 0.3f;
            }
            logitsA.copyFrom(hLog, 0, hLog.length);
            logitsB.copyFrom(hLog, 0, hLog.length);
            gradA.clear();
            gradB.clear();

            float lossSync =
                    TensorOpsGPU.crossEntropySoftmaxGradLossGpuDevice(logitsA, targets, gradA, batch, seqLen, vocab, gradScale, fp16);

            TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceHostFloatTargetsAsync(
                    logitsB, targets, gradB, batch, seqLen, vocab, gradScale, fp16);
            TensorOpsGPU.synchronizeStream();
            float lossAsync = TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost();

            float tol = 1e-4f;
            assertEquals(lossSync, lossAsync, tol, "mean CE");

            float[] ga = new float[logitN];
            float[] gb = new float[logitN];
            gradA.copyTo(ga, 0, logitN);
            gradB.copyTo(gb, 0, logitN);
            for (int i = 0; i < logitN; i++) {
                assertEquals(ga[i], gb[i], tol, "grad " + i);
            }
        }
    }

    @Test
    void pendingLossNonFiniteAfterSyncWhenLogitsNan() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");

        int batch = 1;
        int seqLen = 2;
        int vocab = 5;
        int nrows = batch * seqLen;
        int logitN = nrows * vocab;
        float[] targets = {1f, 2f};
        float gradScale = 1f / (float) nrows;

        try (GpuFloatBuffer logits = GpuFloatBuffer.allocate(logitN);
                GpuFloatBuffer grad = GpuFloatBuffer.allocate(logitN)) {
            float[] hLog = new float[logitN];
            java.util.Arrays.fill(hLog, Float.NaN);
            logits.copyFrom(hLog, 0, hLog.length);
            grad.clear();

            TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceHostFloatTargetsAsync(
                    logits, targets, grad, batch, seqLen, vocab, gradScale, false);
            TensorOpsGPU.synchronizeStream();
            float loss = TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost();
            assertFalse(Float.isFinite(loss));
        }
    }
}
