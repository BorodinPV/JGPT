package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.GpuIntBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOpsBackward;

import java.nio.FloatBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Сверка fused CE+grad на GPU с эталоном на CPU (при достаточном размере буфера). */
class CrossEntropyGpuTest {

    @Test
    void fusedCeMatchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 2;
        int seqLen = 128;
        int vocab = 257;
        int n = batch * seqLen * vocab;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        var rng = new java.util.Random(42);
        Tensor logits = new Tensor(new int[] {batch, seqLen, vocab});
        float[] ld = logits.internalBuffer();
        for (int i = 0; i < ld.length; i++) {
            ld[i] = rng.nextFloat() * 4f - 2f;
        }
        Tensor target = new Tensor(new int[] {batch, seqLen});
        float[] td = target.internalBuffer();
        for (int i = 0; i < td.length; i++) {
            td[i] = rng.nextInt(vocab);
        }

        float refLoss = referenceCeLoss(logits, target);
        Tensor gradRef = TensorOpsBackward.crossEntropySoftmaxBackward(logits, target);

        logits.zeroGrad();
        float[] gradGpu = logits.gradBuffer();
        int totalTokens = batch * seqLen;
        float gradScaleOverTotal = 1f / (float) totalTokens;
        float gpuLoss = TensorOpsGPU.crossEntropySoftmaxGradLossGpu(
                ld, td, gradGpu, batch, seqLen, vocab, gradScaleOverTotal);

        assertEquals(refLoss, gpuLoss, 1e-4f, "mean CE");

        float[] gr = gradRef.gradBuffer();
        for (int i = 0; i < gr.length; i++) {
            assertEquals(gr[i], gradGpu[i], 2e-4f, "grad[" + i + "]");
        }
    }

    @Test
    void fusedCeDirectBuffersMatchFloatArrayPath() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 2;
        int seqLen = 64;
        int vocab = 128;
        int n = batch * seqLen * vocab;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        var rng = new java.util.Random(99);
        Tensor logitsDir = Tensor.allocateDirect(new int[] {batch, seqLen, vocab});
        Tensor targetDir = Tensor.allocateDirect(new int[] {batch, seqLen});
        FloatBuffer ld = logitsDir.directFloatBuffer();
        int ti = 0;
        for (int i = 0; i < n; i++) {
            ld.put(i, rng.nextFloat() * 4f - 2f);
        }
        FloatBuffer tdf = targetDir.directFloatBuffer();
        int nt = batch * seqLen;
        for (int i = 0; i < nt; i++) {
            tdf.put(i, rng.nextInt(vocab));
        }

        Tensor logitsHeap = new Tensor(new int[] {batch, seqLen, vocab});
        System.arraycopy(logitsDir.getDataCopy(), 0, logitsHeap.internalBuffer(), 0, n);
        Tensor targetHeap = new Tensor(new int[] {batch, seqLen});
        System.arraycopy(targetDir.getDataCopy(), 0, targetHeap.internalBuffer(), 0, nt);

        logitsDir.zeroGrad();
        logitsHeap.zeroGrad();
        int totalTokens = batch * seqLen;
        float gradScaleOverTotal = 1f / (float) totalTokens;
        float lossDirect =
                TensorOpsGPU.crossEntropySoftmaxGradLossGpuDirectEx(
                        logitsDir.directByteBuffer(),
                        0L,
                        targetDir.directByteBuffer(),
                        0L,
                        logitsDir.gradBuffer(),
                        batch,
                        seqLen,
                        vocab,
                        gradScaleOverTotal,
                        false);
        float lossHeap =
                TensorOpsGPU.crossEntropySoftmaxGradLossGpuEx(
                        logitsHeap.internalBuffer(),
                        targetHeap.internalBuffer(),
                        logitsHeap.gradBuffer(),
                        batch,
                        seqLen,
                        vocab,
                        gradScaleOverTotal,
                        false);

        assertEquals(lossHeap, lossDirect, 1e-4f, "mean CE direct vs float[]");
        float[] gd = logitsDir.gradBuffer();
        float[] gh = logitsHeap.gradBuffer();
        for (int i = 0; i < n; i++) {
            assertEquals(gh[i], gd[i], 2e-4f, "grad[" + i + "]");
        }
    }

    @Test
    void fusedCeFp16CloseToFp32AndFinite() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 2;
        int seqLen = 128;
        int vocab = 257;
        int n = batch * seqLen * vocab;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        var rng = new java.util.Random(7);
        float[] ld = new float[n];
        for (int i = 0; i < n; i++) {
            ld[i] = rng.nextFloat() * 4f - 2f;
        }
        float[] td = new float[batch * seqLen];
        for (int i = 0; i < td.length; i++) {
            td[i] = rng.nextInt(vocab);
        }

        int totalTokens = batch * seqLen;
        float gradScaleOverTotal = 1f / (float) totalTokens;
        float[] g32 = new float[n];
        float[] g16 = new float[n];
        float loss32 =
                TensorOpsGPU.crossEntropySoftmaxGradLossGpuEx(
                        ld, td, g32, batch, seqLen, vocab, gradScaleOverTotal, false);
        float loss16 =
                TensorOpsGPU.crossEntropySoftmaxGradLossGpuEx(
                        ld, td, g16, batch, seqLen, vocab, gradScaleOverTotal, true);

        assertTrue(Float.isFinite(loss32) && Float.isFinite(loss16));
        assertEquals(loss32, loss16, 0.15f, "mean CE FP16 vs FP32 softmax");
        float maxDiff = 0f;
        for (int i = 0; i < n; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(g32[i] - g16[i]));
        }
        assertTrue(maxDiff < 0.02f, "max |grad_fp32 - grad_fp16|");
    }

    @Test
    void sampledCandidateCeMatchesReferenceAndSkipsMaskedRows() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int rows = 5;
        int vocab = 23;
        int candidates = 6;
        int totalLogits = rows * vocab;
        int totalCandidates = rows * candidates;
        float gradScale = 1f / 4f;

        float[] logits = new float[totalLogits];
        for (int i = 0; i < totalLogits; i++) {
            logits[i] = (i % 19) * 0.13f - 0.9f;
        }
        int[] candidateIds = new int[totalCandidates];
        for (int row = 0; row < rows; row++) {
            int base = row * candidates;
            if (row == rows - 1) {
                for (int j = 0; j < candidates; j++) {
                    candidateIds[base + j] = -1;
                }
                continue;
            }
            candidateIds[base] = (row * 5 + 3) % vocab;
            for (int j = 1; j < candidates; j++) {
                candidateIds[base + j] = (row * 7 + j * 3 + 1) % vocab;
                if (candidateIds[base + j] == candidateIds[base]) {
                    candidateIds[base + j] = (candidateIds[base + j] + 1) % vocab;
                }
            }
        }

        float[] refGrad = new float[totalCandidates];
        float refLoss = referenceSampledCeLoss(logits, candidateIds, rows, vocab, candidates, gradScale, refGrad);

        try (GpuFloatBuffer logitsGpu = GpuFloatBuffer.allocate(totalLogits);
                GpuIntBuffer idsGpu = GpuIntBuffer.allocate(totalCandidates);
                GpuFloatBuffer candLogitsGpu = GpuFloatBuffer.allocate(totalCandidates);
                GpuFloatBuffer candGradGpu = GpuFloatBuffer.allocate(totalCandidates)) {
            logitsGpu.copyFrom(logits, 0, totalLogits);
            idsGpu.copyFrom(candidateIds, 0, totalCandidates);
            candGradGpu.clear();
            TensorOpsGPU.gatherLogitsByIdsGpuDevice(logitsGpu, idsGpu, candLogitsGpu, rows, vocab, candidates);
            float loss = TensorOpsGPU.sampledCrossEntropyGradLossGpuDeviceFirstSlot(
                    candLogitsGpu, idsGpu, candGradGpu, rows, candidates, gradScale);
            assertEquals(refLoss, loss, 1e-4f);
            float[] grad = new float[totalCandidates];
            candGradGpu.copyTo(grad, 0, totalCandidates);
            for (int i = 0; i < totalCandidates; i++) {
                assertEquals(refGrad[i], grad[i], 2e-4f, "sampled grad[" + i + "]");
            }
        }
    }

    private static float referenceCeLoss(Tensor logits, Tensor target) {
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];
        float[] logitData = logits.internalBuffer();
        float[] targetData = target.internalBuffer();
        float totalLoss = 0f;
        int count = 0;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                int logitBase = (b * seqLen + s) * vocabSize;
                int targetToken = (int) targetData[b * seqLen + s];
                if (targetToken < 0 || targetToken >= vocabSize) {
                    continue;
                }
                float max = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < vocabSize; v++) {
                    max = Math.max(max, logitData[logitBase + v]);
                }
                float sumExp = 0f;
                for (int v = 0; v < vocabSize; v++) {
                    sumExp += (float) Math.exp(logitData[logitBase + v] - max);
                }
                float logProb = logitData[logitBase + targetToken] - max - (float) Math.log(sumExp);
                totalLoss -= logProb;
                count++;
            }
        }
        return count == 0 ? 0f : totalLoss / count;
    }

    private static float referenceSampledCeLoss(
            float[] logits,
            int[] candidateIds,
            int rows,
            int vocab,
            int candidates,
            float gradScale,
            float[] gradOut) {
        float totalLoss = 0f;
        int valid = 0;
        for (int row = 0; row < rows; row++) {
            int base = row * candidates;
            if (candidateIds[base] < 0) {
                for (int j = 0; j < candidates; j++) {
                    gradOut[base + j] = 0f;
                }
                continue;
            }
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < candidates; j++) {
                int cid = candidateIds[base + j];
                float v = cid >= 0 ? logits[row * vocab + cid] : Float.NEGATIVE_INFINITY;
                max = Math.max(max, v);
            }
            float sumExp = 0f;
            float[] exp = new float[candidates];
            for (int j = 0; j < candidates; j++) {
                int cid = candidateIds[base + j];
                float e =
                        cid >= 0
                                ? (float) Math.exp(logits[row * vocab + cid] - max)
                                : 0f;
                exp[j] = e;
                sumExp += e;
            }
            float targetProb = exp[0] / sumExp;
            totalLoss -= (float) Math.log(targetProb);
            valid++;
            for (int j = 0; j < candidates; j++) {
                float p = exp[j] / sumExp;
                gradOut[base + j] = ((j == 0) ? (p - 1f) : p) * gradScale;
            }
        }
        return valid == 0 ? 0f : totalLoss / valid;
    }
}
