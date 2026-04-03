package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.TensorOpsGPU;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

class EmbeddingGpuTest {

    @Test
    void tokenEmbeddingForwardMatchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 16;
        int seqLen = 64;
        int dModel = 96;
        int vocabSize = 257;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(11);
        float[] tokens = new float[batch * seqLen];
        float[] weights = new float[vocabSize * dModel];
        float[] outCpu = new float[n];
        float[] outGpu = new float[n];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = rng.nextInt(vocabSize);
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rng.nextFloat() * 2f - 1f;
        }

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                int token = (int) tokens[b * seqLen + s];
                int outBase = (b * seqLen + s) * dModel;
                int wBase = token * dModel;
                for (int j = 0; j < dModel; j++) {
                    outCpu[outBase + j] = weights[wBase + j];
                }
            }
        }

        TensorOpsGPU.embeddingTokenForwardGpuEx(tokens, weights, outGpu, batch, seqLen, dModel, vocabSize, false);

        for (int i = 0; i < n; i++) {
            assertEquals(outCpu[i], outGpu[i], 1e-5f, "fwd[" + i + "]");
        }
    }

    @Test
    void tokenEmbeddingForwardDirectBufferMatchesFloatArray() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 8;
        int seqLen = 16;
        int dModel = 64;
        int vocabSize = 100;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }
        Random rng = new Random(19);
        float[] tokens = new float[batch * seqLen];
        float[] weights = new float[vocabSize * dModel];
        float[] outArray = new float[n];
        float[] outDirect = new float[n];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = rng.nextInt(vocabSize);
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rng.nextFloat() * 2f - 1f;
        }
        ByteBuffer bb =
                ByteBuffer.allocateDirect(tokens.length * Float.BYTES).order(ByteOrder.nativeOrder());
        bb.asFloatBuffer().put(tokens);
        TensorOpsGPU.embeddingTokenForwardGpuEx(tokens, weights, outArray, batch, seqLen, dModel, vocabSize, false);
        TensorOpsGPU.embeddingTokenForwardGpuDirect(bb, weights, outDirect, batch, seqLen, dModel, vocabSize);
        for (int i = 0; i < n; i++) {
            assertEquals(outArray[i], outDirect[i], 1e-5f, "direct[" + i + "]");
        }
    }

    @Test
    void tokenEmbeddingForwardDeviceWeightsMatchesHostWeights() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 8;
        int seqLen = 16;
        int dModel = 64;
        int vocabSize = 100;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }
        Random rng = new Random(29);
        float[] tokens = new float[batch * seqLen];
        float[] weights = new float[vocabSize * dModel];
        float[] outRef = new float[n];
        float[] outDev = new float[n];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = rng.nextInt(vocabSize);
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rng.nextFloat() * 2f - 1f;
        }
        com.veles.llm.jgpt.core.Tensor wHost =
                com.veles.llm.jgpt.core.Tensor.wrap(weights, new int[]{vocabSize, dModel});
        com.veles.llm.jgpt.cuda.GpuTensor wGpu = com.veles.llm.jgpt.cuda.GpuTensor.fromHostTensor(wHost);
        TensorOpsGPU.embeddingTokenForwardGpuEx(tokens, weights, outRef, batch, seqLen, dModel, vocabSize, false);
        TensorOpsGPU.embeddingTokenForwardGpuDeviceWeights(
                tokens, wGpu.devicePointer(), outDev, batch, seqLen, dModel, vocabSize);
        for (int i = 0; i < n; i++) {
            assertEquals(outRef[i], outDev[i], 1e-5f, "devW[" + i + "]");
        }
        wGpu.close();
    }

    @Test
    void tokenEmbeddingForwardFp16CloseToFp32() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 8;
        int seqLen = 32;
        int dModel = 256;
        int vocabSize = 128;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }
        Random rng = new Random(13);
        float[] tokens = new float[batch * seqLen];
        float[] weights = new float[vocabSize * dModel];
        float[] out32 = new float[n];
        float[] out16 = new float[n];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = rng.nextInt(vocabSize);
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rng.nextFloat() * 2f - 1f;
        }
        TensorOpsGPU.embeddingTokenForwardGpuEx(tokens, weights, out32, batch, seqLen, dModel, vocabSize, false);
        TensorOpsGPU.embeddingTokenForwardGpuEx(tokens, weights, out16, batch, seqLen, dModel, vocabSize, true);
        float maxDiff = 0f;
        for (int i = 0; i < n; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(out32[i] - out16[i]));
        }
        assertTrue(maxDiff < 1e-3f, "max |fp32 - fp16 gather|");
    }

    @Test
    void tokenEmbeddingBackwardMatchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 16;
        int seqLen = 64;
        int dModel = 96;
        int vocabSize = 257;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(42);
        float[] tokens = new float[batch * seqLen];
        float[] gradOut = new float[n];
        float[] gradWeightsCpu = new float[vocabSize * dModel];
        float[] gradWeightsGpu = new float[vocabSize * dModel];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = rng.nextInt(vocabSize);
        }
        for (int i = 0; i < gradOut.length; i++) {
            gradOut[i] = rng.nextFloat() * 2f - 1f;
        }

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                int token = (int) tokens[b * seqLen + s];
                int gradBase = (b * seqLen + s) * dModel;
                int weightBase = token * dModel;
                for (int j = 0; j < dModel; j++) {
                    gradWeightsCpu[weightBase + j] += gradOut[gradBase + j];
                }
            }
        }

        TensorOpsGPU.embeddingTokenBackwardGPU(
                tokens, gradOut, gradWeightsGpu, batch, seqLen, dModel, vocabSize);

        for (int i = 0; i < gradWeightsCpu.length; i++) {
            assertEquals(gradWeightsCpu[i], gradWeightsGpu[i], 2e-4f, "token grad[" + i + "]");
        }
    }

    @Test
    void tokenEmbeddingBackwardDeviceGradWeightsMatchesHostPath() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 8;
        int seqLen = 32;
        int dModel = 64;
        int vocabSize = 128;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(99);
        float[] tokens = new float[batch * seqLen];
        float[] gradOut = new float[n];
        float[] gradHostPath = new float[vocabSize * dModel];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = rng.nextInt(vocabSize);
        }
        for (int i = 0; i < gradOut.length; i++) {
            gradOut[i] = rng.nextFloat() * 2f - 1f;
        }

        TensorOpsGPU.embeddingTokenBackwardGPU(tokens, gradOut, gradHostPath, batch, seqLen, dModel, vocabSize);

        GpuTensor gGpu = GpuTensor.allocate(new int[]{vocabSize, dModel});
        try {
            gGpu.zeroGrad();
            TensorOpsGPU.embeddingTokenBackwardGPUDeviceGradWeights(
                    tokens, gradOut, batch, seqLen, dModel, vocabSize, gGpu.gradDevicePointer());
            float[] fromDevice = new float[vocabSize * dModel];
            gGpu.gradBuffer().copyTo(fromDevice, 0, fromDevice.length);
            for (int i = 0; i < gradHostPath.length; i++) {
                assertEquals(gradHostPath[i], fromDevice[i], 2e-4f, "device-grad vs host path[" + i + "]");
            }
        } finally {
            gGpu.close();
        }
    }

    @Test
    void tokenEmbeddingBackwardDeviceGradAccumulatesLikeHostPath() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 8;
        int seqLen = 32;
        int dModel = 64;
        int vocabSize = 128;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(101);
        float[] tokens = new float[batch * seqLen];
        float[] gradOut = new float[n];
        float[] gradInit = new float[vocabSize * dModel];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = rng.nextInt(vocabSize);
        }
        for (int i = 0; i < gradOut.length; i++) {
            gradOut[i] = rng.nextFloat() * 2f - 1f;
        }
        for (int i = 0; i < gradInit.length; i++) {
            gradInit[i] = (rng.nextFloat() - 0.5f) * 0.2f;
        }

        float[] gradHost = Arrays.copyOf(gradInit, gradInit.length);
        TensorOpsGPU.embeddingTokenBackwardGPU(tokens, gradOut, gradHost, batch, seqLen, dModel, vocabSize);

        GpuTensor gGpu = GpuTensor.allocate(new int[]{vocabSize, dModel});
        try {
            gGpu.zeroGrad();
            gGpu.gradBuffer().copyFrom(gradInit, 0, gradInit.length);
            TensorOpsGPU.embeddingTokenBackwardGPUDeviceGradWeights(
                    tokens, gradOut, batch, seqLen, dModel, vocabSize, gGpu.gradDevicePointer());
            float[] fromDevice = new float[vocabSize * dModel];
            gGpu.gradBuffer().copyTo(fromDevice, 0, fromDevice.length);
            for (int i = 0; i < gradHost.length; i++) {
                assertEquals(gradHost[i], fromDevice[i], 2e-4f, "accum device vs host[" + i + "]");
            }
        } finally {
            gGpu.close();
        }
    }

    @Test
    void positionEmbeddingBackwardMatchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 16;
        int seqLen = 64;
        int dModel = 96;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(7);
        float[] gradCombined = new float[n];
        float[] gradWeightsCpu = new float[seqLen * dModel];
        float[] gradWeightsGpu = new float[seqLen * dModel];
        for (int i = 0; i < gradCombined.length; i++) {
            gradCombined[i] = rng.nextFloat() * 2f - 1f;
        }

        int plane = seqLen * dModel;
        for (int s = 0; s < seqLen; s++) {
            for (int j = 0; j < dModel; j++) {
                float sum = 0f;
                for (int b = 0; b < batch; b++) {
                    sum += gradCombined[b * plane + s * dModel + j];
                }
                gradWeightsCpu[s * dModel + j] += sum;
            }
        }

        TensorOpsGPU.embeddingPositionBackwardGPU(gradCombined, gradWeightsGpu, batch, seqLen, dModel);

        for (int i = 0; i < gradWeightsCpu.length; i++) {
            assertEquals(gradWeightsCpu[i], gradWeightsGpu[i], 2e-4f, "position grad[" + i + "]");
        }
    }

    @Test
    void tokenGatherToDevicePlusPositionBuffersMatchesCpuPipeline() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 3;
        int seqLen = 12;
        int dModel = 32;
        int vocabSize = 48;
        int maxSeq = 64;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(31);
        float[] tokens = new float[batch * seqLen];
        float[] weightTok = new float[vocabSize * dModel];
        float[] weightPos = new float[maxSeq * dModel];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = rng.nextInt(vocabSize);
        }
        for (int i = 0; i < weightTok.length; i++) {
            weightTok[i] = (rng.nextFloat() - 0.5f) * 0.3f;
        }
        for (int i = 0; i < weightPos.length; i++) {
            weightPos[i] = (rng.nextFloat() - 0.5f) * 0.2f;
        }

        float[] xCpu = new float[n];
        int strideW = dModel;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                int t = (int) tokens[b * seqLen + s];
                int xBase = (b * seqLen + s) * dModel;
                int wBase = t * strideW;
                for (int j = 0; j < dModel; j++) {
                    xCpu[xBase + j] = weightTok[wBase + j] + weightPos[s * dModel + j];
                }
            }
        }

        GpuTensor dW = GpuTensor.allocate(new int[]{vocabSize, dModel});
        GpuTensor dP = GpuTensor.allocate(new int[]{maxSeq, dModel});
        GpuTensor xGpu = GpuTensor.allocate(new int[]{batch, seqLen, dModel});
        try {
            dW.uploadFrom(weightTok, 0, weightTok.length);
            dP.uploadFrom(weightPos, 0, weightPos.length);
            TensorOpsGPU.embeddingTokenForwardGpuDeviceWeightsToDevice(
                    tokens, dW.devicePointer(), xGpu.dataBuffer(), batch, seqLen, dModel, vocabSize);
            TensorOpsGPU.addPositionEmbeddingGpuDevice(xGpu.dataBuffer(), dP.dataBuffer(), batch, seqLen, dModel);
            float[] xOut = new float[n];
            xGpu.downloadTo(xOut, 0, n);
            for (int i = 0; i < n; i++) {
                assertEquals(xCpu[i], xOut[i], 1e-4f, "fused emb pipeline[" + i + "]");
            }
        } finally {
            dW.close();
            dP.close();
            xGpu.close();
        }
    }

    @Test
    void addPositionEmbeddingGpuDeviceWeightsMatchesCpuAdd() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 4;
        int seqLen = 16;
        int dModel = 48;
        int maxSeq = 64;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(23);
        float[] x = new float[n];
        float[] posTable = new float[maxSeq * dModel];
        for (int i = 0; i < x.length; i++) {
            x[i] = rng.nextFloat();
        }
        for (int i = 0; i < posTable.length; i++) {
            posTable[i] = rng.nextFloat() * 0.5f;
        }

        float[] xCpu = x.clone();
        int plane = seqLen * dModel;
        float[] posSlice = new float[plane];
        System.arraycopy(posTable, 0, posSlice, 0, plane);
        for (int b = 0; b < batch; b++) {
            int base = b * plane;
            for (int k = 0; k < plane; k++) {
                xCpu[base + k] += posSlice[k];
            }
        }

        GpuTensor gW = GpuTensor.allocate(new int[]{maxSeq, dModel});
        try {
            gW.uploadFrom(posTable, 0, posTable.length);
            float[] xGpu = x.clone();
            TensorOpsGPU.addPositionEmbeddingGPUDeviceWeights(
                    xGpu, gW.devicePointer(), batch, seqLen, dModel);
            for (int i = 0; i < xCpu.length; i++) {
                assertEquals(xCpu[i], xGpu[i], 1e-5f, "add pos gpu vs cpu[" + i + "]");
            }
        } finally {
            gW.close();
        }
    }

    @Test
    void positionEmbeddingBackwardDeviceGradWeightsMatchesHostPath() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 8;
        int seqLen = 32;
        int dModel = 64;
        int n = batch * seqLen * dModel;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(17);
        float[] gradCombined = new float[n];
        for (int i = 0; i < gradCombined.length; i++) {
            gradCombined[i] = rng.nextFloat() * 2f - 1f;
        }

        float[] gradHostPath = new float[seqLen * dModel];
        TensorOpsGPU.embeddingPositionBackwardGPU(gradCombined, gradHostPath, batch, seqLen, dModel);

        GpuTensor gGpu = GpuTensor.allocate(new int[]{seqLen, dModel});
        try {
            gGpu.zeroGrad();
            TensorOpsGPU.embeddingPositionBackwardGPUDeviceGradWeights(
                    gradCombined, batch, seqLen, dModel, gGpu.gradDevicePointer());
            float[] fromDevice = new float[seqLen * dModel];
            gGpu.gradBuffer().copyTo(fromDevice, 0, fromDevice.length);
            for (int i = 0; i < gradHostPath.length; i++) {
                assertEquals(gradHostPath[i], fromDevice[i], 2e-4f, "position device-grad vs host[" + i + "]");
            }
        } finally {
            gGpu.close();
        }
    }
}
