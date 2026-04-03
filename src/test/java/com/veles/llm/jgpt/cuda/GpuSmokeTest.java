package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/** Smoke tests for JNI/CUDA primitives (was duplicate name {@code GPUTest} vs main demo). */
class GpuSmokeTest {

    @Test
    void gpuFloatBufferMemorySegmentRoundTrip() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment src = arena.allocate(4L * Float.BYTES, Float.BYTES);
            for (int i = 0; i < 4; i++) {
                src.setAtIndex(ValueLayout.JAVA_FLOAT, i, i + 0.5f);
            }
            try (GpuFloatBuffer g = GpuFloatBuffer.allocate(4)) {
                g.copyFromMemorySegment(src, 0, 4L * Float.BYTES);
                MemorySegment dst = arena.allocate(4L * Float.BYTES, Float.BYTES);
                g.copyToMemorySegment(dst, 0, 4L * Float.BYTES);
                for (int i = 0; i < 4; i++) {
                    assertEquals(i + 0.5f, dst.getAtIndex(ValueLayout.JAVA_FLOAT, i), 1e-5f);
                }
            }
        }
    }

    @Test
    void gpuMetadataIsAccessible() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        assertFalse(TensorOpsGPU.getGpuName().isBlank(), "gpu name");
        assertTrue(TensorOpsGPU.getGpuMemory() > 0, "gpu memory");
        assertTrue(TensorOpsGPU.shouldUseGpuElementwise(1), "small elementwise on GPU when available");
        assertTrue(TensorOpsGPU.shouldUseGpuMatmul(1, 1, 1), "small matmul on GPU when available");
        assertTrue(TensorOpsGPU.shouldUseGpuOptimizer(1), "small optimizer op on GPU when available");
    }

    @Test
    void optimizerAndClipPrimitivesSmoke() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }

        float[] grad = new float[] {3f, 4f};
        assertEquals(25.0, TensorOpsGPU.sumSquaresGPU(grad, grad.length), 1e-5f, "sum squares");

        TensorOpsGPU.scaleInPlaceGPU(grad, grad.length, 0.5f);
        assertArrayEquals(new float[] {1.5f, 2f}, grad, 1e-6f);

        float[] param = new float[] {1f, -2f};
        float[] m = new float[] {0f, 0f};
        float[] v = new float[] {0f, 0f};
        float[] stepGrad = new float[] {0.25f, -0.5f};
        float lr = 1e-3f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float wd = 0.01f;
        float invBias1 = 1f / (1f - beta1);
        float invBias2 = 1f / (1f - beta2);
        TensorOpsGPU.adamWStepGPU(param, stepGrad, m, v, lr, beta1, beta2, eps, wd, invBias1, invBias2, param.length);

        float[] refParam = new float[] {1f, -2f};
        float[] refM = new float[] {0f, 0f};
        float[] refV = new float[] {0f, 0f};
        for (int i = 0; i < refParam.length; i++) {
            refM[i] = beta1 * refM[i] + (1f - beta1) * stepGrad[i];
            refV[i] = beta2 * refV[i] + (1f - beta2) * stepGrad[i] * stepGrad[i];
            float mHat = refM[i] * invBias1;
            float vHat = refV[i] * invBias2;
            refParam[i] -= lr * (mHat / ((float) Math.sqrt(vHat) + eps) + wd * refParam[i]);
        }
        assertArrayEquals(refM, m, 1e-6f);
        assertArrayEquals(refV, v, 1e-6f);
        assertArrayEquals(refParam, param, 1e-6f);
    }

    @Test
    void embeddingBackwardPrimitivesSmoke() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }

        float[] tokens = new float[] {0f, 2f, 1f, 2f};
        float[] gradOut = new float[] {
            1f, 2f,
            3f, 4f,
            5f, 6f,
            7f, 8f
        };
        float[] gradTok = new float[3 * 2];
        TensorOpsGPU.embeddingTokenBackwardGPU(tokens, gradOut, gradTok, 2, 2, 2, 3);
        assertArrayEquals(new float[] {1f, 2f, 5f, 6f, 10f, 12f}, gradTok, 1e-6f);

        float[] gradPos = new float[2 * 2];
        TensorOpsGPU.embeddingPositionBackwardGPU(gradOut, gradPos, 2, 2, 2);
        assertArrayEquals(new float[] {6f, 8f, 10f, 12f}, gradPos, 1e-6f);
    }

    @Test
    void fusedAttentionBackwardPrimitiveSmoke() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }

        int batch = 1;
        int seqLen = 2;
        int d = 2;
        float[] gradOut = new float[] {
            1f, 0f,
            0f, 1f
        };
        float[] probs = new float[] {
            0.8f, 0.2f,
            0.3f, 0.7f
        };
        float[] q = new float[] {
            1f, 2f,
            3f, 4f
        };
        float[] k = new float[] {
            5f, 6f,
            7f, 8f
        };
        float[] v = new float[] {
            1f, 3f,
            2f, 4f
        };
        float[] gradQ = new float[batch * seqLen * d];
        float[] gradK = new float[batch * seqLen * d];
        float[] gradV = new float[batch * seqLen * d];

        TensorOpsGPU.scaledDotProductAttentionBackwardGPU(
                gradOut, probs, q, k, v, gradQ, gradK, gradV, batch, seqLen, d, d, 0.5f, null, false);

        float[] refQ = new float[gradQ.length];
        float[] refK = new float[gradK.length];
        float[] refV = new float[gradV.length];
        referenceAttentionBackward(gradOut, probs, q, k, v, refQ, refK, refV, batch, seqLen, d, 0.5f);

        assertArrayEquals(refQ, gradQ, 1e-5f);
        assertArrayEquals(refK, gradK, 1e-5f);
        assertArrayEquals(refV, gradV, 1e-5f);
    }

    private static void referenceAttentionBackward(
            float[] gradOut,
            float[] probs,
            float[] q,
            float[] k,
            float[] v,
            float[] gradQ,
            float[] gradK,
            float[] gradV,
            int batch,
            int seqLen,
            int d,
            float scale) {
        float[] dScores = new float[batch * seqLen * seqLen];

        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqLen; i++) {
                int rowBase = (b * seqLen + i) * seqLen;
                float dot = 0f;
                for (int j = 0; j < seqLen; j++) {
                    float dp = 0f;
                    int goBase = (b * seqLen + i) * d;
                    int vBase = (b * seqLen + j) * d;
                    for (int t = 0; t < d; t++) {
                        dp += gradOut[goBase + t] * v[vBase + t];
                    }
                    dScores[rowBase + j] = dp;
                    dot += probs[rowBase + j] * dp;
                }
                for (int j = 0; j < seqLen; j++) {
                    dScores[rowBase + j] = probs[rowBase + j] * (dScores[rowBase + j] - dot) * scale;
                }
            }
        }

        for (int b = 0; b < batch; b++) {
            for (int j = 0; j < seqLen; j++) {
                int gvBase = (b * seqLen + j) * d;
                for (int t = 0; t < d; t++) {
                    float sum = 0f;
                    for (int i = 0; i < seqLen; i++) {
                        int pIdx = (b * seqLen + i) * seqLen + j;
                        int goIdx = (b * seqLen + i) * d + t;
                        sum += probs[pIdx] * gradOut[goIdx];
                    }
                    gradV[gvBase + t] += sum;
                }
            }
        }

        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqLen; i++) {
                int gqBase = (b * seqLen + i) * d;
                for (int t = 0; t < d; t++) {
                    float sum = 0f;
                    for (int j = 0; j < seqLen; j++) {
                        int dsIdx = (b * seqLen + i) * seqLen + j;
                        int kIdx = (b * seqLen + j) * d + t;
                        sum += dScores[dsIdx] * k[kIdx];
                    }
                    gradQ[gqBase + t] += sum;
                }
            }
        }

        for (int b = 0; b < batch; b++) {
            for (int j = 0; j < seqLen; j++) {
                int gkBase = (b * seqLen + j) * d;
                for (int t = 0; t < d; t++) {
                    float sum = 0f;
                    for (int i = 0; i < seqLen; i++) {
                        int dsIdx = (b * seqLen + i) * seqLen + j;
                        int qIdx = (b * seqLen + i) * d + t;
                        sum += dScores[dsIdx] * q[qIdx];
                    }
                    gradK[gkBase + t] += sum;
                }
            }
        }
    }
}
