package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TransformerBackward;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import org.junit.jupiter.api.Test;

class AttentionBackwardGpuTest {

    @Test
    void fusedAttentionBackwardMatchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 17;
        int seqLen = 64;
        int d = 241;
        if (!TensorOpsGPU.shouldUseGpuMatmulBatched(batch, seqLen, seqLen, d)) {
            return;
        }

        Random rng = new Random(99);
        Tensor gradOut = new Tensor(new int[] {batch, seqLen, d});
        Tensor q = new Tensor(new int[] {batch, seqLen, d});
        Tensor k = new Tensor(new int[] {batch, seqLen, d});
        Tensor v = new Tensor(new int[] {batch, seqLen, d});
        Tensor probs = new Tensor(new int[] {batch, seqLen, seqLen});
        gradOut.zeroGrad();
        fillRandom(gradOut.gradBuffer(), rng);
        fillRandom(q.internalBuffer(), rng);
        fillRandom(k.internalBuffer(), rng);
        fillRandom(v.internalBuffer(), rng);
        fillRandomProbabilities(probs.internalBuffer(), batch, seqLen, rng);

        float[] refGq = new float[batch * seqLen * d];
        float[] refGk = new float[batch * seqLen * d];
        float[] refGv = new float[batch * seqLen * d];
        float scale = 0.37f;
        referenceBackward(
                gradOut.gradBuffer(),
                probs.internalBuffer(),
                q.internalBuffer(),
                k.internalBuffer(),
                v.internalBuffer(),
                refGq,
                refGk,
                refGv,
                batch,
                seqLen,
                d,
                scale);

        Tensor gradQ = new Tensor(new int[] {batch, seqLen, d});
        Tensor gradK = new Tensor(new int[] {batch, seqLen, d});
        Tensor gradV = new Tensor(new int[] {batch, seqLen, d});
        gradQ.zeroGrad();
        gradK.zeroGrad();
        gradV.zeroGrad();
        TransformerBackward.scaledDotProductAttentionBackward(
                gradOut, q, k, v, null, scale, probs, gradQ, gradK, gradV);

        assertClose(refGq, gradQ.gradBuffer(), 8e-4f, "dQ");
        assertClose(refGk, gradK.gradBuffer(), 8e-4f, "dK");
        assertClose(refGv, gradV.gradBuffer(), 8e-4f, "dV");
    }

    private static void fillRandom(float[] data, Random rng) {
        for (int i = 0; i < data.length; i++) {
            data[i] = rng.nextFloat() * 2f - 1f;
        }
    }

    private static void fillRandomProbabilities(float[] probs, int batch, int seqLen, Random rng) {
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqLen; i++) {
                int base = (b * seqLen + i) * seqLen;
                float sum = 0f;
                for (int j = 0; j < seqLen; j++) {
                    float v = rng.nextFloat();
                    probs[base + j] = v;
                    sum += v;
                }
                float inv = 1f / sum;
                for (int j = 0; j < seqLen; j++) {
                    probs[base + j] *= inv;
                }
            }
        }
    }

    private static void referenceBackward(
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

    private static void assertClose(float[] expected, float[] actual, float eps, String name) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], eps, name + "[" + i + "]");
        }
    }
}
