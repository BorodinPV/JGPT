package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import org.junit.jupiter.api.Test;

class AttentionForwardGpuTest {

    @Test
    void fusedAttentionForwardMatchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 17;
        int seqLen = 64;
        int d = 241;
        if (!TensorOpsGPU.shouldUseGpuMatmulBatched(batch, seqLen, seqLen, d)) {
            return;
        }

        Random rng = new Random(55);
        Tensor q = new Tensor(new int[] {batch, seqLen, d});
        Tensor k = new Tensor(new int[] {batch, seqLen, d});
        Tensor v = new Tensor(new int[] {batch, seqLen, d});
        fillRandom(q.internalBuffer(), rng);
        fillRandom(k.internalBuffer(), rng);
        fillRandom(v.internalBuffer(), rng);
        Tensor mask = TensorOps.createCausalMask(seqLen);
        float scale = 0.19f;

        TensorOps.AttentionForwardResult actual =
                TensorOps.scaledDotProductAttentionWithWeights(q, k, v, mask, scale);
        Reference ref = referenceForward(
                q.internalBuffer(), k.internalBuffer(), v.internalBuffer(), mask.internalBuffer(), batch, seqLen, d, scale);

        assertClose(ref.probs, actual.attentionWeights.internalBuffer(), 8e-4f, "probs");
        assertClose(ref.out, actual.output.internalBuffer(), 8e-4f, "out");
    }

    private static Reference referenceForward(
            float[] q, float[] k, float[] v, float[] mask, int batch, int seqLen, int d, float scale) {
        float[] probs = new float[batch * seqLen * seqLen];
        float[] out = new float[batch * seqLen * d];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqLen; i++) {
                int base = (b * seqLen + i) * seqLen;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < seqLen; j++) {
                    float sum = 0f;
                    for (int t = 0; t < d; t++) {
                        sum += q[(b * seqLen + i) * d + t] * k[(b * seqLen + j) * d + t];
                    }
                    float score = sum * scale + mask[i * seqLen + j];
                    probs[base + j] = score;
                    max = Math.max(max, score);
                }
                float expSum = 0f;
                for (int j = 0; j < seqLen; j++) {
                    float e = (float) Math.exp(probs[base + j] - max);
                    probs[base + j] = e;
                    expSum += e;
                }
                float inv = 1f / expSum;
                for (int j = 0; j < seqLen; j++) {
                    probs[base + j] *= inv;
                }
                for (int t = 0; t < d; t++) {
                    float sum = 0f;
                    for (int j = 0; j < seqLen; j++) {
                        sum += probs[base + j] * v[(b * seqLen + j) * d + t];
                    }
                    out[(b * seqLen + i) * d + t] = sum;
                }
            }
        }
        return new Reference(out, probs);
    }

    private static void fillRandom(float[] data, Random rng) {
        for (int i = 0; i < data.length; i++) {
            data[i] = rng.nextFloat() * 2f - 1f;
        }
    }

    private static void assertClose(float[] expected, float[] actual, float eps, String name) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], eps, name + "[" + i + "]");
        }
    }

    private record Reference(float[] out, float[] probs) {}
}
