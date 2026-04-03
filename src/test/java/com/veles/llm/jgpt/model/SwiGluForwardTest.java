package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;
import org.junit.jupiter.api.Test;

class SwiGluForwardTest {

    @Test
    void swigluForwardMatchesReference() {
        Random rng = new Random(77);
        int batch = 2;
        int seqLen = 3;
        int dModel = 4;
        int dIntermediate = 5;

        Tensor x = new Tensor(new int[] {batch, seqLen, dModel});
        Tensor w1 = new Tensor(new int[] {dModel, dIntermediate});
        Tensor w2 = new Tensor(new int[] {dIntermediate, dModel});
        Tensor w3 = new Tensor(new int[] {dModel, dIntermediate});
        fillRandom(x.internalBuffer(), rng);
        fillRandom(w1.internalBuffer(), rng);
        fillRandom(w2.internalBuffer(), rng);
        fillRandom(w3.internalBuffer(), rng);

        Tensor out = TensorOps.feedForwardSwiGLU(x, w1, w2, w3);
        float[] ref = reference(x, w1, w2, w3);
        float[] actual = out.internalBuffer();
        for (int i = 0; i < ref.length; i++) {
            assertEquals(ref[i], actual[i], 1e-5f, "out[" + i + "]");
        }
    }

    private static float[] reference(Tensor x, Tensor w1, Tensor w2, Tensor w3) {
        int[] xs = x.getShape();
        int batch = xs[0];
        int seqLen = xs[1];
        int dModel = xs[2];
        int dIntermediate = w1.getShape()[1];
        float[] out = new float[batch * seqLen * dModel];
        float[] xb = x.internalBuffer();
        float[] w1b = w1.internalBuffer();
        float[] w2b = w2.internalBuffer();
        float[] w3b = w3.internalBuffer();

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                float[] h1 = new float[dIntermediate];
                float[] gate = new float[dIntermediate];
                for (int j = 0; j < dIntermediate; j++) {
                    float sum1 = 0f;
                    float sum3 = 0f;
                    for (int k = 0; k < dModel; k++) {
                        float xv = xb[(b * seqLen + s) * dModel + k];
                        sum1 += xv * w1b[k * dIntermediate + j];
                        sum3 += xv * w3b[k * dIntermediate + j];
                    }
                    h1[j] = sum1;
                    gate[j] = sum3;
                }
                float[] activated = new float[dIntermediate];
                for (int j = 0; j < dIntermediate; j++) {
                    float sig = 1f / (1f + (float) Math.exp(-gate[j]));
                    activated[j] = h1[j] * gate[j] * sig;
                }
                for (int j = 0; j < dModel; j++) {
                    float sum = 0f;
                    for (int k = 0; k < dIntermediate; k++) {
                        sum += activated[k] * w2b[k * dModel + j];
                    }
                    out[(b * seqLen + s) * dModel + j] = sum;
                }
            }
        }
        return out;
    }

    private static void fillRandom(float[] data, Random rng) {
        for (int i = 0; i < data.length; i++) {
            data[i] = rng.nextFloat() * 2f - 1f;
        }
    }
}
