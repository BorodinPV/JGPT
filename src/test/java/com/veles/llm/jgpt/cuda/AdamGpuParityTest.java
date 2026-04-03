package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.training.AdamOptimizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class AdamGpuParityTest {

    @Test
    void adamGpuStepMatchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 70_000;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(123);
        Tensor param = new Tensor(new int[] {n});
        float[] p = param.internalBuffer();
        float[] refP = new float[n];
        float[] refM = new float[n];
        float[] refV = new float[n];
        for (int i = 0; i < n; i++) {
            float v = rng.nextFloat() * 2f - 1f;
            p[i] = v;
            refP[i] = v;
        }

        float learningRate = 1e-3f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1e-8f;
        float weightDecay = 0.01f;
        AdamOptimizer optimizer = new AdamOptimizer(learningRate, beta1, beta2, epsilon, weightDecay);

        for (int step = 1; step <= 3; step++) {
            Tensor grad = new Tensor(new int[] {n});
            float[] g = grad.internalBuffer();
            for (int i = 0; i < n; i++) {
                g[i] = rng.nextFloat() * 2f - 1f;
            }

            optimizer.step(param, grad);

            float b1t = (float) Math.pow(beta1, step);
            float b2t = (float) Math.pow(beta2, step);
            float invBias1 = 1f / (1f - b1t);
            float invBias2 = 1f / (1f - b2t);
            for (int i = 0; i < n; i++) {
                refM[i] = beta1 * refM[i] + (1f - beta1) * g[i];
                refV[i] = beta2 * refV[i] + (1f - beta2) * g[i] * g[i];
                float mHat = refM[i] * invBias1;
                float vHat = refV[i] * invBias2;
                refP[i] -= learningRate * (mHat / ((float) Math.sqrt(vHat) + epsilon) + weightDecay * refP[i]);
            }
        }

        for (int i = 0; i < n; i++) {
            assertEquals(refP[i], p[i], 2e-5f, "param[" + i + "]");
        }
    }

    @Test
    void gradientClipGpuMatchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 70_000;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(321);
        Tensor t = new Tensor(new int[] {n});
        t.zeroGrad();
        float[] g = t.gradBuffer();
        float[] ref = new float[n];
        float sumSq = 0f;
        for (int i = 0; i < n; i++) {
            float v = rng.nextFloat() * 8f - 4f;
            g[i] = v;
            ref[i] = v;
            sumSq += v * v;
        }
        float maxNorm = 1.75f;
        float refNorm = (float) Math.sqrt(sumSq);
        float scale = maxNorm / refNorm;
        for (int i = 0; i < n; i++) {
            ref[i] *= scale;
        }

        AdamOptimizer optimizer = AdamOptimizer.forTesting();
        float norm = optimizer.clipGradients(List.of(t), maxNorm);

        assertEquals(refNorm, norm, 5e-3f, "norm");
        for (int i = 0; i < n; i++) {
            assertEquals(ref[i], g[i], 2e-5f, "grad[" + i + "]");
        }
    }
}
