package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.training.AdamOptimizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** Склейка параметров в один {@link TensorOpsGPU#adamWStepFusedGPU} — тот же результат, что два {@link AdamOptimizer#stepWithParamGrad} подряд. */
class AdamBatchedGpuTest {

    @Test
    void batchedTwoTensorsMatchesSequentialGpu() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 1_100_000;

        Random rng = new Random(99);
        float[] initA = new float[n];
        float[] initB = new float[n];
        float[] gradA = new float[n];
        float[] gradB = new float[n];
        for (int i = 0; i < n; i++) {
            initA[i] = rng.nextFloat() * 2f - 1f;
            initB[i] = rng.nextFloat() * 2f - 1f;
            gradA[i] = rng.nextFloat() * 2f - 1f;
            gradB[i] = rng.nextFloat() * 2f - 1f;
        }

        float lr = 3e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float wd = 0.01f;

        Tensor aSeq = new Tensor(new int[] {n});
        Tensor bSeq = new Tensor(new int[] {n});
        System.arraycopy(initA, 0, aSeq.internalBuffer(), 0, n);
        System.arraycopy(initB, 0, bSeq.internalBuffer(), 0, n);
        aSeq.zeroGrad();
        bSeq.zeroGrad();
        System.arraycopy(gradA, 0, aSeq.gradBuffer(), 0, n);
        System.arraycopy(gradB, 0, bSeq.gradBuffer(), 0, n);

        AdamOptimizer optSeq = new AdamOptimizer(lr, beta1, beta2, eps, wd);
        optSeq.beginStep();
        optSeq.stepWithParamGrad(aSeq);
        optSeq.stepWithParamGrad(bSeq);

        Tensor aBat = new Tensor(new int[] {n});
        Tensor bBat = new Tensor(new int[] {n});
        System.arraycopy(initA, 0, aBat.internalBuffer(), 0, n);
        System.arraycopy(initB, 0, bBat.internalBuffer(), 0, n);
        aBat.zeroGrad();
        bBat.zeroGrad();
        System.arraycopy(gradA, 0, aBat.gradBuffer(), 0, n);
        System.arraycopy(gradB, 0, bBat.gradBuffer(), 0, n);

        AdamOptimizer optBat = new AdamOptimizer(lr, beta1, beta2, eps, wd);
        optBat.beginStep();
        optBat.stepAllWithParamGrad(List.of(aBat, bBat));

        float[] pa = aSeq.internalBuffer();
        float[] pb = bSeq.internalBuffer();
        float[] qa = aBat.internalBuffer();
        float[] qb = bBat.internalBuffer();
        for (int i = 0; i < n; i++) {
            assertEquals(pa[i], qa[i], 1e-4f, "A[" + i + "]");
            assertEquals(pb[i], qb[i], 1e-4f, "B[" + i + "]");
        }
    }
}
