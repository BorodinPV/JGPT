package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

class GenerateGpuKvSmokeTest {

    @Test
    void generateGpuKv_runsAndPreservesPrompt() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int vocab = 48;
        int maxSeq = 16;
        int dModel = 32;
        int heads = 4;
        int layers = 1;
        int dFf = 64;
        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);
        assertTrue(model.isGpuResident());

        int seqLen = 3;
        int maxNew = 4;
        Tensor prompt = new Tensor(new int[] {1, seqLen});
        float[] p = prompt.internalBuffer();
        p[0] = 5;
        p[1] = 11;
        p[2] = 19;

        Tensor out = model.generateGpuKv(prompt, maxNew, 1.0f, 0);
        assertEquals(1, out.getShape()[0]);
        assertEquals(seqLen + maxNew, out.getShape()[1]);

        float[] o = out.internalBuffer();
        assertEquals(5f, o[0]);
        assertEquals(11f, o[1]);
        assertEquals(19f, o[2]);

        model.closeGpuResidentWeights();
    }

    @Test
    void generateGpuKv_requiresGpuResidentModel() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GPTModel model = new GPTModel(16, 8, 16, 4, 1, 32, false);
        Tensor t = new Tensor(new int[] {1, 2});
        t.internalBuffer()[0] = 1;
        t.internalBuffer()[1] = 2;
        assertThrows(IllegalStateException.class, () -> model.generateGpuKv(t, 1, 1.0f, 0));
    }
}
