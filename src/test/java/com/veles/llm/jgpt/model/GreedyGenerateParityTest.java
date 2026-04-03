package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

/**
 * При {@code temperature <= 0} генерация детерминирована; последовательности токенов с host-KV и VRAM-KV должны
 * совпадать.
 */
class GreedyGenerateParityTest {

    @Test
    void greedyGenerate_matchesBetweenHostKvAndVramKv() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int vocab = 64;
        int maxSeq = 24;
        int dModel = 32;
        int heads = 4;
        int layers = 2;
        int dFf = 64;

        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);

        int seqLen = 4;
        int maxNew = 5;
        Tensor prompt = new Tensor(new int[] {1, seqLen});
        float[] p = prompt.internalBuffer();
        for (int i = 0; i < seqLen; i++) {
            p[i] = (i * 7 + 3) % vocab;
        }

        Tensor outHost = model.generate(prompt, maxNew, 0f, 0);
        Tensor outVram = model.generateGpuKv(prompt, maxNew, 0f, 0);

        assertEquals(outHost.getShape()[1], outVram.getShape()[1]);
        float[] a = outHost.internalBuffer();
        float[] b = outVram.internalBuffer();
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[i], b[i], 1e-6f, "token idx " + i);
        }

        model.closeGpuResidentWeights();
    }

    /**
     * Ветка {@code currentLen > maxSeqLen} в {@link GPTModel#generate}: сброс кэша и prefill среза с
     * {@code ropeOffset}; та же логика в {@link GPTModel#generateGpuKv}.
     */
    @Test
    void greedyGenerate_matchesWhenSlidingWindowTriggers() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int vocab = 64;
        int maxSeq = 12;
        int dModel = 32;
        int heads = 4;
        int layers = 2;
        int dFf = 64;

        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);

        int seqLen = 5;
        int maxNew = 12;
        Tensor prompt = new Tensor(new int[] {1, seqLen});
        float[] p = prompt.internalBuffer();
        for (int i = 0; i < seqLen; i++) {
            p[i] = (i * 11 + 2) % vocab;
        }

        Tensor outHost = model.generate(prompt, maxNew, 0f, 0);
        Tensor outVram = model.generateGpuKv(prompt, maxNew, 0f, 0);

        assertEquals(outHost.getShape()[1], outVram.getShape()[1]);
        float[] a = outHost.internalBuffer();
        float[] b = outVram.internalBuffer();
        for (int i = 0; i < a.length; i++) {
            assertEquals(a[i], b[i], 1e-6f, "token idx " + i);
        }

        model.closeGpuResidentWeights();
    }
}
