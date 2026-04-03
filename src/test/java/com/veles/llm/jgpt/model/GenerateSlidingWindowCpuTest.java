package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

/**
 * Ветка скользящего окна в {@link GPTModel#generate} (host {@link KvCache}) не требует CUDA — регрессия на CI
 * без GPU.
 */
class GenerateSlidingWindowCpuTest {

    @Test
    void greedyGenerate_slidingWindowCompletes() {
        int vocab = 64;
        int maxSeq = 12;
        int dModel = 32;
        int heads = 4;
        int layers = 2;
        int dFf = 64;
        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, false);

        int seqLen = 5;
        int maxNew = 12;
        Tensor prompt = new Tensor(new int[] {1, seqLen});
        float[] p = prompt.internalBuffer();
        for (int i = 0; i < seqLen; i++) {
            p[i] = (i * 11 + 2) % vocab;
        }

        Tensor out = model.generate(prompt, maxNew, 0f, 0);
        assertEquals(1, out.getShape()[0]);
        assertEquals(seqLen + maxNew, out.getShape()[1]);

        float[] o = out.internalBuffer();
        for (int i = 0; i < seqLen; i++) {
            assertEquals(p[i], o[i], 1e-6f);
        }
    }
}
