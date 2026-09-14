package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

class GenerateMaxNewTokensZeroTest {

    @Test
    void generate_zeroNewTokens_returnsPromptCopy() {
        GPTModel model = new GPTModel(32, 8, 16, 2, 1, 32, false, false);
        Tensor prompt = new Tensor(new int[] {1, 3});
        float[] p = prompt.internalBuffer();
        p[0] = 4;
        p[1] = 7;
        p[2] = 9;

        Tensor out = model.generate(prompt, 0, 0f, 0);
        assertEquals(1, out.getShape()[0]);
        assertEquals(3, out.getShape()[1]);
        float[] o = out.internalBuffer();
        assertEquals(4f, o[0]);
        assertEquals(7f, o[1]);
        assertEquals(9f, o[2]);

        Tensor outKv = model.generateGpuKv(prompt, 0, 0f, 0);
        assertEquals(3, outKv.getShape()[1]);
        assertEquals(4f, outKv.internalBuffer()[0]);
    }

    @Test
    void generate_negativeNewTokens_rejected() {
        GPTModel model = new GPTModel(32, 8, 16, 2, 1, 32, false, false);
        Tensor prompt = new Tensor(new int[] {1, 1});
        prompt.internalBuffer()[0] = 1;
        assertThrows(IllegalArgumentException.class, () -> model.generate(prompt, -1, 0f, 0));
    }
}
