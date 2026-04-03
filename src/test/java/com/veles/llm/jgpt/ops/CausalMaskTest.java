package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.core.Tensor;

public class CausalMaskTest {
    public static void main(String[] args) {
        System.out.println("🧪 Testing Causal Mask...");

        int seqLen = 4;
        Tensor mask = TensorOps.createCausalMask(seqLen);

        System.out.println("Causal Mask [" + seqLen + "x" + seqLen + "]:");
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                float v = mask.get(i, j);
                System.out.printf(v == Float.NEGATIVE_INFINITY ? " -inf " : "  %.0f  ", v);
            }
            System.out.println();
        }

        // Строго выше диагонали (j > i) = -inf; диагональ и ниже (j <= i) = 0
        boolean correct = true;
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                float v = mask.get(i, j);
                if (j > i && v != Float.NEGATIVE_INFINITY) {
                    correct = false;
                }
                if (j <= i && v != 0.0f) {
                    correct = false;
                }
            }
        }

        System.out.println(correct ? "✅ Causal mask correct!" : "❌ Causal mask error!");
    }
}
