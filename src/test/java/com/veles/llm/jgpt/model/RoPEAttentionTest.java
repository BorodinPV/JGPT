package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.Arrays;

public class RoPEAttentionTest {
    public static void main(String[] args) {
        System.out.println("🧪 Testing RoPE + Multi-Head Attention...");

        int batch = 2;
        int seqLen = 8;
        int dModel = 64;
        int numHeads = 4;

        Tensor x = randomTensor(new int[]{batch, seqLen, dModel});

        Tensor Wq = randomTensor(new int[]{dModel, dModel}, 0.1f);
        Tensor Wk = randomTensor(new int[]{dModel, dModel}, 0.1f);
        Tensor Wv = randomTensor(new int[]{dModel, dModel}, 0.1f);
        Tensor Wo = randomTensor(new int[]{dModel, dModel}, 0.1f);

        Tensor mask = TensorOps.createCausalMask(seqLen);

        System.out.printf("Input shape: %s%n", Arrays.toString(x.getShape()));
        System.out.printf("Num heads: %d, d_model: %d, d_head: %d%n",
                numHeads, dModel, dModel / numHeads);

        long start = System.nanoTime();
        Tensor output1 = TensorOps.multiHeadAttention(x, Wq, Wk, Wv, Wo, numHeads, mask);
        long end = System.nanoTime();
        System.out.printf("⏱️  Without RoPE: %.2f ms%n", (end - start) / 1_000_000.0);

        start = System.nanoTime();
        Tensor output2 = TensorOps.multiHeadAttentionWithRoPE(x, Wq, Wk, Wv, Wo, numHeads, mask, true);
        end = System.nanoTime();
        System.out.printf("⏱️  With RoPE: %.2f ms%n", (end - start) / 1_000_000.0);

        System.out.printf("✅ Output (no RoPE) shape: %s%n", Arrays.toString(output1.getShape()));
        System.out.printf("✅ Output (with RoPE) shape: %s%n", Arrays.toString(output2.getShape()));

        float[] b1 = output1.internalBuffer();
        float[] b2 = output2.internalBuffer();
        double sumAbs = 0;
        float maxAbs = 0f;
        for (int i = 0; i < b1.length; i++) {
            float ad = Math.abs(b1[i] - b2[i]);
            sumAbs += ad;
            maxAbs = Math.max(maxAbs, ad);
        }
        float meanAbs = (float) (sumAbs / b1.length);
        System.out.printf("📊 Mean |Δ| (RoPE vs no RoPE): %.6f, max |Δ|: %.6f%n", meanAbs, maxAbs);

        if (maxAbs > 1e-5f) {
            System.out.println("✅ RoPE is affecting output (as expected)!");
        } else {
            System.out.println("⚠️  RoPE might not be working correctly");
        }

        System.out.println("\n✅ RoPE + Multi-Head Attention test complete!");
    }

    private static Tensor randomTensor(int[] shape) {
        return randomTensor(shape, 0.5f);
    }

    private static Tensor randomTensor(int[] shape, float std) {
        Tensor t = new Tensor(shape);
        float[] data = t.internalBuffer();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Math.random() * 2 - 1) * std;
        }
        return t;
    }
}
