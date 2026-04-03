package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.Arrays;

public class AttentionTest {
    public static void main(String[] args) {
        int batch = 2;
        int seqLen = 4;
        int d_k = 8;
        int d_v = 8;

        Tensor Q = randomTensor(new int[]{batch, seqLen, d_k});
        Tensor K = randomTensor(new int[]{batch, seqLen, d_k});
        Tensor V = randomTensor(new int[]{batch, seqLen, d_v});

        float scale = 1.0f / (float) Math.sqrt(d_k);

        System.out.println("🧪 Testing Scaled Dot-Product Attention...");
        long start = System.nanoTime();
        Tensor output = TensorOps.scaledDotProductAttention(Q, K, V, scale);
        long end = System.nanoTime();

        System.out.printf("✅ Attention output shape: %s%n", Arrays.toString(output.getShape()));
        System.out.printf("⏱️  Time: %.2f ms%n", (end - start) / 1_000_000.0);
        System.out.printf("📊 Output[0,0,0:4] = [%.4f, %.4f, %.4f, %.4f]%n",
                output.get(0, 0, 0), output.get(0, 0, 1),
                output.get(0, 0, 2), output.get(0, 0, 3));

        System.out.println("✅ Test complete!");
    }

    private static Tensor randomTensor(int[] shape) {
        Tensor t = new Tensor(shape);
        float[] data = t.internalBuffer();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Math.random() * 2 - 1);
        }
        return t;
    }
}
