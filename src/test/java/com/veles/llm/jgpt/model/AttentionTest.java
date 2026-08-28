package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.Arrays;

public class AttentionTest {
    public static void main(String[] unused) {
        int batch = 2;
        int seqLen = 4;
        int dK = 8;
        int dV = 8;

        Tensor q = randomTensor(new int[]{batch, seqLen, dK});
        Tensor k = randomTensor(new int[]{batch, seqLen, dK});
        Tensor v = randomTensor(new int[]{batch, seqLen, dV});

        float scale = 1.0f / (float) Math.sqrt(dK);

        System.out.println("🧪 Testing Scaled Dot-Product Attention...");
        long start = System.nanoTime();
        Tensor output = TensorOps.scaledDotProductAttention(q, k, v, scale);
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
