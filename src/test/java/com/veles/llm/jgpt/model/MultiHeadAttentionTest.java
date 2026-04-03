package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.Arrays;

public class MultiHeadAttentionTest {
    public static void main(String[] args) {
        System.out.println("🧪 Testing Multi-Head Attention...");

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
        Tensor output = TensorOps.multiHeadAttention(x, Wq, Wk, Wv, Wo, numHeads, mask);
        long end = System.nanoTime();

        System.out.printf("✅ Output shape: %s (ожидалось [%d, %d, %d])%n",
                Arrays.toString(output.getShape()), batch, seqLen, dModel);
        System.out.printf("⏱️  Time: %.2f ms%n", (end - start) / 1_000_000.0);
        System.out.printf("📊 Output[0,0,0:4] = [%.4f, %.4f, %.4f, %.4f]%n",
                output.get(0, 0, 0), output.get(0, 0, 1),
                output.get(0, 0, 2), output.get(0, 0, 3));

        if (Arrays.equals(output.getShape(), x.getShape())) {
            System.out.println("✅ Shape check: PASS");
        } else {
            System.out.println("❌ Shape check: FAIL");
        }

        System.out.println("\n✅ Multi-Head Attention test complete!");
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
