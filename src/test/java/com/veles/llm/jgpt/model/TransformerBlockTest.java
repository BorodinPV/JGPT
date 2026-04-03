package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.Arrays;

public class TransformerBlockTest {
    public static void main(String[] args) {
        System.out.println("🧪 Testing Transformer Block...");

        int batch = 2;
        int seqLen = 16;
        int dModel = 128;
        int numHeads = 8;
        int dIntermediate = 512;

        Tensor x = randomTensor(new int[]{batch, seqLen, dModel});

        Tensor Wq = randomTensor(new int[]{dModel, dModel}, 0.1f);
        Tensor Wk = randomTensor(new int[]{dModel, dModel}, 0.1f);
        Tensor Wv = randomTensor(new int[]{dModel, dModel}, 0.1f);
        Tensor Wo = randomTensor(new int[]{dModel, dModel}, 0.1f);

        Tensor W1 = randomTensor(new int[]{dModel, dIntermediate}, 0.1f);
        Tensor W2 = randomTensor(new int[]{dIntermediate, dModel}, 0.1f);
        Tensor W3 = randomTensor(new int[]{dModel, dIntermediate}, 0.1f);

        Tensor norm1 = randomTensor(new int[]{dModel}, 1.0f);
        Tensor norm2 = randomTensor(new int[]{dModel}, 1.0f);

        Tensor mask = TensorOps.createCausalMask(seqLen);

        System.out.printf("Input shape: %s%n", Arrays.toString(x.getShape()));
        System.out.printf("d_model: %d, num_heads: %d, d_intermediate: %d%n",
                dModel, numHeads, dIntermediate);

        long start = System.nanoTime();
        Tensor output1 = TensorOps.transformerBlock(
                x, Wq, Wk, Wv, Wo, W1, W2, W3, norm1, norm2,
                numHeads, mask, false);
        long end = System.nanoTime();
        System.out.printf("⏱️  Without RoPE: %.2f ms%n", (end - start) / 1_000_000.0);

        start = System.nanoTime();
        Tensor output2 = TensorOps.transformerBlock(
                x, Wq, Wk, Wv, Wo, W1, W2, W3, norm1, norm2,
                numHeads, mask, true);
        end = System.nanoTime();
        System.out.printf("⏱️  With RoPE: %.2f ms%n", (end - start) / 1_000_000.0);

        System.out.printf("✅ Output (no RoPE) shape: %s%n", Arrays.toString(output1.getShape()));
        System.out.printf("✅ Output (with RoPE) shape: %s%n", Arrays.toString(output2.getShape()));

        if (Arrays.equals(output1.getShape(), x.getShape())
                && Arrays.equals(output2.getShape(), x.getShape())) {
            System.out.println("✅ Shape check: PASS");
        } else {
            System.out.println("❌ Shape check: FAIL");
        }

        float[] b1 = output1.internalBuffer();
        float[] b2 = output2.internalBuffer();
        double sumAbs = 0;
        for (int i = 0; i < b1.length; i++) {
            sumAbs += Math.abs(b1[i] - b2[i]);
        }
        float meanAbsDiff = (float) (sumAbs / b1.length);
        System.out.printf("📊 Mean |Δ| (RoPE vs no RoPE): %.6f%n", meanAbsDiff);

        if (meanAbsDiff > 0.001f) {
            System.out.println("✅ RoPE is affecting output (as expected)!");
        } else {
            System.out.println("⚠️  RoPE might not be working correctly");
        }

        float sum = 0f;
        for (float v : b1) {
            sum += Math.abs(v);
        }
        float meanAbsOut = sum / b1.length;
        System.out.printf("📊 Mean |output|: %.6f%n", meanAbsOut);

        if (meanAbsOut > 0.01f) {
            System.out.println("✅ Residual connections working (output not vanishing)!");
        } else {
            System.out.println("⚠️  Output might be vanishing");
        }

        System.out.println("\n✅ Transformer Block test complete!");
    }

    private static Tensor randomTensor(int[] shape, float std) {
        Tensor t = new Tensor(shape);
        float[] data = t.internalBuffer();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Math.random() * 2 - 1) * std;
        }
        return t;
    }

    private static Tensor randomTensor(int[] shape) {
        return randomTensor(shape, 0.5f);
    }
}
