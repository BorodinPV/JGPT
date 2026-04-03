package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.Arrays;

public class FFNTest {
    public static void main(String[] args) {
        System.out.println("🧪 Testing Feed-Forward Network...");

        int batch = 2;
        int seqLen = 8;
        int dModel = 64;
        int dIntermediate = 256;

        Tensor x = randomTensor(new int[]{batch, seqLen, dModel});

        Tensor W1 = randomTensor(new int[]{dModel, dIntermediate}, 0.1f);
        Tensor W2 = randomTensor(new int[]{dIntermediate, dModel}, 0.1f);
        Tensor W3 = randomTensor(new int[]{dModel, dIntermediate}, 0.1f);

        Tensor W1Gelu = randomTensor(new int[]{dModel, dIntermediate}, 0.1f);
        Tensor W2Gelu = randomTensor(new int[]{dIntermediate, dModel}, 0.1f);

        System.out.printf("Input shape: %s%n", Arrays.toString(x.getShape()));
        System.out.printf("d_model: %d, d_intermediate: %d%n", dModel, dIntermediate);

        long start = System.nanoTime();
        Tensor outputSwiGLU = TensorOps.feedForwardSwiGLU(x, W1, W2, W3);
        long end = System.nanoTime();
        System.out.printf("⏱️  SwiGLU: %.2f ms%n", (end - start) / 1_000_000.0);
        System.out.printf("✅ SwiGLU output shape: %s%n", Arrays.toString(outputSwiGLU.getShape()));

        start = System.nanoTime();
        Tensor outputGELU = TensorOps.feedForwardGELU(x, W1Gelu, W2Gelu);
        end = System.nanoTime();
        System.out.printf("⏱️  GELU: %.2f ms%n", (end - start) / 1_000_000.0);
        System.out.printf("✅ GELU output shape: %s%n", Arrays.toString(outputGELU.getShape()));

        if (Arrays.equals(outputSwiGLU.getShape(), x.getShape())
                && Arrays.equals(outputGELU.getShape(), x.getShape())) {
            System.out.println("✅ Shape check: PASS");
        } else {
            System.out.println("❌ Shape check: FAIL");
        }

        System.out.println("\n🧪 Testing GELU activation...");
        Tensor testInput = randomTensor(new int[]{4, 4});
        Tensor geluOutput = TensorOps.gelu(testInput);
        System.out.printf("✅ GELU output shape: %s%n", Arrays.toString(geluOutput.getShape()));

        float[] src = testInput.internalBuffer();
        float[] dst = geluOutput.internalBuffer();
        int signMatches = 0;
        for (int i = 0; i < src.length; i++) {
            if ((src[i] >= 0 && dst[i] >= 0) || (src[i] < 0 && dst[i] < 0)) {
                signMatches++;
            }
        }
        System.out.printf("✅ GELU sign preservation: %d/%d (%.1f%%)%n",
                signMatches, src.length, 100.0 * signMatches / src.length);

        System.out.println("\n✅ FFN test complete!");
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
