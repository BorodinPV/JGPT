package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;

import java.util.Arrays;

public class GPTModelTest {
    public static void main(String[] args) {
        System.out.println("🧪 Testing GPT Model...");

        int vocabSize = 1000;
        int maxSeqLen = 64;
        int dModel = 128;
        int numHeads = 8;
        int numLayers = 4;
        int dIntermediate = 512;

        System.out.println("\n📦 Initializing model...");
        GPTModel model =
                new GPTModel(vocabSize, maxSeqLen, dModel, numHeads, numLayers, dIntermediate);

        int batch = 2;
        int seqLen = 16;
        Tensor inputTokens = randomTensor(new int[]{batch, seqLen}, vocabSize, true);

        System.out.printf("%n📊 Input shape: %s%n", Arrays.toString(inputTokens.getShape()));

        System.out.println("\n🔄 Running forward pass...");
        long start = System.nanoTime();
        Tensor logits = model.forward(inputTokens);
        long end = System.nanoTime();

        System.out.printf(
                "✅ Logits shape: %s (ожидалось [%d, %d, %d])%n",
                Arrays.toString(logits.getShape()), batch, seqLen, vocabSize);
        System.out.printf("⏱️  Forward time: %.2f ms%n", (end - start) / 1_000_000.0);

        if (logits.getShape()[0] == batch
                && logits.getShape()[1] == seqLen
                && logits.getShape()[2] == vocabSize) {
            System.out.println("✅ Shape check: PASS");
        } else {
            System.out.println("❌ Shape check: FAIL");
        }

        System.out.println("\n📝 Testing text generation...");
        Tensor genInput = randomTensor(new int[]{1, 8}, vocabSize, true);
        long genStart = System.nanoTime();
        Tensor generated = model.generate(genInput, 10, 1.0f, 50);
        long genEnd = System.nanoTime();

        System.out.printf("✅ Generated shape: %s%n", Arrays.toString(generated.getShape()));
        System.out.printf("⏱️  Generation time: %.2f ms%n", (genEnd - genStart) / 1_000_000.0);

        float[] genData = generated.internalBuffer();
        System.out.print("📄 Generated tokens: [");
        for (int i = 0; i < genData.length; i++) {
            System.out.printf("%.0f", genData[i]);
            if (i < genData.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]");

        System.out.println("\n✅ GPT Model test complete!");
        System.out.println("\n🎯 Next steps:");
        System.out.println("   1. BPE Tokenizer (для работы с реальным текстом)");
        System.out.println("   2. Training loop с checkpointing");
        System.out.println("   3. Save/Load модели");
    }

    private static Tensor randomTensor(int[] shape, int maxVal, boolean integers) {
        Tensor t = new Tensor(shape);
        float[] data = t.internalBuffer();
        for (int i = 0; i < data.length; i++) {
            if (integers) {
                data[i] = (float) (int) (Math.random() * maxVal);
            } else {
                data[i] = (float) (Math.random() * 2 - 1);
            }
        }
        return t;
    }
}
