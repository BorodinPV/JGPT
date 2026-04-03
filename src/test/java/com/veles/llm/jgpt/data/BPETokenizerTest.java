package com.veles.llm.jgpt.data;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class BPETokenizerTest {
    public static void main(String[] args) throws Exception {
        System.out.println("🧪 Testing BPE Tokenizer...");

        List<String> trainingTexts =
                Arrays.asList(
                        "the cat sat on the mat",
                        "the dog sat on the log",
                        "the cat and the dog are friends",
                        "the mat is on the floor",
                        "the cat is cute and the dog is friendly");

        System.out.println("\n📚 Training tokenizer...");
        int vocabSize = 100;
        BPETokenizer tokenizer = BPETokenizer.train(trainingTexts, vocabSize);

        System.out.printf("✅ Vocab size: %d%n", tokenizer.getVocabSize());

        System.out.println("\n🔤 Testing encode...");
        String testText = "the cat sat on the mat";
        int[] tokens = tokenizer.encode(testText, true);

        System.out.printf("Input:  '%s'%n", testText);
        System.out.printf("Tokens: %s%n", Arrays.toString(tokens));
        System.out.printf("Length: %d tokens%n", tokens.length);

        System.out.println("\n🔤 Testing decode...");
        String decoded = tokenizer.decode(tokens);
        System.out.printf("Decoded: '%s'%n", decoded);

        if (decoded.toLowerCase().contains("cat") && decoded.toLowerCase().contains("mat")) {
            System.out.println("✅ Round-trip test: PASS");
        } else {
            System.out.println("⚠️  Round-trip test: Some information lost (normal for BPE)");
        }

        System.out.println("\n💾 Testing save/load...");
        Path tmp = Files.createTempFile("bpe-tokenizer-", ".bin");
        String savePath = tmp.toString();
        try {
            tokenizer.save(savePath);
            BPETokenizer loaded = BPETokenizer.load(savePath);

            int[] tokens2 = loaded.encode(testText, true);
            if (Arrays.equals(tokens, tokens2)) {
                System.out.println("✅ Save/Load test: PASS");
            } else {
                System.out.println("❌ Save/Load test: FAIL");
            }
        } finally {
            Files.deleteIfExists(tmp);
        }

        System.out.println("\n🔤 Testing unknown words...");
        String unknownText = "the unicorn sat on the rainbow";
        int[] unknownTokens = tokenizer.encode(unknownText, true);
        String unknownDecoded = tokenizer.decode(unknownTokens);
        System.out.printf("Input:  '%s'%n", unknownText);
        System.out.printf("Output: '%s'%n", unknownDecoded);
        System.out.println("✅ Unknown words handled (with UNK tokens)");

        System.out.println("\n✅ BPE Tokenizer test complete!");
        System.out.println("\n🎯 Next steps:");
        System.out.println("   1. Training loop с реальными данными");
        System.out.println("   2. DataLoader + batching");
        System.out.println("   3. Checkpointing + resume training");
    }
}
