package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Тест training loop на синтетических данных.
 */
public class TrainingLoopTest {
    public static void main(String[] args) throws Exception {
        System.out.println("🧪 Testing Training Loop...");

        System.out.println("\n📦 Creating tokenizer...");
        BPETokenizer tokenizer = createTestTokenizer();

        System.out.println("\n📦 Creating model & config...");
        TrainingConfig config =
                new TrainingConfig(
                        1000,
                        24,
                        128,
                        8,
                        4,
                        512,
                        2,
                        1,
                        10,
                        0.001f,
                        0.1f,
                        0.01f,
                        1.0f,
                        100,
                        50,
                        LearningRateSchedule.COSINE,
                        0f,
                        "checkpoints",
                        50,
                        0);
        GPTModel model =
                new GPTModel(
                        config.vocabSize,
                        config.maxSeqLen,
                        config.dModel,
                        config.numHeads,
                        config.numLayers,
                        config.dIntermediate);

        System.out.println("\n📦 Creating DataLoader...");
        DataLoader dataLoader = new DataLoader(tokenizer, config.maxSeqLen, config.batchSize);

        String line =
                "the cat sat on the mat. the dog ran in the park. "
                        + "the bird flew over the tree. the fish swam in the lake. "
                        + "the cat and the dog are friends. ";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 80; i++) {
            sb.append(line);
        }
        dataLoader.loadText(sb.toString());

        if (dataLoader.numSequences() < config.batchSize) {
            throw new IllegalStateException(
                    "Not enough sequences: need at least batchSize=" + config.batchSize);
        }

        System.out.println("\n📦 Creating Trainer...");
        LLMTrainer trainer = new LLMTrainer(model, config, dataLoader);

        System.out.println("\n🔄 Testing single training step...");
        dataLoader.shuffle();
        if (dataLoader.hasMore()) {
            DataLoader.Batch batch = dataLoader.nextBatch();
            System.out.printf(
                    "✅ Batch loaded: input shape = %s, target shape = %s%n",
                    Arrays.toString(batch.input.getShape()),
                    Arrays.toString(batch.target.getShape()));

            Tensor logits = model.forward(batch.input);
            System.out.printf("✅ Forward pass: logits shape = %s%n", Arrays.toString(logits.getShape()));

            if (logits.getShape()[2] == config.vocabSize) {
                System.out.println("✅ Shape check: PASS");
            } else {
                System.out.println("❌ Shape check: FAIL");
            }
        } else {
            System.out.println("❌ hasMore() is false — increase text or lower maxSeqLen/batchSize");
        }

        System.out.println("\n💾 Testing checkpoint save/load...");
        Path chkDir = Path.of(config.checkpointDir);
        Files.createDirectories(chkDir);
        trainer.saveCheckpoint("test");
        trainer.loadCheckpoint(config.checkpointDir + "/checkpoint_test.bin");
        System.out.println("✅ Checkpoint test: PASS");

        System.out.println("\n✅ Training Loop test complete!");
        System.out.println("\n🎯 Ready for full training!");
    }

    private static BPETokenizer createTestTokenizer() {
        String[] texts = {
            "the cat sat on the mat",
            "the dog ran in the park",
            "the bird flew over the tree",
            "the fish swam in the lake"
        };
        return BPETokenizer.train(Arrays.asList(texts), 100);
    }
}
