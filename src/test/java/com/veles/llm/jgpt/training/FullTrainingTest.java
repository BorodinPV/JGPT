package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.ops.TensorOpsBackward;

import java.util.Arrays;

/**
 * Ручной прогон: один шаг forward / CE / backward и проверка ненулевых градиентов.
 */
public final class FullTrainingTest {

    public static void main(String[] args) throws Exception {
        System.out.println("🧪 Testing Full GPT Training...");

        TrainingConfig config =
                new TrainingConfig(
                        100,
                        32,
                        64,
                        4,
                        2,
                        256,
                        4,
                        1,
                        3,
                        0.01f,
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

        BPETokenizer tokenizer = createTestTokenizer();

        GPTModel model =
                new GPTModel(
                        config.vocabSize,
                        config.maxSeqLen,
                        config.dModel,
                        config.numHeads,
                        config.numLayers,
                        config.dIntermediate);

        DataLoader dataLoader = new DataLoader(tokenizer, config.maxSeqLen, config.batchSize);
        String testText =
                "the cat sat on the mat. "
                        + "the dog ran in the park. "
                        + "the bird flew over the tree. "
                        + "the cat and the dog are friends. ";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 40; i++) {
            sb.append(testText);
        }
        dataLoader.loadText(sb.toString());

        LLMTrainer trainer = new LLMTrainer(model, config, dataLoader);

        System.out.println("\n🔄 Testing single training step...");
        dataLoader.shuffle();
        if (dataLoader.hasMore()) {
            DataLoader.Batch batch = dataLoader.nextBatch();

            Tensor logits = model.forward(batch.input, true);
            System.out.printf("✅ Forward: logits shape = %s%n", Arrays.toString(logits.getShape()));

            float loss = crossEntropyLoss(logits, batch.target);
            System.out.printf("✅ Initial loss: %.4f%n", loss);

            Tensor grad = TensorOpsBackward.crossEntropySoftmaxBackward(logits, batch.target);
            logits.zeroGrad();
            System.arraycopy(
                    grad.gradBuffer(), 0, logits.gradBuffer(), 0, logits.gradBuffer().length);

            model.backward(logits);
            System.out.println("✅ Backward pass complete");

            boolean hasGradients = false;
            for (Tensor param : model.getParameters()) {
                if (param.hasGrad()) {
                    float[] g = param.gradBuffer();
                    for (float v : g) {
                        if (Math.abs(v) > 1e-6f) {
                            hasGradients = true;
                            break;
                        }
                    }
                }
                if (hasGradients) {
                    break;
                }
            }

            if (hasGradients) {
                System.out.println("✅ Gradients are non-zero (backprop working!)");
            } else {
                System.out.println("⚠️  Gradients might be zero (check backprop)");
            }
        } else {
            System.out.println("⚠️  No batch — увеличьте текст или уменьшите batchSize / maxSeqLen");
        }

        System.out.println("\n✅ Full Training test complete!");
        System.out.printf("%n🎯 Ready for real training! (%s)%n", trainer);
    }

    /** Тот же расчёт, что и в {@link LLMTrainer} (там метод приватный). */
    private static float crossEntropyLoss(Tensor logits, Tensor target) {
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];

        float[] logitData = logits.internalBuffer();
        float[] targetData = target.internalBuffer();

        float totalLoss = 0f;
        int count = 0;

        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                int logitBase = (b * seqLen + s) * vocabSize;
                int targetToken = (int) targetData[b * seqLen + s];

                if (targetToken < 0 || targetToken >= vocabSize) {
                    continue;
                }

                float max = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < vocabSize; v++) {
                    max = Math.max(max, logitData[logitBase + v]);
                }

                float sumExp = 0f;
                for (int v = 0; v < vocabSize; v++) {
                    sumExp += (float) Math.exp(logitData[logitBase + v] - max);
                }

                float logProb = logitData[logitBase + targetToken] - max - (float) Math.log(sumExp);
                totalLoss -= logProb;
                count++;
            }
        }

        return count == 0 ? 0f : totalLoss / count;
    }

    private static BPETokenizer createTestTokenizer() {
        String[] texts = {
            "the cat sat on the mat",
            "the dog ran in the park",
            "the bird flew over the tree"
        };
        return BPETokenizer.train(Arrays.asList(texts), 100);
    }
}
