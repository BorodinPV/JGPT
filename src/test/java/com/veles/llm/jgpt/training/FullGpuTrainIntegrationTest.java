package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;

import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * Интеграционные проверки единого full-GPU шага обучения (VRAM активации, device logits/decoder), накопление
 * микробатчей, конечность loss/весов.
 */
class FullGpuTrainIntegrationTest {

    private static final float WEIGHT_PARITY_EPS = 2.5e-3f;

    @AfterEach
    void tearDownPending() {
        GpuPendingGradients.cleanupThreadLocal();
    }

    private static void restoreDecoderPipelineProperty(String previous) {
        if (previous == null) {
            System.clearProperty("jgpt.decoder.gpu.pipeline");
        } else {
            System.setProperty("jgpt.decoder.gpu.pipeline", previous);
        }
    }

    private static TrainingConfig tinyAcc1(boolean fullGpuTrainStep) {
        return new TrainingConfig(
                48,
                8,
                32,
                4,
                2,
                64,
                2,
                1,
                1,
                0.001f,
                0f,
                0.01f,
                1.0f,
                10_000,
                10_000,
                LearningRateSchedule.CONSTANT,
                0f,
                "checkpoints_test_fullgpu",
                10_000,
                0,
                0,
                false,
                0f,
                0,
                true,
                fullGpuTrainStep,
                fullGpuTrainStep,
                fullGpuTrainStep,
                false);
    }

    private static TrainingConfig tinyAcc2(boolean fullGpuTrainStep) {
        return new TrainingConfig(
                48,
                8,
                32,
                4,
                2,
                64,
                2,
                2,
                1,
                0.001f,
                0f,
                0.01f,
                1.0f,
                10_000,
                10_000,
                LearningRateSchedule.CONSTANT,
                0f,
                "checkpoints_test_fullgpu",
                10_000,
                0,
                0,
                false,
                0f,
                0,
                true,
                fullGpuTrainStep,
                fullGpuTrainStep,
                fullGpuTrainStep,
                false);
    }

    private static DataLoader.Batch fixedBatch(int vocabSize, int batchSize, int seqLen) {
        Tensor input = new Tensor(new int[] {batchSize, seqLen});
        Tensor target = new Tensor(new int[] {batchSize, seqLen});
        float[] in = input.internalBuffer();
        float[] tg = target.internalBuffer();
        for (int i = 0; i < in.length; i++) {
            in[i] = (i * 13 + 7) % vocabSize;
        }
        for (int i = 0; i < tg.length; i++) {
            tg[i] = (i * 5 + 3) % vocabSize;
        }
        return new DataLoader.Batch(input, target);
    }

    private static void copyAllParameters(GPTModel src, GPTModel dst) {
        List<Tensor> a = src.getParameters();
        List<Tensor> b = dst.getParameters();
        for (int i = 0; i < a.size(); i++) {
            Tensor ta = a.get(i);
            Tensor tb = b.get(i);
            System.arraycopy(ta.internalBuffer(), 0, tb.internalBuffer(), 0, ta.size());
        }
        dst.syncGpuResidentWeightsFromHost();
    }

    private static float maxAbsWeightDiff(GPTModel a, GPTModel b) {
        List<Tensor> pa = a.getParameters();
        List<Tensor> pb = b.getParameters();
        float m = 0f;
        for (int i = 0; i < pa.size(); i++) {
            float[] xa = pa.get(i).internalBuffer();
            float[] xb = pb.get(i).internalBuffer();
            for (int j = 0; j < xa.length; j++) {
                m = Math.max(m, Math.abs(xa[j] - xb[j]));
            }
        }
        return m;
    }

    private static boolean anyNonFiniteParameters(GPTModel m) {
        for (Tensor p : m.getParameters()) {
            for (float v : p.internalBuffer()) {
                if (!Float.isFinite(v)) {
                    return true;
                }
            }
        }
        return false;
    }

    @Test
    void oneOptimizerStep_fullGpuPath_updatesWeights() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GpuPendingGradients.cleanupThreadLocal();
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");

            GPTModel model = new GPTModel(48, 8, 32, 4, 2, 64, true);
            assertTrue(model.canFullGpuTrain());

            TrainingConfig cfg = tinyAcc1(true);
            LLMTrainer trainer = new LLMTrainer(model, cfg, dummyLoader());

            DataLoader.Batch batch = fixedBatch(48, 2, 8);

            LLMTrainer.TestMicrobatchResult r = trainer.testHarnessForwardCeBackward(batch, true);
            assertTrue(Float.isFinite(r.ceLoss));

            assertTrue(trainer.testHarnessClipAndOptimizerStep(r.logits, r.ceLoss, 1f));
            assertFalse(anyNonFiniteParameters(model));

            model.closeGpuResidentWeights();
        } finally {
            restoreDecoderPipelineProperty(prevPipe);
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void accumulationTwoMicrobatches_unifiedPath_deterministicAcrossTrainers() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GpuPendingGradients.cleanupThreadLocal();
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");

            GPTModel modelA = new GPTModel(48, 8, 32, 4, 2, 64, true);
            GPTModel modelB = new GPTModel(48, 8, 32, 4, 2, 64, true);
            copyAllParameters(modelA, modelB);

            TrainingConfig cfg = tinyAcc2(true);
            LLMTrainer trainerA = new LLMTrainer(modelA, cfg, dummyLoader());
            LLMTrainer trainerB = new LLMTrainer(modelB, cfg, dummyLoader());

            DataLoader.Batch b1 = fixedBatch(48, 2, 8);
            DataLoader.Batch b2 = fixedBatch(48, 2, 8);
            float[] id2 = b2.input.internalBuffer();
            for (int i = 0; i < id2.length; i++) {
                id2[i] = (id2[i] + 3) % 48;
            }

            LLMTrainer.TestMicrobatchResult r1A = trainerA.testHarnessForwardCeBackward(b1, true);
            LLMTrainer.TestMicrobatchResult r2A = trainerA.testHarnessForwardCeBackward(b2, false);
            float avgA = (r1A.ceLoss + r2A.ceLoss) * 0.5f;
            assertTrue(trainerA.testHarnessClipAndOptimizerStep(r2A.logits, avgA, 1f));
            GpuPendingGradients.cleanupThreadLocal();

            LLMTrainer.TestMicrobatchResult r1B = trainerB.testHarnessForwardCeBackward(b1, true);
            LLMTrainer.TestMicrobatchResult r2B = trainerB.testHarnessForwardCeBackward(b2, false);
            float avgB = (r1B.ceLoss + r2B.ceLoss) * 0.5f;
            assertEquals(avgA, avgB, 1e-5f);
            assertTrue(trainerB.testHarnessClipAndOptimizerStep(r2B.logits, avgB, 1f));

            float md = maxAbsWeightDiff(modelA, modelB);
            assertTrue(md < WEIGHT_PARITY_EPS, "accum=2: max abs diff " + md);

            modelA.closeGpuResidentWeights();
            modelB.closeGpuResidentWeights();
        } finally {
            restoreDecoderPipelineProperty(prevPipe);
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void lastBatchPartialAccumulation_fullGpu_appliesScaleInOptimizerHarness() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");

            GPTModel model = new GPTModel(48, 8, 32, 4, 2, 64, true);
            TrainingConfig cfgFull =
                    new TrainingConfig(
                            48,
                            8,
                            32,
                            4,
                            2,
                            64,
                            2,
                            3,
                            1,
                            0.001f,
                            0f,
                            0.01f,
                            1.0f,
                            10_000,
                            10_000,
                            LearningRateSchedule.CONSTANT,
                            0f,
                            "checkpoints_test_fullgpu",
                            10_000,
                            0,
                            0,
                            false,
                            0f,
                            0,
                            true,
                            true,
                            true,
                            true,
                            false);

            LLMTrainer trainerFull = new LLMTrainer(model, cfgFull, dummyLoader());

            DataLoader.Batch b1 = fixedBatch(48, 2, 8);
            float partial = 3f / 2f;

            LLMTrainer.TestMicrobatchResult u1F = trainerFull.testHarnessForwardCeBackward(b1, true);
            LLMTrainer.TestMicrobatchResult u2F = trainerFull.testHarnessForwardCeBackward(b1, false);
            float avgF = (u1F.ceLoss + u2F.ceLoss) * 0.5f;
            assertTrue(trainerFull.testHarnessClipAndOptimizerStep(u2F.logits, avgF, partial));
            assertFalse(anyNonFiniteParameters(model));

            model.closeGpuResidentWeights();
        } finally {
            restoreDecoderPipelineProperty(prevPipe);
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    private static DataLoader dummyLoader() {
        return new DataLoader(BPETokenizer.train(java.util.List.of("a b"), 48), 8, 2);
    }

    /**
     * Ровно три шага оптимизатора за эпоху (6 последовательностей, batch 2) — hot {@link LLMTrainer#train()}
     * с {@link TrainingConfig#fullGpuTrainStep}.
     */
    private static DataLoader trainLoaderThreeOptimizerSteps() {
        DataLoader dl =
                new DataLoader(BPETokenizer.train(java.util.List.of("hello world foo bar baz qux"), 48), 8, 2);
        dl.setMaxSequences(6);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 50000; i++) {
            sb.append(" hello world foo bar");
        }
        dl.loadText(sb.toString());
        if (dl.numBatches() != 3) {
            throw new IllegalStateException("expected 3 batches, got " + dl.numBatches());
        }
        return dl;
    }

    @Test
    void trainMethod_threeOptimizerSteps_fullGpuCompletes() throws Exception {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GpuPendingGradients.cleanupThreadLocal();
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel model = new GPTModel(48, 8, 32, 4, 2, 64, true);
            TrainingConfig cfg = tinyAcc1(true);
            DataLoader dl = trainLoaderThreeOptimizerSteps();
            LLMTrainer trainer = new LLMTrainer(model, cfg, dl);
            trainer.train();
            assertEquals(3, trainer.getGlobalStep());
            assertFalse(anyNonFiniteParameters(model));
            model.closeGpuResidentWeights();
        } finally {
            restoreDecoderPipelineProperty(prevPipe);
            GpuPendingGradients.cleanupThreadLocal();
        }
    }
}

