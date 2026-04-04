package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.ops.GpuWorkspaceCleanup;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class SampledTrainLossGpuTest {

    @BeforeEach
    void setUpGpuIsolation() {
        GpuPendingGradients.cleanupThreadLocal();
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
    }

    @AfterEach
    void tearDownGpuIsolation() {
        GpuPendingGradients.cleanupThreadLocal();
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
    }

    private static boolean envTruthy(String key) {
        String v = System.getenv(key);
        if (v == null) {
            return false;
        }
        String t = v.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    private static TrainingConfig sampledFullGpu(Path checkpointDir) {
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
                checkpointDir.toString(),
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
                false,
                TrainLossMode.SAMPLED,
                8,
                SampledNegativeMode.BATCH_SHARED_UNIFORM);
    }

    private static DataLoader dummyLoader() {
        return new DataLoader(BPETokenizer.train(List.of("a b"), 48), 8, 2);
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
            tg[i] = (i == tg.length - 1) ? -1f : ((i * 5 + 3) % vocabSize);
        }
        return new DataLoader.Batch(input, target);
    }

    @Test
    void sampledPathKeepsDenseDeviceLogitsGradBufferUnused() throws IOException {
        assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        assumeFalse(envTruthy("JGPT_CE_ASYNC"), "sampled mode intentionally rejects async CE");
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        GPTModel model = null;
        LLMTrainer trainer = null;
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            model = new GPTModel(48, 8, 32, 4, 2, 64, true);
            trainer =
                    new LLMTrainer(model, sampledFullGpu(Files.createTempDirectory("jgpt-sampled-grad")), dummyLoader());
            LLMTrainer.TestMicrobatchResult r = trainer.testHarnessForwardCeBackward(fixedBatch(48, 2, 8), true);
            assertTrue(Float.isFinite(r.ceLoss));
            assertEquals(2 * 8, r.logits.size(), "sampled path should keep only lightweight host stub metadata");
            assertNull(model.deviceLogitsBuffer(), "sampled path should not allocate full VRAM logits");
            assertNull(model.deviceLogitsGradBuffer(), "sampled path should not materialize dense device logits grad");

            Map<Tensor, GpuTensor> map = model.gpuTensorByTrainableParameter();
            GpuPendingGradients.flushMergeToGpuGrads(map);
            GpuTensor lm = map.get(model.getLmHead());
            assertTrue(lm != null && lm.hasGradBuffer());
            float[] grad = new float[lm.size()];
            lm.gradBuffer().copyTo(grad, 0, grad.length);
            float maxAbs = 0f;
            for (float v : grad) {
                maxAbs = Math.max(maxAbs, Math.abs(v));
            }
            assertTrue(Float.isFinite(maxAbs) && maxAbs > 0f);
        } finally {
            if (trainer != null) {
                trainer.releaseGpuResourcesAfterBook();
            } else if (model != null) {
                model.closeGpuResidentWeights();
            }
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
        }
    }

    @Test
    void sampledCheckpointRoundTripKeepsTrainingUsable() throws Exception {
        assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        assumeFalse(envTruthy("JGPT_CE_ASYNC"), "sampled mode intentionally rejects async CE");
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        GPTModel model = null;
        GPTModel restored = null;
        LLMTrainer trainer = null;
        LLMTrainer restoredTrainer = null;
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            Path dir = Files.createTempDirectory("jgpt-sampled-ckpt");
            TrainingConfig config = sampledFullGpu(dir);
            model = new GPTModel(48, 8, 32, 4, 2, 64, true);
            trainer = new LLMTrainer(model, config, dummyLoader());

            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            LLMTrainer.TestMicrobatchResult r = trainer.testHarnessForwardCeBackward(batch, true);
            assertTrue(Float.isFinite(r.ceLoss));
            assertTrue(trainer.testHarnessClipAndOptimizerStep(r.logits, r.ceLoss, 1f));
            trainer.saveCheckpoint("sampled");
            trainer.awaitPendingCheckpointWrites();

            restored = new GPTModel(48, 8, 32, 4, 2, 64, true);
            restoredTrainer = new LLMTrainer(restored, config, dummyLoader());
            restoredTrainer.loadModelWeights("sampled");
            restoredTrainer.loadCheckpoint(dir.resolve("checkpoint_sampled.bin").toString());
            assertEquals(0, restoredTrainer.getGlobalStep());

            LLMTrainer.TestMicrobatchResult resumed =
                    restoredTrainer.testHarnessForwardCeBackward(fixedBatch(48, 2, 8), true);
            assertTrue(Float.isFinite(resumed.ceLoss));
        } finally {
            if (trainer != null) {
                trainer.releaseGpuResourcesAfterBook();
            } else if (model != null) {
                model.closeGpuResidentWeights();
            }
            if (restoredTrainer != null) {
                restoredTrainer.releaseGpuResourcesAfterBook();
            } else if (restored != null) {
                restored.closeGpuResidentWeights();
            }
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
        }
    }
}
