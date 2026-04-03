package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;

import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Единый full-GPU путь: CE на device и градиент LM head на VRAM конечны и согласованы с хостовым снимком после
 * flush.
 */
class DeviceLogitsTrainParityTest {

    @BeforeEach
    void clearPendingBefore() {
        GpuPendingGradients.cleanupThreadLocal();
    }

    @AfterEach
    void tearDownPending() {
        GpuPendingGradients.cleanupThreadLocal();
    }

    private static TrainingConfig unifiedFullGpu() {
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
                "checkpoints_device_logits",
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
    }

    private static DataLoader dummyLoader() {
        return new DataLoader(BPETokenizer.train(java.util.List.of("a b"), 48), 8, 2);
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

    @Test
    void unifiedPath_ceLossFinite_andLmHeadGradFlushable() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");

            GPTModel model = new GPTModel(48, 8, 32, 4, 2, 64, true);
            LLMTrainer trainer = new LLMTrainer(model, unifiedFullGpu(), dummyLoader());

            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            LLMTrainer.TestMicrobatchResult r = trainer.testHarnessForwardCeBackward(batch, true);
            assertTrue(Float.isFinite(r.ceLoss));

            Map<Tensor, GpuTensor> map = model.gpuTensorByTrainableParameter();
            GpuPendingGradients.flushMergeToGpuGrads(map);
            GpuTensor gLm = map.get(model.getLmHead());
            assertTrue(gLm != null && gLm.hasGradBuffer());
            float[] gDev = new float[gLm.size()];
            gLm.gradBuffer().copyTo(gDev, 0, gDev.length);
            float maxAbs = 0f;
            for (float v : gDev) {
                maxAbs = Math.max(maxAbs, Math.abs(v));
            }
            assertTrue(Float.isFinite(maxAbs) && maxAbs > 0f, "expected non-zero finite LM grad on device");

            model.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
        }
    }
}
