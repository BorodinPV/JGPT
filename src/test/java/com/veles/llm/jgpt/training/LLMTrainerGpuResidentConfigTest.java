package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class LLMTrainerGpuResidentConfigTest {

    private static TrainingConfig gpuResidentConfig() {
        return new TrainingConfig(
                64,
                16,
                32,
                4,
                1,
                64,
                2,
                1,
                1,
                0.001f,
                0f,
                0.01f,
                1.0f,
                100,
                50,
                LearningRateSchedule.COSINE,
                0f,
                "checkpoints",
                50,
                0,
                0,
                false,
                0f,
                0,
                true);
    }

    private static DataLoader tinyLoader() {
        BPETokenizer tok = BPETokenizer.train(Arrays.asList("a b c d e f g h"), 32);
        DataLoader dl = new DataLoader(tok, 8, 2);
        dl.loadText("a b c d e f g h ".repeat(20));
        return dl;
    }

    @Test
    void rejectsWhenCudaUnavailable() {
        assumeFalse(TensorOpsGPU.isGpuAvailable());
        GPTModel model = new GPTModel(64, 16, 32, 4, 1, 64, false);
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new LLMTrainer(model, gpuResidentConfig(), tinyLoader()));
        assertTrue(ex.getMessage().contains("CUDA"));
    }

    @Test
    void rejectsNonGpuResidentModelWhenCudaAvailable() {
        assumeTrue(TensorOpsGPU.isGpuAvailable());
        GPTModel model = new GPTModel(64, 16, 32, 4, 1, 64, false);
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new LLMTrainer(model, gpuResidentConfig(), tinyLoader()));
        assertTrue(ex.getMessage().contains("gpuResident"));
    }
}
