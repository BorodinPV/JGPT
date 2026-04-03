package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertFalse;
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

class FullGpuTrainConfigTest {

    private static TrainingConfig fullGpuConfig() {
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
                true,
                true,
                true,
                true,
                false);
    }

    private static DataLoader tinyLoader() {
        BPETokenizer tok = BPETokenizer.train(Arrays.asList("a b c d e f g h"), 32);
        DataLoader dl = new DataLoader(tok, 8, 2);
        dl.loadText("a b c d e f g h ".repeat(20));
        return dl;
    }

    @Test
    void rejectsFullGpuWhenDecoderPipelineOff() {
        assumeTrue(TensorOpsGPU.isGpuAvailable());
        System.clearProperty("jgpt.decoder.gpu.pipeline");
        GPTModel model = new GPTModel(64, 16, 32, 4, 1, 64, true);
        // Если в окружении задан JGPT_DECODER_GPU_PIPELINE=1, негативный сценарий недоступен.
        assumeFalse(model.canFullGpuTrain());
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new LLMTrainer(model, fullGpuConfig(), tinyLoader()));
        assertTrue(ex.getMessage().contains("fullGpuTrainStep") || ex.getMessage().contains("decoder"));
    }

    @Test
    void fullGpuTrainStepRequiresUseGpuResident() {
        TrainingConfig bad =
                new TrainingConfig(
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
                        false,
                        true,
                        false,
                        false,
                        false);
        GPTModel model = new GPTModel(64, 16, 32, 4, 1, 64, true);
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new LLMTrainer(model, bad, tinyLoader()));
        assertTrue(ex.getMessage().contains("useGpuResident"));
        model.closeGpuResidentWeights();
    }

    @Test
    void trainerConstructsWithFullGpuWhenDecoderPipelineAvailable() {
        assumeTrue(TensorOpsGPU.isGpuAvailable());
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel model = new GPTModel(64, 16, 32, 4, 1, 64, true);
            assumeTrue(model.canFullGpuTrain());
            new LLMTrainer(model, fullGpuConfig(), tinyLoader());
            model.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
        }
    }

    @Test
    void gpuResidentWithoutFullGpuFlag_llmTrainerRejects() {
        assumeTrue(TensorOpsGPU.isGpuAvailable());
        TrainingConfig residentOnly =
                new TrainingConfig(
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
                        true,
                        false,
                        false,
                        false,
                        false);
        assertTrue(residentOnly.useGpuResident);
        assertFalse(residentOnly.fullGpuTrainStep);
        GPTModel model = new GPTModel(64, 16, 32, 4, 1, 64, true);
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new LLMTrainer(model, residentOnly, tinyLoader()));
        assertTrue(ex.getMessage().contains("fullGpuTrainStep") || ex.getMessage().contains("deviceDecoder"));
        model.closeGpuResidentWeights();
    }

    @Test
    void deviceLogitsTrainStepRequiresFullGpuConstructorClamps() {
        TrainingConfig c =
                new TrainingConfig(
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
                        true,
                        false,
                        true,
                        false,
                        false);
        assertFalse(c.deviceLogitsTrainStep);
    }
}
