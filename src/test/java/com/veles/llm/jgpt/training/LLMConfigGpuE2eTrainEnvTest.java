package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * {@link LLMConfig#toTrainingConfig(String, int)} требует CUDA; наследие {@code JGPT_GPU_E2E_TRAIN} /
 * pipeline-флагов не выбирает путь.
 */
class LLMConfigGpuE2eTrainEnvTest {

    @AfterEach
    void clearProperty() {
        System.clearProperty("jgpt.gpu.e2eTrain");
        System.clearProperty("jgpt.fullGpuTrain");
        System.clearProperty("jgpt.decoder.gpu.pipeline");
    }

    @Test
    void toTrainingConfig_withoutCuda_throws() {
        if (TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        IllegalStateException ex =
                assertThrows(
                        IllegalStateException.class,
                        () -> LLMConfig.nano().toTrainingConfig("ck_e2e_nocuda", 500));
        assertTrue(ex.getMessage().contains("CUDA") || ex.getMessage().contains("GPU"));
    }

    @Test
    void toTrainingConfig_withCuda_fullPathEvenIfLegacyPipelineOff() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        System.setProperty("jgpt.gpu.e2eTrain", "true");
        System.setProperty("jgpt.decoder.gpu.pipeline", "false");
        TrainingConfig cfg = LLMConfig.nano().toTrainingConfig("ck_e2e_autores", 500);
        assertTrue(cfg.useGpuResident);
        assertTrue(cfg.fullGpuTrainStep);
        assertTrue(cfg.deviceLogitsTrainStep);
        assertTrue(cfg.deviceDecoderBackward);
    }
}
