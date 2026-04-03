package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.TensorOpsGPU;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * {@link LLMConfig#gpuE2eTrainFromEnv()}, {@link LLMConfig#effectiveFullGpuTrainStepFromEnv()} и fail-fast в
 * {@link LLMConfig#toTrainingConfig(String, int)} (в т.ч. decoder pipeline при полном GPU-шаге).
 */
class LLMConfigGpuE2eTrainEnvTest {

    @AfterEach
    void clearProperty() {
        System.clearProperty("jgpt.gpu.e2eTrain");
        System.clearProperty("jgpt.fullGpuTrain");
        System.clearProperty("jgpt.decoder.gpu.pipeline");
    }

    private static void assumeEnvBlank(String key) {
        String v = System.getenv(key);
        assumeTrue(v == null || v.isBlank());
    }

    @Test
    void gpuE2eTrainProperty_withoutCuda_throws() {
        if (TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        System.setProperty("jgpt.gpu.e2eTrain", "true");
        IllegalStateException ex =
                assertThrows(
                        IllegalStateException.class,
                        () -> LLMConfig.nano().toTrainingConfig("ck_e2e_nocuda", 500));
        assertTrue(ex.getMessage().contains("CUDA") || ex.getMessage().contains("GPU"));
    }

    @Test
    void gpuE2eTrainProperty_withCuda_blankResident_usesGpuResident() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String resident = System.getenv("JGPT_TRAIN_GPU_RESIDENT");
        assumeTrue(
                resident == null
                        || resident.isBlank()
                        || (!"0".equals(resident.trim())
                                && !"false".equalsIgnoreCase(resident.trim())));
        assumeEnvBlank("JGPT_GPU_E2E_TRAIN");
        System.setProperty("jgpt.gpu.e2eTrain", "true");
        TrainingConfig cfg = LLMConfig.nano().toTrainingConfig("ck_e2e_autores", 500);
        assertTrue(cfg.useGpuResident);
        assertTrue(cfg.fullGpuTrainStep);
    }

    @Test
    void gpuE2eTrainProperty_withCuda_pipelinePropertyOff_throws() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        assumeEnvBlank("JGPT_GPU_E2E_TRAIN");
        assumeEnvBlank("JGPT_DECODER_GPU_PIPELINE");
        System.setProperty("jgpt.gpu.e2eTrain", "true");
        System.setProperty("jgpt.decoder.gpu.pipeline", "false");
        IllegalStateException ex =
                assertThrows(
                        IllegalStateException.class,
                        () -> LLMConfig.nano().toTrainingConfig("ck_e2e_nopipe", 500));
        String msg = ex.getMessage();
        assertTrue(
                msg.contains("JGPT_DECODER_GPU_PIPELINE")
                        || msg.contains("jgpt.decoder.gpu.pipeline"));
    }

    @Test
    void fullGpuTrainProperty_withCuda_pipelinePropertyOff_throws() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        assumeEnvBlank("JGPT_GPU_E2E_TRAIN");
        assumeEnvBlank("JGPT_FULL_GPU_TRAIN");
        assumeEnvBlank("JGPT_DECODER_GPU_PIPELINE");
        System.setProperty("jgpt.fullGpuTrain", "true");
        System.setProperty("jgpt.decoder.gpu.pipeline", "false");
        IllegalStateException ex =
                assertThrows(
                        IllegalStateException.class,
                        () -> LLMConfig.nano().toTrainingConfig("ck_fullgpu_nopipe", 500));
        String msg = ex.getMessage();
        assertTrue(
                msg.contains("JGPT_DECODER_GPU_PIPELINE")
                        || msg.contains("jgpt.decoder.gpu.pipeline"));
    }
}
