package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.model.GPTModel;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * Канонический GPU-train: при CUDA полный путь без пяти env-флагов; периодический VRAM cleanup выкл.
 */
class CanonicalGpuTrainPathTest {

    @AfterEach
    void clearProperties() {
        System.clearProperty("jgpt.fullGpuTrain");
        System.clearProperty("jgpt.gpu.e2eTrain");
        System.clearProperty("jgpt.deviceLogitsTrain");
        System.clearProperty("jgpt.deviceDecoderBackward");
        System.clearProperty("jgpt.decoder.gpu.pipeline");
    }

    @Test
    void canonicalGpuTrain_matchesCudaAvailability() {
        assertEquals(TensorOpsGPU.isGpuAvailable(), LLMConfig.canonicalGpuTrain());
        assertEquals(LLMConfig.canonicalGpuTrain(), LLMConfig.effectiveGpuResidentTraining());
        assertEquals(LLMConfig.canonicalGpuTrain(), LLMConfig.effectiveFullGpuTrainStepFromEnv());
    }

    @Test
    void toTrainingConfig_withCuda_isFullGpuPath_ignoringLegacyOffFlags() {
        assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        System.setProperty("jgpt.fullGpuTrain", "false");
        System.setProperty("jgpt.gpu.e2eTrain", "false");
        System.setProperty("jgpt.deviceLogitsTrain", "false");
        System.setProperty("jgpt.deviceDecoderBackward", "false");
        System.setProperty("jgpt.decoder.gpu.pipeline", "false");

        TrainingConfig tc = LLMConfig.nano().toTrainingConfig("checkpoints_canonical", 500);
        assertTrue(tc.useGpuResident);
        assertTrue(tc.fullGpuTrainStep);
        assertTrue(tc.deviceLogitsTrainStep);
        assertTrue(tc.deviceDecoderBackward);
        assertFalse(tc.mergeFirstGpuResidentTrain);
    }

    @Test
    void gptModel_gpuResident_enablesDecoderPipelineByDefault() {
        assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        GPTModel on = new GPTModel(64, 16, 32, 4, 1, 64, true);
        try {
            assertTrue(on.isGpuResident());
            assertTrue(on.isDecoderGpuPipeline());
            assertTrue(on.canFullGpuTrain());
        } finally {
            on.closeGpuResidentWeights();
        }
        GPTModel off = new GPTModel(64, 16, 32, 4, 1, 64, true, false);
        try {
            assertTrue(off.isGpuResident());
            assertFalse(off.isDecoderGpuPipeline());
            assertFalse(off.canFullGpuTrain());
        } finally {
            off.closeGpuResidentWeights();
        }
    }

    @Test
    void periodicVramCleanupAndCudaTrimDefaultOffWhenUnset() {
        String vram = System.getenv("JGPT_VRAM_CLEANUP_EVERY_STEPS");
        if (vram == null || vram.isBlank()) {
            assertEquals(0, LlmTrainerEnvUtils.readVramCleanupEveryStepsFromEnv());
        }
        String trim = System.getenv("JGPT_CUDA_TRIM_EVERY_STEPS");
        if (trim == null || trim.isBlank()) {
            assertEquals(0, LlmTrainerEnvUtils.readCudaTrimEveryOptimizerStepsFromEnv());
        }
    }
}
