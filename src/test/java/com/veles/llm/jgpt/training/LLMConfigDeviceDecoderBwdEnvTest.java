package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.TensorOpsGPU;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * {@link LLMConfig#deviceDecoderBackwardFromEnv()} и влияние на {@link LLMConfig#toTrainingConfig(String, int)}:
 * env {@code JGPT_DEVICE_DECODER_BWD}, затем {@code -Djgpt.deviceDecoderBackward}.
 */
class LLMConfigDeviceDecoderBwdEnvTest {

    @AfterEach
    void clearProperties() {
        System.clearProperty("jgpt.deviceDecoderBackward");
        System.clearProperty("jgpt.deviceLogitsTrain");
        System.clearProperty("jgpt.fullGpuTrain");
        System.clearProperty("jgpt.gpu.e2eTrain");
        System.clearProperty("jgpt.decoder.gpu.pipeline");
    }

    private static void assumeEnvBlank(String key) {
        String v = System.getenv(key);
        assumeTrue(v == null || v.isBlank());
    }

    @Test
    void propertyTrue_enablesDeviceDecoderBackwardRequest() {
        assumeEnvBlank("JGPT_DEVICE_DECODER_BWD");
        System.setProperty("jgpt.deviceDecoderBackward", "true");
        assertTrue(LLMConfig.deviceDecoderBackwardFromEnv());
    }

    @Test
    void propertyFalse_disablesDeviceDecoderBackwardRequest() {
        assumeEnvBlank("JGPT_DEVICE_DECODER_BWD");
        System.setProperty("jgpt.deviceDecoderBackward", "false");
        assertFalse(LLMConfig.deviceDecoderBackwardFromEnv());
    }

    @Test
    void toTrainingConfig_deviceDecoderOnlyWhenFullGpuAndDeviceLogits() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        assumeEnvBlank("JGPT_GPU_E2E_TRAIN");
        assumeEnvBlank("JGPT_FULL_GPU_TRAIN");
        assumeEnvBlank("JGPT_DEVICE_LOGITS_TRAIN");
        assumeEnvBlank("JGPT_DEVICE_DECODER_BWD");
        assumeEnvBlank("JGPT_DECODER_GPU_PIPELINE");

        System.setProperty("jgpt.fullGpuTrain", "true");
        System.setProperty("jgpt.deviceLogitsTrain", "true");
        System.setProperty("jgpt.deviceDecoderBackward", "true");
        System.setProperty("jgpt.decoder.gpu.pipeline", "true");

        TrainingConfig tc = LLMConfig.nano().toTrainingConfig("checkpoints_env_dec", 500);
        assertTrue(tc.fullGpuTrainStep);
        assertTrue(tc.deviceLogitsTrainStep);
        assertTrue(tc.deviceDecoderBackward);
    }

    @Test
    void toTrainingConfig_whenFullGpu_forcesDeviceLogitsAndDecoder() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        assumeEnvBlank("JGPT_GPU_E2E_TRAIN");
        assumeEnvBlank("JGPT_FULL_GPU_TRAIN");
        assumeEnvBlank("JGPT_DEVICE_LOGITS_TRAIN");
        assumeEnvBlank("JGPT_DEVICE_DECODER_BWD");
        assumeEnvBlank("JGPT_DECODER_GPU_PIPELINE");

        System.setProperty("jgpt.fullGpuTrain", "true");
        System.setProperty("jgpt.deviceLogitsTrain", "false");
        System.setProperty("jgpt.deviceDecoderBackward", "true");
        System.setProperty("jgpt.decoder.gpu.pipeline", "true");

        TrainingConfig tc = LLMConfig.nano().toTrainingConfig("checkpoints_env_dec2", 500);
        assertTrue(tc.fullGpuTrainStep);
        assertTrue(tc.deviceLogitsTrainStep);
        assertTrue(tc.deviceDecoderBackward);
    }

    @Test
    void whenEnvExplicitlySetsDecoderBwd_itDominatesConflictingProperty() {
        String env = System.getenv("JGPT_DEVICE_DECODER_BWD");
        if (env == null || env.isBlank()) {
            return;
        }
        String t = env.trim();
        boolean on = "1".equals(t) || "true".equalsIgnoreCase(t);
        boolean off = "0".equals(t) || "false".equalsIgnoreCase(t);
        if (!on && !off) {
            return;
        }
        System.setProperty("jgpt.deviceDecoderBackward", Boolean.toString(!on));
        assertEquals(on, LLMConfig.deviceDecoderBackwardFromEnv());
    }
}
