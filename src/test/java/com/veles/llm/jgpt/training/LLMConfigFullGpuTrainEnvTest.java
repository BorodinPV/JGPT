package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.TensorOpsGPU;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * {@link LLMConfig#fullGpuTrainStepFromEnv()} и {@link LLMConfig#effectiveFullGpuTrainStepFromEnv()}:
 * приоритет env {@code JGPT_FULL_GPU_TRAIN}, затем {@code -Djgpt.fullGpuTrain}.
 */
class LLMConfigFullGpuTrainEnvTest {

    @AfterEach
    void clearProperty() {
        System.clearProperty("jgpt.fullGpuTrain");
    }

    @Test
    void propertyTrue_enablesFullGpuTrainRequest() {
        assumeTrue(
                System.getenv("JGPT_FULL_GPU_TRAIN") == null
                        || System.getenv("JGPT_FULL_GPU_TRAIN").isBlank());
        System.setProperty("jgpt.fullGpuTrain", "true");
        assertTrue(LLMConfig.fullGpuTrainStepFromEnv());
        if (TensorOpsGPU.isGpuAvailable()) {
            assertTrue(LLMConfig.effectiveFullGpuTrainStepFromEnv());
        } else {
            assertFalse(LLMConfig.effectiveFullGpuTrainStepFromEnv());
        }
    }

    @Test
    void propertyFalse_disablesFullGpuTrainRequest() {
        assumeTrue(
                System.getenv("JGPT_FULL_GPU_TRAIN") == null
                        || System.getenv("JGPT_FULL_GPU_TRAIN").isBlank());
        System.setProperty("jgpt.fullGpuTrain", "false");
        assertFalse(LLMConfig.fullGpuTrainStepFromEnv());
        assertFalse(LLMConfig.effectiveFullGpuTrainStepFromEnv());
    }

    @Test
    void whenEnvExplicitlyRequestsFullGpu_itDominatesProperty() {
        String env = System.getenv("JGPT_FULL_GPU_TRAIN");
        if (env == null || env.isBlank()) {
            return;
        }
        String t = env.trim();
        boolean wantsFull = "1".equals(t) || "true".equalsIgnoreCase(t);
        System.setProperty("jgpt.fullGpuTrain", "false");
        assertTrue(
                LLMConfig.fullGpuTrainStepFromEnv() == wantsFull,
                "env JGPT_FULL_GPU_TRAIN должен определять запрос при несовпадении с -Djgpt.fullGpuTrain=false");
    }
}
