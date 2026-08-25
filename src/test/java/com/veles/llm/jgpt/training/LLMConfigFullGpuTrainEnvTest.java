package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * Сырой разбор {@code JGPT_FULL_GPU_TRAIN} / {@code -Djgpt.fullGpuTrain} сохранён для совместимости,
 * но {@link LLMConfig#effectiveFullGpuTrainStepFromEnv()} равен {@link LLMConfig#canonicalGpuTrain()}.
 */
class LLMConfigFullGpuTrainEnvTest {

    @AfterEach
    void clearProperty() {
        System.clearProperty("jgpt.fullGpuTrain");
    }

    @Test
    void propertyTrue_rawReaderStillSeesRequest() {
        assumeTrue(
                System.getenv("JGPT_FULL_GPU_TRAIN") == null
                        || System.getenv("JGPT_FULL_GPU_TRAIN").isBlank());
        System.setProperty("jgpt.fullGpuTrain", "true");
        assertTrue(LLMConfig.fullGpuTrainStepFromEnv());
    }

    @Test
    void propertyFalse_rawReaderSeesOff_effectiveFollowsCanonical() {
        assumeTrue(
                System.getenv("JGPT_FULL_GPU_TRAIN") == null
                        || System.getenv("JGPT_FULL_GPU_TRAIN").isBlank());
        System.setProperty("jgpt.fullGpuTrain", "false");
        assertFalse(LLMConfig.fullGpuTrainStepFromEnv());
        assertEqualsCanonical();
    }

    @Test
    void whenEnvExplicitlyRequestsFullGpu_rawReaderDominatesProperty() {
        String env = System.getenv("JGPT_FULL_GPU_TRAIN");
        if (env == null || env.isBlank()) {
            return;
        }
        String t = env.trim();
        boolean wantsFull = "1".equals(t) || "true".equalsIgnoreCase(t);
        System.setProperty("jgpt.fullGpuTrain", "false");
        assertTrue(
                LLMConfig.fullGpuTrainStepFromEnv() == wantsFull,
                "env JGPT_FULL_GPU_TRAIN должен определять сырой разбор при несовпадении с property");
    }

    private static void assertEqualsCanonical() {
        assertTrue(LLMConfig.effectiveFullGpuTrainStepFromEnv() == LLMConfig.canonicalGpuTrain());
    }
}
