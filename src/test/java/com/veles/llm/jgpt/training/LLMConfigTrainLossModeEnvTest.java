package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class LLMConfigTrainLossModeEnvTest {

    @AfterEach
    void clearProperties() {
        System.clearProperty("jgpt.trainLossMode");
        System.clearProperty("jgpt.sampledCe.candidates");
        System.clearProperty("jgpt.sampledCe.negativeMode");
    }

    @Test
    void sampledPropertiesAreParsed() {
        assumeTrue(
                System.getenv("JGPT_TRAIN_LOSS_MODE") == null || System.getenv("JGPT_TRAIN_LOSS_MODE").isBlank());
        assumeTrue(
                System.getenv("JGPT_SAMPLED_CE_CANDIDATES") == null
                        || System.getenv("JGPT_SAMPLED_CE_CANDIDATES").isBlank());
        assumeTrue(
                System.getenv("JGPT_SAMPLED_CE_NEGATIVE_MODE") == null
                        || System.getenv("JGPT_SAMPLED_CE_NEGATIVE_MODE").isBlank());
        System.setProperty("jgpt.trainLossMode", "sampled");
        System.setProperty("jgpt.sampledCe.candidates", "17");
        System.setProperty("jgpt.sampledCe.negativeMode", "batch_shared_uniform");
        assertEquals(TrainLossMode.SAMPLED, LLMConfig.trainLossModeFromEnvOrProp());
        assertEquals(17, LLMConfig.sampledCeCandidatesFromEnv());
        assertEquals(
                SampledNegativeMode.BATCH_SHARED_UNIFORM,
                LLMConfig.sampledCeNegativeModeFromEnvOrProp());
    }

    @Test
    void invalidNegativeModeThrows() {
        assumeTrue(
                System.getenv("JGPT_SAMPLED_CE_NEGATIVE_MODE") == null
                        || System.getenv("JGPT_SAMPLED_CE_NEGATIVE_MODE").isBlank());
        System.setProperty("jgpt.sampledCe.negativeMode", "per_row");
        assertThrows(IllegalArgumentException.class, LLMConfig::sampledCeNegativeModeFromEnvOrProp);
    }
}
