package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class LearningRateSchedulerTest {

    @Test
    void cosineWarmupPeakAndZeroEnd() {
        int total = 100;
        int wu = 10;
        float base = 1f;
        float lr1 = LearningRateScheduler.learningRateAtStep(1, total, wu, base, LearningRateSchedule.COSINE, 0f);
        assertTrue(lr1 > 0f && lr1 < base);
        float lrPeak = LearningRateScheduler.learningRateAtStep(10, total, wu, base, LearningRateSchedule.COSINE, 0f);
        assertEquals(base, lrPeak, 1e-5f);
        float lrEnd =
                LearningRateScheduler.learningRateAtStep(100, total, wu, base, LearningRateSchedule.COSINE, 0f);
        assertEquals(0f, lrEnd, 1e-5f);
    }

    @Test
    void cosineMinRatioFloor() {
        int total = 100;
        int wu = 10;
        float base = 1f;
        float lrEnd =
                LearningRateScheduler.learningRateAtStep(100, total, wu, base, LearningRateSchedule.COSINE, 0.1f);
        assertEquals(0.1f, lrEnd, 1e-5f);
    }

    @Test
    void constantAfterWarmup() {
        int total = 50;
        int wu = 5;
        float lr =
                LearningRateScheduler.learningRateAtStep(
                        49, total, wu, 0.01f, LearningRateSchedule.CONSTANT, 0f);
        assertEquals(0.01f, lr, 1e-6f);
    }

    @Test
    void linearDecaysToMinRatio() {
        int total = 100;
        int wu = 10;
        float base = 1f;
        float lrEnd =
                LearningRateScheduler.learningRateAtStep(100, total, wu, base, LearningRateSchedule.LINEAR, 0.2f);
        assertEquals(0.2f, lrEnd, 1e-5f);
    }

    @Test
    void inverseSqrtDecreases() {
        int total = 1000;
        int wu = 10;
        float base = 1f;
        float a = LearningRateScheduler.learningRateAtStep(100, total, wu, base, LearningRateSchedule.INVERSE_SQRT, 0f);
        float b = LearningRateScheduler.learningRateAtStep(200, total, wu, base, LearningRateSchedule.INVERSE_SQRT, 0f);
        assertTrue(a > b);
    }
}
