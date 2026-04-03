package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class Fp16MetricsTest {

    @Test
    void countersAndReset() {
        Fp16Metrics m = new Fp16Metrics();
        m.recordStep();
        m.recordStep();
        m.recordOverflow();
        assertEquals(2, m.getTotalSteps());
        assertEquals(1, m.getOverflowCount());
        m.reset();
        assertEquals(0, m.getTotalSteps());
        assertEquals(0, m.getOverflowCount());
    }

    @Test
    void printStatsWithZeroStepsDoesNotThrow() {
        Fp16Metrics m = new Fp16Metrics();
        m.printStats();
    }
}
