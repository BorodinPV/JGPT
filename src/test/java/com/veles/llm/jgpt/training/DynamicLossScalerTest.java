package com.veles.llm.jgpt.training;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DynamicLossScalerTest {

    @Test
    void overflowHalvesScaleAndSkipsStep() {
        DynamicLossScaler s = new DynamicLossScaler(1024f, 1000, 1f, 65536f);
        assertFalse(s.step(true));
        assertEquals(512f, s.getScale(), 1e-5f);
        assertEquals(1f / 512f, s.getInvScale(), 1e-7f);
    }

    @Test
    void consecutiveSuccessesGrowScale() {
        DynamicLossScaler s = new DynamicLossScaler(1024f, 3, 1f, 8192f);
        assertTrue(s.step(false));
        assertTrue(s.step(false));
        assertTrue(s.step(false));
        assertEquals(2048f, s.getScale(), 1e-5f);
    }

    @Test
    void scaleClampedToMax() {
        DynamicLossScaler s = new DynamicLossScaler(4096f, 1, 1f, 4096f);
        assertTrue(s.step(false));
        assertEquals(4096f, s.getScale(), 1e-5f);
    }

    @Test
    void recoveryResetsBaselineAfterMinStreak() {
        DynamicLossScaler s = new DynamicLossScaler(64f, 1000, 1f, 65536f, 4);
        assertEquals(64f, s.getScale(), 1e-5f);
        while (s.getScale() > 1f) {
            assertFalse(s.step(true));
        }
        assertEquals(1f, s.getScale(), 1e-5f);
        assertFalse(s.step(true));
        assertFalse(s.step(true));
        assertFalse(s.step(true));
        assertFalse(s.step(true));
        assertEquals(64f, s.getScale(), 1e-5f);
    }
}
