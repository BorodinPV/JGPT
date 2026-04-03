package com.veles.llm.jgpt.ops;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class GpuWorkspaceTest {

    @AfterEach
    void releaseLocals() {
        GpuAttentionBackwardWorkspace.releaseThreadLocal();
        GpuAttentionResidentWorkspace.releaseThreadLocal();
        GpuBlockWorkspace.releaseThreadLocal();
    }

    @Test
    void attentionBackwardEnsure_rejectsIndivisibleDModel() {
        GpuAttentionBackwardWorkspace w = GpuAttentionBackwardWorkspace.local();
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> w.ensure(2, 7, 128, 256));
        assertTrue(ex.getMessage().contains("divisible"), ex.getMessage());
    }

    @Test
    void attentionResident_reusesHostArrayWhenNewSizeFits() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GpuAttentionResidentWorkspace w = GpuAttentionResidentWorkspace.local();
        w.ensure(10, 64);
        float[] h1 = w.getHostOut();
        w.ensure(5, 64);
        float[] h2 = w.getHostOut();
        assertSame(h1, h2, "reuse float[] when length >= rows*dModel");
        w.ensure(10, 64);
        assertSame(h1, w.getHostOut());
    }

    @Test
    void blockWorkspace_rejectsNonPositiveRows() {
        GpuBlockWorkspace w = GpuBlockWorkspace.local();
        assertThrows(IllegalArgumentException.class, () -> w.ensureFfnNorm2(0, 64, 256));
    }
}
