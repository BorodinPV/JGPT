package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

class GpuTensorTest {

    @Test
    void toHostTensor_snapshotNotAliasedToDeviceUpdates() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try (GpuTensor g = GpuTensor.allocate(new int[] {3})) {
            g.uploadFrom(new float[] {1f, 2f, 3f}, 0, 3);
            Tensor host = g.toHostTensor();
            g.uploadFrom(new float[] {9f, 9f, 9f}, 0, 3);
            assertEquals(1f, host.getLinear(0), 1e-6f);
            assertEquals(2f, host.getLinear(1), 1e-6f);
            assertEquals(3f, host.getLinear(2), 1e-6f);
        }
    }

    @Test
    void gradErrorsIncludeShape() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try (GpuTensor g = GpuTensor.allocate(new int[] {2, 3})) {
            IllegalStateException ex = assertThrows(IllegalStateException.class, g::gradDevicePointer);
            assertTrue(ex.getMessage().contains("shape="), ex.getMessage());
            assertTrue(ex.getMessage().contains("zeroGrad"), ex.getMessage());
        }
    }

    @Test
    void uploadFromFullArray() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try (GpuTensor g = GpuTensor.allocate(new int[] {2})) {
            g.uploadFrom(new float[] {5f, -5f});
            float[] out = new float[2];
            g.downloadTo(out, 0, 2);
            assertEquals(5f, out[0], 0f);
            assertEquals(-5f, out[1], 0f);
        }
    }

    @Test
    void equalsSameOpenTensorsWithSameDeviceIdentity() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try (GpuTensor a = GpuTensor.allocate(new int[] {1})) {
            assertEquals(a, a);
            assertNotEquals(a, GpuTensor.allocate(new int[] {1}));
        }
    }
}
