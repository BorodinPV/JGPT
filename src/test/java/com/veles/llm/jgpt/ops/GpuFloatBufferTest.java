package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class GpuFloatBufferTest {

    @Test
    public void testGpuFloatBufferMatmulMatchesCpu() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        Tensor a = Tensor.fromArray(
                new float[]{1, 2, 3, 4, 5, 6},
                new int[]{2, 3}
        );
        Tensor b = Tensor.fromArray(
                new float[]{7, 8, 9, 10, 11, 12},
                new int[]{3, 2}
        );
        try (GpuFloatBuffer ga = GpuFloatBuffer.allocate(6);
                GpuFloatBuffer gb = GpuFloatBuffer.allocate(6);
                GpuFloatBuffer gc = GpuFloatBuffer.allocate(4)) {
            ga.copyFrom(a.internalBuffer(), 0, 6);
            gb.copyFrom(b.internalBuffer(), 0, 6);
            TensorOpsGPU.matmulGpuDevice(ga, gb, gc, 2, 3, 2);
            Tensor c = new Tensor(new int[]{2, 2});
            gc.copyTo(c.internalBuffer(), 0, 4);
            Tensor cpu = TensorOps.matmul(a, b);
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(cpu.get(i, j), c.get(i, j), 1e-4f);
                }
            }
        }
    }

    @Test
    public void testMatmulAddReluLargeGpuMatchesReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int d = 128;
        Tensor a = new Tensor(new int[]{d, d});
        Tensor b = new Tensor(new int[]{d, d});
        Tensor bias = new Tensor(new int[]{d});
        java.util.Random r = new java.util.Random(42);
        for (int i = 0; i < d * d; i++) {
            a.internalBuffer()[i] = r.nextFloat() * 0.02f - 0.01f;
            b.internalBuffer()[i] = r.nextFloat() * 0.02f - 0.01f;
        }
        for (int j = 0; j < d; j++) {
            bias.internalBuffer()[j] = r.nextFloat() * 0.1f - 0.05f;
        }

        Tensor fused = TensorOps.matmulAddRelu(a, b, bias);

        Tensor ref = new Tensor(new int[]{d, d});
        TensorOps.matmulInto(a, b, ref);
        float[] biasData = bias.internalBuffer();
        float[] rd = ref.internalBuffer();
        for (int i = 0; i < d * d; i++) {
            int col = i % d;
            float v = rd[i] + biasData[col];
            rd[i] = v > 0f ? v : 0f;
        }

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                assertEquals(ref.get(i, j), fused.get(i, j), 1e-3f);
            }
        }
    }
}
