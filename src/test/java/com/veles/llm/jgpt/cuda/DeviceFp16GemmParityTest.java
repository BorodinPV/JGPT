package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;

import java.util.Random;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Device GEMM ({@link TensorOpsGPU#matmulGpuDeviceEx}) vs host double reference.
 * При {@code JGPT_FP16_MATMUL=1} допуск шире (Tensor Cores FP16).
 */
class DeviceFp16GemmParityTest {

    @Test
    void deviceGemmNnMatchesHost() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        compare(32, 48, 24, false, false, 0x51);
    }

    @Test
    void deviceGemmNtAndTnMatchHost() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        compare(16, 32, 20, false, true, 0x52);
        compare(16, 32, 20, true, false, 0x53);
    }

    private static void compare(int m, int k, int n, boolean trA, boolean trB, int seed) {
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        Random r = new Random(seed);
        for (int i = 0; i < a.length; i++) {
            a[i] = r.nextFloat() * 2f - 1f;
        }
        for (int i = 0; i < b.length; i++) {
            b[i] = r.nextFloat() * 2f - 1f;
        }
        float[] expected = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int t = 0; t < k; t++) {
                    float av = trA ? a[t * m + i] : a[i * k + t];
                    float bv = trB ? b[j * k + t] : b[t * n + j];
                    s += (double) av * (double) bv;
                }
                expected[i * n + j] = (float) s;
            }
        }

        try (GpuFloatBuffer dA = GpuFloatBuffer.allocate(a.length);
                GpuFloatBuffer dB = GpuFloatBuffer.allocate(b.length);
                GpuFloatBuffer dC = GpuFloatBuffer.allocate(m * n)) {
            dA.copyFrom(a, 0, a.length);
            dB.copyFrom(b, 0, b.length);
            TensorOpsGPU.matmulGpuDeviceEx(dA, dB, dC, m, k, n, trA, trB);
            TensorOpsGPU.synchronizeStream();
            float[] got = new float[m * n];
            dC.copyTo(got, 0, got.length);
            float tol = TensorOpsGPU.useFp16Matmul() ? 5e-2f : 2e-3f;
            float maxAbs = 0f;
            for (int i = 0; i < got.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(got[i] - expected[i]));
                assertEquals(expected[i], got[i], tol, "i=" + i);
            }
            assertTrue(maxAbs < tol * 4f || maxAbs < 0.15f, "maxAbs=" + maxAbs);
        }
    }
}
