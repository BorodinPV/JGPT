package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import org.junit.jupiter.api.Test;

/**
 * Паритет FP32 / FP16 forward RMSNorm на GPU (FP16 — округление x, γ до half; RMS в FP32).
 */
public class RmsNormFp16GpuTest {

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    private static void rmsNormCpu(float[] src, float[] gamma, float[] dst, int outer, int lastDim, float eps) {
        for (int i = 0; i < outer; i++) {
            int base = i * lastDim;
            float sumSq = 0f;
            for (int j = 0; j < lastDim; j++) {
                float v = src[base + j];
                sumSq += v * v;
            }
            float rms = (float) Math.sqrt(sumSq / lastDim + eps);
            for (int j = 0; j < lastDim; j++) {
                dst[base + j] = (src[base + j] / rms) * gamma[j];
            }
        }
    }

    @Test
    public void gpuRmsNormFp32MatchesCpuFp32AndFp16Close() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int lastDim = 256;
        int outer = 256;
        int n = outer * lastDim;

        float[] x = new float[n];
        float[] g = new float[lastDim];
        java.util.Random rnd = new java.util.Random(42);
        for (int i = 0; i < n; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        for (int j = 0; j < lastDim; j++) {
            g[j] = 0.5f + rnd.nextFloat();
        }

        float eps = 1e-6f;
        float[] ref = new float[n];
        rmsNormCpu(x, g, ref, outer, lastDim, eps);

        float[] out32 = new float[n];
        float[] out16 = new float[n];
        TensorOpsGPU.rmsNormGPU(x, g, out32, outer, lastDim, eps, false);
        TensorOpsGPU.rmsNormGPU(x, g, out16, outer, lastDim, eps, true);

        assertTrue(maxAbsDiff(out32, ref) < 1e-4f, "FP32 GPU vs CPU ref");
        assertTrue(maxAbsDiff(out16, ref) < 5e-2f, "FP16-roundtrip GPU vs CPU ref (relaxed)");
    }
}
