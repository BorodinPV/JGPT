package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** Контракт device-dropout, на который опирается {@code GpuDropout}: маска = f(seed, index), inverted scaling. */
class DropoutGpuDeviceTest {

    private static final int N = 1 << 18;
    private static final float P = 0.1f;

    private static float[] randomData(long seed) {
        Random r = new Random(seed);
        float[] d = new float[N];
        for (int i = 0; i < N; i++) {
            d[i] = (float) r.nextGaussian() + 0.5f;
        }
        return d;
    }

    @Test
    void invertedDropout_keepRateAndScale() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        float[] src = randomData(1);
        float[] out = new float[N];
        try (GpuFloatBuffer a = GpuFloatBuffer.allocate(N); GpuFloatBuffer b = GpuFloatBuffer.allocate(N)) {
            a.copyFrom(src, 0, N);
            TensorOpsGPU.dropoutGpuDevice(a, b, N, P, 0x1234_5678_9ABCL);
            b.copyTo(out, 0, N);
        }
        int zeros = 0;
        float scale = 1f / (1f - P);
        for (int i = 0; i < N; i++) {
            if (out[i] == 0f) {
                zeros++;
            } else {
                assertEquals(src[i] * scale, out[i], 1e-5f, "kept element must be scaled by 1/(1-p) at " + i);
            }
        }
        double dropRate = zeros / (double) N;
        assertTrue(Math.abs(dropRate - P) < 0.01, "drop rate " + dropRate + " should be ≈ " + P);
    }

    @Test
    void sameSeed_sameMask_inPlaceEqualsOutOfPlace() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        float[] src = randomData(2);
        float[] outOfPlace = new float[N];
        float[] inPlace = new float[N];
        long seed = -987654321L;
        try (GpuFloatBuffer a = GpuFloatBuffer.allocate(N); GpuFloatBuffer b = GpuFloatBuffer.allocate(N)) {
            a.copyFrom(src, 0, N);
            TensorOpsGPU.dropoutGpuDevice(a, b, N, P, seed);
            b.copyTo(outOfPlace, 0, N);
            TensorOpsGPU.dropoutGpuDevice(a, a, N, P, seed);
            a.copyTo(inPlace, 0, N);
        }
        assertArrayEquals(outOfPlace, inPlace, 0f);
    }

    @Test
    void differentSeeds_differentMasks() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        float[] src = randomData(3);
        float[] o1 = new float[N];
        float[] o2 = new float[N];
        try (GpuFloatBuffer a = GpuFloatBuffer.allocate(N); GpuFloatBuffer b = GpuFloatBuffer.allocate(N)) {
            a.copyFrom(src, 0, N);
            TensorOpsGPU.dropoutGpuDevice(a, b, N, P, 1L);
            b.copyTo(o1, 0, N);
            TensorOpsGPU.dropoutGpuDevice(a, b, N, P, 2L);
            b.copyTo(o2, 0, N);
        }
        int diff = 0;
        for (int i = 0; i < N; i++) {
            if ((o1[i] == 0f) != (o2[i] == 0f)) {
                diff++;
            }
        }
        assertFalse(diff == 0, "masks for different seeds must differ");
        // Independent Bernoulli(p) masks differ in ≈ 2p(1-p) of positions.
        double expected = 2 * P * (1 - P);
        assertTrue(Math.abs(diff / (double) N - expected) < 0.01, "mask difference rate " + diff / (double) N);
    }

    /** Backward = тот же kernel на градиенте: grad_in = mask ⊙ grad_out / (1-p) ⇒ J·v совпадает для forward и backward. */
    @Test
    void backwardWithSameSeed_isSameLinearMap() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        float[] x = randomData(4);
        float[] g = randomData(5);
        float[] y = new float[N];
        float[] gx = new float[N];
        long seed = 42L;
        try (GpuFloatBuffer a = GpuFloatBuffer.allocate(N); GpuFloatBuffer b = GpuFloatBuffer.allocate(N)) {
            a.copyFrom(x, 0, N);
            TensorOpsGPU.dropoutGpuDevice(a, b, N, P, seed);
            b.copyTo(y, 0, N);
            a.copyFrom(g, 0, N);
            TensorOpsGPU.dropoutGpuDevice(a, b, N, P, seed);
            b.copyTo(gx, 0, N);
        }
        // <y, g> == <x, gx>  (симметричная диагональная матрица)
        double lhs = 0, rhs = 0;
        for (int i = 0; i < N; i++) {
            lhs += (double) y[i] * g[i];
            rhs += (double) x[i] * gx[i];
            assertEquals(y[i] == 0f, gx[i] == 0f, "mask mismatch between forward and backward at " + i);
        }
        assertEquals(lhs, rhs, 1e-3 * Math.max(1.0, Math.abs(lhs)));
    }
}
