package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;

import java.util.Random;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * FlashAttention native vs causal softmax reference. При {@code JGPT_FP16_MATMUL=1} — WMMA QK^T + FP16 I/O.
 */
class FlashAttnFp16ParityTest {

    private static final int DH = TensorOpsGPU.FLASH_ATTENTION_D_HEAD;

    @Test
    void forwardMatchesCausalSoftmax() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        compareForward(2, 64, 0x61);
        compareForward(1, 128, 0x62);
        compareForwardHeads(2, 4, 64, 0x63);
    }

    @Test
    void backwardMatchesReference() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        compareBackward(2, 64, 0x71);
        compareBackwardHeads(2, 4, 64, 0x72);
    }

    private static void compareForward(int bh, int s, int seed) {
        compareForwardHeads(bh, 1, s, seed);
    }

    private static void compareForwardHeads(int batch, int heads, int s, int seed) {
        int bh = batch * heads;
        int n = bh * s * DH;
        float[] q = randn(n, seed);
        float[] k = randn(n, seed + 1);
        float[] v = randn(n, seed + 2);
        float scale = 1f / (float) Math.sqrt(DH);
        float[] expO = new float[n];
        float[] expLse = new float[bh * s];
        cpuForward(q, k, v, expO, expLse, bh, s, scale);

        try (GpuFloatBuffer dQ = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer dK = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer dV = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer dO = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer dLse = GpuFloatBuffer.allocate(bh * s)) {
            dQ.copyFrom(q, 0, n);
            dK.copyFrom(k, 0, n);
            dV.copyFrom(v, 0, n);
            TensorOpsGPU.flashAttentionForwardGpuDeviceResident(dQ, dK, dV, dO, dLse, bh, s, DH, scale, heads);
            TensorOpsGPU.synchronizeStream();
            float[] gotO = new float[n];
            float[] gotLse = new float[bh * s];
            dO.copyTo(gotO, 0, n);
            dLse.copyTo(gotLse, 0, bh * s);
            float tol = TensorOpsGPU.cudnnSdpaAvailable() || TensorOpsGPU.useFp16Matmul() ? 4e-2f : 2e-3f;
            assertClose(expO, gotO, tol, "O");
            assertClose(expLse, gotLse, tol, "LSE");
        }
    }

    private static void compareBackward(int bh, int s, int seed) {
        compareBackwardHeads(bh, 1, s, seed);
    }

    private static void compareBackwardHeads(int batch, int heads, int s, int seed) {
        int bh = batch * heads;
        int n = bh * s * DH;
        float[] q = randn(n, seed);
        float[] k = randn(n, seed + 1);
        float[] v = randn(n, seed + 2);
        float[] dO = randn(n, seed + 3);
        float scale = 1f / (float) Math.sqrt(DH);
        float[] o = new float[n];
        float[] lse = new float[bh * s];
        cpuForward(q, k, v, o, lse, bh, s, scale);
        float[] expDq = new float[n];
        float[] expDk = new float[n];
        float[] expDv = new float[n];
        cpuBackward(q, k, v, o, dO, lse, expDq, expDk, expDv, bh, s, scale);

        try (GpuFloatBuffer gQ = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer gK = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer gV = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer gO = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer gDo = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer gLse = GpuFloatBuffer.allocate(bh * s);
                GpuFloatBuffer gDq = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer gDk = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer gDv = GpuFloatBuffer.allocate(n)) {
            gQ.copyFrom(q, 0, n);
            gK.copyFrom(k, 0, n);
            gV.copyFrom(v, 0, n);
            gO.copyFrom(o, 0, n);
            gDo.copyFrom(dO, 0, n);
            gLse.copyFrom(lse, 0, bh * s);
            TensorOpsGPU.flashAttentionBackwardGpuDeviceResident(
                    gQ, gK, gV, gO, gDo, gLse, gDq, gDk, gDv, bh, s, DH, scale, heads);
            TensorOpsGPU.synchronizeStream();
            float[] gotDq = new float[n];
            float[] gotDk = new float[n];
            float[] gotDv = new float[n];
            gDq.copyTo(gotDq, 0, n);
            gDk.copyTo(gotDk, 0, n);
            gDv.copyTo(gotDv, 0, n);
            float tol = TensorOpsGPU.cudnnSdpaAvailable() || TensorOpsGPU.useFp16Matmul() ? 8e-2f : 3e-3f;
            assertClose(expDq, gotDq, tol, "dQ");
            assertClose(expDk, gotDk, tol, "dK");
            assertClose(expDv, gotDv, tol, "dV");
        }
    }

    private static void cpuForward(
            float[] q, float[] k, float[] v, float[] o, float[] lse, int bh, int s, float scale) {
        for (int b = 0; b < bh; b++) {
            for (int i = 0; i < s; i++) {
                float max = Float.NEGATIVE_INFINITY;
                float[] scores = new float[s];
                for (int j = 0; j <= i; j++) {
                    float dot = 0f;
                    for (int d = 0; d < DH; d++) {
                        dot += q[(b * s + i) * DH + d] * k[(b * s + j) * DH + d];
                    }
                    scores[j] = dot * scale;
                    max = Math.max(max, scores[j]);
                }
                float sum = 0f;
                for (int j = 0; j <= i; j++) {
                    scores[j] = (float) Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                float inv = 1f / sum;
                lse[b * s + i] = max + (float) Math.log(sum);
                for (int d = 0; d < DH; d++) {
                    float acc = 0f;
                    for (int j = 0; j <= i; j++) {
                        acc += scores[j] * inv * v[(b * s + j) * DH + d];
                    }
                    o[(b * s + i) * DH + d] = acc;
                }
            }
        }
    }

    private static void cpuBackward(
            float[] q,
            float[] k,
            float[] v,
            float[] o,
            float[] dO,
            float[] lse,
            float[] dQ,
            float[] dK,
            float[] dV,
            int bh,
            int s,
            float scale) {
        for (int b = 0; b < bh; b++) {
            for (int i = 0; i < s; i++) {
                float di = 0f;
                for (int d = 0; d < DH; d++) {
                    di += dO[(b * s + i) * DH + d] * o[(b * s + i) * DH + d];
                }
                for (int j = 0; j <= i; j++) {
                    float dot = 0f;
                    for (int t = 0; t < DH; t++) {
                        dot += q[(b * s + i) * DH + t] * k[(b * s + j) * DH + t];
                    }
                    float p = (float) Math.exp(dot * scale - lse[b * s + i]);
                    float dp = 0f;
                    for (int t = 0; t < DH; t++) {
                        dp += dO[(b * s + i) * DH + t] * v[(b * s + j) * DH + t];
                    }
                    float ds = p * (dp - di) * scale;
                    for (int t = 0; t < DH; t++) {
                        dQ[(b * s + i) * DH + t] += ds * k[(b * s + j) * DH + t];
                        dK[(b * s + j) * DH + t] += ds * q[(b * s + i) * DH + t];
                        dV[(b * s + j) * DH + t] += p * dO[(b * s + i) * DH + t];
                    }
                }
            }
        }
    }

    private static float[] randn(int n, int seed) {
        Random r = new Random(seed);
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = r.nextFloat() * 2f - 1f;
        }
        return a;
    }

    private static void assertClose(float[] expected, float[] actual, float eps, String name) {
        float maxAbs = 0f;
        for (int i = 0; i < expected.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(expected[i] - actual[i]));
            assertEquals(expected[i], actual[i], eps, name + "[" + i + "]");
        }
        assertTrue(maxAbs < eps * 8f || maxAbs < 0.2f, name + " maxAbs=" + maxAbs);
    }
}
