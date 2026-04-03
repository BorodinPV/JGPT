package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.GpuIntBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for new device-pointer GPU operations (Phase 3c of full-GPU training plan).
 */
class GpuDeviceOpsTest {

    @Test
    void scaleInPlaceGpuDevice_matchesHostScale() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        int n = data.length;
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.copyFrom(data, 0, n);
            TensorOpsGPU.scaleInPlaceGpuDevice(buf, n, 2.5f);
            float[] result = new float[n];
            buf.copyTo(result, 0, n);
            float[] expected = {2.5f, 5.0f, 7.5f, 10.0f, 12.5f};
            assertArrayEquals(expected, result, 1e-5f);
        }
    }

    @Test
    void sumSquaresGpuDevice_matchesHostSum() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        float expectedSumSq = 1 + 4 + 9 + 16;
        int n = data.length;
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.copyFrom(data, 0, n);
            double result = TensorOpsGPU.sumSquaresGpuDevice(buf, n);
            assertEquals(expectedSumSq, result, 1e-4f, "sum of squares must match");
        }
    }

    @Test
    void sumSquaresGpuDeviceFused_matchesSumOfParts() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        float[] a = {1.0f, 2.0f, 0.5f};
        float[] b = {3.0f, 1.0f};
        try (GpuFloatBuffer bufA = GpuFloatBuffer.allocate(a.length);
                GpuFloatBuffer bufB = GpuFloatBuffer.allocate(b.length)) {
            bufA.copyFrom(a, 0, a.length);
            bufB.copyFrom(b, 0, b.length);
            double s1 = TensorOpsGPU.sumSquaresGpuDevice(bufA, a.length);
            double s2 = TensorOpsGPU.sumSquaresGpuDevice(bufB, b.length);
            long[] ptrs = {bufA.devicePointer(), bufB.devicePointer()};
            int[] lens = {a.length, b.length};
            double fused = TensorOpsGPU.sumSquaresGPUDeviceFused(ptrs, lens, 2);
            assertEquals(s1 + s2, fused, 1e-4f);
        }
    }

    @Test
    void anyNonFiniteGpuDevice_detectsNaN() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        float[] data = {1.0f, Float.NaN, 3.0f};
        int n = data.length;
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.copyFrom(data, 0, n);
            assertTrue(TensorOpsGPU.anyNonFiniteGpuDevice(buf, n), "must detect NaN");
        }
    }

    @Test
    void anyNonFiniteGpuDevice_finiteReturnsFalse() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        float[] data = {1.0f, 2.0f, -3.0f, 0.0f};
        int n = data.length;
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.copyFrom(data, 0, n);
            assertFalse(TensorOpsGPU.anyNonFiniteGpuDevice(buf, n), "all finite must return false");
        }
    }

    @Test
    void anyNonFiniteGpuDevice_detectsInf() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        float[] data = {1.0f, Float.POSITIVE_INFINITY, 3.0f};
        int n = data.length;
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.copyFrom(data, 0, n);
            assertTrue(TensorOpsGPU.anyNonFiniteGpuDevice(buf, n), "must detect Inf");
        }
    }

    @Test
    void crossEntropySoftmaxGradLossGpuDevice_matchesHostCe() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        int batch = 1, seqLen = 2, vocab = 4;
        float[] logits = {
            1.0f, 2.0f, 3.0f, 4.0f,
            2.0f, 1.0f, 0.0f, 3.0f
        };
        float[] targets = {2.0f, 3.0f};
        float gradScale = 1.0f / (batch * seqLen);

        float[] hostGrad = new float[batch * seqLen * vocab];
        float hostLoss = TensorOpsGPU.crossEntropySoftmaxGradLossGpuEx(
                logits.clone(), targets, hostGrad, batch, seqLen, vocab, gradScale, false);

        int total = batch * seqLen * vocab;
        try (GpuFloatBuffer dLogits = GpuFloatBuffer.allocate(total);
             GpuFloatBuffer dGrad = GpuFloatBuffer.allocate(total)) {
            dLogits.copyFrom(logits, 0, total);
            float deviceLoss = TensorOpsGPU.crossEntropySoftmaxGradLossGpuDevice(
                    dLogits, targets, dGrad, batch, seqLen, vocab, gradScale, false);
            assertEquals(hostLoss, deviceLoss, 1e-4f, "CE loss must match");

            float[] deviceGradResult = new float[total];
            dGrad.copyTo(deviceGradResult, 0, total);
            assertArrayEquals(hostGrad, deviceGradResult, 1e-5f, "CE grad must match");
        }
    }

    @Test
    void crossEntropySoftmaxGradLossGpuDevice_intTargetsOnDevice_matchesFloatTargetsPath() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        int batch = 1, seqLen = 2, vocab = 4;
        float[] logits = {
            1.0f, 2.0f, 3.0f, 4.0f,
            2.0f, 1.0f, 0.0f, 3.0f
        };
        float[] targetsF = {2.0f, 3.0f};
        int[] targetsI = {2, 3};
        float gradScale = 1.0f / (batch * seqLen);

        int total = batch * seqLen * vocab;
        try (GpuFloatBuffer dLogits = GpuFloatBuffer.allocate(total);
             GpuFloatBuffer dGradA = GpuFloatBuffer.allocate(total);
             GpuFloatBuffer dGradB = GpuFloatBuffer.allocate(total);
             GpuIntBuffer dTgt = GpuIntBuffer.allocate(batch * seqLen)) {
            dLogits.copyFrom(logits, 0, total);
            dGradA.clear();
            float lossA =
                    TensorOpsGPU.crossEntropySoftmaxGradLossGpuDevice(
                            dLogits,
                            targetsF,
                            dGradA,
                            batch,
                            seqLen,
                            vocab,
                            gradScale,
                            false);
            dLogits.copyFrom(logits, 0, total);
            dTgt.copyFrom(targetsI, 0, batch * seqLen);
            dGradB.clear();
            float lossB =
                    TensorOpsGPU.crossEntropySoftmaxGradLossGpuDeviceTargetsDevice(
                            dLogits,
                            dTgt,
                            dGradB,
                            batch,
                            seqLen,
                            vocab,
                            gradScale,
                            false);
            assertEquals(lossA, lossB, 1e-5f);
            float[] ga = new float[total];
            float[] gb = new float[total];
            dGradA.copyTo(ga, 0, total);
            dGradB.copyTo(gb, 0, total);
            assertArrayEquals(ga, gb, 1e-5f);
        }
    }

    @Test
    void gpuIntBuffer_copyRoundTrip_andDirectBuffer() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        int n = 16;
        int[] src = new int[n];
        for (int i = 0; i < n; i++) {
            src[i] = i * 7;
        }
        try (GpuIntBuffer buf = GpuIntBuffer.allocate(n)) {
            buf.copyFrom(src, 0, n);
            int[] out = new int[n];
            buf.copyTo(out, 0, n);
            assertArrayEquals(src, out);
            int subLen = n - 2;
            buf.copyFrom(src, 2, subLen);
            buf.copyTo(out, 0, subLen);
            int[] expectedMid = new int[subLen];
            System.arraycopy(src, 2, expectedMid, 0, subLen);
            assertArrayEquals(expectedMid, Arrays.copyOf(out, subLen));
        }
        try (GpuIntBuffer buf = GpuIntBuffer.allocate(n)) {
            ByteBuffer bb = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder());
            for (int i = 0; i < n; i++) {
                bb.putInt(i * 4, src[i]);
            }
            buf.copyFromDirect(bb, 0, n * 4L);
            int[] out = new int[n];
            buf.copyTo(out, 0, n);
            assertArrayEquals(src, out);
            ByteBuffer bb2 = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder());
            buf.copyToDirect(bb2, 0, n * 4L);
            for (int i = 0; i < n; i++) {
                assertEquals(src[i], bb2.getInt(i * 4));
            }
        }
    }

    @Test
    void accumulateAddGpuFromHost_addsCorrectly() {
        if (!TensorOpsGPU.isGpuAvailable()) return;
        float[] deviceInit = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] hostDelta = {10.0f, 20.0f, 30.0f, 40.0f};
        int n = deviceInit.length;
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.copyFrom(deviceInit, 0, n);
            TensorOpsGPU.accumulateAddGpuFromHost(buf, hostDelta, 0, n);
            float[] result = new float[n];
            buf.copyTo(result, 0, n);
            float[] expected = {11.0f, 22.0f, 33.0f, 44.0f};
            assertArrayEquals(expected, result, 1e-5f);
        }
    }
}
