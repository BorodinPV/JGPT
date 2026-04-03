package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import org.junit.jupiter.api.Test;

/**
 * Контрактные тесты для H2D/D2H, {@link GpuFloatBuffer#clear()} и согласованности потоков CUDA
 * ({@code kTensorCudaStream} vs default stream). Дополняют {@link GpuFloatBufferTest} (GEMM).
 */
class GpuFloatBufferCopyContractTest {

    private static void assertAllZero(float[] a) {
        for (int i = 0; i < a.length; i++) {
            assertEquals(0f, a[i], 0f, "index " + i);
        }
    }

    @Test
    void clearThenDownload_largeBuffer_allZeros() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 65_536;
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.clear();
            float[] host = new float[n];
            buf.copyTo(host, 0, n);
            assertAllZero(host);
        }
    }

    @Test
    void clearDownload_repeated_sameBuffer_staysZero() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 256;
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            float[] host = new float[n];
            for (int iter = 0; iter < 48; iter++) {
                buf.clear();
                buf.copyTo(host, 0, n);
                assertAllZero(host);
            }
        }
    }

    @Test
    void uploadThenClearThenDownload_allZeros() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 512;
        float[] pattern = new float[n];
        for (int i = 0; i < n; i++) {
            pattern[i] = (float) Math.sin(i) * 3.7f;
        }
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.copyFrom(pattern, 0, n);
            float[] tmp = new float[n];
            buf.copyTo(tmp, 0, n);
            assertArrayEquals(pattern, tmp, 1e-5f);

            buf.clear();
            buf.copyTo(tmp, 0, n);
            assertAllZero(tmp);
        }
    }

    @Test
    void floatArray_roundTrip_withOffset() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int total = 128;
        int off = 17;
        int len = 41;
        float[] src = new float[total];
        for (int i = 0; i < len; i++) {
            src[off + i] = i * 0.25f - 5f;
        }
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(len)) {
            buf.copyFrom(src, off, len);
            float[] dst = new float[len];
            buf.copyTo(dst, 0, len);
            for (int i = 0; i < len; i++) {
                assertEquals(src[off + i], dst[i], 1e-5f);
            }
        }
    }

    @Test
    void floatBuffer_roundTrip_copyFromHost_copyToHost() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 12;
        int srcOff = 3;
        int copyLen = 5;
        ByteBuffer bb = ByteBuffer.allocateDirect(64 * Float.BYTES).order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        for (int i = 0; i < fb.capacity(); i++) {
            fb.put(i, i * 0.5f - 2f);
        }
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(copyLen)) {
            buf.copyFromHost(fb, srcOff, copyLen);
            FloatBuffer out = ByteBuffer.allocateDirect(32 * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
            buf.copyToHost(out, 7, copyLen);
            for (int i = 0; i < copyLen; i++) {
                assertEquals(fb.get(srcOff + i), out.get(7 + i), 1e-5f);
            }
            buf.copyToHost(out.clear());
            for (int i = 0; i < copyLen; i++) {
                assertEquals(fb.get(srcOff + i), out.get(i), 1e-5f);
            }
        }
    }

    @Test
    void directByteBuffer_roundTrip() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 8;
        ByteBuffer bb = ByteBuffer.allocateDirect(n * Float.BYTES).order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        for (int i = 0; i < n; i++) {
            fb.put(i, (float) (i * i) - 0.125f);
        }
        try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            buf.copyFromDirect(bb, 0, (long) n * Float.BYTES);
            ByteBuffer out = ByteBuffer.allocateDirect(n * Float.BYTES).order(ByteOrder.nativeOrder());
            buf.copyToDirect(out, 0, (long) n * Float.BYTES);
            FloatBuffer fo = out.asFloatBuffer();
            for (int i = 0; i < n; i++) {
                assertEquals(fb.get(i), fo.get(i), 1e-5f);
            }
        }
    }

    @Test
    void memorySegment_matchesFloatArrayPayload() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 32;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = 1f / (1 + i);
        }
        try (Arena arena = Arena.ofConfined();
                GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
            MemorySegment seg = arena.allocate((long) n * Float.BYTES, Float.BYTES);
            for (int i = 0; i < n; i++) {
                seg.setAtIndex(ValueLayout.JAVA_FLOAT, i, data[i]);
            }
            buf.copyFromMemorySegment(seg, 0, (long) n * Float.BYTES);
            float[] fromGpuAfterSeg = new float[n];
            buf.copyTo(fromGpuAfterSeg, 0, n);
            assertArrayEquals(data, fromGpuAfterSeg, 1e-6f);

            buf.clear();
            buf.copyFrom(data, 0, n);
            MemorySegment dst = arena.allocate((long) n * Float.BYTES, Float.BYTES);
            buf.copyToMemorySegment(dst, 0, (long) n * Float.BYTES);
            float[] fromGpuAfterFloatArray = new float[n];
            for (int i = 0; i < n; i++) {
                fromGpuAfterFloatArray[i] = dst.getAtIndex(ValueLayout.JAVA_FLOAT, i);
            }
            assertArrayEquals(data, fromGpuAfterFloatArray, 1e-6f);
        }
    }

    @Test
    void copyFromDevice_copiesBetweenGpuBuffers() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 64;
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = i * 0.1f;
        }
        try (GpuFloatBuffer src = GpuFloatBuffer.allocate(n);
                GpuFloatBuffer dst = GpuFloatBuffer.allocate(n)) {
            src.copyFrom(a, 0, n);
            dst.clear();
            dst.copyFromDevice(src, n);
            float[] out = new float[n];
            dst.copyTo(out, 0, n);
            assertArrayEquals(a, out, 1e-5f);
        }
    }

    @Test
    void pinnedStaging_growsAcrossSuccessiveSizes() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int[] sizes = {4, 1024, 16, 8192, 32};
        float[] host = new float[8192];
        for (int n : sizes) {
            try (GpuFloatBuffer buf = GpuFloatBuffer.allocate(n)) {
                for (int i = 0; i < n; i++) {
                    host[i] = (float) (n + i);
                }
                buf.copyFrom(host, 0, n);
                float[] out = new float[n];
                buf.copyTo(out, 0, n);
                for (int i = 0; i < n; i++) {
                    assertEquals(host[i], out[i], 1e-4f, "n=" + n + " i=" + i);
                }
            }
        }
    }
}
