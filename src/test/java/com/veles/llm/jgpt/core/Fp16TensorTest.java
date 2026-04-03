package com.veles.llm.jgpt.core;

import com.veles.llm.jgpt.ops.Fp16Ops;
import com.veles.llm.jgpt.ops.TensorOps;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class Fp16TensorTest {

    @Test
    void roundTripNear() {
        Tensor t = new Tensor(new int[]{2, 3});
        float[] f = t.internalBuffer();
        for (int i = 0; i < f.length; i++) {
            f[i] = (i - 3) * 0.25f;
        }
        Fp16Tensor h = t.toFp16();
        Tensor back = h.toTensor();
        float[] g = back.internalBuffer();
        float maxErr = 0f;
        for (int i = 0; i < f.length; i++) {
            maxErr = Math.max(maxErr, Math.abs(f[i] - g[i]));
        }
        assertTrue(maxErr < 1e-3f, "max fp16 roundtrip err " + maxErr);
    }

    @Test
    void matmulMatchesFp32Pipeline() {
        Fp16Tensor a = Fp16Ops.randomTensor(new int[]{4, 8}, 0.1f);
        Fp16Tensor b = Fp16Ops.randomTensor(new int[]{8, 3}, 0.1f);
        Fp16Tensor c16 = Fp16Ops.matmul(a, b);
        Tensor cref = TensorOps.matmul(a.toTensor(), b.toTensor());
        float maxDiff = 0f;
        float[] p = c16.toTensor().internalBuffer();
        float[] q = cref.internalBuffer();
        for (int i = 0; i < p.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(p[i] - q[i]));
        }
        assertTrue(maxDiff < 1e-3f, "matmul fp16 path vs fp32 max diff " + maxDiff);
    }

    @Test
    void sizeBytes() {
        Fp16Tensor h = new Fp16Tensor(new int[]{10, 20});
        assertEquals(200L * 2L, h.sizeBytes());
    }

    @Test
    void fromTensorRejectsNaN() {
        Tensor t = Tensor.fromArray(new float[] {1f, Float.NaN}, new int[] {2});
        assertThrows(IllegalArgumentException.class, () -> Fp16Tensor.fromTensor(t));
    }

    @Test
    void fromFloat16ArrayCopiesBuffer() {
        short[] bits = new short[] {Float.floatToFloat16(1f), Float.floatToFloat16(2f)};
        Fp16Tensor q = Fp16Tensor.fromFloat16Array(bits, new int[] {2});
        bits[0] = 0;
        assertEquals(1f, q.get(0), 1e-6f);
    }

    @Test
    void toFloatArrayMatchesToTensor() {
        Tensor t = Tensor.fromArray(new float[] {1f, -2f, 0.5f}, new int[] {3});
        Fp16Tensor h = Fp16Tensor.fromTensor(t);
        float[] a = h.toFloatArray();
        float[] b = h.toTensor().internalBuffer();
        assertEquals(a.length, b.length);
        for (int i = 0; i < a.length; i++) {
            assertEquals(b[i], a[i], 1e-6f);
        }
    }

    @Test
    void shapeProductOverflowThrows() {
        int[] huge = {4096, 4096, 4096};
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> new Fp16Tensor(huge));
        assertTrue(ex.getMessage().contains("Integer.MAX_VALUE"), ex.getMessage());
        assertTrue(ex.getMessage().contains(Arrays.toString(huge)), ex.getMessage());
    }

    @Test
    void wrapSharesBufferWithCaller() {
        short[] bits = new short[] {Float.floatToFloat16(3f), Float.floatToFloat16(4f)};
        Fp16Tensor w = Fp16Tensor.wrap(bits, new int[] {2});
        assertEquals(3f, w.get(0), 1e-6f);
        bits[0] = Float.floatToFloat16(9f);
        assertEquals(9f, w.get(0), 1e-6f);
    }

    @Test
    void equalsAndHashCode_respectShapeAndData() {
        short[] bits = new short[] {Float.floatToFloat16(1f), Float.floatToFloat16(2f)};
        Fp16Tensor a = Fp16Tensor.fromFloat16Array(bits, new int[] {2});
        Fp16Tensor b = Fp16Tensor.fromFloat16Array(bits, new int[] {2});
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        Fp16Tensor c = Fp16Tensor.fromFloat16Array(new short[] {bits[0], 0}, new int[] {2});
        assertNotEquals(a, c);
    }

    @Test
    void fromTensorRejectsInfinity() {
        Tensor t = Tensor.fromArray(new float[] {1f, Float.POSITIVE_INFINITY}, new int[] {2});
        assertThrows(IllegalArgumentException.class, () -> Fp16Tensor.fromTensor(t));
    }
}
