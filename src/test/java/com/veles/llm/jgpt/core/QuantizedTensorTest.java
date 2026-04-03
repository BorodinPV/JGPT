package com.veles.llm.jgpt.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QuantizedTensorTest {

    @Test
    public void roundTripSmall() {
        Tensor t = Tensor.fromArray(new float[]{-1f, 0f, 0.5f, 1f}, new int[]{2, 2});
        QuantizedTensor q = QuantizedTensor.fromTensor(t);
        Tensor back = q.toTensor();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(t.get(i, j), back.get(i, j), 0.02f);
            }
        }
    }

    @Test
    public void sameElementCountByteStorage() {
        Tensor t = new Tensor(new int[]{100, 100});
        QuantizedTensor q = QuantizedTensor.fromTensor(t);
        assertEquals(t.internalBuffer().length, q.internalBuffer().length);
    }

    @Test
    public void fromTensorRejectsNaN() {
        Tensor t = Tensor.fromArray(new float[] {1f, Float.NaN}, new int[] {2});
        assertThrows(IllegalArgumentException.class, () -> QuantizedTensor.fromTensor(t));
    }

    @Test
    public void tinyMaxAbsUsesThatScale() {
        float v = 1e-13f;
        Tensor t = Tensor.fromArray(new float[] {v, -v * 0.5f}, new int[] {2});
        QuantizedTensor q = QuantizedTensor.fromTensor(t);
        assertEquals(v, q.getScale(), v * 1e-5f);
        Tensor back = q.toTensor();
        assertEquals(v, back.get(0), v * 0.02f);
        assertEquals(-v * 0.5f, back.get(1), v * 0.02f);
    }

    @Test
    public void allZerosScaleOne() {
        Tensor t = new Tensor(new int[] {3});
        QuantizedTensor q = QuantizedTensor.fromTensor(t);
        assertEquals(1f, q.getScale());
        assertEquals(0f, q.get(0), 0f);
        assertEquals(0f, q.get(1), 0f);
        assertEquals(0f, q.get(2), 0f);
    }

    @Test
    public void setRejectsNonFinite() {
        QuantizedTensor q = QuantizedTensor.fromTensor(Tensor.fromArray(new float[] {1f}, new int[] {1}));
        assertThrows(IllegalArgumentException.class, () -> q.set(Float.NaN, 0));
        assertThrows(IllegalArgumentException.class, () -> q.set(Float.POSITIVE_INFINITY, 0));
    }

    @Test
    void fromBytesShapeProductOverflowThrows() {
        int[] huge = {4096, 4096, 4096};
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> QuantizedTensor.fromBytes(new byte[1], huge, 1f));
        assertTrue(ex.getMessage().contains("Integer.MAX_VALUE"), ex.getMessage());
    }

    @Test
    void symmetricClampTo127() {
        QuantizedTensor q = new QuantizedTensor(new int[] {2});
        float s = 1f;
        q = QuantizedTensor.fromBytes(new byte[2], new int[] {2}, s);
        q.set(1.5f * s, 0);
        q.set(-1.5f * s, 1);
        assertEquals(127, q.internalBuffer()[0]);
        assertEquals(-127, q.internalBuffer()[1]);
    }

    @Test
    void equalsAndHashCode() {
        byte[] b = new byte[] {0, 127, -127};
        QuantizedTensor a = QuantizedTensor.fromBytes(b, new int[] {3}, 0.5f);
        QuantizedTensor c = QuantizedTensor.fromBytes(b, new int[] {3}, 0.5f);
        assertEquals(a, c);
        assertEquals(a.hashCode(), c.hashCode());
        QuantizedTensor d = QuantizedTensor.fromBytes(b, new int[] {3}, 1f);
        assertNotEquals(a, d);
    }

    @Test
    void fromBytesRequiresNonNullBuffer() {
        assertThrows(NullPointerException.class, () -> QuantizedTensor.fromBytes(null, new int[] {1}, 1f));
    }

    @Test
    void fromTensorRequiresNonNull() {
        assertThrows(NullPointerException.class, () -> QuantizedTensor.fromTensor(null));
    }

    @Test
    void numElementsAndSizeBytes() {
        QuantizedTensor q = new QuantizedTensor(new int[] {4, 5});
        assertEquals(20, q.numElements());
        assertEquals(20L, q.sizeBytes());
    }
}
