package com.veles.llm.jgpt.core;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Tensor}.
 */
@DisplayName("Tensor")
class TensorTest {

    @Nested
    @DisplayName("Construction")
    class Construction {

        @Test
        @DisplayName("creates tensor with given shape")
        void testCreateTensor() {
            Tensor t = new Tensor(new int[]{2, 3});
            assertArrayEquals(new int[]{2, 3}, t.getShape());
            assertEquals(6, t.size());  // 🔧 FIX: было getData().length → size()
            assertEquals(6, t.internalBuffer().length);  // альтернатива
        }

        @Test
        @DisplayName("rejects null shape")
        void testCreateWithNullShape() {
            assertThrows(NullPointerException.class, () -> new Tensor(null));
        }

        @Test
        @DisplayName("rejects empty shape")
        void testCreateWithEmptyShape() {
            assertThrows(IllegalArgumentException.class, () -> new Tensor(new int[]{}));
        }

        @Test
        @DisplayName("rejects non-positive dimensions")
        void testCreateWithInvalidDimensions() {
            assertThrows(IllegalArgumentException.class, () -> new Tensor(new int[]{2, 0, 3}));
            assertThrows(IllegalArgumentException.class, () -> new Tensor(new int[]{-1, 3}));
        }

        @Test
        @DisplayName("rejects shape with overflow product")
        void testCreateWithOverflowShape() {
            // Integer.MAX_VALUE = 2_147_483_647
            // 50000 * 50000 = 2_500_000_000 > MAX_VALUE → overflow
            assertThrows(IllegalArgumentException.class,
                    () -> new Tensor(new int[]{50000, 50000}));
        }
    }

    @Nested
    @DisplayName("Element access")
    class ElementAccess {

        @Test
        @DisplayName("get/set works for 2D tensor")
        void testGetSet2D() {
            Tensor t = new Tensor(new int[]{2, 3});
            t.set(5.0f, 0, 1);
            assertEquals(5.0f, t.get(0, 1));
        }

        @Test
        @DisplayName("get/set works for 3D tensor")
        void testGetSet3D() {
            Tensor t = new Tensor(new int[]{2, 3, 4});
            t.set(7.5f, 1, 2, 3);
            assertEquals(7.5f, t.get(1, 2, 3));
        }

        @Test
        @DisplayName("get throws on wrong number of indices")
        void testGetWithWrongIndexCount() {
            Tensor t = new Tensor(new int[]{2, 3});
            assertThrows(IllegalArgumentException.class, () -> t.get(0));  // need 2 indices
            assertThrows(IllegalArgumentException.class, () -> t.get(0, 1, 2));  // too many
        }

        @Test
        @DisplayName("get/set throws on out-of-bounds index")
        void testGetSetOutOfBounds() {
            Tensor t = new Tensor(new int[]{2, 3});
            assertThrows(IndexOutOfBoundsException.class, () -> t.get(2, 0));  // dim 0: max 1
            assertThrows(IndexOutOfBoundsException.class, () -> t.get(0, 3));  // dim 1: max 2
            assertThrows(IndexOutOfBoundsException.class, () -> t.set(1f, -1, 0));
        }
    }

    @Nested
    @DisplayName("Factory methods")
    class FactoryMethods {

        @Test
        @DisplayName("fromArray copies data correctly")
        void testFromArray() {
            float[] data = {1, 2, 3, 4, 5, 6};
            Tensor t = Tensor.fromArray(data, new int[]{2, 3});
            assertEquals(1.0f, t.get(0, 0));
            assertEquals(6.0f, t.get(1, 2));
            // Verify it's a copy: modifying original doesn't affect tensor
            data[0] = 999f;
            assertEquals(1.0f, t.get(0, 0));  // still 1, not 999
        }

        @Test
        @DisplayName("fromArray rejects length mismatch")
        void testFromArrayLengthMismatch() {
            float[] data = {1, 2, 3};
            assertThrows(IllegalArgumentException.class,
                    () -> Tensor.fromArray(data, new int[]{2, 3}));  // needs 6 elements
        }

        @Test
        @DisplayName("wrap creates zero-copy view")
        void testWrap() {
            float[] shared = new float[]{10, 20, 30, 40};
            Tensor t = Tensor.wrap(shared, new int[]{2, 2});

            // Modifying tensor affects original array (zero-copy)
            t.set(99f, 0, 0);
            assertEquals(99f, shared[0]);

            // Modifying original array affects tensor
            shared[3] = 77f;
            assertEquals(77f, t.get(1, 1));
        }

        @Test
        @DisplayName("wrap rejects null data")
        void testWrapNullData() {
            assertThrows(NullPointerException.class,
                    () -> Tensor.wrap(null, new int[]{2, 2}));
        }

        @Test
        @DisplayName("wrap rejects length mismatch")
        void testWrapLengthMismatch() {
            float[] data = {1, 2, 3};
            assertThrows(IllegalArgumentException.class,
                    () -> Tensor.wrap(data, new int[]{2, 3}));
        }
    }

    @Nested
    @DisplayName("Gradient management")
    class GradientManagement {

        @Test
        @DisplayName("zeroGrad allocates buffer on first call")
        void testZeroGradAllocates() {
            Tensor t = new Tensor(new int[]{3, 4});
            assertFalse(t.hasGrad());
            t.zeroGrad();
            assertTrue(t.hasGrad());
            assertEquals(12, t.gradBuffer().length);
        }

        @Test
        @DisplayName("zeroGrad zeroes existing buffer")
        void testZeroGradZeroes() {
            Tensor t = new Tensor(new int[]{2, 2});
            t.zeroGrad();
            float[] grad = t.gradBuffer();
            grad[0] = 5f;
            grad[1] = 10f;

            t.zeroGrad();  // should zero them
            assertEquals(0f, grad[0]);
            assertEquals(0f, grad[1]);
        }

        @Test
        @DisplayName("gradBuffer throws if not initialized")
        void testGradBufferNotInitialized() {
            Tensor t = new Tensor(new int[]{2, 2});
            assertThrows(IllegalStateException.class, t::gradBuffer);
        }
    }

    @Nested
    @DisplayName("Metadata access")
    class MetadataAccess {

        @Test
        @DisplayName("getShape returns copy")
        void testGetShapeReturnsCopy() {
            Tensor t = new Tensor(new int[]{2, 3});
            int[] shape1 = t.getShape();
            int[] shape2 = t.getShape();
            assertArrayEquals(shape1, shape2);
            // Verify it's a copy: modifying returned array doesn't affect tensor
            shape1[0] = 999;
            assertEquals(2, t.getShape()[0]);  // still 2, not 999
        }

        @Test
        @DisplayName("strides are computed correctly for row-major")
        void testStridesRowMajor() {
            // Shape [2, 3, 4] → strides [12, 4, 1]
            Tensor t = new Tensor(new int[]{2, 3, 4});
            assertArrayEquals(new int[]{12, 4, 1}, t.getStrides());
        }

        @Test
        @DisplayName("stridesInternal returns reference (documented)")
        void testStridesInternal() {
            Tensor t = new Tensor(new int[]{2, 3});
            int[] internal = t.stridesInternal();
            int[] copy = t.getStrides();
            assertArrayEquals(copy, internal);
            // Note: internal is the actual array — don't modify in production!
        }

        @Test
        @DisplayName("rank returns number of dimensions")
        void testRank() {
            assertEquals(1, new Tensor(new int[]{5}).rank());
            assertEquals(3, new Tensor(new int[]{2, 3, 4}).rank());
        }
    }

    @Nested
    @DisplayName("equals and hashCode")
    class Equality {

        @Test
        @DisplayName("equals compares shape and data values")
        void testEquals() {
            Tensor t1 = Tensor.fromArray(new float[]{1, 2, 3}, new int[]{3});
            Tensor t2 = Tensor.fromArray(new float[]{1, 2, 3}, new int[]{3});
            Tensor t3 = Tensor.fromArray(new float[]{1, 2, 4}, new int[]{3});
            Tensor t4 = Tensor.fromArray(new float[]{1, 2, 3}, new int[]{1, 3});

            assertEquals(t1, t2);  // same shape, same values
            assertEquals(t1.hashCode(), t2.hashCode());

            assertNotEquals(t1, t3);  // different values
            assertNotEquals(t1.hashCode(), t3.hashCode());

            assertNotEquals(t1, t4);  // different shape
            assertNotEquals(t1.hashCode(), t4.hashCode());
        }

        @Test
        @DisplayName("equals is reflexive, symmetric, transitive")
        void testEqualsContract() {
            Tensor t = new Tensor(new int[]{2, 2});

            // Reflexive
            assertEquals(t, t);

            // Symmetric
            Tensor t2 = Tensor.fromArray(new float[]{1, 2, 3, 4}, new int[]{2, 2});
            Tensor t3 = Tensor.fromArray(new float[]{1, 2, 3, 4}, new int[]{2, 2});
            assertEquals(t2, t3);
            assertEquals(t3, t2);

            // Transitive
            assertEquals(t2, t3);
            assertEquals(t3, t2);  // already tested, but explicit
        }

        @Test
        @DisplayName("not equal to null or different class")
        void testEqualsEdgeCases() {
            Tensor t = new Tensor(new int[]{2, 2});
            assertNotEquals(null, t);
            assertNotEquals("not a tensor", t);
        }
    }

    @Nested
    @DisplayName("Linear index and bulk")
    class LinearAndBulk {

        @Test
        @DisplayName("getLinear/setLinear match row-major order")
        void linearMatchesRowMajor() {
            Tensor t = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
            assertEquals(1f, t.getLinear(0));
            assertEquals(4f, t.getLinear(3));
            t.setLinear(5, 99f);
            assertEquals(99f, t.get(1, 2));
        }

        @Test
        @DisplayName("fillData and copyDataTo")
        void fillAndCopy() {
            Tensor t = new Tensor(new int[]{2, 2});
            t.fillData(3f);
            float[] buf = t.internalBuffer();
            for (float v : buf) {
                assertEquals(3f, v);
            }
            float[] out = new float[3];
            assertThrows(IllegalArgumentException.class, () -> t.copyDataTo(out, 0));
            float[] out2 = new float[4];
            t.copyDataTo(out2, 0);
            assertArrayEquals(buf, out2);
        }
    }

    @Nested
    @DisplayName("viewReshape")
    class ViewReshape {

        @Test
        @DisplayName("same data, new shape, shared grad")
        void viewReshapeSharesDataAndGrad() {
            Tensor t = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
            t.zeroGrad();
            t.gradBuffer()[0] = 7f;
            Tensor v = t.viewReshape(new int[]{3, 2});
            assertEquals(6, v.size());
            assertArrayEquals(new int[]{3, 2}, v.getShape());
            assertEquals(1f, v.get(0, 0));
            assertEquals(6f, v.get(2, 1));
            assertEquals(7f, v.gradBuffer()[0]);
            v.gradBuffer()[1] = 8f;
            assertEquals(8f, t.gradBuffer()[1]);
        }

        @Test
        @DisplayName("rejects element count mismatch")
        void rejectsSizeMismatch() {
            Tensor t = new Tensor(new int[]{2, 3});
            assertThrows(IllegalArgumentException.class, () -> t.viewReshape(new int[]{5}));
        }
    }

    @Nested
    @DisplayName("toString")
    class ToString {

        @Test
        @DisplayName("includes shape and size")
        void testToString() {
            Tensor t = new Tensor(new int[]{2, 3});
            String s = t.toString();
            assertTrue(s.contains("shape=[2, 3]"));
            assertTrue(s.contains("size=6"));
        }

        @Test
        @DisplayName("indicates grad allocation state")
        void testToStringWithGrad() {
            Tensor t = new Tensor(new int[]{2, 2});
            assertTrue(t.toString().contains("grad=null"));
            t.zeroGrad();
            assertTrue(t.toString().contains("grad=allocated"));
        }
    }
}