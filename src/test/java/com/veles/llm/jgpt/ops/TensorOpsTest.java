package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.core.QuantizedTensor;
import com.veles.llm.jgpt.core.Tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TensorOpsTest {

    // ========== Element-wise operations ==========

    @Test
    public void testAdd() {
        Tensor a = Tensor.fromArray(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        Tensor b = Tensor.fromArray(new float[]{5, 6, 7, 8}, new int[]{2, 2});
        Tensor result = TensorOps.add(a, b);

        assertEquals(6.0f, result.get(0, 0));
        assertEquals(8.0f, result.get(0, 1));
        assertEquals(10.0f, result.get(1, 0));
        assertEquals(12.0f, result.get(1, 1));
    }

    @Test
    public void testAddNullInputs() {
        Tensor a = new Tensor(new int[]{2, 2});
        assertThrows(NullPointerException.class, () -> TensorOps.add(null, a));
        assertThrows(NullPointerException.class, () -> TensorOps.add(a, null));
    }

    @Test
    public void testAddShapeMismatch() {
        Tensor a = new Tensor(new int[]{2, 2});
        Tensor b = new Tensor(new int[]{2, 3});
        assertThrows(IllegalArgumentException.class, () -> TensorOps.add(a, b));
    }

    @Test
    public void testSubtract() {
        Tensor a = Tensor.fromArray(new float[]{10, 20}, new int[]{2});
        Tensor b = Tensor.fromArray(new float[]{3, 4}, new int[]{2});
        Tensor result = TensorOps.subtract(a, b);
        assertEquals(7.0f, result.get(0));
        assertEquals(16.0f, result.get(1));
    }

    @Test
    public void testMultiplyScalar() {
        Tensor a = Tensor.fromArray(new float[]{1, 2, 3}, new int[]{3});
        Tensor result = TensorOps.multiplyScalar(a, 2.0f);
        assertEquals(2.0f, result.get(0));
        assertEquals(4.0f, result.get(1));
        assertEquals(6.0f, result.get(2));
    }

    // ========== Activations ==========

    @Test
    public void testRelu() {
        Tensor a = Tensor.fromArray(new float[]{-2, -1, 0, 1, 2}, new int[]{5});
        Tensor result = TensorOps.relu(a);
        assertEquals(0.0f, result.get(0));
        assertEquals(0.0f, result.get(1));
        assertEquals(0.0f, result.get(2));
        assertEquals(1.0f, result.get(3));
        assertEquals(2.0f, result.get(4));
    }

    @Test
    public void testSigmoid() {
        Tensor a = Tensor.fromArray(new float[]{0}, new int[]{1});
        Tensor result = TensorOps.sigmoid(a);
        assertEquals(0.5f, result.get(0), 0.001f);
    }

    @Test
    public void testSigmoidEdgeCases() {
        Tensor largePos = Tensor.fromArray(new float[]{20f}, new int[]{1});
        Tensor largeNeg = Tensor.fromArray(new float[]{-20f}, new int[]{1});
        assertEquals(1.0f, TensorOps.sigmoid(largePos).get(0), 1e-5f);
        assertEquals(0.0f, TensorOps.sigmoid(largeNeg).get(0), 1e-5f);
    }

    @Test
    public void testGelu() {
        Tensor zero = Tensor.fromArray(new float[]{0}, new int[]{1});
        assertEquals(0.0f, TensorOps.gelu(zero).get(0), 1e-5f);

        Tensor one = Tensor.fromArray(new float[]{1}, new int[]{1});
        assertEquals(0.841f, TensorOps.gelu(one).get(0), 0.01f);
    }

    // ========== Normalization ==========

    @Test
    public void testRmsNorm() {
        Tensor x = Tensor.fromArray(new float[]{3f, 4f}, new int[]{2});
        Tensor gamma = Tensor.fromArray(new float[]{1f, 1f}, new int[]{2});
        Tensor result = TensorOps.rmsNorm(x, gamma, 1e-6f);

        // 🔧 FIX: правильный расчёт RMS = sqrt(mean(x²))
        // x² = [9, 16], mean = (9+16)/2 = 12.5, RMS = sqrt(12.5) ≈ 3.5355
        float rms = (float) Math.sqrt((3f*3f + 4f*4f) / 2f);  // ≈ 3.5355339

        // Ожидаемые значения: [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        assertEquals(3f / rms, result.get(0), 1e-5f);  // ≈ 0.8485281
        assertEquals(4f / rms, result.get(1), 1e-5f);  // ≈ 1.1313708
    }

    // ========== Matrix operations ==========

    @Test
    public void testMatmul() {
        Tensor a = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
        Tensor b = Tensor.fromArray(new float[]{7, 8, 9, 10, 11, 12}, new int[]{3, 2});
        Tensor result = TensorOps.matmul(a, b);

        assertEquals(58.0f, result.get(0, 0));
        assertEquals(64.0f, result.get(0, 1));
        assertEquals(139.0f, result.get(1, 0));
        assertEquals(154.0f, result.get(1, 1));
    }

    @Test
    public void testMatmulAddReluMatchesReference() {
        Tensor a = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
        Tensor b = Tensor.fromArray(new float[]{7, 8, 9, 10, 11, 12}, new int[]{3, 2});
        Tensor bias = Tensor.fromArray(new float[]{0.1f, -0.5f}, new int[]{2});
        Tensor fused = TensorOps.matmulAddRelu(a, b, bias);

        Tensor raw = TensorOps.matmul(a, b);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                float v = raw.get(i, j) + bias.get(j);
                float expected = v > 0f ? v : 0f;
                assertEquals(expected, fused.get(i, j), 1e-5f);
            }
        }
    }

    @Test
    public void testMatmulAddReluNegativeBiasZeros() {
        Tensor a = Tensor.fromArray(new float[]{1, 0, 0, 0, 1, 0}, new int[]{2, 3});
        Tensor b = Tensor.fromArray(new float[]{1, 0, 0, 1, 0, 0}, new int[]{3, 2});
        Tensor bias = Tensor.fromArray(new float[]{-100f, -100f}, new int[]{2});
        Tensor out = TensorOps.matmulAddRelu(a, b, bias);
        assertEquals(0f, out.get(0, 0));
        assertEquals(0f, out.get(0, 1));
        assertEquals(0f, out.get(1, 0));
        assertEquals(0f, out.get(1, 1));
    }

    // ========== Attention helpers ==========

    @Test
    public void testSplitConcatHeadsRoundtrip() {
        Tensor x = Tensor.fromArray(new float[]{1,2,3,4, 5,6,7,8}, new int[]{1, 2, 4});
        int numHeads = 2;

        Tensor split = TensorOps.splitHeads(x, numHeads);
        Tensor concat = TensorOps.concatHeads(split, numHeads);

        for (int i = 0; i < 8; i++) {
            assertEquals(x.internalBuffer()[i], concat.internalBuffer()[i], 1e-6f);
        }
    }

    @Test
    public void testSoftmaxLastDim() {
        // 🔧 FIX: softmaxLastDim требует 3D тензор [batch, mid, inner]
        // Создаём [1, 2, 3]: 1 батч, 2 строки, 3 элемента в каждой
        Tensor x = Tensor.fromArray(
                new float[]{1, 2, 3,   // первая строка
                        4, 5, 6},  // вторая строка
                new int[]{1, 2, 3}     // ← 3D shape!
        );

        Tensor result = TensorOps.softmaxLastDim(x);

        // Проверка: сумма по последней оси = 1 для каждой строки
        assertEquals(1.0f,
                result.get(0, 0, 0) + result.get(0, 0, 1) + result.get(0, 0, 2),
                1e-5f);
        assertEquals(1.0f,
                result.get(0, 1, 0) + result.get(0, 1, 1) + result.get(0, 1, 2),
                1e-5f);

        // Дополнительная проверка: значения в [0, 1]
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                float v = result.get(0, i, j);
                assertTrue(v >= 0f && v <= 1f, "softmax output out of range: " + v);
            }
        }
    }

    // ========== Consistency tests ==========

    @Test
    public void testVectorizedVsScalarConsistency() {
        Tensor a = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4});
        Tensor b = Tensor.fromArray(new float[]{0.5f, -1f, 2f, -0.5f, 1f, 0f, -2f, 3f}, new int[]{2, 4});

        Tensor resultVec = TensorOps.add(a, b);

        float[] expected = new float[8];
        float[] da = a.internalBuffer();
        float[] db = b.internalBuffer();
        for (int i = 0; i < 8; i++) expected[i] = da[i] + db[i];

        for (int i = 0; i < 8; i++) {
            assertEquals(expected[i], resultVec.internalBuffer()[i], 1e-6f);
        }
    }

    // ========== Quantized ==========

    @Test
    public void testMatmulQuantizedMatchesFloat() {
        Tensor a = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
        Tensor b = Tensor.fromArray(new float[]{7, 8, 9, 10, 11, 12}, new int[]{3, 2});

        QuantizedTensor qa = QuantizedTensor.fromTensor(a);
        QuantizedTensor qb = QuantizedTensor.fromTensor(b);

        Tensor expected = TensorOps.matmul(a, b);
        Tensor actual = TensorOps.matmul(qa, qb);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expected.get(i, j), actual.get(i, j), 0.5f);
            }
        }
    }
}