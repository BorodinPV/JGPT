package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("TensorOpsBackward")
public class TensorOpsBackwardTest {

    // ========== Add Backward ==========

    @Test
    @DisplayName("addBackward accumulates gradients correctly")
    public void addBackwardAccumulates() {
        Tensor gradC = new Tensor(new int[]{3});
        gradC.zeroGrad();
        gradC.gradBuffer()[0] = 1f;
        gradC.gradBuffer()[1] = 2f;
        gradC.gradBuffer()[2] = 3f;

        Tensor gradA = new Tensor(new int[]{3});
        Tensor gradB = new Tensor(new int[]{3});
        gradA.zeroGrad();
        gradB.zeroGrad();

        TensorOpsBackward.addBackward(gradC, gradA, gradB);

        assertEquals(1f, gradA.gradBuffer()[0]);
        assertEquals(2f, gradA.gradBuffer()[1]);
        assertEquals(3f, gradA.gradBuffer()[2]);
        assertEquals(1f, gradB.gradBuffer()[0]);
        assertEquals(2f, gradB.gradBuffer()[1]);
        assertEquals(3f, gradB.gradBuffer()[2]);
    }

    @Test
    @DisplayName("subtractBackward accumulates with negation")
    public void subtractBackwardAccumulates() {
        Tensor gradC = new Tensor(new int[]{3});
        gradC.zeroGrad();
        gradC.gradBuffer()[0] = 1f;
        gradC.gradBuffer()[1] = 2f;
        gradC.gradBuffer()[2] = 3f;

        Tensor gradA = new Tensor(new int[]{3});
        Tensor gradB = new Tensor(new int[]{3});
        gradA.zeroGrad();
        gradB.zeroGrad();

        TensorOpsBackward.subtractBackward(gradC, gradA, gradB);

        assertEquals(1f, gradA.gradBuffer()[0]);
        assertEquals(-1f, gradB.gradBuffer()[0]);
        assertEquals(2f, gradA.gradBuffer()[1]);
        assertEquals(-2f, gradB.gradBuffer()[1]);
        assertEquals(3f, gradA.gradBuffer()[2]);
        assertEquals(-3f, gradB.gradBuffer()[2]);
    }

    @Test
    @DisplayName("addBackward rejects null inputs")
    public void addBackwardRejectsNull() {
        Tensor t = new Tensor(new int[]{3});
        t.zeroGrad();
        assertThrows(NullPointerException.class,
                () -> TensorOpsBackward.addBackward(null, t, t));
        assertThrows(NullPointerException.class,
                () -> TensorOpsBackward.addBackward(t, null, t));
        assertThrows(NullPointerException.class,
                () -> TensorOpsBackward.addBackward(t, t, null));
    }

    @Test
    @DisplayName("addBackward rejects shape mismatch")
    public void addBackwardRejectsShapeMismatch() {
        Tensor gradC = new Tensor(new int[]{3});
        gradC.zeroGrad();
        Tensor gradA = new Tensor(new int[]{3});
        gradA.zeroGrad();
        Tensor gradB = new Tensor(new int[]{4});
        gradB.zeroGrad();

        assertThrows(IllegalArgumentException.class,
                () -> TensorOpsBackward.addBackward(gradC, gradA, gradB));
    }

    // ========== Matmul Backward ==========

    @Test
    @DisplayName("matmulBackward matches analytical gradients")
    public void matmulBackwardMatchesAnalytic() {
        Tensor a = Tensor.fromArray(new float[]{1f, 2f}, new int[]{1, 2});
        Tensor b = Tensor.fromArray(new float[]{3f, 4f}, new int[]{2, 1});

        Tensor gradC = new Tensor(new int[]{1, 1});
        gradC.zeroGrad();
        gradC.gradBuffer()[0] = 1f;

        Tensor gradA = new Tensor(new int[]{1, 2});
        Tensor gradB = new Tensor(new int[]{2, 1});
        gradA.zeroGrad();
        gradB.zeroGrad();

        TensorOpsBackward.matmulBackward(gradC, a, b, gradA, gradB);

        assertEquals(3f, gradA.gradBuffer()[0], 1e-5f);
        assertEquals(4f, gradA.gradBuffer()[1], 1e-5f);
        assertEquals(1f, gradB.gradBuffer()[0], 1e-5f);
        assertEquals(2f, gradB.gradBuffer()[1], 1e-5f);
    }

    // ========== Transpose ==========

    @Test
    @DisplayName("transpose swaps dimensions correctly")
    public void transposeSwapsDims() {
        Tensor t = Tensor.fromArray(new float[]{1, 2, 3, 4}, new int[]{2, 2});
        Tensor tr = TensorOpsBackward.transpose(t);
        assertEquals(2, tr.getShape()[0]);
        assertEquals(2, tr.getShape()[1]);
        assertEquals(1f, tr.get(0, 0));
        assertEquals(3f, tr.get(0, 1));
        assertEquals(2f, tr.get(1, 0));
        assertEquals(4f, tr.get(1, 1));
    }

    @Test
    @DisplayName("transpose rejects non-2D tensor")
    public void transposeRejectsNon2D() {
        Tensor t = new Tensor(new int[]{2, 2, 2});
        assertThrows(IllegalArgumentException.class,
                () -> TensorOpsBackward.transpose(t));
    }

    // ========== Accumulate Gradient ==========

    @Test
    @DisplayName("accumulateGradientInto adds update to target")
    public void accumulateGradientIntoAdds() {
        Tensor target = new Tensor(new int[]{2});
        target.zeroGrad();

        // 🔧 FIX: заполняем gradBuffer(), а не internalBuffer()
        target.gradBuffer()[0] = 1f;
        target.gradBuffer()[1] = 2f;

        Tensor update = Tensor.fromArray(new float[]{0.5f, 0.5f}, new int[]{2});

        TensorOpsBackward.accumulateGradientInto(target, update);

        assertEquals(1.5f, target.gradBuffer()[0]);
        assertEquals(2.5f, target.gradBuffer()[1]);
    }

    @Test
    @DisplayName("accumulateGradientInto initializes grad if needed")
    public void accumulateGradientIntoInitializesGrad() {
        Tensor target = new Tensor(new int[]{2});
        // Don't call zeroGrad() - should be initialized by method
        Tensor update = Tensor.fromArray(new float[]{1f, 2f}, new int[]{2});

        TensorOpsBackward.accumulateGradientInto(target, update);

        assertTrue(target.hasGrad());
        assertEquals(1f, target.gradBuffer()[0]);
        assertEquals(2f, target.gradBuffer()[1]);
    }

    // ========== ReLU Backward ==========

    @Test
    @DisplayName("reluBackward only passes positive gradients")
    public void reluBackwardOnlyPositive() {
        Tensor gradOut = Tensor.fromArray(new float[]{1f, 2f, 3f}, new int[]{3});
        Tensor input = Tensor.fromArray(new float[]{-1f, 0f, 1f}, new int[]{3});
        Tensor gradIn = new Tensor(new int[]{3});
        gradIn.zeroGrad();

        TensorOpsBackward.reluBackward(gradOut, input, gradIn);

        assertEquals(0f, gradIn.gradBuffer()[0]);  // negative → 0
        assertEquals(0f, gradIn.gradBuffer()[1]);  // zero → 0
        assertEquals(3f, gradIn.gradBuffer()[2]);  // positive → passes through
    }

    // ========== Cross-Entropy Backward ==========

    @Test
    @DisplayName("crossEntropySoftmaxBackward basic case")
    public void crossEntropySoftmaxBackwardBasic() {
        Tensor logits = Tensor.fromArray(new float[]{0f, 0f}, new int[]{1, 1, 2});
        Tensor target = Tensor.fromArray(new float[]{1f}, new int[]{1, 1});

        Tensor grad = TensorOpsBackward.crossEntropySoftmaxBackward(logits, target);

        // For uniform probs [0.5, 0.5], gradient = [0.5-0, 0.5-1] / 1 = [0.5, -0.5]
        assertEquals(0.5f, grad.gradBuffer()[0], 1e-5f);
        assertEquals(-0.5f, grad.gradBuffer()[1], 1e-5f);
    }

    @Test
    @DisplayName("crossEntropySoftmaxBackward rejects null")
    public void crossEntropySoftmaxBackwardRejectsNull() {
        Tensor t = new Tensor(new int[]{1, 1, 2});
        assertThrows(NullPointerException.class,
                () -> TensorOpsBackward.crossEntropySoftmaxBackward(null, t));
        assertThrows(NullPointerException.class,
                () -> TensorOpsBackward.crossEntropySoftmaxBackward(t, null));
    }
}