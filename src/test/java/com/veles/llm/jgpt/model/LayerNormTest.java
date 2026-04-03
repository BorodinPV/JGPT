package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;
import com.veles.llm.jgpt.ops.TensorOpsBackward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LayerNormTest {

    @Test
    public void layerNormForwardShapeAndFinite() {
        Tensor x = Tensor.fromArray(
                new float[]{1, 2, 3, 4, 5, 6},
                new int[]{2, 3});
        Tensor gamma = Tensor.fromArray(new float[]{1, 1, 1}, new int[]{3});
        Tensor beta = Tensor.fromArray(new float[]{0, 0, 0}, new int[]{3});
        Tensor y = TensorOps.layerNorm(x, gamma, beta, 1e-5f);
        assertEquals(2, y.getShape()[0]);
        assertEquals(3, y.getShape()[1]);
        for (int i = 0; i < 6; i++) {
            assertTrue(Float.isFinite(y.internalBuffer()[i]));
        }
    }

    @Test
    public void layerNormBackwardBetaMatchesOnesGrad() {
        int lastDim = 3;
        int outer = 2;
        Tensor x = Tensor.fromArray(
                new float[]{1, 2, 3, 4, 5, 6},
                new int[]{outer, lastDim});
        Tensor gamma = Tensor.fromArray(new float[]{1, 1, 1}, new int[]{lastDim});
        Tensor beta = Tensor.fromArray(new float[]{0, 0, 0}, new int[]{lastDim});

        Tensor gradOut = new Tensor(new int[]{outer, lastDim});
        gradOut.zeroGrad();
        float[] go = gradOut.gradBuffer();
        for (int i = 0; i < go.length; i++) {
            go[i] = 1f;
        }

        Tensor gradX = new Tensor(x.getShape());
        Tensor gradGamma = new Tensor(new int[]{lastDim});
        Tensor gradBeta = new Tensor(new int[]{lastDim});

        TensorOpsBackward.layerNormBackward(gradOut, x, gamma, beta, 1e-5f, gradX, gradGamma, gradBeta);

        for (int j = 0; j < lastDim; j++) {
            assertEquals(outer, gradBeta.gradBuffer()[j], 1e-4f,
                    "dL/dBeta_j = sum over outer of gradOut");
        }
    }
}
