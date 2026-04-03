package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOpsBackward;

import java.util.Random;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

@DisplayName("subtractBackward (GPU path)")
class SubtractBackwardGpuTest {

    @Test
    @DisplayName("subtractBackward на GPU: ∂A += ∂C, ∂B -= ∂C совпадает с эталоном")
    void subtractBackwardGpuMatchesReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = TensorOpsGPU.GPU_ELEMENTWISE_MIN;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Random rng = new Random(42);
        Tensor gradC = new Tensor(new int[] {n});
        gradC.zeroGrad();
        float[] gc = gradC.gradBuffer();
        for (int i = 0; i < n; i++) {
            gc[i] = rng.nextFloat() * 4f - 2f;
        }

        Tensor gradA = new Tensor(new int[] {n});
        Tensor gradB = new Tensor(new int[] {n});
        gradA.zeroGrad();
        gradB.zeroGrad();

        TensorOpsBackward.subtractBackward(gradC, gradA, gradB);

        float[] ga = gradA.gradBuffer();
        float[] gb = gradB.gradBuffer();
        for (int i = 0; i < n; i++) {
            assertEquals(gc[i], ga[i], 1e-5f, "gradA[" + i + "]");
            assertEquals(-gc[i], gb[i], 1e-5f, "gradB[" + i + "]");
        }
    }

    @Test
    @DisplayName("subtractBackward на GPU: накопление += / -=")
    void subtractBackwardGpuAccumulates() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = TensorOpsGPU.GPU_ELEMENTWISE_MIN;
        if (!TensorOpsGPU.shouldUseGpuElementwise(n)) {
            return;
        }

        Tensor gradC = new Tensor(new int[] {n});
        gradC.zeroGrad();
        float[] gc = gradC.gradBuffer();
        gc[0] = 1f;
        gc[1] = 2f;
        gc[2] = 3f;
        for (int i = 3; i < n; i++) {
            gc[i] = 0f;
        }

        Tensor gradA = new Tensor(new int[] {n});
        Tensor gradB = new Tensor(new int[] {n});
        gradA.zeroGrad();
        gradB.zeroGrad();
        float[] ga = gradA.gradBuffer();
        float[] gb = gradB.gradBuffer();
        ga[0] = 10f;
        gb[0] = -5f;

        TensorOpsBackward.subtractBackward(gradC, gradA, gradB);

        assertEquals(11f, ga[0], 1e-5f);
        assertEquals(-6f, gb[0], 1e-5f);
        assertEquals(2f, ga[1], 1e-5f);
        assertEquals(-2f, gb[1], 1e-5f);
        assertEquals(0f, ga[99], 1e-5f);
        assertEquals(0f, gb[99], 1e-5f);
    }
}
