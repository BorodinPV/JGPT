package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.model.BlockActivationCache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

class FfnNormResidualGpuTest {
    @Test
    void fusedFfnNormResidualBackwardMatchesReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }

        int batch = 8;
        int seqLen = 64;
        int dModel = 128;
        int dInt = 512;
        int rows = batch * seqLen;
        if (!TensorOpsGPU.shouldUseGpuMatmul(rows, dModel, dInt)
                || !TensorOpsGPU.shouldUseGpuElementwise(rows * dInt)) {
            return;
        }

        Random rng = new Random(1234);
        Tensor xRes1 = randomTensor(new int[] {batch, seqLen, dModel}, rng);
        Tensor norm2 = randomTensor(new int[] {dModel}, rng);
        Tensor xNorm2 = TensorOps.rmsNorm(xRes1, norm2, 1e-6f);
        Tensor w1 = randomTensor(new int[] {dModel, dInt}, rng);
        Tensor w2 = randomTensor(new int[] {dInt, dModel}, rng);
        Tensor w3 = randomTensor(new int[] {dModel, dInt}, rng);
        Tensor gradOut = new Tensor(new int[] {batch, seqLen, dModel});
        gradOut.zeroGrad();
        fillRandom(gradOut.gradBuffer(), rng);

        BlockActivationCache cache = new BlockActivationCache();
        TensorOps.feedForwardSwiGLU(xNorm2, w1, w2, w3, cache);

        Tensor refGradXRes1 = new Tensor(new int[] {batch, seqLen, dModel});
        refGradXRes1.zeroGrad();
        System.arraycopy(gradOut.gradBuffer(), 0, refGradXRes1.gradBuffer(), 0, gradOut.gradBuffer().length);
        Tensor refGradXNorm2 = new Tensor(new int[] {batch, seqLen, dModel});
        refGradXNorm2.zeroGrad();
        Tensor refGradW1 = new Tensor(new int[] {dModel, dInt});
        Tensor refGradW2 = new Tensor(new int[] {dInt, dModel});
        Tensor refGradW3 = new Tensor(new int[] {dModel, dInt});
        Tensor refGradNorm2 = new Tensor(new int[] {dModel});
        TransformerBackward.feedForwardSwiGLUBackward(
                gradOut,
                xNorm2,
                w1,
                w2,
                w3,
                refGradXNorm2,
                refGradW1,
                refGradW2,
                refGradW3,
                cache.ffnH1.getTensor(),
                cache.ffnGate.getTensor(),
                cache.ffnSig.getTensor(),
                cache.ffnGateSwish.getTensor(),
                cache.ffnHActivated.getTensor());
        TensorOpsBackward.rmsNormBackward(refGradXNorm2, xRes1, norm2, 1e-6f, refGradXRes1, refGradNorm2);

        Tensor gradXRes1 = new Tensor(new int[] {batch, seqLen, dModel});
        gradXRes1.zeroGrad();
        Tensor gradW1 = new Tensor(new int[] {dModel, dInt});
        Tensor gradW2 = new Tensor(new int[] {dInt, dModel});
        Tensor gradW3 = new Tensor(new int[] {dModel, dInt});
        Tensor gradNorm2 = new Tensor(new int[] {dModel});
        boolean used =
                TransformerBackward.tryFusedFfnNormResidualBackwardGpu(
                        gradOut,
                        cache,
                        xRes1,
                        xNorm2,
                        w1,
                        w2,
                        w3,
                        norm2,
                        gradXRes1,
                        gradW1,
                        gradW2,
                        gradW3,
                        gradNorm2,
                        null);
        assertTrue(used, "gpu fused path should be used");
        // gradW* и gradNorm2 идут через GpuPendingGradients (device); без flush host-градиент пустой.
        GpuPendingGradients.flushAllToHost();

        assertClose(refGradXRes1.gradBuffer(), gradXRes1.gradBuffer(), 2e-2f, "gradXRes1");
        assertClose(refGradNorm2.gradBuffer(), gradNorm2.gradBuffer(), 2e-2f, "gradNorm2");
        assertClose(refGradW1.gradBuffer(), gradW1.gradBuffer(), 2e-2f, "gradW1");
        assertClose(refGradW2.gradBuffer(), gradW2.gradBuffer(), 2e-2f, "gradW2");
        assertClose(refGradW3.gradBuffer(), gradW3.gradBuffer(), 2e-2f, "gradW3");
    }

    private static Tensor randomTensor(int[] shape, Random rng) {
        Tensor t = new Tensor(shape);
        fillRandom(t.internalBuffer(), rng);
        return t;
    }

    private static void fillRandom(float[] data, Random rng) {
        for (int i = 0; i < data.length; i++) {
            data[i] = rng.nextFloat() * 2f - 1f;
        }
    }

    private static void assertClose(float[] expected, float[] actual, float eps, String name) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], eps, name + "[" + i + "]");
        }
    }
}
