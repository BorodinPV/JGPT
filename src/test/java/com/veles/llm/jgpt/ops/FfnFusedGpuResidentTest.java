package com.veles.llm.jgpt.ops;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import org.junit.jupiter.api.Test;

/**
 * Сравнение fused FFN: копия весов с хоста на каждый вызов vs уже загруженные device-буферы.
 */
class FfnFusedGpuResidentTest {

    @Test
    void residentMatchesHostCopyFusedPath() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 2;
        int seqLen = 3;
        int dModel = 16;
        int dInt = 32;
        int[] xShape = new int[] {batch, seqLen, dModel};

        Tensor xRes1 = new Tensor(xShape);
        float[] xb = xRes1.internalBuffer();
        for (int i = 0; i < xb.length; i++) {
            xb[i] = (float) Math.sin(i * 0.07) * 0.5f;
        }

        Tensor norm2 = TensorOps.randomTensor(new int[] {dModel}, 1.0f);
        Tensor w1 = TensorOps.randomTensor(new int[] {dModel, dInt}, 0.02f);
        Tensor w2 = TensorOps.randomTensor(new int[] {dInt, dModel}, 0.02f);
        Tensor w3 = TensorOps.randomTensor(new int[] {dModel, dInt}, 0.02f);

        TensorOps.FfnForwardResult copyPath =
                TensorOps.tryFusedNormResidualSwiGLUForwardGpu(xRes1, norm2, w1, w2, w3, null);
        assertNotNull(copyPath);

        GpuTensor gn = GpuTensor.fromHostTensor(norm2);
        GpuTensor g1 = GpuTensor.fromHostTensor(w1);
        GpuTensor g2 = GpuTensor.fromHostTensor(w2);
        GpuTensor g3 = GpuTensor.fromHostTensor(w3);
        try {
            TensorOps.GpuFfnResidentBuffers resident =
                    new TensorOps.GpuFfnResidentBuffers(
                            gn.dataBuffer(), g1.dataBuffer(), g2.dataBuffer(), g3.dataBuffer());
            TensorOps.FfnForwardResult resPath =
                    TensorOps.tryFusedNormResidualSwiGLUForwardGpuResident(xRes1, resident, null);
            assertNotNull(resPath);

            float[] a = copyPath.out.internalBuffer();
            float[] b = resPath.out.internalBuffer();
            float maxAbs = 0f;
            for (int i = 0; i < a.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
            }
            assertTrue(maxAbs < 1e-4f, "max abs diff " + maxAbs);
        } finally {
            gn.close();
            g1.close();
            g2.close();
            g3.close();
        }
    }
}
