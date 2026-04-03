package com.veles.llm.jgpt.ops;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import org.junit.jupiter.api.Test;

class AttnGpuResidentTest {

    @Test
    void residentMatchesCpuMultiHeadAttentionWithRoPE() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 1;
        int seqLen = 4;
        int dModel = 16;
        int numHeads = 4;
        int[] xShape = new int[] {batch, seqLen, dModel};

        Tensor x = new Tensor(xShape);
        float[] xb = x.internalBuffer();
        for (int i = 0; i < xb.length; i++) {
            xb[i] = (float) Math.sin(i * 0.11) * 0.4f;
        }

        float eps = 1e-6f;
        Tensor norm1 = TensorOps.randomTensor(new int[] {dModel}, 1.0f);
        Tensor wq = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wk = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wv = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wo = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);

        Tensor xNormCpu = TensorOps.rmsNorm(x, norm1, eps);
        Tensor ref =
                TensorOps.multiHeadAttentionWithRoPE(
                        xNormCpu, wq, wk, wv, wo, numHeads, TensorOps.createCausalMask(seqLen), true, null);

        GpuTensor gn = GpuTensor.fromHostTensor(norm1);
        GpuTensor gq = GpuTensor.fromHostTensor(wq);
        GpuTensor gk = GpuTensor.fromHostTensor(wk);
        GpuTensor gv = GpuTensor.fromHostTensor(wv);
        GpuTensor go = GpuTensor.fromHostTensor(wo);
        try {
            TensorOps.GpuAttnResidentBuffers res =
                    new TensorOps.GpuAttnResidentBuffers(
                            gn.dataBuffer(), gq.dataBuffer(), gk.dataBuffer(), gv.dataBuffer(), go.dataBuffer());
            TensorOps.AttnGpuResidentResult gpu =
                    TensorOps.tryMultiHeadAttentionWithRoPEGpuResident(
                            x, eps, res, numHeads, TensorOps.createCausalMask(seqLen), true, null);
            assertNotNull(gpu);

            float[] a = ref.internalBuffer();
            float[] b = gpu.out().internalBuffer();
            float maxAbs = 0f;
            for (int i = 0; i < a.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
            }
            assertTrue(maxAbs < 2e-3f, "max abs diff " + maxAbs);
        } finally {
            gn.close();
            gq.close();
            gk.close();
            gv.close();
            go.close();
        }
    }
}
