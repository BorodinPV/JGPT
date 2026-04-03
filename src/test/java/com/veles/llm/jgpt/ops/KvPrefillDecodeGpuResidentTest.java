package com.veles.llm.jgpt.ops;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.model.KvCache;
import com.veles.llm.jgpt.model.KvCacheGpu;
import org.junit.jupiter.api.Test;

class KvPrefillDecodeGpuResidentTest {

    @Test
    void prefillResidentMatchesCpuPrefill() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 1;
        int seqLen = 4;
        int dModel = 16;
        int numHeads = 4;
        int dHead = dModel / numHeads;
        int ropeOffset = 2;
        float eps = 1e-6f;

        int[] xShape = new int[] {batch, seqLen, dModel};
        Tensor x = new Tensor(xShape);
        float[] xb = x.internalBuffer();
        for (int i = 0; i < xb.length; i++) {
            xb[i] = (float) Math.cos(i * 0.09) * 0.35f;
        }

        Tensor norm1 = TensorOps.randomTensor(new int[] {dModel}, 1.0f);
        Tensor wq = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wk = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wv = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wo = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        KvCache cacheCpu = new KvCache(1, numHeads, dHead, 32);
        KvCache cacheGpu = new KvCache(1, numHeads, dHead, 32);

        Tensor xNorm = TensorOps.rmsNorm(x, norm1, eps);
        Tensor ref =
                TensorOps.multiHeadAttentionWithRoPEPrefill(
                        xNorm,
                        wq,
                        wk,
                        wv,
                        wo,
                        numHeads,
                        mask,
                        cacheCpu.getK(0),
                        cacheCpu.getV(0),
                        ropeOffset);

        GpuTensor gn = GpuTensor.fromHostTensor(norm1);
        GpuTensor gq = GpuTensor.fromHostTensor(wq);
        GpuTensor gk = GpuTensor.fromHostTensor(wk);
        GpuTensor gv = GpuTensor.fromHostTensor(wv);
        GpuTensor go = GpuTensor.fromHostTensor(wo);
        try {
            TensorOps.GpuAttnResidentBuffers res =
                    new TensorOps.GpuAttnResidentBuffers(
                            gn.dataBuffer(), gq.dataBuffer(), gk.dataBuffer(), gv.dataBuffer(), go.dataBuffer());
            Tensor gpu =
                    TensorOps.tryMultiHeadAttentionWithRoPEPrefillGpuResident(
                            x, eps, res, numHeads, mask, cacheGpu.getK(0), cacheGpu.getV(0), ropeOffset);
            assertNotNull(gpu);

            float[] a = ref.internalBuffer();
            float[] b = gpu.internalBuffer();
            float maxAbs = 0f;
            for (int i = 0; i < a.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
            }
            assertTrue(maxAbs < 3e-3f, "prefill max abs diff " + maxAbs);

            float[] kc = cacheCpu.getK(0).internalBuffer();
            float[] kg = cacheGpu.getK(0).internalBuffer();
            float maxK = 0f;
            for (int i = 0; i < kc.length; i++) {
                maxK = Math.max(maxK, Math.abs(kc[i] - kg[i]));
            }
            assertTrue(maxK < 3e-3f, "K cache max abs diff " + maxK);
        } finally {
            gn.close();
            gq.close();
            gk.close();
            gv.close();
            go.close();
        }
    }

    @Test
    void prefillVramMatchesCpuPrefill() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 1;
        int seqLen = 4;
        int dModel = 16;
        int numHeads = 4;
        int dHead = dModel / numHeads;
        int maxSeqLen = 32;
        int ropeOffset = 2;
        float eps = 1e-6f;

        int[] xShape = new int[] {batch, seqLen, dModel};
        Tensor x = new Tensor(xShape);
        float[] xb = x.internalBuffer();
        for (int i = 0; i < xb.length; i++) {
            xb[i] = (float) Math.cos(i * 0.09) * 0.35f;
        }

        Tensor norm1 = TensorOps.randomTensor(new int[] {dModel}, 1.0f);
        Tensor wq = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wk = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wv = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wo = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        KvCache cacheCpu = new KvCache(1, numHeads, dHead, maxSeqLen);

        Tensor xNorm = TensorOps.rmsNorm(x, norm1, eps);
        Tensor ref =
                TensorOps.multiHeadAttentionWithRoPEPrefill(
                        xNorm,
                        wq,
                        wk,
                        wv,
                        wo,
                        numHeads,
                        mask,
                        cacheCpu.getK(0),
                        cacheCpu.getV(0),
                        ropeOffset);

        GpuTensor gn = GpuTensor.fromHostTensor(norm1);
        GpuTensor gq = GpuTensor.fromHostTensor(wq);
        GpuTensor gk = GpuTensor.fromHostTensor(wk);
        GpuTensor gv = GpuTensor.fromHostTensor(wv);
        GpuTensor go = GpuTensor.fromHostTensor(wo);
        try (KvCacheGpu cacheVram = new KvCacheGpu(1, numHeads, dHead, maxSeqLen)) {
            TensorOps.GpuAttnResidentBuffers res =
                    new TensorOps.GpuAttnResidentBuffers(
                            gn.dataBuffer(), gq.dataBuffer(), gk.dataBuffer(), gv.dataBuffer(), go.dataBuffer());
            Tensor gpu =
                    TensorOps.tryMultiHeadAttentionWithRoPEPrefillGpuResident(
                            x,
                            eps,
                            res,
                            numHeads,
                            mask,
                            cacheVram.getK(0),
                            cacheVram.getV(0),
                            maxSeqLen,
                            ropeOffset);
            assertNotNull(gpu);

            float[] a = ref.internalBuffer();
            float[] b = gpu.internalBuffer();
            float maxAbs = 0f;
            for (int i = 0; i < a.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
            }
            assertTrue(maxAbs < 3e-3f, "prefill VRAM max abs diff " + maxAbs);

            int n = cacheCpu.getK(0).internalBuffer().length;
            float[] kc = cacheCpu.getK(0).internalBuffer();
            float[] kg = new float[n];
            cacheVram.getK(0).copyTo(kg, 0, n);
            float maxK = 0f;
            for (int i = 0; i < n; i++) {
                maxK = Math.max(maxK, Math.abs(kc[i] - kg[i]));
            }
            assertTrue(maxK < 3e-3f, "K cache VRAM max abs diff " + maxK);
        } finally {
            gn.close();
            gq.close();
            gk.close();
            gv.close();
            go.close();
        }
    }

    @Test
    void decodeResidentMatchesCpuAfterPrefill() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int dModel = 16;
        int numHeads = 4;
        int dHead = dModel / numHeads;
        int seqLen = 3;
        float eps = 1e-6f;

        int[] xShape = new int[] {1, seqLen, dModel};
        Tensor xPrefill = new Tensor(xShape);
        float[] xb = xPrefill.internalBuffer();
        for (int i = 0; i < xb.length; i++) {
            xb[i] = (float) Math.sin(i * 0.12) * 0.4f;
        }

        Tensor norm1 = TensorOps.randomTensor(new int[] {dModel}, 1.0f);
        Tensor wq = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wk = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wv = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wo = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        KvCache cacheCpu = new KvCache(1, numHeads, dHead, 32);
        KvCache cacheGpu = new KvCache(1, numHeads, dHead, 32);

        Tensor xNormP = TensorOps.rmsNorm(xPrefill, norm1, eps);
        TensorOps.multiHeadAttentionWithRoPEPrefill(
                xNormP,
                wq,
                wk,
                wv,
                wo,
                numHeads,
                mask,
                cacheCpu.getK(0),
                cacheCpu.getV(0),
                0);

        int kvLen = cacheCpu.getK(0).internalBuffer().length;
        System.arraycopy(cacheCpu.getK(0).internalBuffer(), 0, cacheGpu.getK(0).internalBuffer(), 0, kvLen);
        System.arraycopy(cacheCpu.getV(0).internalBuffer(), 0, cacheGpu.getV(0).internalBuffer(), 0, kvLen);

        cacheCpu.setLength(seqLen);
        cacheGpu.setLength(seqLen);

        Tensor xOne = new Tensor(new int[] {1, 1, dModel});
        float[] ob = xOne.internalBuffer();
        for (int i = 0; i < ob.length; i++) {
            ob[i] = (float) Math.sin(i * 0.31 + 1.0) * 0.5f;
        }

        GpuTensor gn = GpuTensor.fromHostTensor(norm1);
        GpuTensor gq = GpuTensor.fromHostTensor(wq);
        GpuTensor gk = GpuTensor.fromHostTensor(wk);
        GpuTensor gv = GpuTensor.fromHostTensor(wv);
        GpuTensor go = GpuTensor.fromHostTensor(wo);
        try {
            TensorOps.GpuAttnResidentBuffers res =
                    new TensorOps.GpuAttnResidentBuffers(
                            gn.dataBuffer(), gq.dataBuffer(), gk.dataBuffer(), gv.dataBuffer(), go.dataBuffer());

            Tensor xNorm1 = TensorOps.rmsNorm(xOne, norm1, eps);
            Tensor ref =
                    TensorOps.multiHeadAttentionWithRoPEDecode(
                            xNorm1,
                            wq,
                            wk,
                            wv,
                            wo,
                            numHeads,
                            cacheCpu.getK(0),
                            cacheCpu.getV(0),
                            seqLen,
                            seqLen);

            Tensor gpu =
                    TensorOps.tryMultiHeadAttentionWithRoPEDecodeGpuResident(
                            xOne, eps, res, numHeads, cacheGpu.getK(0), cacheGpu.getV(0), seqLen, seqLen);

            assertNotNull(gpu);
            float[] a = ref.internalBuffer();
            float[] b = gpu.internalBuffer();
            float maxAbs = 0f;
            for (int i = 0; i < a.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
            }
            assertTrue(maxAbs < 3e-3f, "decode max abs diff " + maxAbs);
        } finally {
            gn.close();
            gq.close();
            gk.close();
            gv.close();
            go.close();
        }
    }

    @Test
    void decodeVramMatchesCpuAfterPrefill() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int dModel = 16;
        int numHeads = 4;
        int dHead = dModel / numHeads;
        int seqLen = 3;
        int maxSeqLen = 32;
        float eps = 1e-6f;

        int[] xShape = new int[] {1, seqLen, dModel};
        Tensor xPrefill = new Tensor(xShape);
        float[] xb = xPrefill.internalBuffer();
        for (int i = 0; i < xb.length; i++) {
            xb[i] = (float) Math.sin(i * 0.12) * 0.4f;
        }

        Tensor norm1 = TensorOps.randomTensor(new int[] {dModel}, 1.0f);
        Tensor wq = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wk = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wv = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);
        Tensor wo = TensorOps.randomTensor(new int[] {dModel, dModel}, 0.03f);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        KvCache cacheCpu = new KvCache(1, numHeads, dHead, maxSeqLen);

        Tensor xNormP = TensorOps.rmsNorm(xPrefill, norm1, eps);
        TensorOps.multiHeadAttentionWithRoPEPrefill(
                xNormP,
                wq,
                wk,
                wv,
                wo,
                numHeads,
                mask,
                cacheCpu.getK(0),
                cacheCpu.getV(0),
                0);

        int kvLen = cacheCpu.getK(0).internalBuffer().length;

        Tensor xOne = new Tensor(new int[] {1, 1, dModel});
        float[] ob = xOne.internalBuffer();
        for (int i = 0; i < ob.length; i++) {
            ob[i] = (float) Math.sin(i * 0.31 + 1.0) * 0.5f;
        }

        GpuTensor gn = GpuTensor.fromHostTensor(norm1);
        GpuTensor gq = GpuTensor.fromHostTensor(wq);
        GpuTensor gk = GpuTensor.fromHostTensor(wk);
        GpuTensor gv = GpuTensor.fromHostTensor(wv);
        GpuTensor go = GpuTensor.fromHostTensor(wo);
        try (KvCacheGpu cacheVram = new KvCacheGpu(1, numHeads, dHead, maxSeqLen)) {
            cacheVram.getK(0).copyFrom(cacheCpu.getK(0).internalBuffer(), 0, kvLen);
            cacheVram.getV(0).copyFrom(cacheCpu.getV(0).internalBuffer(), 0, kvLen);
            cacheVram.setLength(seqLen);

            TensorOps.GpuAttnResidentBuffers res =
                    new TensorOps.GpuAttnResidentBuffers(
                            gn.dataBuffer(), gq.dataBuffer(), gk.dataBuffer(), gv.dataBuffer(), go.dataBuffer());

            Tensor xNorm1 = TensorOps.rmsNorm(xOne, norm1, eps);
            Tensor ref =
                    TensorOps.multiHeadAttentionWithRoPEDecode(
                            xNorm1,
                            wq,
                            wk,
                            wv,
                            wo,
                            numHeads,
                            cacheCpu.getK(0),
                            cacheCpu.getV(0),
                            seqLen,
                            seqLen);

            Tensor gpu =
                    TensorOps.tryMultiHeadAttentionWithRoPEDecodeGpuResident(
                            xOne,
                            eps,
                            res,
                            numHeads,
                            cacheVram.getK(0),
                            cacheVram.getV(0),
                            maxSeqLen,
                            seqLen,
                            seqLen);

            assertNotNull(gpu);
            float[] a = ref.internalBuffer();
            float[] b = gpu.internalBuffer();
            float maxAbs = 0f;
            for (int i = 0; i < a.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
            }
            assertTrue(maxAbs < 3e-3f, "decode VRAM max abs diff " + maxAbs);
        } finally {
            gn.close();
            gq.close();
            gk.close();
            gv.close();
            go.close();
        }
    }
}
