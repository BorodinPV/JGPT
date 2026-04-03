package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.model.BlockActivationCache;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

class AttentionBlockBackwardGpuTest {

    @Test
    void fusedAttentionBackwardMatchesDecomposedReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int batch = 8;
        int seqLen = 64;
        int numHeads = 8;
        int dModel = 512;
        int dHead = dModel / numHeads;
        int batchHeads = batch * numHeads;
        if (!TensorOpsGPU.shouldUseGpuMatmul(batch * seqLen, dModel, dModel)
                || !TensorOpsGPU.shouldUseGpuMatmulBatched(batchHeads, seqLen, seqLen, dHead)) {
            return;
        }

        Random rng = new Random(1234);
        Tensor xNorm = new Tensor(new int[] {batch, seqLen, dModel});
        Tensor wq = new Tensor(new int[] {dModel, dModel});
        Tensor wk = new Tensor(new int[] {dModel, dModel});
        Tensor wv = new Tensor(new int[] {dModel, dModel});
        Tensor wo = new Tensor(new int[] {dModel, dModel});
        fillRandom(xNorm.internalBuffer(), rng);
        fillRandom(wq.internalBuffer(), rng);
        fillRandom(wk.internalBuffer(), rng);
        fillRandom(wv.internalBuffer(), rng);
        fillRandom(wo.internalBuffer(), rng);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        BlockActivationCache cache = new BlockActivationCache();
        TensorOps.multiHeadAttentionWithRoPE(xNorm, wq, wk, wv, wo, numHeads, mask, true, cache);

        Tensor gradOut = new Tensor(new int[] {batch, seqLen, dModel});
        gradOut.zeroGrad();
        fillRandom(gradOut.gradBuffer(), rng);

        Tensor gradXFused = new Tensor(new int[] {batch, seqLen, dModel});
        Tensor gradWqFused = new Tensor(new int[] {dModel, dModel});
        Tensor gradWkFused = new Tensor(new int[] {dModel, dModel});
        Tensor gradWvFused = new Tensor(new int[] {dModel, dModel});
        Tensor gradWoFused = new Tensor(new int[] {dModel, dModel});
        assertTrue(
                TransformerBackward.tryFusedAttentionBackwardGpu(
                        gradOut,
                        xNorm,
                        wq,
                        wk,
                        wv,
                        wo,
                        numHeads,
                        cache,
                        gradXFused,
                        gradWqFused,
                        gradWkFused,
                        gradWvFused,
                        gradWoFused,
                        null));
        GpuPendingGradients.flushAllToHost();

        Tensor gradXRef = new Tensor(new int[] {batch, seqLen, dModel});
        Tensor gradWqRef = new Tensor(new int[] {dModel, dModel});
        Tensor gradWkRef = new Tensor(new int[] {dModel, dModel});
        Tensor gradWvRef = new Tensor(new int[] {dModel, dModel});
        Tensor gradWoRef = new Tensor(new int[] {dModel, dModel});
        referenceMultiHeadAttentionWithRoPEBackward(
                gradOut, xNorm, wq, wk, wv, wo, numHeads, mask, cache, gradXRef, gradWqRef, gradWkRef, gradWvRef, gradWoRef);

        assertClose(gradXRef.gradBuffer(), gradXFused.gradBuffer(), 2e-2f, "gradX");
        assertClose(gradWqRef.gradBuffer(), gradWqFused.gradBuffer(), 3e-2f, "gradWq");
        assertClose(gradWkRef.gradBuffer(), gradWkFused.gradBuffer(), 3e-2f, "gradWk");
        assertClose(gradWvRef.gradBuffer(), gradWvFused.gradBuffer(), 3e-2f, "gradWv");
        assertClose(gradWoRef.gradBuffer(), gradWoFused.gradBuffer(), 3e-2f, "gradWo");
    }

    private static void referenceMultiHeadAttentionWithRoPEBackward(
            Tensor gradOut,
            Tensor xNorm,
            Tensor Wq,
            Tensor Wk,
            Tensor Wv,
            Tensor Wo,
            int numHeads,
            Tensor mask,
            BlockActivationCache cache,
            Tensor gradX,
            Tensor gradWq,
            Tensor gradWk,
            Tensor gradWv,
            Tensor gradWo) {
        int[] xs = xNorm.getShape();
        int batch = xs[0];
        int seqLen = xs[1];
        int dModel = xs[2];
        int dHead = dModel / numHeads;
        float attScale = 1.0f / (float) Math.sqrt(dHead);

        Tensor qHeads = cache.attnQHeads.getTensor();
        Tensor kHeads = cache.attnKHeads.getTensor();
        Tensor vHeads = cache.attnVHeads.getTensor();
        Tensor concat = cache.attnConcat.getTensor();

        Tensor xFlat = Tensor.wrap(xNorm.internalBuffer(), new int[] {batch * seqLen, dModel});
        Tensor gradOutFlat = Tensor.wrap(gradOut.gradBuffer(), new int[] {batch * seqLen, dModel});
        Tensor gradConcatFlat = TensorOps.matmul(gradOutFlat, TensorOpsBackward.transpose(Wo));
        Tensor gradConcatData = Tensor.wrap(gradConcatFlat.internalBuffer(), xs);

        Tensor dHeads4 = TensorOps.splitHeads(gradConcatData, numHeads);
        Tensor dHeads3 = Tensor.wrap(dHeads4.internalBuffer(), new int[] {batch * numHeads, seqLen, dHead});
        dHeads3.zeroGrad();
        System.arraycopy(dHeads4.internalBuffer(), 0, dHeads3.gradBuffer(), 0, dHeads4.internalBuffer().length);

        Tensor qHeads3 = Tensor.wrap(qHeads.internalBuffer(), new int[] {batch * numHeads, seqLen, dHead});
        Tensor kHeads3 = Tensor.wrap(kHeads.internalBuffer(), new int[] {batch * numHeads, seqLen, dHead});
        Tensor vHeads3 = Tensor.wrap(vHeads.internalBuffer(), new int[] {batch * numHeads, seqLen, dHead});

        Tensor gradQh3 = new Tensor(new int[] {batch * numHeads, seqLen, dHead});
        Tensor gradKh3 = new Tensor(new int[] {batch * numHeads, seqLen, dHead});
        Tensor gradVh3 = new Tensor(new int[] {batch * numHeads, seqLen, dHead});
        gradQh3.zeroGrad();
        gradKh3.zeroGrad();
        gradVh3.zeroGrad();

        TransformerBackward.scaledDotProductAttentionBackward(
                dHeads3,
                qHeads3,
                kHeads3,
                vHeads3,
                mask,
                attScale,
                cache.attnProbs.getTensor(),
                gradQh3,
                gradKh3,
                gradVh3);

        Tensor gradQh = Tensor.wrap(gradQh3.gradBuffer(), qHeads.getShape());
        Tensor gradKh = Tensor.wrap(gradKh3.gradBuffer(), kHeads.getShape());

        Tensor dqMerge = new Tensor(gradQh.getShape());
        Tensor dkMerge = new Tensor(gradKh.getShape());
        dqMerge.zeroGrad();
        dkMerge.zeroGrad();
        TransformerBackward.applyRoPEBackward(gradQh, dqMerge, null);
        TransformerBackward.applyRoPEBackward(gradKh, dkMerge, null);

        Tensor dQ = TensorOps.concatHeads(Tensor.wrap(dqMerge.gradBuffer(), dqMerge.getShape()), numHeads);
        Tensor dK = TensorOps.concatHeads(Tensor.wrap(dkMerge.gradBuffer(), dkMerge.getShape()), numHeads);
        Tensor dV = TensorOps.concatHeads(Tensor.wrap(gradVh3.gradBuffer(), vHeads.getShape()), numHeads);

        gradX.zeroGrad();
        Tensor dQFlat = Tensor.wrap(dQ.internalBuffer(), new int[] {batch * seqLen, dModel});
        Tensor dKFlat = Tensor.wrap(dK.internalBuffer(), new int[] {batch * seqLen, dModel});
        Tensor dVFlat = Tensor.wrap(dV.internalBuffer(), new int[] {batch * seqLen, dModel});
        Tensor gradXQ = TensorOps.matmul(dQFlat, TensorOpsBackward.transpose(Wq));
        Tensor gradXK = TensorOps.matmul(dKFlat, TensorOpsBackward.transpose(Wk));
        Tensor gradXV = TensorOps.matmul(dVFlat, TensorOpsBackward.transpose(Wv));
        float[] gx = gradX.gradBuffer();
        float[] gxq = gradXQ.internalBuffer();
        float[] gxk = gradXK.internalBuffer();
        float[] gxv = gradXV.internalBuffer();
        for (int i = 0; i < gx.length; i++) {
            gx[i] += gxq[i] + gxk[i] + gxv[i];
        }

        TensorOpsBackward.accumulateGradientInto(gradWq, TensorOps.matmul(TensorOpsBackward.transpose(xFlat), dQFlat));
        TensorOpsBackward.accumulateGradientInto(gradWk, TensorOps.matmul(TensorOpsBackward.transpose(xFlat), dKFlat));
        TensorOpsBackward.accumulateGradientInto(gradWv, TensorOps.matmul(TensorOpsBackward.transpose(xFlat), dVFlat));
        Tensor concatFlat = Tensor.wrap(concat.internalBuffer(), new int[] {batch * seqLen, dModel});
        TensorOpsBackward.accumulateGradientInto(
                gradWo, TensorOps.matmul(TensorOpsBackward.transpose(concatFlat), gradOutFlat));
    }

    private static void fillRandom(float[] data, Random rng) {
        for (int i = 0; i < data.length; i++) {
            data[i] = rng.nextFloat() * 2f - 1f;
        }
    }

    private static void assertClose(float[] expected, float[] actual, float eps, String name) {
        for (int i = 0; i < expected.length; i++) {
            org.junit.jupiter.api.Assertions.assertEquals(expected[i], actual[i], eps, name + "[" + i + "]");
        }
    }
}
