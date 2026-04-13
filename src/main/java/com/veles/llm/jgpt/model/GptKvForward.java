package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;

/** Prefill/decode с KV (хост и VRAM). */
final class GptKvForward {

    private GptKvForward() {}

    static Tensor forwardPrefillHost(GPTModel m, Tensor inputTokens, KvCache cache, int ropeOffset) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape[0] != 1) {
            throw new IllegalArgumentException("forwardPrefill supports batch_size=1 only");
        }
        int seqLen = inputShape[1];
        if (seqLen > m.maxSeqLen) {
            throw new IllegalArgumentException("seq_len " + seqLen + " > max_seq_len " + m.maxSeqLen);
        }
        if (cache.numLayers() != m.numLayers) {
            throw new IllegalArgumentException("KvCache layer count mismatch");
        }

        Tensor x = m.tokenEmbedding.forward(inputTokens);
        m.positionEmbedding.addToActivationsInPlace(x, seqLen, ropeOffset);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        for (int i = 0; i < m.numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[i].attnBuffers() : null;
            x = m.blocks[i].forwardKvPrefill(
                    x, mask, null, cache.getK(i), cache.getV(i), ropeOffset, ffnResident, attnResident);
        }

        x = TensorOps.rmsNorm(x, m.layerNormFinal, TensorOpsGPU.rmsNormEps());
        Tensor logits = new Tensor(new int[] {1, seqLen, m.vocabSize});
        GptTensorBatchPlanes.copyBatchPlane(
                logits, TensorOps.matmul(GptTensorBatchPlanes.sliceBatch3D(x, 0), m.lmHead), 0);
        cache.setLength(seqLen);
        return logits;
    }

    static Tensor forwardPrefillGpu(GPTModel m, Tensor inputTokens, KvCacheGpu cache, int ropeOffset) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape[0] != 1) {
            throw new IllegalArgumentException("forwardPrefill supports batch_size=1 only");
        }
        int seqLen = inputShape[1];
        if (seqLen > m.maxSeqLen) {
            throw new IllegalArgumentException("seq_len " + seqLen + " > max_seq_len " + m.maxSeqLen);
        }
        if (cache.numLayers() != m.numLayers) {
            throw new IllegalArgumentException("KvCacheGpu layer count mismatch");
        }
        if (seqLen > cache.maxSeqLen()) {
            throw new IllegalArgumentException("seq_len " + seqLen + " > KvCacheGpu.maxSeqLen " + cache.maxSeqLen());
        }

        Tensor x = m.tokenEmbedding.forward(inputTokens);
        m.positionEmbedding.addToActivationsInPlace(x, seqLen, ropeOffset);

        Tensor mask = TensorOps.createCausalMask(seqLen);
        for (int i = 0; i < m.numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[i].attnBuffers() : null;
            x =
                    m.blocks[i].forwardKvPrefillVram(
                            x,
                            mask,
                            null,
                            cache.getK(i),
                            cache.getV(i),
                            cache.maxSeqLen(),
                            ropeOffset,
                            ffnResident,
                            attnResident);
        }

        x = TensorOps.rmsNorm(x, m.layerNormFinal, TensorOpsGPU.rmsNormEps());
        Tensor logits = new Tensor(new int[] {1, seqLen, m.vocabSize});
        GptTensorBatchPlanes.copyBatchPlane(
                logits, TensorOps.matmul(GptTensorBatchPlanes.sliceBatch3D(x, 0), m.lmHead), 0);
        cache.setLength(seqLen);
        return logits;
    }

    static Tensor forwardDecodeHost(GPTModel m, Tensor inputTokens, KvCache cache, int cacheLenBefore, int ropePosition) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape[0] != 1 || inputShape[1] != 1) {
            throw new IllegalArgumentException("forwardDecode expects [1, 1] token indices");
        }
        if (cache.length() != cacheLenBefore) {
            throw new IllegalStateException(
                    "KV cache length " + cache.length() + " != cacheLenBefore " + cacheLenBefore);
        }

        Tensor x = m.tokenEmbedding.forward(inputTokens);
        m.positionEmbedding.addToActivationsInPlace(x, 1, ropePosition);

        for (int i = 0; i < m.numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[i].attnBuffers() : null;
            x = m.blocks[i].forwardKvDecode(
                    x,
                    null,
                    cache.getK(i),
                    cache.getV(i),
                    cacheLenBefore,
                    ropePosition,
                    ffnResident,
                    attnResident);
        }

        x = TensorOps.rmsNorm(x, m.layerNormFinal, TensorOpsGPU.rmsNormEps());
        Tensor logits = new Tensor(new int[] {1, 1, m.vocabSize});
        GptTensorBatchPlanes.copyBatchPlane(
                logits, TensorOps.matmul(GptTensorBatchPlanes.sliceBatch3D(x, 0), m.lmHead), 0);
        cache.setLength(cacheLenBefore + 1);
        return logits;
    }

    static Tensor forwardDecodeGpu(
            GPTModel m, Tensor inputTokens, KvCacheGpu cache, int cacheLenBefore, int ropePosition) {
        int[] inputShape = inputTokens.getShape();
        if (inputShape[0] != 1 || inputShape[1] != 1) {
            throw new IllegalArgumentException("forwardDecode expects [1, 1] token indices");
        }
        if (cache.length() != cacheLenBefore) {
            throw new IllegalStateException(
                    "KV cache length " + cache.length() + " != cacheLenBefore " + cacheLenBefore);
        }
        if (cacheLenBefore >= cache.maxSeqLen()) {
            throw new IllegalStateException("KV cache full (cacheLenBefore " + cacheLenBefore + ")");
        }

        Tensor x = m.tokenEmbedding.forward(inputTokens);
        m.positionEmbedding.addToActivationsInPlace(x, 1, ropePosition);

        for (int i = 0; i < m.numLayers; i++) {
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[i].ffnBuffers() : null;
            TensorOps.GpuAttnResidentBuffers attnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[i].attnBuffers() : null;
            x =
                    m.blocks[i].forwardKvDecodeVram(
                            x,
                            null,
                            cache.getK(i),
                            cache.getV(i),
                            cache.maxSeqLen(),
                            cacheLenBefore,
                            ropePosition,
                            ffnResident,
                            attnResident);
        }

        x = TensorOps.rmsNorm(x, m.layerNormFinal, TensorOpsGPU.rmsNormEps());
        Tensor logits = new Tensor(new int[] {1, 1, m.vocabSize});
        GptTensorBatchPlanes.copyBatchPlane(
                logits, TensorOps.matmul(GptTensorBatchPlanes.sliceBatch3D(x, 0), m.lmHead), 0);
        cache.setLength(cacheLenBefore + 1);
        return logits;
    }
}
