package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

/**
 * Сравнение полного инференса с {@link KvCache} (K/V в host-тензорах в attention) и {@link KvCacheGpu}
 * (K/V только на VRAM в attention). Требуется GPU и {@code gpuResident} веса декодера.
 */
class GptKvCacheGpuParityTest {

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    @Test
    void prefill_logitsMatchBetweenHostKvAndVramKv() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int vocab = 64;
        int maxSeq = 32;
        int dModel = 32;
        int heads = 4;
        int layers = 2;
        int dFf = 64;
        int seqLen = 5;

        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);
        assertTrue(model.isGpuResident(), "test expects gpu-resident decoder weights");

        Tensor input = new Tensor(new int[] {1, seqLen});
        float[] id = input.internalBuffer();
        for (int i = 0; i < seqLen; i++) {
            id[i] = (i + 3) % vocab;
        }

        KvCache cacheHost = new KvCache(layers, heads, dModel / heads, maxSeq);
        Tensor logitsHost = model.forwardPrefill(input, cacheHost, 0);

        try (KvCacheGpu cacheGpu = new KvCacheGpu(layers, heads, dModel / heads, maxSeq)) {
            Tensor logitsGpu = model.forwardPrefill(input, cacheGpu, 0);
            float d = maxAbsDiff(logitsHost.internalBuffer(), logitsGpu.internalBuffer());
            assertTrue(d < 5e-3f, "prefill logits host-KV vs VRAM-KV max abs diff " + d);
        } finally {
            model.closeGpuResidentWeights();
        }
    }

    @Test
    void decode_logitsMatchBetweenHostKvAndVramKv() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int vocab = 64;
        int maxSeq = 32;
        int dModel = 32;
        int heads = 4;
        int layers = 2;
        int dFf = 64;

        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);
        assertTrue(model.isGpuResident(), "test expects gpu-resident decoder weights");

        Tensor first = new Tensor(new int[] {1, 1});
        first.internalBuffer()[0] = 7;
        Tensor second = new Tensor(new int[] {1, 1});
        second.internalBuffer()[0] = 11;

        KvCache cacheHost = new KvCache(layers, heads, dModel / heads, maxSeq);
        model.forwardPrefill(first, cacheHost, 0);
        Tensor logitsHost = model.forwardDecode(second, cacheHost, 1, 1);

        try (KvCacheGpu cacheGpu = new KvCacheGpu(layers, heads, dModel / heads, maxSeq)) {
            model.forwardPrefill(first, cacheGpu, 0);
            Tensor logitsGpu = model.forwardDecode(second, cacheGpu, 1, 1);
            float d = maxAbsDiff(logitsHost.internalBuffer(), logitsGpu.internalBuffer());
            assertTrue(d < 5e-3f, "decode logits host-KV vs VRAM-KV max abs diff " + d);
        } finally {
            model.closeGpuResidentWeights();
        }
    }
}
