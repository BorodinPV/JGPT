package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.concurrent.ThreadLocalRandom;

/** Авторегрессивная генерация и сэмплинг следующего токена (batch=1). */
final class GptAutoregressiveGenerator {

    private static final Logger log = LoggerFactory.getLogger(GptAutoregressiveGenerator.class);

    private GptAutoregressiveGenerator() {}

    static Tensor generateHost(GPTModel m, Tensor inputTokens, int maxNewTokens, float temperature, int topK) {
        int[] inputShape = inputTokens.getShape();
        int batch = inputShape[0];
        int seqLen = inputShape[1];

        if (batch != 1) {
            throw new IllegalArgumentException("generate currently supports batch_size=1 only");
        }

        Tensor output = new Tensor(new int[] {1, seqLen + maxNewTokens});
        float[] outData = output.internalBuffer();
        float[] inData = inputTokens.internalBuffer();
        System.arraycopy(inData, 0, outData, 0, seqLen);

        int dHead = m.dModel / m.numHeads;
        KvCache cache = new KvCache(m.numLayers, m.numHeads, dHead, m.maxSeqLen);

        Tensor logitsPrefill = GptKvForward.forwardPrefillHost(m, inputTokens, cache, 0);
        Tensor lastPlane = GptTensorBatchPlanes.sliceBatch3D(logitsPrefill, 0);
        float[] lastLogitData = lastPlane.internalBuffer();
        int lastRowOffset = (seqLen - 1) * m.vocabSize;

        int nextToken = sampleNextToken(m, lastLogitData, lastRowOffset, m.vocabSize, temperature, topK);
        outData[seqLen] = nextToken;
        if (nextToken == 0) {
            return output;
        }

        for (int j = 1; j < maxNewTokens; j++) {
            int currentLen = seqLen + j;

            if (currentLen > m.maxSeqLen) {
                int startIdx = currentLen - m.maxSeqLen;
                int sliceLen = m.maxSeqLen - startIdx;
                if (sliceLen <= 0) {
                    throw new IllegalStateException(
                            "sliding window: startIdx="
                                    + startIdx
                                    + " maxSeqLen="
                                    + m.maxSeqLen
                                    + " (увеличьте max_seq_len модели или уменьшите длину генерации)");
                }
                cache.clear();
                log.warn(
                        "Скользящее окно KV (кэш на хосте): полный prefill по {} токенам (позиции {}..{}). "
                                + "Каждое срабатывание — O(окно²); для длинных прогонов увеличьте max_seq_len или используйте paged/rolling KV.",
                        sliceLen,
                        startIdx,
                        startIdx + sliceLen - 1);
                if (m.reusableSlidingPrefillInput == null
                        || m.reusableSlidingPrefillInput.getShape()[0] != 1
                        || m.reusableSlidingPrefillInput.getShape()[1] != sliceLen) {
                    m.reusableSlidingPrefillInput = new Tensor(new int[] {1, sliceLen});
                }
                float[] sliceData = m.reusableSlidingPrefillInput.internalBuffer();
                for (int t = 0; t < sliceLen; t++) {
                    sliceData[t] = outData[startIdx + t];
                }
                logitsPrefill = GptKvForward.forwardPrefillHost(m, m.reusableSlidingPrefillInput, cache, startIdx);
                lastPlane = GptTensorBatchPlanes.sliceBatch3D(logitsPrefill, 0);
                lastLogitData = lastPlane.internalBuffer();
                lastRowOffset = (sliceLen - 1) * m.vocabSize;
                nextToken = sampleNextToken(m, lastLogitData, lastRowOffset, m.vocabSize, temperature, topK);
                outData[currentLen] = nextToken;
                if (nextToken == 0) {
                    break;
                }
                continue;
            }

            if (m.reusableDecodeOneToken == null) {
                m.reusableDecodeOneToken = new Tensor(new int[] {1, 1});
            }
            m.reusableDecodeOneToken.internalBuffer()[0] = outData[seqLen + j - 1];
            Tensor logitsDec =
                    GptKvForward.forwardDecodeHost(m, m.reusableDecodeOneToken, cache, cache.length(), seqLen + j - 1);
            lastPlane = GptTensorBatchPlanes.sliceBatch3D(logitsDec, 0);
            lastLogitData = lastPlane.internalBuffer();
            nextToken = sampleNextToken(m, lastLogitData, 0, m.vocabSize, temperature, topK);
            outData[currentLen] = nextToken;
            if (nextToken == 0) {
                break;
            }
        }

        return output;
    }

    static Tensor generateGpuKv(GPTModel m, Tensor inputTokens, int maxNewTokens, float temperature, int topK) {
        if (!m.isGpuResident()) {
            throw new IllegalStateException("generateGpuKv requires GPU-resident weights");
        }
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("generateGpuKv requires CUDA");
        }
        int[] inputShape = inputTokens.getShape();
        int batch = inputShape[0];
        int seqLen = inputShape[1];

        if (batch != 1) {
            throw new IllegalArgumentException("generateGpuKv currently supports batch_size=1 only");
        }

        Tensor output = new Tensor(new int[] {1, seqLen + maxNewTokens});
        float[] outData = output.internalBuffer();
        float[] inData = inputTokens.internalBuffer();
        System.arraycopy(inData, 0, outData, 0, seqLen);

        int dHead = m.dModel / m.numHeads;
        try (KvCacheGpu cache = new KvCacheGpu(m.numLayers, m.numHeads, dHead, m.maxSeqLen)) {
            try {
                Tensor logitsPrefill = GptKvForward.forwardPrefillGpu(m, inputTokens, cache, 0);
                Tensor lastPlane = GptTensorBatchPlanes.sliceBatch3D(logitsPrefill, 0);
                float[] lastLogitData = lastPlane.internalBuffer();
                int lastRowOffset = (seqLen - 1) * m.vocabSize;

                int nextToken = sampleNextToken(m, lastLogitData, lastRowOffset, m.vocabSize, temperature, topK);
                outData[seqLen] = nextToken;
                if (nextToken == 0) {
                    return output;
                }

                for (int j = 1; j < maxNewTokens; j++) {
                    int currentLen = seqLen + j;

                    if (currentLen > m.maxSeqLen) {
                        int startIdx = currentLen - m.maxSeqLen;
                        int sliceLen = m.maxSeqLen - startIdx;
                        if (sliceLen <= 0) {
                            throw new IllegalStateException(
                                    "sliding window: startIdx="
                                            + startIdx
                                            + " maxSeqLen="
                                            + m.maxSeqLen
                                            + " (увеличьте max_seq_len модели или уменьшите длину генерации)");
                        }
                        cache.clear();
                        log.warn(
                                "Скользящее окно KV (кэш в VRAM): полный prefill по {} токенам (позиции {}..{}). "
                                        + "Каждое срабатывание — O(окно²); для длинных прогонов увеличьте max_seq_len или используйте paged/rolling KV.",
                                sliceLen,
                                startIdx,
                                startIdx + sliceLen - 1);
                        if (m.reusableSlidingPrefillInput == null
                                || m.reusableSlidingPrefillInput.getShape()[0] != 1
                                || m.reusableSlidingPrefillInput.getShape()[1] != sliceLen) {
                            m.reusableSlidingPrefillInput = new Tensor(new int[] {1, sliceLen});
                        }
                        float[] sliceData = m.reusableSlidingPrefillInput.internalBuffer();
                        for (int t = 0; t < sliceLen; t++) {
                            sliceData[t] = outData[startIdx + t];
                        }
                        logitsPrefill = GptKvForward.forwardPrefillGpu(m, m.reusableSlidingPrefillInput, cache, startIdx);
                        lastPlane = GptTensorBatchPlanes.sliceBatch3D(logitsPrefill, 0);
                        lastLogitData = lastPlane.internalBuffer();
                        lastRowOffset = (sliceLen - 1) * m.vocabSize;
                        nextToken =
                                sampleNextToken(m, lastLogitData, lastRowOffset, m.vocabSize, temperature, topK);
                        outData[currentLen] = nextToken;
                        if (nextToken == 0) {
                            break;
                        }
                        continue;
                    }

                    if (m.reusableDecodeOneToken == null) {
                        m.reusableDecodeOneToken = new Tensor(new int[] {1, 1});
                    }
                    m.reusableDecodeOneToken.internalBuffer()[0] = outData[seqLen + j - 1];
                    Tensor logitsDec =
                            GptKvForward.forwardDecodeGpu(
                                    m, m.reusableDecodeOneToken, cache, cache.length(), seqLen + j - 1);
                    lastPlane = GptTensorBatchPlanes.sliceBatch3D(logitsDec, 0);
                    lastLogitData = lastPlane.internalBuffer();
                    nextToken = sampleNextToken(m, lastLogitData, 0, m.vocabSize, temperature, topK);
                    outData[currentLen] = nextToken;
                    if (nextToken == 0) {
                        break;
                    }
                }
            } finally {
                TensorOpsGPU.synchronizeStream();
            }
        }

        return output;
    }

    static int sampleNextToken(
            GPTModel m, float[] logits, int offset, int vocabSize, float temperature, int topK) {
        if (m.sampleLogitsScratch == null || m.sampleLogitsScratch.length < vocabSize) {
            m.sampleLogitsScratch = new float[vocabSize];
        }
        System.arraycopy(logits, offset, m.sampleLogitsScratch, 0, vocabSize);

        if (temperature != 1.0f && temperature > 0) {
            for (int i = 0; i < vocabSize; i++) {
                m.sampleLogitsScratch[i] /= temperature;
            }
        }

        if (topK > 0 && topK < vocabSize) {
            PriorityQueue<Integer> worstOfTop =
                    new PriorityQueue<>(
                            topK,
                            (a, b) -> {
                                int c = Float.compare(m.sampleLogitsScratch[a], m.sampleLogitsScratch[b]);
                                if (c != 0) {
                                    return c;
                                }
                                return Integer.compare(b, a);
                            });
            for (int i = 0; i < vocabSize; i++) {
                if (worstOfTop.size() < topK) {
                    worstOfTop.offer(i);
                } else {
                    int w = worstOfTop.peek();
                    if (isBetterLogit(m.sampleLogitsScratch, i, w)) {
                        worstOfTop.poll();
                        worstOfTop.offer(i);
                    }
                }
            }
            if (m.sampleTopKMember == null || m.sampleTopKMember.length < vocabSize) {
                m.sampleTopKMember = new boolean[vocabSize];
            }
            Arrays.fill(m.sampleTopKMember, 0, vocabSize, false);
            while (!worstOfTop.isEmpty()) {
                m.sampleTopKMember[worstOfTop.poll()] = true;
            }
            for (int i = 0; i < vocabSize; i++) {
                if (!m.sampleTopKMember[i]) {
                    m.sampleLogitsScratch[i] = Float.NEGATIVE_INFINITY;
                }
            }
        }

        if (temperature <= 0f) {
            return argmaxLogitsGreedy(m.sampleLogitsScratch, vocabSize);
        }

        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocabSize; i++) {
            max = Math.max(max, m.sampleLogitsScratch[i]);
        }

        float sum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            float e = (float) Math.exp(m.sampleLogitsScratch[i] - max);
            m.sampleLogitsScratch[i] = e;
            sum += e;
        }
        for (int i = 0; i < vocabSize; i++) {
            m.sampleLogitsScratch[i] /= sum;
        }

        float rand = (float) ThreadLocalRandom.current().nextDouble();
        float cumsum = 0f;
        for (int i = 0; i < vocabSize; i++) {
            cumsum += m.sampleLogitsScratch[i];
            if (rand <= cumsum) {
                return i;
            }
        }

        return vocabSize - 1;
    }

    private static int argmaxLogitsGreedy(float[] logits, int vocabSize) {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < vocabSize; i++) {
            float v = logits[i];
            if (v > bestVal || (v == bestVal && i < best)) {
                bestVal = v;
                best = i;
            }
        }
        return best;
    }

    private static boolean isBetterLogit(float[] vals, int i, int j) {
        int c = Float.compare(vals[i], vals[j]);
        if (c != 0) {
            return c > 0;
        }
        return i < j;
    }
}
