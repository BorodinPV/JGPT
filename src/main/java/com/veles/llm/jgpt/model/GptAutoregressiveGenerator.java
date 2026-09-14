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

    private static Tensor copyPromptSequence(Tensor inputTokens, int seqLen) {
        Tensor output = new Tensor(new int[] {1, seqLen});
        System.arraycopy(inputTokens.internalBuffer(), 0, output.internalBuffer(), 0, seqLen);
        return output;
    }

    static Tensor generateHost(GPTModel m, Tensor inputTokens, int maxNewTokens, float temperature, int topK) {
        return generateHost(m, inputTokens, maxNewTokens, DecodeSampling.of(temperature, topK));
    }

    static Tensor generateHost(GPTModel m, Tensor inputTokens, int maxNewTokens, DecodeSampling sampling) {
        int[] inputShape = inputTokens.getShape();
        int batch = inputShape[0];
        int seqLen = inputShape[1];

        if (batch != 1) {
            throw new IllegalArgumentException("generate currently supports batch_size=1 only");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must be >= 0");
        }
        if (maxNewTokens == 0) {
            return copyPromptSequence(inputTokens, seqLen);
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

        int nextToken =
                sampleNextToken(m, lastLogitData, lastRowOffset, m.vocabSize, sampling, outData, seqLen);
        outData[seqLen] = nextToken;
        if (isGenerationStopToken(m, nextToken)) {
            return output;
        }

        for (int j = 1; j < maxNewTokens; j++) {
            int currentLen = seqLen + j;

            if (currentLen > m.maxSeqLen) {
                int startIdx = currentLen - m.maxSeqLen;
                int sliceLen = m.maxSeqLen;
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
                        "Скользящее окно KV (кэш на хосте): полный prefill по {} последним токенам ({}..{}), позиции 0..{}. "
                                + "Каждое срабатывание — O(окно²); для длинных прогонов увеличьте max_seq_len или используйте paged/rolling KV.",
                        sliceLen,
                        startIdx,
                        startIdx + sliceLen - 1,
                        sliceLen - 1);
                if (m.reusableSlidingPrefillInput == null
                        || m.reusableSlidingPrefillInput.getShape()[0] != 1
                        || m.reusableSlidingPrefillInput.getShape()[1] != sliceLen) {
                    m.reusableSlidingPrefillInput = new Tensor(new int[] {1, sliceLen});
                }
                float[] sliceData = m.reusableSlidingPrefillInput.internalBuffer();
                for (int t = 0; t < sliceLen; t++) {
                    sliceData[t] = outData[startIdx + t];
                }
                /*
                 * Окно с ropeOffset=0: E_pos имеет ровно maxSeqLen строк (startIdx+S вышло бы за таблицу).
                 * RoPE относителен — сдвиг всего окна на -startIdx не меняет attention внутри окна.
                 * continue: после этого prefill decode с глобальной позицией currentLen не вызывается.
                 */
                logitsPrefill = GptKvForward.forwardPrefillHost(m, m.reusableSlidingPrefillInput, cache, 0);
                lastPlane = GptTensorBatchPlanes.sliceBatch3D(logitsPrefill, 0);
                lastLogitData = lastPlane.internalBuffer();
                lastRowOffset = (sliceLen - 1) * m.vocabSize;
                nextToken =
                        sampleNextToken(
                                m, lastLogitData, lastRowOffset, m.vocabSize, sampling, outData, currentLen);
                outData[currentLen] = nextToken;
                if (isGenerationStopToken(m, nextToken)) {
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
            nextToken = sampleNextToken(m, lastLogitData, 0, m.vocabSize, sampling, outData, currentLen);
            outData[currentLen] = nextToken;
            if (isGenerationStopToken(m, nextToken)) {
                break;
            }
        }

        return output;
    }

    static Tensor generateGpuKv(GPTModel m, Tensor inputTokens, int maxNewTokens, float temperature, int topK) {
        return generateGpuKv(m, inputTokens, maxNewTokens, DecodeSampling.of(temperature, topK));
    }

    static Tensor generateGpuKv(GPTModel m, Tensor inputTokens, int maxNewTokens, DecodeSampling sampling) {
        int[] inputShape = inputTokens.getShape();
        int batch = inputShape[0];
        int seqLen = inputShape[1];

        if (batch != 1) {
            throw new IllegalArgumentException("generateGpuKv currently supports batch_size=1 only");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must be >= 0");
        }
        if (maxNewTokens == 0) {
            return copyPromptSequence(inputTokens, seqLen);
        }
        if (!m.isGpuResident()) {
            throw new IllegalStateException("generateGpuKv requires GPU-resident weights");
        }
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("generateGpuKv requires CUDA");
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

                int nextToken =
                        sampleNextToken(m, lastLogitData, lastRowOffset, m.vocabSize, sampling, outData, seqLen);
                outData[seqLen] = nextToken;
                if (isGenerationStopToken(m, nextToken)) {
                    return output;
                }

                for (int j = 1; j < maxNewTokens; j++) {
                    int currentLen = seqLen + j;

                    if (currentLen > m.maxSeqLen) {
                        int startIdx = currentLen - m.maxSeqLen;
                        int sliceLen = m.maxSeqLen;
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
                                "Скользящее окно KV (кэш в VRAM): полный prefill по {} последним токенам ({}..{}), позиции 0..{}. "
                                        + "Каждое срабатывание — O(окно²); для длинных прогонов увеличьте max_seq_len или используйте paged/rolling KV.",
                                sliceLen,
                                startIdx,
                                startIdx + sliceLen - 1,
                                sliceLen - 1);
                        if (m.reusableSlidingPrefillInput == null
                                || m.reusableSlidingPrefillInput.getShape()[0] != 1
                                || m.reusableSlidingPrefillInput.getShape()[1] != sliceLen) {
                            m.reusableSlidingPrefillInput = new Tensor(new int[] {1, sliceLen});
                        }
                        float[] sliceData = m.reusableSlidingPrefillInput.internalBuffer();
                        for (int t = 0; t < sliceLen; t++) {
                            sliceData[t] = outData[startIdx + t];
                        }
                        // ropeOffset=0, затем continue (см. host-вариант).
                        logitsPrefill = GptKvForward.forwardPrefillGpu(m, m.reusableSlidingPrefillInput, cache, 0);
                        lastPlane = GptTensorBatchPlanes.sliceBatch3D(logitsPrefill, 0);
                        lastLogitData = lastPlane.internalBuffer();
                        lastRowOffset = (sliceLen - 1) * m.vocabSize;
                        nextToken =
                                sampleNextToken(
                                        m, lastLogitData, lastRowOffset, m.vocabSize, sampling, outData, currentLen);
                        outData[currentLen] = nextToken;
                        if (isGenerationStopToken(m, nextToken)) {
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
                    nextToken = sampleNextToken(m, lastLogitData, 0, m.vocabSize, sampling, outData, currentLen);
                    outData[currentLen] = nextToken;
                    if (isGenerationStopToken(m, nextToken)) {
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
        return sampleNextToken(
                m, logits, offset, vocabSize, DecodeSampling.of(temperature, topK), null, 0);
    }

    static int sampleNextToken(
            GPTModel m,
            float[] logits,
            int offset,
            int vocabSize,
            DecodeSampling sampling,
            float[] tokens,
            int tokenLen) {
        if (m.sampleLogitsScratch == null || m.sampleLogitsScratch.length < vocabSize) {
            m.sampleLogitsScratch = new float[vocabSize];
        }
        System.arraycopy(logits, offset, m.sampleLogitsScratch, 0, vocabSize);

        applyRepetitionPenalty(m, vocabSize, sampling.repetitionPenalty, tokens, tokenLen);
        banRepeatingNgrams(m.sampleLogitsScratch, vocabSize, sampling.noRepeatNgramSize, tokens, tokenLen);

        float temperature = sampling.temperature;
        int topK = sampling.topK;
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

        applyNucleus(m, vocabSize, sampling.topP);

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
        if (sum <= 0f || !Float.isFinite(sum)) {
            return argmaxLogitsGreedy(m.sampleLogitsScratch, vocabSize);
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

    private static void applyRepetitionPenalty(
            GPTModel m, int vocabSize, float penalty, float[] tokens, int tokenLen) {
        if (penalty == 1f || tokens == null || tokenLen <= 0) {
            return;
        }
        if (m.sampleSeenScratch == null || m.sampleSeenScratch.length < vocabSize) {
            m.sampleSeenScratch = new boolean[vocabSize];
        }
        Arrays.fill(m.sampleSeenScratch, 0, vocabSize, false);
        for (int t = 0; t < tokenLen; t++) {
            int id = (int) tokens[t];
            if (id >= 0 && id < vocabSize) {
                m.sampleSeenScratch[id] = true;
            }
        }
        float[] scores = m.sampleLogitsScratch;
        for (int i = 0; i < vocabSize; i++) {
            if (!m.sampleSeenScratch[i]) {
                continue;
            }
            float s = scores[i];
            scores[i] = s < 0f ? s * penalty : s / penalty;
        }
    }

    private static void banRepeatingNgrams(
            float[] logits, int vocabSize, int ngram, float[] tokens, int tokenLen) {
        if (ngram < 2 || tokens == null || tokenLen < ngram - 1) {
            return;
        }
        int prefix = ngram - 1;
        for (int i = 0; i + ngram <= tokenLen; i++) {
            boolean match = true;
            for (int k = 0; k < prefix; k++) {
                if ((int) tokens[i + k] != (int) tokens[tokenLen - prefix + k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                int ban = (int) tokens[i + prefix];
                if (ban >= 0 && ban < vocabSize) {
                    logits[ban] = Float.NEGATIVE_INFINITY;
                }
            }
        }
    }

    private static void applyNucleus(GPTModel m, int vocabSize, float topP) {
        if (!(topP > 0f && topP < 1f)) {
            return;
        }
        if (m.sampleIndexScratch == null || m.sampleIndexScratch.length < vocabSize) {
            m.sampleIndexScratch = new int[vocabSize];
        }
        float[] logits = m.sampleLogitsScratch;
        int n = 0;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocabSize; i++) {
            float v = logits[i];
            if (!Float.isFinite(v)) {
                continue;
            }
            m.sampleIndexScratch[n++] = i;
            max = Math.max(max, v);
        }
        if (n <= 1) {
            return;
        }
        float sum = 0f;
        for (int k = 0; k < n; k++) {
            int i = m.sampleIndexScratch[k];
            float e = (float) Math.exp(logits[i] - max);
            logits[i] = e;
            sum += e;
        }
        if (sum <= 0f || !Float.isFinite(sum)) {
            return;
        }
        for (int k = 0; k < n; k++) {
            int i = m.sampleIndexScratch[k];
            logits[i] /= sum;
        }
        quicksortIdxDesc(m.sampleIndexScratch, logits, 0, n - 1);
        float cum = 0f;
        int keep = 0;
        while (keep < n) {
            cum += logits[m.sampleIndexScratch[keep]];
            keep++;
            if (cum >= topP) {
                break;
            }
        }
        for (int k = keep; k < n; k++) {
            logits[m.sampleIndexScratch[k]] = Float.NEGATIVE_INFINITY;
        }
        for (int k = 0; k < keep; k++) {
            int i = m.sampleIndexScratch[k];
            logits[i] = (float) Math.log(Math.max(logits[i], 1e-20f));
        }
    }

    private static void quicksortIdxDesc(int[] idx, float[] vals, int lo, int hi) {
        while (lo < hi) {
            int p = partitionIdxDesc(idx, vals, lo, hi);
            if (p - lo < hi - p) {
                quicksortIdxDesc(idx, vals, lo, p - 1);
                lo = p + 1;
            } else {
                quicksortIdxDesc(idx, vals, p + 1, hi);
                hi = p - 1;
            }
        }
    }

    private static int partitionIdxDesc(int[] idx, float[] vals, int lo, int hi) {
        int pivotId = idx[hi];
        float pivot = vals[pivotId];
        int i = lo;
        for (int j = lo; j < hi; j++) {
            int a = idx[j];
            int c = Float.compare(vals[a], pivot);
            if (c > 0 || (c == 0 && a < pivotId)) {
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
                i++;
            }
        }
        int tmp = idx[i];
        idx[i] = idx[hi];
        idx[hi] = tmp;
        return i;
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

    /**
     * {@code <pad>=0} (хвост буфера) и {@code <eos>=3} — конец реплики, как при SFT; плюс
     * {@link GPTModel#setExtraGenerationStopTokens} (ролевые токены при чат-шаблоне).
     */
    static boolean isGenerationStopToken(GPTModel m, int token) {
        if (token == 0 || token == 3) {
            return true;
        }
        for (int t : m.extraGenerationStopTokens()) {
            if (t == token) {
                return true;
            }
        }
        return false;
    }

    private static boolean isBetterLogit(float[] vals, int i, int j) {
        int c = Float.compare(vals[i], vals[j]);
        if (c != 0) {
            return c > 0;
        }
        return i < j;
    }
}
