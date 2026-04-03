package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class KvCacheTest {

    @Test
    void prefillMatchesForward() {
        int vocab = 64;
        int maxSeq = 32;
        int dModel = 32;
        int heads = 4;
        int layers = 2;
        int dFf = 64;
        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf);
        int seqLen = 5;
        Tensor input = new Tensor(new int[]{1, seqLen});
        float[] id = input.internalBuffer();
        for (int i = 0; i < seqLen; i++) {
            id[i] = (i + 3) % vocab;
        }
        Tensor logitsFull = model.forward(input, false);
        KvCache cache = new KvCache(layers, heads, dModel / heads, maxSeq);
        Tensor logitsPre = model.forwardPrefill(input, cache, 0);
        float maxDiff = maxAbsDiff(logitsFull.internalBuffer(), logitsPre.internalBuffer());
        assertTrue(maxDiff < 1e-3f, "prefill vs forward max diff " + maxDiff);
    }

    @Test
    void decodeMatchesSecondTokenLogits() {
        int vocab = 64;
        int maxSeq = 32;
        int dModel = 32;
        int heads = 4;
        int layers = 2;
        int dFf = 64;
        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf);

        Tensor both = new Tensor(new int[]{1, 2});
        both.internalBuffer()[0] = 7;
        both.internalBuffer()[1] = 11;

        Tensor logitsFull = model.forward(both, false);
        float[] full = logitsFull.internalBuffer();
        int vs = vocab;
        float[] row1 = new float[vs];
        System.arraycopy(full, vs, row1, 0, vs);

        KvCache cache = new KvCache(layers, heads, dModel / heads, maxSeq);
        Tensor first = new Tensor(new int[]{1, 1});
        first.internalBuffer()[0] = 7;
        model.forwardPrefill(first, cache, 0);
        Tensor second = new Tensor(new int[]{1, 1});
        second.internalBuffer()[0] = 11;
        Tensor logitsDec = model.forwardDecode(second, cache, 1, 1);
        float[] dec = logitsDec.internalBuffer();

        float maxDiff = 0f;
        for (int i = 0; i < vs; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(row1[i] - dec[i]));
        }
        assertTrue(maxDiff < 1e-3f, "decode vs forward pos1 max diff " + maxDiff);
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    @Test
    void rejectsInvalidConstructor() {
        assertThrows(IllegalArgumentException.class, () -> new KvCache(0, 2, 8, 32));
        assertThrows(IllegalArgumentException.class, () -> new KvCache(2, 0, 8, 32));
        assertThrows(IllegalArgumentException.class, () -> new KvCache(2, 2, 0, 32));
        assertThrows(IllegalArgumentException.class, () -> new KvCache(2, 2, 8, 0));
    }

    @Test
    void getKgetVBounds() {
        KvCache c = new KvCache(2, 2, 8, 32);
        assertThrows(IndexOutOfBoundsException.class, () -> c.getK(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> c.getK(2));
    }

    @Test
    void setLengthAndCapacity() {
        KvCache c = new KvCache(1, 2, 8, 10);
        assertEquals(10, c.maxSeqLen());
        assertEquals(0, c.length());
        assertEquals(10, c.remainingCapacity());
        assertFalse(c.isFull());
        c.setLength(7);
        assertEquals(7, c.length());
        assertEquals(3, c.remainingCapacity());
        assertThrows(IllegalArgumentException.class, () -> c.setLength(11));
        assertThrows(IllegalArgumentException.class, () -> c.setLength(-1));
    }

    @Test
    void clearZerosBuffers() {
        KvCache c = new KvCache(1, 1, 2, 4);
        float[] kb = c.getK(0).internalBuffer();
        kb[0] = 3f;
        c.setLength(2);
        c.clear();
        assertEquals(0, c.length());
        assertEquals(0f, kb[0]);
    }
}
