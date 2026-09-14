package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** Token-mean CE across an accumulation window: provisional denom and post-window rescale. */
class LlmTrainerAccumTokenMeanTest {

    @Test
    void windowDenom_isBatchTimesSeqTimesAccum() {
        assertEquals(4 * 1024 * 8, LlmTrainerCrossEntropy.accumWindowTokenDenom(4, 1024, 8));
        assertEquals(1, LlmTrainerCrossEntropy.accumWindowTokenDenom(0, 0, 0));
    }

    @Test
    void packedFullWindow_rescaleIsOne() {
        int window = LlmTrainerCrossEntropy.accumWindowTokenDenom(4, 1024, 8);
        assertEquals(1f, LlmTrainerCrossEntropy.tokenMeanRescale(window, window), 0f);
    }

    @Test
    void incompleteWindowAllValid_matchesOldPartialAccumScale() {
        int batch = 2;
        int seq = 8;
        int accum = 3;
        int micros = 2;
        int window = LlmTrainerCrossEntropy.accumWindowTokenDenom(batch, seq, accum);
        int nValid = batch * seq * micros;
        assertEquals(
                (float) accum / (float) micros,
                LlmTrainerCrossEntropy.tokenMeanRescale(window, nValid),
                1e-6f);
    }

    @Test
    void shortVsLongMicrobatch_weightsByTokenCount() {
        int window = LlmTrainerCrossEntropy.accumWindowTokenDenom(4, 1024, 2);
        int nShort = 20;
        int nLong = 400;
        float scale = LlmTrainerCrossEntropy.tokenMeanRescale(window, nShort + nLong);
        assertEquals(window / 420f, scale, 1e-5f);
    }

    @Test
    void emptyWindow_noRescale() {
        assertEquals(1f, LlmTrainerCrossEntropy.tokenMeanRescale(32768, 0), 0f);
    }

    @Test
    void validCount_skipsNegativeTargets() {
        int[] ids = {1, -1, 3, -1, 5};
        assertEquals(3, LlmTrainerCrossEntropy.countValidCeTargets(ids, ids.length));
        assertEquals(1, LlmTrainerCrossEntropy.countValidCeTargets(ids, 2));
    }
}
