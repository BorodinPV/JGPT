package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class LlmTrainerEvalTokenMeanTest {

    @Test
    void tokenMean_weightsBatchesByValidCount() {
        LlmTrainerEvalAndSample.TokenMeanAcc acc = new LlmTrainerEvalAndSample.TokenMeanAcc();
        acc.addBatchMean(2f, 10);
        acc.addBatchMean(4f, 90);
        assertEquals(100, acc.validTokens());
        assertEquals(3.8f, acc.mean(), 1e-6f);
    }

    @Test
    void tokenMean_skipsEmptyAndNonFinite() {
        LlmTrainerEvalAndSample.TokenMeanAcc acc = new LlmTrainerEvalAndSample.TokenMeanAcc();
        acc.addBatchMean(9f, 0);
        acc.addBatchMean(Float.NaN, 50);
        acc.addBatchMean(1f, 4);
        assertEquals(4, acc.validTokens());
        assertEquals(1f, acc.mean(), 0f);
    }

    @Test
    void tokenMean_emptyIsNaN() {
        LlmTrainerEvalAndSample.TokenMeanAcc acc = new LlmTrainerEvalAndSample.TokenMeanAcc();
        assertTrue(Float.isNaN(acc.mean()));
    }
}
