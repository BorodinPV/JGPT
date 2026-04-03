package com.veles.llm.jgpt.data;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Контракт {@link DataLoader#buildBatchNoAdvance()} + {@link DataLoader#advanceAfterPreparedBatch()} ≡ {@link DataLoader#nextBatch()}.
 */
class DataLoaderBatchContractTest {

    @Test
    void buildBatchNoAdvancePlusAdvanceMatchesNextBatch() {
        BPETokenizer tokenizer =
                BPETokenizer.train(
                        Arrays.asList(
                                "the cat sat on the mat",
                                "the dog ran in the park",
                                "the bird flew over the tree",
                                "the fish swam in the lake"),
                        100);

        int maxSeqLen = 16;
        int batchSize = 2;
        DataLoader loader = new DataLoader(tokenizer, maxSeqLen, batchSize);

        String chunk =
                "the cat sat on the mat. the dog ran in the park. the bird flew over the tree. ";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 30; i++) {
            sb.append(chunk);
        }
        loader.loadText(sb.toString());
        loader.shuffle();

        DataLoader.Batch fromNext = loader.nextBatch();
        loader.setCurrentIndex(0);

        DataLoader.Batch fromPrep = loader.buildBatchNoAdvance();
        loader.advanceAfterPreparedBatch();

        assertArrayEquals(
                fromNext.input.internalBuffer(),
                fromPrep.input.internalBuffer(),
                1e-6f,
                "input");
        assertArrayEquals(
                fromNext.target.internalBuffer(),
                fromPrep.target.internalBuffer(),
                1e-6f,
                "target");
    }
}
