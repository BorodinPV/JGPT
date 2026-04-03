package com.veles.llm.jgpt.data;

import static org.junit.jupiter.api.Assertions.assertNotSame;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class DataLoaderBatchReuseTest {

    @Test
    void consecutiveBatchesUseAlternatingTensorSlots() {
        BPETokenizer tokenizer =
                BPETokenizer.train(
                        Arrays.asList(
                                "the cat sat on the mat",
                                "the dog ran in the park",
                                "the bird flew over the tree",
                                "the fish swam in the lake"),
                        100);

        int maxSeqLen = 8;
        int batchSize = 2;
        DataLoader loader = new DataLoader(tokenizer, maxSeqLen, batchSize);

        String chunk = "the cat sat on the mat. the dog ran in the park. ";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 40; i++) {
            sb.append(chunk);
        }
        loader.loadText(sb.toString());
        loader.shuffle();

        DataLoader.Batch first = loader.nextBatch();
        DataLoader.Batch second = loader.nextBatch();

        assertNotSame(first.input, second.input);
        assertNotSame(first.target, second.target);
    }
}
