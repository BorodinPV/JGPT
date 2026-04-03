package com.veles.llm.jgpt.data;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.TensorCudaLibrary;
import java.util.List;
import org.junit.jupiter.api.Test;

class DataLoaderPinnedHostTest {

    @Test
    void pinnedFlag_impliesDirect() {
        BPETokenizer tok = BPETokenizer.train(List.of("hello world", "foo bar"), 32);
        DataLoader loader = new DataLoader(tok, 4, 2, false, true);
        assertTrue(loader.usesPinnedHostBatchBuffers());
        assertTrue(loader.usesDirectBatchBuffers());
    }

    @Test
    void buildBatch_direct_whenPinnedRequested() {
        try {
            TensorCudaLibrary.load();
        } catch (UnsatisfiedLinkError e) {
            assumeTrue(false, "skip: no libjgpt_cuda");
        }
        BPETokenizer tok = BPETokenizer.train(List.of("the cat sat on the mat the dog"), 48);
        DataLoader loader = new DataLoader(tok, 8, 2, false, true);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 20; i++) {
            sb.append("the cat sat on the mat. ");
        }
        loader.loadText(sb.toString());
        DataLoader.Batch b = loader.nextBatch();
        assertTrue(b.input.isDirectStorage());
        assertTrue(b.target.isDirectStorage());
    }

    @Test
    void allocatePinnedHost_reportsPinned_whenSupported() {
        try {
            TensorCudaLibrary.load();
        } catch (UnsatisfiedLinkError e) {
            assumeTrue(false, "skip: no libjgpt_cuda");
        }
        Tensor t = Tensor.allocatePinnedHost(new int[]{2, 3});
        assertTrue(t.isDirectStorage());
    }
}
