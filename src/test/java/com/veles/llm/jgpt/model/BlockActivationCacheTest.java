package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

class BlockActivationCacheTest {

    @Test
    void clearResetsSlots() {
        BlockActivationCache cache = new BlockActivationCache();
        cache.xIn.store(Tensor.fromArray(new float[] {1f, 2f}, new int[] {2}), false);
        assertTrue(cache.xIn.getTensor() != null);
        cache.clear();
        assertNull(cache.xIn.getTensor());
    }

    @Test
    void toStringMentionsFlags() {
        BlockActivationCache cache = new BlockActivationCache();
        cache.useFp16ActivationStorage = true;
        cache.preferFp32ForFusedGpuBackwardSlots = false;
        assertTrue(cache.toString().contains("useFp16ActivationStorage=true"));
    }
}
