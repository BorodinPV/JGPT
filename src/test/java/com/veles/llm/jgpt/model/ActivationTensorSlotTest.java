package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

class ActivationTensorSlotTest {

    @Test
    void getTensorCachesFp16Decompression() {
        ActivationTensorSlot slot = new ActivationTensorSlot();
        Tensor a = Tensor.fromArray(new float[] {1f, 2f, 3f, 4f}, new int[] {2, 2});
        slot.store(a, true);

        Tensor t1 = slot.getTensor();
        Tensor t2 = slot.getTensor();
        assertNotNull(t1);
        assertSame(t1, t2, "second getTensor should reuse fp32Cache");
    }

    @Test
    void storeFp16FailureMessageIncludesShape() {
        ActivationTensorSlot slot = new ActivationTensorSlot();
        Tensor bad = Tensor.fromArray(new float[] {1f, Float.NaN}, new int[] {1, 2});
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> slot.store(bad, true));
        assertTrue(ex.getMessage().contains("[shape=[1, 2]]"), ex.getMessage());
    }

    @Test
    void clearThenGetTensorNull() {
        ActivationTensorSlot slot = new ActivationTensorSlot();
        slot.store(Tensor.fromArray(new float[] {0f, 0f}, new int[] {2}), false);
        slot.clear();
        assertNull(slot.getTensor());
        assertEquals(ActivationTensorSlot.StorageMode.EMPTY, slot.mode());
    }

    @Test
    void ensureTensorReuseSameBuffer() {
        ActivationTensorSlot slot = new ActivationTensorSlot();
        int[] sh = {3};
        Tensor t1 = slot.ensureTensor(sh);
        t1.internalBuffer()[0] = 7f;
        Tensor t2 = slot.ensureTensor(sh);
        assertSame(t1, t2);
        assertEquals(7f, t2.internalBuffer()[0]);
    }

    @Test
    void finalizeAfterWriteCompressLeavesFp32OnConversionError() {
        ActivationTensorSlot slot = new ActivationTensorSlot();
        Tensor buf = slot.ensureTensor(new int[] {1});
        buf.internalBuffer()[0] = Float.NaN;
        assertThrows(IllegalArgumentException.class, () -> slot.finalizeAfterWrite(true));
        assertNotNull(slot.getTensor());
        assertEquals(ActivationTensorSlot.StorageMode.FP32, slot.mode());
    }
}
