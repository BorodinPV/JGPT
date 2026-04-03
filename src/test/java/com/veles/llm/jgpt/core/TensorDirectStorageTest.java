package com.veles.llm.jgpt.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.FloatBuffer;
import org.junit.jupiter.api.Test;

class TensorDirectStorageTest {

    @Test
    void allocateDirect_roundTripAndInternalBuffer() {
        Tensor t = Tensor.allocateDirect(new int[]{2, 3});
        assertTrue(t.isDirectStorage());
        assertNotNull(t.directByteBuffer());
        assertTrue(t.directByteBuffer().isDirect());
        for (int i = 0; i < 6; i++) {
            t.setLinear(i, i * 0.5f);
        }
        float[] h = t.internalBuffer();
        assertEquals(6, h.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(i * 0.5f, h[i], 1e-6f);
        }
    }

    @Test
    void directBufferForJNI_mutationsVisibleInGet() {
        Tensor t = Tensor.allocateDirect(new int[]{2, 2});
        FloatBuffer fb = t.directBufferForJNI();
        fb.put(0, 3.25f);
        fb.put(3, -1f);
        assertEquals(3.25f, t.getLinear(0), 1e-6f);
        assertEquals(-1f, t.getLinear(3), 1e-6f);
    }

    @Test
    void directBufferForJNI_notOnHeap() {
        Tensor t = new Tensor(new int[]{2});
        assertThrows(IllegalStateException.class, t::directBufferForJNI);
    }

    @Test
    void storageModeReflectsAllocation() {
        assertEquals(Tensor.StorageMode.HEAP, new Tensor(new int[]{1}).storageMode());
        assertEquals(Tensor.StorageMode.DIRECT, Tensor.allocateDirect(new int[]{1}).storageMode());
    }

    @Test
    void copyDataTo_bulkMatchesGetDataCopy() {
        Tensor t = Tensor.allocateDirect(new int[]{4});
        for (int i = 0; i < 4; i++) {
            t.setLinear(i, i + 0.5f);
        }
        float[] viaCopy = t.getDataCopy();
        float[] viaBulk = new float[4];
        t.copyDataTo(viaBulk, 0);
        assertEquals(viaCopy.length, viaBulk.length);
        for (int i = 0; i < 4; i++) {
            assertEquals(viaCopy[i], viaBulk[i], 1e-6f);
        }
    }

    @Test
    void fillData_uniformDirect() {
        Tensor t = Tensor.allocateDirect(new int[]{100});
        t.fillData(2.5f);
        for (int i = 0; i < 100; i++) {
            assertEquals(2.5f, t.getLinear(i), 1e-6f);
        }
    }

    @Test
    void equalsAndHashCode_directVsHeapSameValues() {
        Tensor heap = Tensor.fromArray(new float[] {1f, -2f, 0f, 4f}, new int[]{2, 2});
        Tensor direct = Tensor.allocateDirect(new int[]{2, 2});
        for (int i = 0; i < 4; i++) {
            direct.setLinear(i, heap.getLinear(i));
        }
        assertEquals(heap, direct);
        assertEquals(direct, heap);
        assertEquals(heap.hashCode(), direct.hashCode());
    }
}
