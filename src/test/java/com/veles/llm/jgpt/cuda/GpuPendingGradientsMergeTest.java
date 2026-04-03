package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Детальные проверки {@link GpuPendingGradients#flushMergeToGpuGrads} и согласованности с {@link GpuPendingGradients#flushAllToHost}. */
class GpuPendingGradientsMergeTest {

    @Test
    void flushMergeToGpuMatchesFlushAllToHost() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try {
            float[] deltaData = new float[] {0.25f, -0.5f, 1f, 0f};
            Tensor param = new Tensor(new int[] {4});
            param.zeroGrad();
            try (GpuTensor gt = GpuTensor.fromHostTensor(param)) {
                gt.zeroGrad();

                GpuFloatBuffer delta = GpuFloatBuffer.allocate(4);
                delta.copyFrom(deltaData, 0, 4);
                GpuPendingGradients.accumulate(param, delta, 4);

                Map<Tensor, GpuTensor> map = new IdentityHashMap<>();
                map.put(param, gt);
                GpuPendingGradients.flushMergeToGpuGrads(map);
                float[] fromDevice = new float[4];
                gt.gradBuffer().copyTo(fromDevice, 0, 4);

                param.zeroGrad();
                GpuPendingGradients.accumulate(param, delta, 4);
                GpuPendingGradients.flushAllToHost();

                assertArrayEquals(param.gradBuffer(), fromDevice, 1e-5f, "pending → host vs pending → VRAM grad");
            }
        } finally {
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void twoSequentialAccumulatesOnSameParameterSumBeforeFlush() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try {
            Tensor param = new Tensor(new int[] {3});
            param.zeroGrad();
            try (GpuTensor gt = GpuTensor.fromHostTensor(param)) {
                gt.zeroGrad();
                GpuFloatBuffer d1 = GpuFloatBuffer.allocate(3);
                GpuFloatBuffer d2 = GpuFloatBuffer.allocate(3);
                d1.copyFrom(new float[] {1f, 0f, -1f}, 0, 3);
                d2.copyFrom(new float[] {0f, 2f, 0.5f}, 0, 3);
                GpuPendingGradients.accumulate(param, d1, 3);
                GpuPendingGradients.accumulate(param, d2, 3);
                Map<Tensor, GpuTensor> map = new IdentityHashMap<>();
                map.put(param, gt);
                GpuPendingGradients.flushMergeToGpuGrads(map);
                float[] g = new float[3];
                gt.gradBuffer().copyTo(g, 0, 3);
                assertArrayEquals(new float[] {1f, 2f, -0.5f}, g, 1e-5f);
            }
        } finally {
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void twoIndependentParametersDoNotCrossContaminate() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try {
            Tensor pA = new Tensor(new int[] {2});
            Tensor pB = new Tensor(new int[] {2});
            try (GpuTensor gA = GpuTensor.fromHostTensor(pA);
                    GpuTensor gB = GpuTensor.fromHostTensor(pB)) {
                gA.zeroGrad();
                gB.zeroGrad();
                GpuFloatBuffer dA = GpuFloatBuffer.allocate(2);
                GpuFloatBuffer dB = GpuFloatBuffer.allocate(2);
                dA.copyFrom(new float[] {10f, -1f}, 0, 2);
                dB.copyFrom(new float[] {3f, 4f}, 0, 2);
                GpuPendingGradients.accumulate(pA, dA, 2);
                GpuPendingGradients.accumulate(pB, dB, 2);
                Map<Tensor, GpuTensor> map = new IdentityHashMap<>();
                map.put(pA, gA);
                map.put(pB, gB);
                GpuPendingGradients.flushMergeToGpuGrads(map);
                float[] outA = new float[2];
                float[] outB = new float[2];
                gA.gradBuffer().copyTo(outA, 0, 2);
                gB.gradBuffer().copyTo(outB, 0, 2);
                assertArrayEquals(new float[] {10f, -1f}, outA, 1e-5f);
                assertArrayEquals(new float[] {3f, 4f}, outB, 1e-5f);
            }
        } finally {
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void flushMergeAddsOntoExistingDeviceGradient() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try {
            Tensor param = new Tensor(new int[] {4});
            param.zeroGrad();
            try (GpuTensor gt = GpuTensor.fromHostTensor(param)) {
                gt.zeroGrad();
                float[] pre = {100f, -200f, 0.25f, 0f};
                TensorOpsGPU.accumulateAddGpuFromHost(gt.gradBuffer(), pre, 0, 4);

                GpuFloatBuffer delta = GpuFloatBuffer.allocate(4);
                delta.copyFrom(new float[] {1f, 1f, 1f, 1f}, 0, 4);
                GpuPendingGradients.accumulate(param, delta, 4);
                Map<Tensor, GpuTensor> map = new IdentityHashMap<>();
                map.put(param, gt);
                GpuPendingGradients.flushMergeToGpuGrads(map);
                float[] out = new float[4];
                gt.gradBuffer().copyTo(out, 0, 4);
                assertArrayEquals(new float[] {101f, -199f, 1.25f, 1f}, out, 1e-4f);
            }
        } finally {
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void flushMergeThrowsWhenParameterMissingFromMap() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try {
            Tensor param = new Tensor(new int[] {2});
            GpuFloatBuffer delta = GpuFloatBuffer.allocate(2);
            delta.copyFrom(new float[] {1f, 2f}, 0, 2);
            GpuPendingGradients.accumulate(param, delta, 2);
            Map<Tensor, GpuTensor> empty = new IdentityHashMap<>();
            IllegalStateException ex =
                    assertThrows(IllegalStateException.class, () -> GpuPendingGradients.flushMergeToGpuGrads(empty));
            assertTrue(ex.getMessage().contains("GpuTensor"), ex.getMessage());
            assertTrue(ex.getMessage().contains("shape="), ex.getMessage());
        } finally {
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void scopeTryWithResourcesRunsCleanup() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        Tensor param = new Tensor(new int[] {2});
        try (GpuFloatBuffer delta = GpuFloatBuffer.allocate(2)) {
            delta.copyFrom(new float[] {1f, 2f}, 0, 2);
            try (GpuPendingGradients.Scope ignored = GpuPendingGradients.acquire()) {
                GpuPendingGradients.accumulate(param, delta, 2);
                assertTrue(GpuPendingGradients.isDirty(param));
                assertTrue(GpuPendingGradients.currentThreadDebugSummary().contains("dirtyEntries=1"));
            }
            assertFalse(GpuPendingGradients.isDirty(param));
        }
    }

    @Test
    void anyNonFinitePending_trueWhenAccumulatedDeltaHasNaN() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try {
            Tensor param = new Tensor(new int[] {2});
            GpuFloatBuffer delta = GpuFloatBuffer.allocate(2);
            delta.copyFrom(new float[] {Float.NaN, 1f}, 0, 2);
            GpuPendingGradients.accumulate(param, delta, 2);
            assertTrue(GpuPendingGradients.anyNonFinitePending());
        } finally {
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void anyNonFinitePending_falseForFiniteDelta() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try {
            Tensor param = new Tensor(new int[] {2});
            GpuFloatBuffer delta = GpuFloatBuffer.allocate(2);
            delta.copyFrom(new float[] {1f, 2f}, 0, 2);
            GpuPendingGradients.accumulate(param, delta, 2);
            assertFalse(GpuPendingGradients.anyNonFinitePending());
        } finally {
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void secondFlushAfterCleanPendingLeavesDeviceGradUnchanged() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try {
            Tensor param = new Tensor(new int[] {2});
            try (GpuTensor gt = GpuTensor.fromHostTensor(param)) {
                gt.zeroGrad();
                GpuFloatBuffer delta = GpuFloatBuffer.allocate(2);
                delta.copyFrom(new float[] {5f, -5f}, 0, 2);
                GpuPendingGradients.accumulate(param, delta, 2);
                Map<Tensor, GpuTensor> map = new IdentityHashMap<>();
                map.put(param, gt);
                GpuPendingGradients.flushMergeToGpuGrads(map);
                float[] mid = new float[2];
                gt.gradBuffer().copyTo(mid, 0, 2);
                GpuPendingGradients.flushMergeToGpuGrads(map);
                float[] after = new float[2];
                gt.gradBuffer().copyTo(after, 0, 2);
                assertArrayEquals(mid, after, 0f);
            }
        } finally {
            GpuPendingGradients.cleanupThreadLocal();
        }
    }
}
