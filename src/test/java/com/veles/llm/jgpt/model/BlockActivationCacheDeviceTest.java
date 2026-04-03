package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import org.junit.jupiter.api.Test;

class BlockActivationCacheDeviceTest {

    @Test
    void ensureThrowsIllegalArgumentOnSizeOverflow() {
        BlockActivationCacheDevice c = new BlockActivationCacheDevice();
        IllegalArgumentException ex =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> c.ensure(Integer.MAX_VALUE, Integer.MAX_VALUE, 128, 1, 128));
        assertTrue(ex.getMessage().contains("overflow"), ex.getMessage());
    }

    @Test
    void slotReadBeforeEnsureThrows() {
        BlockActivationCacheDevice c = new BlockActivationCacheDevice();
        IllegalStateException ex =
                assertThrows(
                        IllegalStateException.class,
                        () -> c.copySlotToHostFloat(BlockActivationCacheDevice.SlotId.X_IN, new float[4], 0, 1));
        assertTrue(ex.getMessage().contains("ensure") || ex.getMessage().contains("Call ensure"), ex.getMessage());
    }

    @Test
    void ensureAllocatesWhenGpuAvailable() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try (BlockActivationCacheDevice c = new BlockActivationCacheDevice()) {
            c.ensure(1, 2, 64, 4, 128);
            assertTrue(c.isAllocated());
        }
    }

    /**
     * При {@code -Djgpt.activationCache.fp16=true} и без явного отключения в {@code JGPT_ACTIVATION_CACHE_FP16}
     * слоты выделяются как FP16 на device.
     */
    @Test
    void ensureWithFp16Property_usesHalfStorage() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String env = System.getenv("JGPT_ACTIVATION_CACHE_FP16");
        if (env != null && !env.isBlank()) {
            String t = env.trim();
            if ("0".equals(t) || "false".equalsIgnoreCase(t) || "no".equalsIgnoreCase(t)) {
                return;
            }
        }
        String prev = System.getProperty("jgpt.activationCache.fp16");
        try {
            System.setProperty("jgpt.activationCache.fp16", "true");
            org.junit.jupiter.api.Assumptions.assumeTrue(BlockActivationCacheDevice.activationCacheFp16StorageFromEnv());
            try (BlockActivationCacheDevice c = new BlockActivationCacheDevice()) {
                c.ensure(1, 2, 64, 4, 128);
                assertTrue(c.isFp16ActivationStorage(), "ожидается GpuHalfBuffer-слоты");
            }
        } finally {
            if (prev == null) {
                System.clearProperty("jgpt.activationCache.fp16");
            } else {
                System.setProperty("jgpt.activationCache.fp16", prev);
            }
        }
    }
}
