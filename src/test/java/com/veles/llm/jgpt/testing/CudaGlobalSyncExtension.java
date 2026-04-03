package com.veles.llm.jgpt.testing;

import com.veles.llm.jgpt.TensorOpsGPU;
import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * Ограждает тесты от гонок на одном CUDA-stream: синхронизирует до и после каждого теста. Иначе
 * {@code @AfterEach}, освобождающий VRAM, иногда выполняется до {@link AfterEachCallback} расширения,
 * и буфер может быть освобождён при ещё незавершённых ядрах предыдущего шага.
 */
public final class CudaGlobalSyncExtension implements BeforeEachCallback, AfterEachCallback {

    private static void syncIfGpu() {
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeDevice();
        }
    }

    @Override
    public void beforeEach(ExtensionContext context) {
        syncIfGpu();
    }

    @Override
    public void afterEach(ExtensionContext context) {
        syncIfGpu();
    }
}
