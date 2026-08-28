package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;

/** Общие GPU-барьеры и env для FP16 aux soften (тренер + вспомогательный инференс). */
final class LlmTrainerGpuUtils {

    private LlmTrainerGpuUtils() {
        // utility class
    }

    /**
     * После overflow: без барьера следующий forward/backward может пересечься с async zeroGrad (ложные NaN).
     */
    static void synchronizeGpuAfterOverflowSkip() {
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
        }
    }

    /**
     * Env {@code JGPT_FP16_AUX_SOFTEN} / {@code -Djgpt.fp16.aux.soften}: делить loss scale после eval/генерации.
     */
    static boolean fp16AuxSoftenScaleAfterInfer() {
        String env = readTrimmed(() -> System.getenv("JGPT_FP16_AUX_SOFTEN"));
        if (env != null) {
            return isNotDisabledFlag(env);
        }
        String prop = readTrimmed(() -> System.getProperty("jgpt.fp16.aux.soften"));
        if (prop != null) {
            return isNotDisabledFlag(prop);
        }
        return true;
    }

    private static boolean isNotDisabledFlag(String s) {
        return !("0".equals(s) || "false".equalsIgnoreCase(s));
    }

    private static String readTrimmed(java.util.function.Supplier<String> src) {
        try {
            String raw = src.get();
            if (raw == null || raw.isBlank()) {
                return null;
            }
            return raw.trim();
        } catch (Exception _) {
            return null;
        }
    }
}
