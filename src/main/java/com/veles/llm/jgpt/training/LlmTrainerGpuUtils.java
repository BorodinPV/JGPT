package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;

/** Общие GPU-барьеры и env для FP16 aux soften (тренер + вспомогательный инференс). */
final class LlmTrainerGpuUtils {

    private LlmTrainerGpuUtils() {}

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
        try {
            String e = System.getenv("JGPT_FP16_AUX_SOFTEN");
            if (e != null && !e.isBlank()) {
                String s = e.trim();
                if ("0".equals(s) || "false".equalsIgnoreCase(s)) {
                    return false;
                }
                return true;
            }
        } catch (Exception ignored) {
        }
        try {
            String p = System.getProperty("jgpt.fp16.aux.soften");
            if (p != null && !p.isBlank()) {
                String s = p.trim();
                if ("0".equals(s) || "false".equalsIgnoreCase(s)) {
                    return false;
                }
            }
        } catch (Exception ignored) {
        }
        return true;
    }
}
