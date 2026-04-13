package com.veles.llm.jgpt;

/** Env/property разбор при статической инициализации {@link TensorOpsGPU}. */
final class TensorOpsGpuInit {

    private TensorOpsGpuInit() {}

    static boolean allowNoGpuOverride() {
        try {
            if (Boolean.getBoolean("jgpt.allow.no.gpu")) {
                return true;
            }
        } catch (Exception ignored) {
            // ignore
        }
        try {
            String e = System.getenv("JGPT_ALLOW_NO_GPU");
            if (e != null) {
                String t = e.trim();
                if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
            // ignore
        }
        return false;
    }

    static float resolveRmsNormEps(boolean fp16MatmulEnabled) {
        try {
            String e = System.getenv("JGPT_RMSNORM_EPS");
            if (e != null && !e.isBlank()) {
                float v = Float.parseFloat(e.trim());
                if (v > 0f && Float.isFinite(v) && v <= 1e-2f) {
                    return v;
                }
            }
        } catch (Exception ignored) {
            // ignore
        }
        try {
            String p = System.getProperty("jgpt.rmsnorm.eps");
            if (p != null && !p.isBlank()) {
                float v = Float.parseFloat(p.trim());
                if (v > 0f && Float.isFinite(v) && v <= 1e-2f) {
                    return v;
                }
            }
        } catch (Exception ignored) {
            // ignore
        }
        return fp16MatmulEnabled ? 1e-5f : 1e-6f;
    }

    static int resolveCeGpuMinElements() {
        try {
            String e = System.getenv("JGPT_CE_GPU_MIN_ELEMENTS");
            if (e != null && !e.isBlank()) {
                int v = Integer.parseInt(e.trim());
                if (v >= 0) {
                    return v;
                }
            }
        } catch (Exception ignored) {
            // ignore
        }
        return 0;
    }

    /**
     * FP16 Tensor Cores для host GEMM ({@code JGPT_FP16_MATMUL} / {@code -Djgpt.fp16.matmul=true}); эффективно только
     * при {@code gpuAvailable}.
     */
    static boolean resolveFp16Matmul(boolean gpuAvailable) {
        Boolean fp16FromEnv = null;
        try {
            String v = System.getenv("JGPT_FP16_MATMUL");
            if (v != null) {
                String t = v.trim();
                if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                    fp16FromEnv = true;
                } else if ("0".equals(t) || "false".equalsIgnoreCase(t)) {
                    fp16FromEnv = false;
                }
            }
        } catch (Exception ignored) {
            // ignore
        }
        boolean fp16Matmul = fp16FromEnv != null ? fp16FromEnv : false;
        if (fp16FromEnv == null) {
            try {
                if (Boolean.getBoolean("jgpt.fp16.matmul")) {
                    fp16Matmul = true;
                }
            } catch (Exception ignored) {
                // ignore
            }
        }
        return fp16Matmul && gpuAvailable;
    }

    /** FlashAttention-2 ({@code JGPT_FLASH_ATTENTION=1}); эффективно только при {@code gpuAvailable}. */
    static boolean resolveFlashAttention(boolean gpuAvailable) {
        boolean flashAttn = false;
        try {
            String v = System.getenv("JGPT_FLASH_ATTENTION");
            if (v != null && ("1".equals(v.trim()) || "true".equalsIgnoreCase(v.trim()))) {
                flashAttn = true;
            }
        } catch (Exception ignored) {
            // ignore
        }
        return flashAttn && gpuAvailable;
    }
}
