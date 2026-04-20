package com.veles.llm.jgpt.training;

/** Чтение env/prop для конструктора тренера и сводок в лог. */
final class LlmTrainerEnvUtils {

    private LlmTrainerEnvUtils() {}

    static boolean readBooleanEnv(String key, boolean defaultValue) {
        try {
            String e = System.getenv(key);
            if (e != null) {
                String t = e.trim();
                if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                    return true;
                }
                if ("0".equals(t) || "false".equalsIgnoreCase(t)) {
                    return false;
                }
            }
        } catch (Exception ignored) {
        }
        return defaultValue;
    }

    static String envTrimOrDefault(String key, String defaultValue) {
        String v = System.getenv(key);
        return (v != null && !v.isBlank()) ? v.trim() : defaultValue;
    }

    static String envRawOrDash(String key) {
        String v = System.getenv(key);
        if (v == null) {
            return "—";
        }
        String t = v.trim();
        return t.isEmpty() ? "—" : t;
    }

    static String envCudaLibSummary() {
        String v = System.getenv("JGPT_CUDA_LIB");
        if (v == null || v.isBlank()) {
            return "—";
        }
        return "(задан)";
    }

    static int readPositiveEnvInt(String key, int defaultValue) {
        try {
            String e = System.getenv(key);
            if (e != null && !e.isBlank()) {
                int v = Integer.parseInt(e.trim());
                if (v > 0) {
                    return v;
                }
            }
        } catch (Exception ignored) {
        }
        return defaultValue;
    }

    static int readCudaTrimEveryOptimizerStepsFromEnv() {
        try {
            String e = System.getenv("JGPT_CUDA_TRIM_EVERY_STEPS");
            if (e != null && !e.isBlank()) {
                int v = Integer.parseInt(e.trim());
                return Math.max(0, v);
            }
        } catch (Exception ignored) {
        }
        return 500;
    }

    static int readVramCleanupEveryStepsFromEnv() {
        try {
            String e = System.getenv("JGPT_VRAM_CLEANUP_EVERY_STEPS");
            if (e != null && !e.isBlank()) {
                int v = Integer.parseInt(e.trim());
                return Math.max(0, v);
            }
        } catch (Exception ignored) {
        }
        return 100;
    }

    static boolean batchPrefetchEnabled() {
        if (Boolean.getBoolean("jgpt.batch.prefetch")) {
            return true;
        }
        String p = System.getProperty("jgpt.batch.prefetch");
        if (p != null && ("0".equals(p.trim()) || "false".equalsIgnoreCase(p.trim()))) {
            return false;
        }
        String v = System.getenv("JGPT_BATCH_PREFETCH");
        if (v != null) {
            String t = v.trim();
            if ("0".equals(t) || "false".equalsIgnoreCase(t) || "no".equalsIgnoreCase(t)) {
                return false;
            }
        }
        return true;
    }
}
