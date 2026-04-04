package com.veles.llm.jgpt.util;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Opt-in JSONL-диагностика GPU-обучения и тяжёлые проверки по слоям. По умолчанию выключено.
 *
 * <p>{@code JGPT_DEBUG_GPU_TRAIN=1}: запись событий в файл (путь — {@code JGPT_DEBUG_GPU_TRAIN_LOG} или
 * {@code java.io.tmpdir/jgpt-debug-gpu-train.jsonl}).
 *
 * <p>{@code JGPT_BWD_LAYER_FINITE_CHECK=1}: после каждого decoder-блока на GPU — {@link
 * com.veles.llm.jgpt.TensorOpsGPU#anyNonFiniteGpuDevice} (дороже, локализация NaN по слою); без
 * записи в файл, если debug train выключен.
 */
public final class DebugGpuTrain {

    private DebugGpuTrain() {}

    public static boolean isEnabled() {
        String e = System.getenv("JGPT_DEBUG_GPU_TRAIN");
        if (e == null || e.isBlank()) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t) || "yes".equalsIgnoreCase(t);
    }

    public static Path logPath() {
        String p = System.getenv("JGPT_DEBUG_GPU_TRAIN_LOG");
        if (p != null && !p.isBlank()) {
            return Path.of(p.trim());
        }
        return Path.of(System.getProperty("java.io.tmpdir", "."), "jgpt-debug-gpu-train.jsonl");
    }

    /**
     * Проверка ∂ после каждого слоя декодера на GPU. Включено при {@link #isEnabled()} или при
     * {@code JGPT_BWD_LAYER_FINITE_CHECK=1}.
     */
    public static boolean perLayerFiniteCheck() {
        if (isEnabled()) {
            return true;
        }
        String e = System.getenv("JGPT_BWD_LAYER_FINITE_CHECK");
        if (e == null || e.isBlank()) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    /** Одна строка JSON + перевод строки; без эффекта, если {@link #isEnabled()} == false. */
    public static void appendJsonLine(String jsonLine) {
        if (!isEnabled()) {
            return;
        }
        String s = jsonLine;
        if (!s.endsWith("\n")) {
            s = s + "\n";
        }
        try (BufferedWriter w =
                Files.newBufferedWriter(
                        logPath(), StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            w.write(s);
        } catch (IOException ignored) {
        }
    }
}
