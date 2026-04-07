package com.veles.llm.jgpt.util;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

/**
 * Сессионная NDJSON-диагностика (debug mode): {@code /home/pavel/StudioProjects/.cursor/debug-b39372.log}.
 *
 * <p>Verbose pre-launch: env {@code JGPT_DEBUG_CURSOR_B39372=1}. События при сбое graph launch пишутся всегда (Java
 * + native). Пошаговый VRAM вокруг decoder: {@code JGPT_TRAIN_VRAM_STEP_PROBE=1} (см. {@code LLMConfig}).
 * Проактивный отказ от graph при низком free: {@code JGPT_DECODER_GRAPH_MIN_FREE_MIB} (см. {@code LLMConfig}).
 */
public final class CursorDebugB39372 {

    private static final Path LOG = Path.of("/home/pavel/StudioProjects/.cursor/debug-b39372.log");

    private CursorDebugB39372() {}

    /** Подробные логи перед каждым {@code cudaGraphExecLaunch} (поток/ключ/зонд). */
    public static boolean verboseDecoderGraph() {
        String e = System.getenv("JGPT_DEBUG_CURSOR_B39372");
        if (e == null || e.isBlank()) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    public static void appendJson(String hypothesisId, String location, String message, String dataJsonInner) {
        long ts = System.currentTimeMillis();
        String line =
                String.format(
                        Locale.ROOT,
                        "{\"sessionId\":\"b39372\",\"timestamp\":%d,\"hypothesisId\":\"%s\",\"location\":\"%s\",\"message\":\"%s\",\"data\":{%s}}\n",
                        ts,
                        esc(hypothesisId),
                        esc(location),
                        esc(message),
                        dataJsonInner == null ? "" : dataJsonInner);
        try {
            Files.writeString(
                    LOG, line, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException ignored) {
            // debug ingest best-effort
        }
    }

    private static String esc(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ").replace("\r", " ");
    }
}
