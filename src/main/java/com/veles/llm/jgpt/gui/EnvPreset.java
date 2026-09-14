package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Разбор {@code env/*.env} (строки {@code export KEY=value}) — как {@code Import-BashEnvFile} в .ps1. */
record EnvPreset(Path file, Map<String, String> vars) {

    private static final Pattern EXPORT = Pattern.compile("^export\\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$");

    static EnvPreset load(Path file) throws IOException {
        Map<String, String> vars = new LinkedHashMap<>();
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        for (String raw : lines) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            Matcher m = EXPORT.matcher(line);
            if (!m.matches()) {
                continue;
            }
            String val = m.group(2).trim();
            if (val.length() >= 2) {
                char q = val.charAt(0);
                if ((q == '"' || q == '\'') && val.charAt(val.length() - 1) == q) {
                    val = val.substring(1, val.length() - 1);
                }
            }
            vars.put(m.group(1), val);
        }
        return new EnvPreset(file, vars);
    }

    String name() {
        String n = file.getFileName().toString();
        return n.endsWith(".env") ? n.substring(0, n.length() - 4) : n;
    }

    String get(String key, String def) {
        String v = vars.get(key);
        return v == null || v.isBlank() ? def : v.trim();
    }

    int getInt(String key, int def) {
        try {
            return Integer.parseInt(get(key, Integer.toString(def)));
        } catch (NumberFormatException _) {
            return def;
        }
    }

    boolean flag(String key) {
        String v = get(key, "0");
        return "1".equals(v) || "true".equalsIgnoreCase(v);
    }

    /** Подкаталог чекпоинтов ({@code JGPT_CHECKPOINT_SUBDIR}), {@code null} если не задан. */
    String checkpointSubdir() {
        String v = get("JGPT_CHECKPOINT_SUBDIR", "");
        return v.isEmpty() ? null : v.replace('/', '\\').replace("\\", "/");
    }

    /** Геометрия модели для инференса; вокаб уточняется по токенизатору. */
    ModelGeometry geometry() {
        return new ModelGeometry(
                getInt("JGPT_VOCAB_SIZE", 16000),
                getInt("JGPT_MAX_SEQ_LEN", 1024),
                getInt("JGPT_D_MODEL", 512),
                getInt("JGPT_NUM_HEADS", 32),
                getInt("JGPT_PRESET_NUM_LAYERS", 28),
                getInt("JGPT_D_INTERMEDIATE", 2048),
                get("JGPT_TOKENIZER_PATH", "checkpoints/tokenizer_wide_16k.bin"),
                flag("JGPT_SFT"));
    }

    record ModelGeometry(
            int vocabSize,
            int maxSeqLen,
            int dModel,
            int numHeads,
            int numLayers,
            int dIntermediate,
            String tokenizerRel,
            boolean sft) {

        String describe() {
            return numLayers + "L d=" + dModel + " h=" + numHeads + " seq=" + maxSeqLen + " vocab=" + vocabSize;
        }
    }
}
