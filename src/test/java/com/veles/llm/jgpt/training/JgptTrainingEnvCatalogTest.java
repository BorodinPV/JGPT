package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;

/**
 * Единый реестр {@code JGPT_*} в {@code jgpt-training-env-keys.txt}: формат, сортировка и покрытие всех
 * {@code System.getenv("JGPT_…")} в {@code src/main/java} и {@code src/test/java}.
 */
class JgptTrainingEnvCatalogTest {

    private static final Pattern GETENV_JGPT =
            Pattern.compile("System\\.getenv\\(\"(JGPT_[A-Z0-9_]+)\"");
    private static final Pattern KEY_LINE = Pattern.compile("^JGPT_[A-Z0-9_]+$");

    private static List<String> loadCatalogFromResource() throws IOException {
        List<String> keys = new ArrayList<>();
        try (BufferedReader br =
                new BufferedReader(
                        new InputStreamReader(
                                JgptTrainingEnvCatalogTest.class
                                        .getResourceAsStream("/jgpt-training-env-keys.txt"),
                                StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                keys.add(line);
            }
        }
        return keys;
    }

    private static Path moduleRoot() {
        Path cwd = Path.of("").toAbsolutePath();
        if (Files.isRegularFile(cwd.resolve("pom.xml"))) {
            return cwd;
        }
        Path jgpt = cwd.resolve("JGPT");
        if (Files.isRegularFile(jgpt.resolve("pom.xml"))) {
            return jgpt;
        }
        return cwd;
    }

    private static Set<String> jgptGetenvKeysUnder(Path dir) throws IOException {
        Set<String> out = new TreeSet<>();
        if (!Files.isDirectory(dir)) {
            return out;
        }
        try (Stream<Path> walk = Files.walk(dir)) {
            List<Path> javaFiles = walk.filter(x -> x.toString().endsWith(".java")).toList();
            for (Path p : javaFiles) {
                String s = Files.readString(p);
                Matcher m = GETENV_JGPT.matcher(s);
                while (m.find()) {
                    out.add(m.group(1));
                }
            }
        }
        return out;
    }

    @Test
    void catalogLinesAreValidJgptKeysAndSorted() throws IOException {
        List<String> keys = loadCatalogFromResource();
        assertTrue(keys.size() >= 30, "catalog should list main JGPT env keys");
        List<String> sorted = new ArrayList<>(keys);
        sorted.sort(String::compareTo);
        assertEquals(sorted, keys, "jgpt-training-env-keys.txt must stay alphabetically sorted");
        for (String k : keys) {
            assertTrue(KEY_LINE.matcher(k).matches(), "invalid key line: " + k);
        }
    }

    @Test
    void catalogCoversAllGetenvReadsInSources() throws IOException {
        Set<String> catalog = new TreeSet<>(loadCatalogFromResource());
        Path root = moduleRoot();
        Set<String> fromMain = jgptGetenvKeysUnder(root.resolve("src/main/java"));
        Set<String> fromTest = jgptGetenvKeysUnder(root.resolve("src/test/java"));
        TreeSet<String> missing = new TreeSet<>();
        for (String k : fromMain) {
            if (!catalog.contains(k)) {
                missing.add(k);
            }
        }
        for (String k : fromTest) {
            if (!catalog.contains(k)) {
                missing.add(k);
            }
        }
        assertTrue(
                missing.isEmpty(),
                "Add to jgpt-training-env-keys.txt (sorted): " + missing);
    }

    @Test
    void llmConfigEnvReadersDoNotThrow() {
        LLMConfig.nano();
        var cfg =
                LLMConfig.applyLearningRateOverrideFromEnv(
                        LLMConfig.applyBatchSizeOverrideFromEnv(LLMConfig.nano()));
        LLMConfig.gpuResidentTrainingExplicitlyOn();
        LLMConfig.effectiveGpuResidentTraining();
        LLMConfig.fullGpuTrainStepFromEnv();
        LLMConfig.effectiveFullGpuTrainStepFromEnv();
        LLMConfig.deviceLogitsTrainStepFromEnv();
        LLMConfig.deviceDecoderBackwardFromEnv();
        LLMConfig.trainLossModeFromEnvOrProp();
        LLMConfig.sampledCeCandidatesFromEnv();
        try {
            LLMConfig.sampledCeNegativeModeFromEnvOrProp();
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("sampled CE negative mode"), e.getMessage());
        }
        LLMConfig.gpuE2eTrainFromEnv();
        LLMConfig.decoderGpuPipelineFromEnvOrProp();
        int vs = 100;
        try {
            cfg.toTrainingConfig("x", vs);
        } catch (IllegalStateException e) {
            assertTrue(
                    e.getMessage().contains("JGPT_") || e.getMessage().contains("CUDA") || e.getMessage().contains("pipeline"),
                    e.getMessage());
        }
        TensorOpsGPU.isGpuAvailable();
    }
}
