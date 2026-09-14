package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Один запускатель обучения {@code scripts/windows/jgpt-train-*.cmd} и связанные с ним лог, пресет и каталог
 * чекпоинтов (вычитываются из строк {@code $LogFile = ...}, {@code $EnvFile = ...}, {@code $CkptDir = ...} в .ps1).
 */
record TrainRun(String name, Path cmd, Path logFile, Path envFile, Path ckptDir, EnvPreset preset) {

    private static final Pattern ASSIGN =
            Pattern.compile("^\\$(LogFile|EnvFile|CkptDir)\\s*=\\s*Join-Path\\s+\\$Root\\s+\"([^\"]+)\"");

    @Override
    public String toString() {
        return name;
    }

    boolean isSft() {
        return preset != null && preset.flag("JGPT_SFT");
    }

    static List<TrainRun> discover(ProjectPaths paths) {
        List<TrainRun> runs = new ArrayList<>();
        Path dir = paths.scriptsWindows();
        if (!Files.isDirectory(dir)) {
            return runs;
        }
        try (Stream<Path> st = Files.list(dir)) {
            List<Path> ps1 =
                    st.filter(p -> p.getFileName().toString().matches("jgpt-train-.*\\.ps1"))
                            .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                            .toList();
            for (Path p : ps1) {
                parse(paths, p).ifPresent(runs::add);
            }
        } catch (IOException _) {
            // каталог недоступен — пустой список
        }
        return runs;
    }

    private static Optional<TrainRun> parse(ProjectPaths paths, Path ps1) {
        String log = null;
        String env = null;
        String ckpt = null;
        try {
            for (String line : Files.readAllLines(ps1, StandardCharsets.UTF_8)) {
                Matcher m = ASSIGN.matcher(line.trim());
                if (!m.find()) {
                    continue;
                }
                switch (m.group(1)) {
                    case "LogFile" -> log = m.group(2);
                    case "EnvFile" -> env = m.group(2);
                    case "CkptDir" -> ckpt = m.group(2);
                    default -> {
                    }
                }
            }
        } catch (IOException _) {
            return Optional.empty();
        }
        if (log == null || env == null || ckpt == null) {
            return Optional.empty();
        }
        String base = ps1.getFileName().toString();
        base = base.substring(0, base.length() - ".ps1".length());
        Path cmd = ps1.resolveSibling(base + ".cmd");
        if (!Files.isRegularFile(cmd)) {
            return Optional.empty();
        }
        Path root = paths.root();
        Path envFile = root.resolve(env.replace('\\', '/'));
        EnvPreset preset = null;
        try {
            if (Files.isRegularFile(envFile)) {
                preset = EnvPreset.load(envFile);
            }
        } catch (IOException _) {
            preset = null;
        }
        String name = base.replaceFirst("^jgpt-train-", "");
        return Optional.of(
                new TrainRun(
                        name,
                        cmd,
                        root.resolve(log.replace('\\', '/')),
                        envFile,
                        root.resolve(ckpt.replace('\\', '/')),
                        preset));
    }
}
