package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Сопоставление каталога чекпоинтов и пресета (геометрия модели, токенизатор): по {@link TrainRun} или по
 * {@code JGPT_CHECKPOINT_SUBDIR} в любом {@code env/*.env}.
 */
final class ModelCatalog {

    private final ProjectPaths paths;
    private final Map<String, EnvPreset> presetBySubdir = new LinkedHashMap<>();

    ModelCatalog(ProjectPaths paths, List<TrainRun> runs) {
        this.paths = paths;
        for (TrainRun r : runs) {
            if (r.preset() != null) {
                presetBySubdir.putIfAbsent(r.ckptDir().getFileName().toString(), r.preset());
            }
        }
        if (Files.isDirectory(paths.envDir())) {
            try (Stream<Path> st = Files.list(paths.envDir())) {
                for (Path p : st.filter(x -> x.getFileName().toString().endsWith(".env")).sorted().toList()) {
                    try {
                        EnvPreset e = EnvPreset.load(p);
                        String sub = e.checkpointSubdir();
                        if (sub != null) {
                            presetBySubdir.putIfAbsent(Path.of(sub).getFileName().toString(), e);
                        }
                    } catch (IOException _) {
                        // пропускаем нечитаемый пресет
                    }
                }
            } catch (IOException _) {
                // без env — только пресеты запускателей
            }
        }
    }

    /** Модель для чата: файл весов + пресет с геометрией. */
    record ModelEntry(Path file, EnvPreset preset, long sizeBytes, long modifiedMs) {
        @Override
        public String toString() {
            return file.getParent().getFileName() + "\\" + file.getFileName();
        }

        Path tokenizer(ProjectPaths paths) {
            return paths.root().resolve(preset.geometry().tokenizerRel().replace('\\', '/'));
        }
    }

    Optional<EnvPreset> presetFor(Path ckptDir) {
        return Optional.ofNullable(presetBySubdir.get(ckptDir.getFileName().toString()));
    }

    /** Все {@code model_*.bin} в подкаталогах {@code checkpoints} с известной геометрией, новые сверху. */
    List<ModelEntry> models() {
        List<ModelEntry> out = new ArrayList<>();
        Path root = paths.checkpointsDir();
        if (!Files.isDirectory(root)) {
            return out;
        }
        try (Stream<Path> dirs = Files.list(root)) {
            for (Path dir : dirs.filter(Files::isDirectory).toList()) {
                EnvPreset preset = presetBySubdir.get(dir.getFileName().toString());
                if (preset == null) {
                    continue;
                }
                try (Stream<Path> files = Files.list(dir)) {
                    for (Path f : files.filter(x -> x.getFileName().toString().matches("model_.*\\.bin")).toList()) {
                        out.add(new ModelEntry(f, preset, Files.size(f), Files.getLastModifiedTime(f).toMillis()));
                    }
                }
            }
        } catch (IOException _) {
            // частичный список
        }
        out.sort(Comparator.comparingLong(ModelEntry::modifiedMs).reversed());
        return out;
    }

    Optional<ModelEntry> entryFor(Path modelFile) {
        return presetFor(modelFile.getParent())
                .map(p -> {
                    try {
                        return new ModelEntry(modelFile, p, Files.size(modelFile), Files.getLastModifiedTime(modelFile).toMillis());
                    } catch (IOException _) {
                        return new ModelEntry(modelFile, p, 0, 0);
                    }
                });
    }
}
