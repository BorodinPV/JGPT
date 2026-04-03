package com.veles.llm.jgpt.training;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Пути и приоритеты артефактов одной книги в {@code checkpoints/books/&lt;id&gt;/}.
 *
 * <p>Два сценария выбора весов без чекпоинта Adam:
 *
 * <ul>
 *   <li>{@link #pickBestElseFinal(Path)} — для цепочки книг после завершённого прогона: сначала eval-best,
 *       иначе финальные веса.
 *   <li>{@link #findWeightsForResumeWithoutCheckpoint(Path)} — при {@code JGPT_RESUME_WEIGHTS_ONLY}:
 *       {@code model_final} → {@code model_best} → максимальный {@code model_step_*}.
 * </ul>
 */
public final class BookCheckpointPaths {

    private BookCheckpointPaths() {}

    public static Path modelFinal(Path bookDir) {
        return bookDir.resolve("model_final.bin");
    }

    public static Path modelBest(Path bookDir) {
        return bookDir.resolve("model_best.bin");
    }

    public static Path checkpointFinal(Path bookDir) {
        return bookDir.resolve("checkpoint_final.bin");
    }

    /**
     * Для передачи весов следующей книге: {@code model_best.bin} если есть, иначе {@code model_final.bin}
     * если есть.
     */
    public static Optional<Path> pickBestElseFinal(Path bookDir) {
        Path best = modelBest(bookDir);
        if (Files.isRegularFile(best)) {
            return Optional.of(best);
        }
        Path fin = modelFinal(bookDir);
        if (Files.isRegularFile(fin)) {
            return Optional.of(fin);
        }
        return Optional.empty();
    }

    /**
     * Если в каталоге файла есть {@code model_best.bin}, вернуть его; иначе исходный путь (если он
     * существует).
     */
    public static Path preferModelBestInSameDir(Path weightsPath) {
        if (weightsPath == null || !Files.isRegularFile(weightsPath)) {
            return weightsPath;
        }
        Path parent = weightsPath.getParent();
        if (parent == null) {
            return weightsPath;
        }
        Path best = modelBest(parent);
        return Files.isRegularFile(best) ? best : weightsPath;
    }

    /**
     * Без чекпоинта: {@code model_final} → {@code model_best} → максимальный {@code model_step_*}.
     */
    public static Path findWeightsForResumeWithoutCheckpoint(Path bookDir) throws IOException {
        if (!Files.isDirectory(bookDir)) {
            return null;
        }
        Path fin = modelFinal(bookDir);
        if (Files.isRegularFile(fin)) {
            return fin;
        }
        Path best = modelBest(bookDir);
        if (Files.isRegularFile(best)) {
            return best;
        }
        return maxNumberedFile(bookDir, "model_step_", ".bin");
    }

    /**
     * Приоритет: последний {@code checkpoint_step_*.bin}, иначе {@code checkpoint_best.bin}, иначе
     * максимальный {@code checkpoint_epoch_*.bin}. Если есть {@code checkpoint_final.bin} — {@code null}.
     */
    public static Path findResumeCheckpoint(Path bookDir) throws IOException {
        if (!Files.isDirectory(bookDir)) {
            return null;
        }
        if (Files.isRegularFile(checkpointFinal(bookDir))) {
            return null;
        }
        Path stepCk = maxNumberedFile(bookDir, "checkpoint_step_", ".bin");
        if (stepCk != null) {
            return stepCk;
        }
        Path best = bookDir.resolve("checkpoint_best.bin");
        if (Files.isRegularFile(best)) {
            return best;
        }
        return maxNumberedFile(bookDir, "checkpoint_epoch_", ".bin");
    }

    /** Суффикс имени чекпоинта для пары {@code model_&lt;suffix&gt;.bin} (как в {@link LLMTrainer}). */
    public static String checkpointSuffix(Path checkpointFile) {
        String n = checkpointFile.getFileName().toString();
        if (!n.startsWith("checkpoint_") || !n.endsWith(".bin")) {
            throw new IllegalArgumentException("Expected checkpoint_*.bin, got: " + n);
        }
        return n.substring("checkpoint_".length(), n.length() - ".bin".length());
    }

    /**
     * Файл с максимальным числом в имени {@code prefix + number + suffix} в каталоге.
     */
    private static Path maxNumberedFile(Path dir, String prefix, String suffix) throws IOException {
        long max = -1;
        Path best = null;
        try (Stream<Path> s = Files.list(dir)) {
            List<Path> list = s.toList();
            for (Path p : list) {
                String n = p.getFileName().toString();
                if (n.startsWith(prefix) && n.endsWith(suffix)) {
                    String num = n.substring(prefix.length(), n.length() - suffix.length());
                    try {
                        long v = Long.parseLong(num);
                        if (v > max) {
                            max = v;
                            best = p;
                        }
                    } catch (NumberFormatException ignored) {
                    }
                }
            }
        }
        return best;
    }
}
