package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.util.LogFmt;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Удаляет старые файлы {@code checkpoint_*} / {@code model_*} / {@code tokenizer_*} с числовым суффиксом
 * (шаг или эпоха), оставляя несколько самых свежих. Не трогает {@code final}, {@code best}, {@code emergency},
 * глобальный токенизатор в родительских каталогах и т.п.
 *
 * <p>Семантика resume для {@link com.veles.llm.jgpt.app.AllBooksTrain}: без {@code checkpoint_final.bin}
 * подхватывается последний {@code checkpoint_epoch_*}; промежуточные {@code checkpoint_step_*} для него не
 * используются. Для цепочки по книгам см. {@link BookCheckpointPaths#findResumeCheckpoint(Path)} — там при
 * отсутствии final важен последний {@code checkpoint_step_*}, поэтому по умолчанию сохраняем несколько
 * последних шаговых снимков.
 *
 * <p>Переменные окружения (см. {@link #pruningEnabled()}, {@link #keepStepSnapshots()}, {@link #keepEpochSnapshots()}).
 */
public final class CheckpointPruner {

    private static final Logger log = LoggerFactory.getLogger(CheckpointPruner.class);

    private CheckpointPruner() {}

    public static boolean pruningEnabled() {
        String e = System.getenv("JGPT_CKPT_PRUNE");
        if (e == null || e.isBlank()) {
            return true;
        }
        String t = e.trim();
        return !("0".equals(t) || "false".equalsIgnoreCase(t) || "no".equalsIgnoreCase(t));
    }

    /**
     * Сколько последних шаговых троек оставить. {@code 0} — не удалять {@code *_step_*}.
     */
    public static int keepStepSnapshots() {
        return readNonNegativeIntEnv("JGPT_CKPT_KEEP_STEP_SNAPSHOTS", 2);
    }

    /**
     * Сколько последних эпоховых троек оставить. {@code 0} — не удалять {@code *_epoch_*}.
     */
    public static int keepEpochSnapshots() {
        return readNonNegativeIntEnv("JGPT_CKPT_KEEP_EPOCH_SNAPSHOTS", 2);
    }

    private static int readNonNegativeIntEnv(String name, int defaultValue) {
        String e = System.getenv(name);
        if (e == null || e.isBlank()) {
            return defaultValue;
        }
        try {
            int v = Integer.parseInt(e.trim().replace(',', '.'));
            return Math.max(0, v);
        } catch (NumberFormatException ex) {
            return defaultValue;
        }
    }

    public static void pruneStepTriples(Path checkpointDir, int keepLast) throws IOException {
        if (keepLast <= 0 || !Files.isDirectory(checkpointDir)) {
            return;
        }
        pruneNumberedTriples(checkpointDir, "checkpoint_step_", "step_", keepLast);
    }

    public static void pruneEpochTriples(Path checkpointDir, int keepLast) throws IOException {
        if (keepLast <= 0 || !Files.isDirectory(checkpointDir)) {
            return;
        }
        pruneNumberedTriples(checkpointDir, "checkpoint_epoch_", "epoch_", keepLast);
    }

    private static void pruneNumberedTriples(
            Path dir, String checkpointPrefix, String tripleSuffix, int keepLast) throws IOException {
        List<long[]> found = new ArrayList<>();
        try (Stream<Path> s = Files.list(dir)) {
            for (Path p : s.toList()) {
                if (!Files.isRegularFile(p)) {
                    continue;
                }
                String n = p.getFileName().toString();
                if (!n.startsWith(checkpointPrefix) || !n.endsWith(".bin")) {
                    continue;
                }
                String mid = n.substring(checkpointPrefix.length(), n.length() - ".bin".length());
                long num;
                try {
                    num = Long.parseLong(mid);
                } catch (NumberFormatException e) {
                    continue;
                }
                found.add(new long[] {num, 0});
            }
        }
        if (found.size() <= keepLast) {
            return;
        }
        found.sort(Comparator.comparingLong(a -> a[0]));
        List<Long> remove = new ArrayList<>();
        for (int i = 0; i < found.size() - keepLast; i++) {
            remove.add(found.get(i)[0]);
        }
        int deleted = 0;
        for (Long num : remove) {
            String suffix = tripleSuffix + num;
            for (String stem : List.of("checkpoint_", "model_", "tokenizer_")) {
                Path f = dir.resolve(stem + suffix + ".bin");
                if (Files.isRegularFile(f)) {
                    Files.delete(f);
                    deleted++;
                }
            }
        }
        if (deleted > 0) {
            log.info(
                    "{} удалено устаревших файлов чекпоинта (префикс «{}», оставлено последних {}): {}",
                    LogFmt.badge("CKPT"),
                    checkpointPrefix,
                    keepLast,
                    deleted);
        }
    }

    /**
     * Один раз после сохранения: подчищает шаговые и (при необходимости) эпоховые снимки.
     *
     * @param checkpointName суффикс без {@code checkpoint_} / расширения (как в {@link LLMTrainer#saveCheckpoint})
     */
    public static void pruneAfterSave(Path checkpointDir, String checkpointName) throws IOException {
        if (!pruningEnabled() || !Files.isDirectory(checkpointDir)) {
            return;
        }
        int ks = keepStepSnapshots();
        if (ks > 0) {
            pruneStepTriples(checkpointDir, ks);
        }
        int ke = keepEpochSnapshots();
        if (ke > 0 && shouldPruneEpochsAfter(checkpointName)) {
            pruneEpochTriples(checkpointDir, ke);
        }
    }

    static boolean shouldPruneEpochsAfter(String checkpointName) {
        if (checkpointName == null || checkpointName.isBlank()) {
            return false;
        }
        if (checkpointName.startsWith("epoch_")) {
            return true;
        }
        String lower = checkpointName.toLowerCase(Locale.ROOT);
        return "final".equals(lower) || "best".equals(lower) || "emergency".equals(lower);
    }
}
