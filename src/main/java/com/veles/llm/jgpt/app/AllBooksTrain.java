package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.training.LLMConfig;
import com.veles.llm.jgpt.training.LLMTrainer;
import com.veles.llm.jgpt.training.TrainingConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Обучение на ВСЕХ книгах одновременно (единый датасет, один прогон).
 * Решает проблему катастрофического забывания по сравнению с {@link MultiBookTrain}.
 *
 * <p>Все {@code .txt} из {@code data/books/} (или {@code --data-dir}) объединяются в один
 * {@link DataLoader}. Чекпоинты — в {@code checkpoints/all_books/}.
 *
 * <h3>Режимы запуска:</h3>
 * <ul>
 *   <li><b>Первый запуск</b> — тренирует с нуля.</li>
 *   <li><b>Resume</b> — при повторном запуске автоматически подхватывает последний чекпоинт
 *       (ищет {@code checkpoint_final.bin}, затем последний {@code checkpoint_epoch_N.bin}).</li>
 *   <li><b>JGPT_EPOCHS=40</b> — увеличить плановое число эпох (полезно для дообучения:
 *       старый {@code globalStep} меньше нового {@code totalTrainingSteps} → обучение
 *       продолжится с того места, где остановилось).</li>
 *   <li><b>JGPT_FINETUNE=1</b> — сбросить {@code globalStep} в 0; веса и Adam-состояние
 *       при этом сохраняются. Используйте, если добавили новые книги и хотите
 *       переобучить полный цикл эпох заново.</li>
 * </ul>
 *
 * <p>Пример дообучения после добавления новых книг:
 * <pre>
 *   JGPT_TRAIN_LOSS_MODE=sampled JGPT_SAMPLED_CE_CANDIDATES=512 \
 *   JGPT_MAX_SEQ_LEN=1024 JGPT_CE_ASYNC=0 JGPT_INTERACTIVE_EVERY=0 \
 *   JGPT_FINETUNE=1 JGPT_EPOCHS=40 \
 *   ./scripts/train-e2e-gpu.sh allbooks
 * </pre>
 */
public final class AllBooksTrain {

    private static final Logger log = LoggerFactory.getLogger(AllBooksTrain.class);

    public static void main(String[] args) throws Exception {
        TensorOpsGPU.requireCuda("AllBooksTrain");

        String booRoot = ".";
        String dataDirArg = null;
        for (int i = 0; i < args.length; i++) {
            if ("--boo".equals(args[i]) && i + 1 < args.length) {
                booRoot = args[++i];
            } else if ("--data-dir".equals(args[i]) && i + 1 < args.length) {
                dataDirArg = args[++i];
            }
        }

        Path root = Path.of(booRoot).toAbsolutePath().normalize();
        Path dataDir = dataDirArg != null
                ? Path.of(dataDirArg).toAbsolutePath().normalize()
                : root.resolve("data").resolve("books");
        Path checkpointsDir = root.resolve("checkpoints").resolve("all_books");
        Path tokenizerPath  = root.resolve("checkpoints").resolve("tokenizer_global.bin");

        log.info("=".repeat(60));
        log.info("[ALL-BOOKS] обучение на всём корпусе (единый датасет)");
        log.info("[DATA] каталог с текстами: {}", dataDir);
        log.info("[CKPT] чекпоинты: {}", checkpointsDir);
        log.info("=".repeat(60));

        List<Path> books = listTxtFilesSorted(dataDir);
        if (books.isEmpty()) {
            log.error("В каталоге нет .txt файлов: {}", dataDir);
            System.exit(1);
        }
        log.info("[DATA] книг найдено: {}", books.size());

        LLMConfig llm = LLMConfig.applyEpochsOverrideFromEnv(
                LLMConfig.applySeqLenOverrideFromEnv(
                        LLMConfig.applyBatchSizeOverrideFromEnv(LLMConfig.smart50M())));
        log.info("[CFG] seq={}, d_model={}, layers={}, heads={}, vocab={}",
                llm.maxSeqLen, llm.dModel, llm.numLayers, llm.numHeads, llm.vocabSize);

        // --- токенизатор ---
        BPETokenizer tokenizer;
        if (Files.isRegularFile(tokenizerPath)) {
            log.info("[DATA] загрузка токенизатора: {}", tokenizerPath.getFileName());
            tokenizer = BPETokenizer.load(tokenizerPath.toString());
        } else {
            log.info("[DATA] обучение BPE-токенизатора на всём корпусе...");
            List<String> allTexts = new ArrayList<>();
            for (Path p : books) {
                allTexts.add(readUtf8(p));
            }
            tokenizer = BPETokenizer.train(allTexts, llm.vocabSize);
            Files.createDirectories(tokenizerPath.getParent());
            tokenizer.save(tokenizerPath.toString());
            log.info("[DATA] токенизатор сохранён: {} (vocab={})",
                    tokenizerPath.getFileName(), tokenizer.getVocabSize());
        }
        int vocabSize = tokenizer.getVocabSize();
        log.info("[DATA] размер словаря: {}", vocabSize);

        // --- датасет: все книги в один DataLoader ---
        Files.createDirectories(checkpointsDir);
        DataLoader dataLoader = new DataLoader(tokenizer, llm.maxSeqLen, llm.batchSize);

        int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        log.info("[DATA] параллельное кодирование: {} потоков", threads);
        ExecutorService pool = Executors.newFixedThreadPool(threads);

        // Запускаем кодирование всех книг параллельно
        List<Future<int[]>> futures = new ArrayList<>(books.size());
        for (Path p : books) {
            futures.add(pool.submit((Callable<int[]>) () -> {
                String text = readUtf8(p);
                return tokenizer.encode(text, true);
            }));
        }
        pool.shutdown();

        long totalChars = 0;
        int skipped = 0;
        int minTokens = llm.maxSeqLen + 1;
        for (int i = 0; i < books.size(); i++) {
            Path p = books.get(i);
            int[] tokens;
            try {
                tokens = futures.get(i).get();
            } catch (Exception e) {
                log.warn("[DATA] ошибка кодирования {}: {}", p.getFileName(), e.getMessage());
                skipped++;
                continue;
            }
            totalChars += p.toFile().length();
            log.info("[DATA]   {} → {} токенов", p.getFileName(), tokens.length);
            if (tokens.length < minTokens) {
                log.warn("[DATA] пропущена (слишком короткая): {} ({} токенов < {})",
                        p.getFileName(), tokens.length, minTokens);
                skipped++;
                continue;
            }
            dataLoader.loadTokens(tokens);
            log.info("[DATA]   {} → +{} окон (итого {})",
                    p.getFileName(), tokens.length / llm.maxSeqLen, dataLoader.numSequences());
        }
        log.info("[DATA] итого: {} символов, {} книг загружено, {} пропущено",
                String.format("%,d", totalChars), books.size() - skipped, skipped);
        int nSeq = dataLoader.numSequences();
        log.info("[DATA] всего последовательностей: {} (~{} батчей/эпоха)",
                nSeq, dataLoader.numBatches());
        if (nSeq == 0) {
            log.error("[DATA] нет последовательностей — все тексты слишком короткие (нужно >{})",
                    llm.maxSeqLen);
            System.exit(1);
        }

        // --- модель ---
        boolean gpuResident = LLMConfig.effectiveGpuResidentTraining();
        GPTModel model = new GPTModel(vocabSize, llm.maxSeqLen, llm.dModel,
                llm.numHeads, llm.numLayers, llm.dIntermediate, gpuResident);

        Path modelFinal = checkpointsDir.resolve("model_final.bin");
        if (Files.isRegularFile(modelFinal)) {
            log.info("[CKPT] продолжение: загрузка весов из {}", modelFinal.getFileName());
            model.loadWeights(modelFinal.toString());
        }

        // --- тренировка ---
        boolean finetune = isFinetuneMode();
        TrainingConfig trainConfig = llm.toTrainingConfig(checkpointsDir.toString(), vocabSize);
        LLMTrainer trainer = new LLMTrainer(model, trainConfig, dataLoader);

        // Ищем чекпоинт для resume: сначала checkpoint_final.bin, затем последний checkpoint_epoch_N.bin
        Optional<Path> resumeCkpt = findResumeCheckpoint(checkpointsDir);
        if (resumeCkpt.isPresent()) {
            log.info("[CKPT] загрузка состояния (Adam + globalStep): {}",
                    resumeCkpt.get().getFileName());
            trainer.loadCheckpoint(resumeCkpt.get().toString());
            if (finetune) {
                log.info("[CKPT] JGPT_FINETUNE=1 — globalStep сброшен в 0 (дообучение с начала эпох)");
                trainer.resetGlobalStep();
            }
        } else if (finetune) {
            log.info("[CKPT] JGPT_FINETUNE=1 — чекпоинт не найден, обучение с нуля");
        }

        // Graceful shutdown: при Ctrl+C сохранить финальный checkpoint
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            log.info("[SHUTDOWN] Получен сигнал остановки — сохраняем checkpoint...");
            try {
                trainer.saveCheckpoint("final");
                log.info("[SHUTDOWN] checkpoint сохранён. Возобновление: ./scripts/jgpt-start.sh");
            } catch (Exception e) {
                log.warn("[SHUTDOWN] Не удалось сохранить checkpoint: {}", e.getMessage());
            }
        }, "shutdown-ckpt"));

        log.info("=".repeat(60));
        trainer.train();

        trainer.saveCheckpoint("final");
        log.info("[ALL-BOOKS] обучение завершено. Лучший eval loss: {}",
                String.format("%.4f", trainer.getBestLoss()));
    }

    /**
     * Ищет чекпоинт для resume в порядке приоритета:
     * <ol>
     *   <li>{@code checkpoint_final.bin} — сохраняется в конце обучения;</li>
     *   <li>последний {@code checkpoint_epoch_N.bin} по номеру N.</li>
     * </ol>
     */
    static Optional<Path> findResumeCheckpoint(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return Optional.empty();

        Path fin = dir.resolve("checkpoint_final.bin");
        if (Files.isRegularFile(fin)) return Optional.of(fin);

        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(Files::isRegularFile)
                    .filter(p -> {
                        String n = p.getFileName().toString();
                        return n.startsWith("checkpoint_epoch_") && n.endsWith(".bin");
                    })
                    .max(Comparator.comparingInt(p -> {
                        String n = p.getFileName().toString();
                        try {
                            return Integer.parseInt(
                                    n.replace("checkpoint_epoch_", "").replace(".bin", ""));
                        } catch (NumberFormatException e) {
                            return -1;
                        }
                    }));
        }
    }

    /** {@code JGPT_FINETUNE=1} / {@code true} — сброс globalStep для дообучения. */
    static boolean isFinetuneMode() {
        String e = System.getenv("JGPT_FINETUNE");
        if (e == null || e.isBlank()) return false;
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    private static String readUtf8(Path p) throws IOException {
        return new String(Files.readAllBytes(p), StandardCharsets.UTF_8);
    }

    private static List<Path> listTxtFilesSorted(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return List.of();
        try (Stream<Path> s = Files.walk(dir)) {
            return s.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".txt"))
                    .sorted(Comparator.comparing(Path::toString))
                    .collect(Collectors.toCollection(ArrayList::new));
        }
    }
}
