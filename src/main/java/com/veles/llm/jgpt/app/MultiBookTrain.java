package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.data.TextDataset;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.training.BookCheckpointPaths;
import com.veles.llm.jgpt.training.BookTrainingState;
import com.veles.llm.jgpt.training.LLMConfig;
import com.veles.llm.jgpt.training.LLMTrainer;
import com.veles.llm.jgpt.training.TrainingConfig;
import com.veles.llm.jgpt.util.LogFmt;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Обучение по одной книге за раз: все {@code .txt} под {@code --data-dir} (по умолчанию
 * {@code <корень>/data/books}). Корень артефактов — {@code --boo} (по умолчанию текущая директория
 * процесса, обычно корень проекта {@code JGPT}). Чекпоинты: {@code <корень>/checkpoints/}.
 * Опции: {@code --max-sequences N} или env {@code JGPT_MAX_SEQUENCES} — ограничить число
 * окон по книге (меньше RAM, часть текста не используется).
 * <p>
 * После каждой завершённой книги (до освобождения VRAM) можно включить интерактивную генерацию:
 * {@code JGPT_INTERACTIVE_AFTER_BOOK=1}. Нужна реальная консоль (TTY); иначе шаг пропускается с
 * предупреждением в логе. Опционально: {@code JGPT_INTERACTIVE_GEN_MAX_NEW}, {@code
 * JGPT_INTERACTIVE_GEN_TEMP}, {@code JGPT_INTERACTIVE_GEN_TOP_K}.
 */
public final class MultiBookTrain {

    private static final Logger log = LoggerFactory.getLogger(MultiBookTrain.class);

    public static void main(String[] args) throws Exception {
        TensorOpsGPU.requireCuda("MultiBookTrain");
        Instant wallStart = Instant.now();
        String booRoot = ".";
        String dataDirArg = null;
        int maxSequencesPerBook = 0;
        for (int i = 0; i < args.length; i++) {
            if ("--boo".equals(args[i]) && i + 1 < args.length) {
                booRoot = args[++i];
            } else if ("--data-dir".equals(args[i]) && i + 1 < args.length) {
                dataDirArg = args[++i];
            } else if ("--max-sequences".equals(args[i]) && i + 1 < args.length) {
                maxSequencesPerBook = Integer.parseInt(args[++i]);
            }
        }
        String envMax = System.getenv("JGPT_MAX_SEQUENCES");
        if (maxSequencesPerBook <= 0 && envMax != null && !envMax.isBlank()) {
            maxSequencesPerBook = Integer.parseInt(envMax.trim());
        }

        Path root = Path.of(booRoot).toAbsolutePath().normalize();
        Path dataDir =
                dataDirArg != null
                        ? Path.of(dataDirArg).toAbsolutePath().normalize()
                        : root.resolve("data").resolve("books");
        Path checkpointsRoot = root.resolve("checkpoints");
        Path statePath = checkpointsRoot.resolve("training_state.properties");
        Path tokenizerPath = checkpointsRoot.resolve("tokenizer_global.bin");

        log.info("{} обучение по нескольким книгам (цепочка книг)", LogFmt.badge("BOOK"));
        log.info("{} время старта: {}", LogFmt.badge("CFG"), wallStart);
        log.info("{} каталог с текстами: {}", LogFmt.badge("DATA"), dataDir.toAbsolutePath());
        log.info("{} файл состояния очереди: {}", LogFmt.badge("CFG"), statePath.toAbsolutePath());
        if (maxSequencesPerBook > 0) {
            log.info(
                    "{} ограничение окон на книгу: {} (см. --max-sequences / JGPT_MAX_SEQUENCES)",
                    LogFmt.badge("CFG"),
                    maxSequencesPerBook);
        }
        log.info("{}", LogFmt.muted("=".repeat(60)));

        List<Path> books = listTxtFilesSorted(dataDir);
        if (books.isEmpty()) {
            log.error("В каталоге нет файлов .txt: {}", dataDir.toAbsolutePath());
            System.exit(1);
        }

        BookTrainingState state = BookTrainingState.load(statePath);
        LLMConfig llm = LLMConfig.applySeqLenOverrideFromEnv(
                LLMConfig.applyBatchSizeOverrideFromEnv(LLMConfig.smart50M()));
        log.info("{} эффективный размер батча: {}", LogFmt.badge("CFG"), llm.batchSize);
        if (System.getenv("JGPT_MAX_SEQ_LEN") != null) {
            log.info("{} maxSeqLen переопределён через JGPT_MAX_SEQ_LEN: {}", LogFmt.badge("CFG"), llm.maxSeqLen);
        }

        BPETokenizer tokenizer;
        if (Files.isRegularFile(tokenizerPath)) {
            log.info("{} загрузка общего токенизатора с диска", LogFmt.badge("DATA"));
            tokenizer = BPETokenizer.load(tokenizerPath.toString());
        } else {
            TextDataset allTexts = new TextDataset();
            for (Path p : books) {
                allTexts.loadFile(p.toString());
            }
            log.info("{} обучение общего BPE-токенизатора на всех книгах", LogFmt.badge("DATA"));
            tokenizer = BPETokenizer.train(allTexts.getTexts(), llm.vocabSize);
            Files.createDirectories(tokenizerPath.getParent());
            tokenizer.save(tokenizerPath.toString());
            allTexts.clearTexts();
        }
        int vocabSize = tokenizer.getVocabSize();
        log.info("{} размер словаря: {}", LogFmt.badge("DATA"), vocabSize);
        log.info(
                "{} подготовка (список книг + токенизатор) заняла: {}",
                LogFmt.badge("DATA"),
                humanDuration(Duration.between(wallStart, Instant.now())));

        Path lastModelPath = null;
        Path dataRootAbs = dataDir.toAbsolutePath().normalize();
        for (Path bookPath : books) {
            String bookKey =
                    dataRootAbs
                            .relativize(bookPath.toAbsolutePath().normalize())
                            .toString()
                            .replace('\\', '/');
            String bookDirId = sanitizeFileName(bookKey.replace('/', '_'));
            Path bookDir = checkpointsRoot.resolve("books").resolve(bookDirId);
            Files.createDirectories(bookDir);

            if (state.isCompleted(bookKey)) {
                log.info("{} пропуск уже завершённой книги: {}", LogFmt.badge("BOOK"), bookKey);
                Path carry = BookCheckpointPaths.findWeightsForResumeWithoutCheckpoint(bookDir);
                if (carry != null) {
                    lastModelPath = carry;
                    log.info("{} для следующей книги тёплый старт: {}", LogFmt.badge("BOOK"), carry.getFileName());
                } else {
                    log.warn(
                            "{} нет model_final / model_best / model_step_* — следующая книга стартует без весов с этой",
                            LogFmt.badge("BOOK"));
                }
                continue;
            }

            if (Files.isRegularFile(BookCheckpointPaths.checkpointFinal(bookDir))) {
                log.info(
                        "{} найден завершённый прогон (финальный чекпоинт) для «{}» — помечаю книгу выполненной",
                        LogFmt.badge("BOOK"),
                        bookKey);
                state.markCompleted(bookKey);
                state.setCurrent(null);
                state.save(statePath);
                Optional<Path> carryDone = BookCheckpointPaths.pickBestElseFinal(bookDir);
                if (carryDone.isPresent()) {
                    lastModelPath = carryDone.get();
                }
                continue;
            }

            state.setCurrent(bookKey);
            state.save(statePath);

            log.info("{} книга: {}", LogFmt.badge("BOOK"), bookKey);
            log.info("{} каталог чекпоинтов: {}", LogFmt.badge("CKPT"), bookDir.toAbsolutePath());

            Path resumeCkpt = BookCheckpointPaths.findResumeCheckpoint(bookDir);
            boolean resumeStateAllowed = false;

            if (LLMConfig.gpuResidentTrainingExplicitlyOn() && !TensorOpsGPU.isGpuAvailable()) {
                log.warn(
                        "JGPT_TRAIN_GPU_RESIDENT задан, но CUDA недоступна — обучение без GPU-резидентных весов.");
            }
            boolean gpuResident = LLMConfig.effectiveGpuResidentTraining();
            GPTModel model =
                    new GPTModel(
                            vocabSize,
                            llm.maxSeqLen,
                            llm.dModel,
                            llm.numHeads,
                            llm.numLayers,
                            llm.dIntermediate,
                            gpuResident);

            if (resumeCkpt != null) {
                String ckptName = BookCheckpointPaths.checkpointSuffix(resumeCkpt);
                Path weights = bookDir.resolve("model_" + ckptName + ".bin");
                if (!Files.isRegularFile(weights)) {
                    throw new IOException("Нет файла весов для продолжения: " + weights);
                }
                log.info("{} продолжение с чекпоинта: {}", LogFmt.badge("CKPT"), resumeCkpt.getFileName());
                if (tryLoadWeights(
                        model,
                        weights,
                        "resume " + resumeCkpt.getFileName() + " for " + bookKey)) {
                    resumeStateAllowed = true;
                } else {
                    resumeCkpt = null;
                }
            } else if (lastModelPath != null && Files.isRegularFile(lastModelPath)) {
                Path ws = BookCheckpointPaths.preferModelBestInSameDir(lastModelPath);
                if (!ws.equals(lastModelPath)) {
                    log.info(
                            "{} тёплый старт: в каталоге предыдущей книги найден model_best.bin — берём его вместо {}",
                            LogFmt.badge("BOOK"),
                            lastModelPath.getFileName());
                }
                log.info("{} тёплый старт весами из предыдущей книги: {}", LogFmt.badge("BOOK"), ws);
                tryLoadWeights(model, ws, "warm start for " + bookKey);
            }

            DataLoader dataLoader = new DataLoader(tokenizer, llm.maxSeqLen, llm.batchSize);
            dataLoader.setMaxSequences(maxSequencesPerBook);
            dataLoader.loadTextFile(bookPath.toString());

            int nSeq = dataLoader.numSequences();
            log.info(
                    "{} последовательностей: {} (около {} батчей на эпоху)",
                    LogFmt.badge("DATA"),
                    nSeq,
                    dataLoader.numBatches());
            if (nSeq <= 0) {
                log.warn(
                        "{} книга «{}» пропущена: нет окон для обучения (0 последовательностей). "
                                + "Нужно минимум {} токенов, а в тексте только {}. "
                                + "Удалите/замените слишком короткий .txt или объедините его с другим материалом.",
                        LogFmt.badge("BOOK"),
                        bookKey,
                        llm.maxSeqLen + 1,
                        dataLoader.numSequences());
                model.closeGpuResidentWeights();
                state.markCompleted(bookKey);
                state.setCurrent(null);
                state.save(statePath);
                continue;
            }

            TrainingConfig trainConfig = llm.toTrainingConfig(bookDir.toString(), vocabSize);
            LLMTrainer trainer = new LLMTrainer(model, trainConfig, dataLoader);

            if (resumeStateAllowed && resumeCkpt != null) {
                trainer.loadCheckpoint(resumeCkpt.toString());
            }

            Instant tBookTrain = Instant.now();
            boolean stopChain = false;
            try {
                trainer.train();
                trainer.saveCheckpoint("final");
                stopChain = maybeInteractiveAfterBook(model, tokenizer, bookKey);
            } finally {
                trainer.releaseGpuResourcesAfterBook();
            }
            Duration bookDur = Duration.between(tBookTrain, Instant.now());

            state.markCompleted(bookKey);
            state.setCurrent(null);
            state.save(statePath);

            Path bestPath = BookCheckpointPaths.modelBest(bookDir);
            lastModelPath =
                    BookCheckpointPaths.pickBestElseFinal(bookDir)
                            .orElse(BookCheckpointPaths.modelFinal(bookDir));
            if (Files.isRegularFile(bestPath) && lastModelPath.equals(bestPath)) {
                log.info("  для следующей книги в цепочке будет использован model_best.bin");
            } else {
                log.info("  для следующей книги в цепочке будет использован model_final.bin");
            }
            log.info("Книга «{}» обучена за {}", bookKey, humanDuration(bookDur));
            if (stopChain) {
                log.info(
                        "{} остановка цепочки книг: запрошено пользователем ({} в интерактиве после книги).",
                        LogFmt.badge("BOOK"),
                        "quit/exit");
                break;
            }
        }

        Duration total = Duration.between(wallStart, Instant.now());
        log.info("{}", "=".repeat(60));
        log.info("Все запланированные книги обработаны.");
        log.info("Общее реальное время с запуска: {}", humanDuration(total));
    }

    /**
     * После сохранения финального чекпоинта книги: цикл ввода промптов в консоль. Вызывается до {@link
     * LLMTrainer#releaseGpuResourcesAfterBook()}, пока веса ещё на GPU.
     *
     * @return true — прервать цепочку {@code MultiBookTrain} (пользователь ввёл quit/exit); иначе false
     */
    private static boolean maybeInteractiveAfterBook(
            GPTModel model, BPETokenizer tokenizer, String bookKey) {
        if (!truthyEnv("JGPT_INTERACTIVE_AFTER_BOOK")) {
            return false;
        }
        if (System.console() == null) {
            log.warn(
                    "{} JGPT_INTERACTIVE_AFTER_BOOK=1, но нет интерактивной консоли (System.console()==null) — шаг пропущен.",
                    LogFmt.badge("BOOK"));
            return false;
        }
        int maxNew = intEnvOr("JGPT_INTERACTIVE_GEN_MAX_NEW", 64);
        float temp = floatEnvOr("JGPT_INTERACTIVE_GEN_TEMP", 0.9f);
        int topK = intEnvOr("JGPT_INTERACTIVE_GEN_TOP_K", 40);

        log.info("{}", "=".repeat(60));
        log.info(
                "{} после «{}»: интерактивная проверка (JGPT_INTERACTIVE_AFTER_BOOK=1).",
                LogFmt.badge("BOOK"),
                bookKey);
        log.info(
                "  Строка-подсказка → генерация продолжения; пустая строка → следующая книга; quit или exit → выход из цепочки.");
        log.info("{}", "=".repeat(60));

        java.io.Console console = System.console();
        while (true) {
            String line = console.readLine("prompt> ");
            if (line == null) {
                return false;
            }
            String trimmed = line.strip();
            if (trimmed.isEmpty()) {
                log.info("{} пустой ввод — переходим к следующей книге.", LogFmt.badge("BOOK"));
                return false;
            }
            if ("quit".equalsIgnoreCase(trimmed) || "exit".equalsIgnoreCase(trimmed)) {
                log.info("{} выход из цепочки книг по запросу пользователя.", LogFmt.badge("BOOK"));
                return true;
            }
            try {
                model.zeroGradParameters();
                String out =
                        LlmTextGeneration.generateText(
                                model, tokenizer, trimmed, maxNew, temp, topK);
                log.info("{} сгенерировано: {}", LogFmt.badge("BOOK"), out);
            } catch (Exception e) {
                log.warn("{} генерация не удалась: {}", LogFmt.badge("BOOK"), e.getMessage());
            } finally {
                if (TensorOpsGPU.isGpuAvailable()) {
                    TensorOpsGPU.synchronizeStream();
                }
            }
        }
    }

    private static boolean truthyEnv(String name) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) {
            return false;
        }
        String t = v.strip();
        return "1".equals(t) || "true".equalsIgnoreCase(t) || "yes".equalsIgnoreCase(t);
    }

    private static int intEnvOr(String name, int def) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) {
            return def;
        }
        try {
            return Integer.parseInt(v.strip());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static float floatEnvOr(String name, float def) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) {
            return def;
        }
        try {
            return Float.parseFloat(v.strip().replace(',', '.'));
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static String humanDuration(Duration d) {
        long totalSec = d.getSeconds();
        long h = totalSec / 3600;
        long m = (totalSec % 3600) / 60;
        long s = totalSec % 60;
        if (h > 0) {
            return String.format("%d ч %d мин %d с", h, m, s);
        }
        if (m > 0) {
            return String.format("%d мин %d с", m, s);
        }
        if (totalSec > 0) {
            return String.format("%d с", totalSec);
        }
        return String.format("%.2f с", d.toNanos() / 1_000_000_000.0);
    }

    private static boolean tryLoadWeights(GPTModel model, Path weights, String context) {
        try {
            model.loadWeights(weights.toString());
            return true;
        } catch (IOException | ClassNotFoundException e) {
            log.warn("Пропуск несовместимых весов ({}): {}", context, e.getMessage());
            return false;
        }
    }

    static String sanitizeFileName(String fileName) {
        String base = fileName.replaceAll("[^a-zA-Z0-9._-]", "_");
        if (base.isEmpty()) {
            return "book";
        }
        return base.length() > 120 ? base.substring(0, 120) : base;
    }

    private static List<Path> listTxtFilesSorted(Path dataDir) throws IOException {
        if (!Files.isDirectory(dataDir)) {
            return List.of();
        }
        try (Stream<Path> walk = Files.walk(dataDir)) {
            return walk.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(".txt"))
                    .sorted(Comparator.comparing(Path::toString))
                    .collect(Collectors.toCollection(ArrayList::new));
        }
    }

}
