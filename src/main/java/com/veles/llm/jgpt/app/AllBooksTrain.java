package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.data.SftCorpus;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.training.CheckpointPruner;
import com.veles.llm.jgpt.training.LLMConfig;
import com.veles.llm.jgpt.training.LLMTrainer;
import com.veles.llm.jgpt.training.TrainingConfig;
import com.veles.llm.jgpt.training.TrainingPlanExhaustedException;
import com.veles.llm.jgpt.training.TrainingStopFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
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
 *
 * <p>Все {@code .txt} из {@code data/books/} (или {@code --data-dir}) объединяются в один
 * {@link DataLoader}. Чекпоинты — в {@code checkpoints/all_books/} или
 * {@code checkpoints/$JGPT_CHECKPOINT_SUBDIR/}.
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
 *   <li><b>JGPT_VAL_FRACTION</b> — доля под hold-out validation (например {@code 0.05}); {@code JGPT_VAL_SEED}
 *       — seed перемешивания при split (по умолчанию 42). Без env eval считается на train-потоке, как раньше.
 *       При {@code JGPT_SFT=1} по умолчанию режутся уже упакованные окна; {@code JGPT_SFT_SPLIT=dialog} —
 *       сначала откладываются диалоги, затем train/val упаковываются отдельно.</li>
 *   <li><b>JGPT_TRAIN_SHUFFLE_SEED</b> — seed {@link com.veles.llm.jgpt.data.DataLoader#shuffle()} на каждой эпохе
 *       (по умолчанию 42); другой seed — другой порядок батчей при том же корпусе.</li>
 *   <li><b>JGPT_IF_STEP_BEYOND_PLAN</b> — если из чекпоинта {@code globalStep} не меньше нового
 *       {@code totalTrainingSteps} (типично после смены пресета/батча): {@code skip} (по умолчанию вне smart),
 *       {@code restart_schedule} (сброс шага и LR-цикла, веса/Adam/best eval сохраняются;
 *       задаётся по умолчанию в {@code scripts/linux/jgpt-smart.sh}), {@code fail} — выход с кодом 2.</li>
 *   <li><b>JGPT_LEARNING_RATE</b> или <b>JGPT_LR</b> — базовый learning rate пресета (дообучение на плато).</li>
 *   <li><b>JGPT_SFT=1</b> — JSONL instruction/chat: лосс только на токенах ассистента,
 *       упаковка нескольких диалогов в окно {@code seq}. Ищутся файлы {@code .jsonl}.</li>
 *   <li><b>JGPT_VOCAB_SIZE</b> — целевой размер BPE, если токенизатор ещё не сохранён.</li>
 *   <li><b>JGPT_TOKENIZER_PATH</b> — файл BPE (относительно {@code --boo} или абсолютный);
 *       иначе {@code checkpoints/tokenizer_global.bin}.</li>
 *   <li><b>JGPT_CHECKPOINT_SUBDIR</b> — подкаталог в {@code checkpoints/} (по умолчанию {@code all_books}).</li>
 *   <li><b>JGPT_CKPT_PRUNE</b> — автоматическое удаление старых {@code checkpoint_step_*}/{@code model_step_*}
 *       и эпоховых снимков (см. {@link CheckpointPruner}); {@code 0} — выключить.
 *       {@code JGPT_CKPT_KEEP_STEP_SNAPSHOTS} (по умолчанию 2), {@code JGPT_CKPT_KEEP_EPOCH_SNAPSHOTS}
 *       (по умолчанию 2); {@code 0} = не трогать соответствующий вид файлов.</li>
 * </ul>
 *
 * <p>Пример дообучения после добавления новых книг:
 * <pre>
 *   JGPT_TRAIN_LOSS_MODE=sampled JGPT_SAMPLED_CE_CANDIDATES=512 \
 *   JGPT_MAX_SEQ_LEN=1024 JGPT_CE_ASYNC=0 JGPT_INTERACTIVE_EVERY=0 \
 *   JGPT_FINETUNE=1 JGPT_EPOCHS=40 \
 *   ./scripts/linux/jgpt-smart.sh
 * </pre>
 */
public final class AllBooksTrain {

    private static final Logger log = LoggerFactory.getLogger(AllBooksTrain.class);
    private static final String CHECKPOINTS_DIR = "checkpoints";

    public static void main(String[] args) throws Exception {
        TensorOpsGPU.requireCuda("AllBooksTrain");
        TrainingStopFile.clearStaleAtStartup();

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

        log.info("=".repeat(60));
        log.info("[ALL-BOOKS] обучение на всём корпусе (единый датасет)");
        log.info("[DATA] каталог с текстами: {}", dataDir);
        log.info("[CKPT] чекпоинты: {}", resolveCheckpointsDir(root));
        log.info("=".repeat(60));

        boolean sft = isSftMode();
        List<Path> books = sft ? listJsonlFilesSorted(dataDir) : listTxtFilesSorted(dataDir);
        if (books.isEmpty()) {
            log.error(
                    sft ? "В каталоге нет .jsonl файлов: {}" : "В каталоге нет .txt файлов: {}",
                    dataDir);
            System.exit(1);
        }
        log.info("[DATA] {} найдено: {}", sft ? "jsonl" : "книг", books.size());
        if (sft) {
            log.info("[SFT] лосс только на ответах ассистента (роли <user>/<assistant> если есть в BPE)");
        }

        LLMConfig llm = LLMConfig.applyLearningRateOverrideFromEnv(
                LLMConfig.applyAccumulationStepsOverrideFromEnv(
                        LLMConfig.applyEpochsOverrideFromEnv(
                                LLMConfig.applyWidthOverrideFromEnv(
                                        LLMConfig.applyPresetNumLayersOverrideFromEnv(
                                                LLMConfig.applyVocabSizeOverrideFromEnv(
                                                        LLMConfig.applySeqLenOverrideFromEnv(
                                                                LLMConfig.applyBatchSizeOverrideFromEnv(
                                                                        LLMConfig.canonical()))))))));
        runCore(root, books, llm);
    }

    private static void runCore(Path root, List<Path> books, LLMConfig llm)
            throws Exception {
        Path checkpointsDir = resolveCheckpointsDir(root);
        Path tokenizerPath = resolveTokenizerPath(root);

        log.info(
                "[CFG] seq={}, d_model={}, d_ff={}, layers={}, heads={}, vocab={}, lr={}, ~{} params",
                llm.maxSeqLen,
                llm.dModel,
                llm.dIntermediate,
                llm.numLayers,
                llm.numHeads,
                llm.vocabSize,
                String.format(Locale.ROOT, "%.4g", llm.learningRate),
                String.format(Locale.US, "%,d", llm.estimateParameters()));

        // --- токенизатор ---
        BPETokenizer tokenizer;
        if (Files.isRegularFile(tokenizerPath)) {
            log.info("[DATA] загрузка токенизатора: {}", tokenizerPath);
            tokenizer = BPETokenizer.load(tokenizerPath.toString());
            if (tokenizer.getVocabSize() != llm.vocabSize) {
                log.warn(
                        "[DATA] vocab токенизатора {} ≠ JGPT_VOCAB_SIZE/canonical {} — модель берёт размер из файла",
                        tokenizer.getVocabSize(),
                        llm.vocabSize);
            }
        } else {
            log.info("[DATA] обучение BPE-токенизатора на всём корпусе...");
            List<String> allTexts =
                    isSftMode() ? SftCorpus.collectPlainTexts(books) : new ArrayList<>();
            if (!isSftMode()) {
                for (Path p : books) {
                    allTexts.add(readUtf8(p));
                }
            }
            tokenizer = BPETokenizer.train(allTexts, llm.vocabSize);
            allTexts.clear();
            Files.createDirectories(tokenizerPath.getParent());
            tokenizer.save(tokenizerPath.toString());
            log.info("[DATA] токенизатор сохранён: {} (vocab={})",
                    tokenizerPath.getFileName(), tokenizer.getVocabSize());
        }
        int vocabSize = tokenizer.getVocabSize();
        log.info(
                "[DATA] размер словаря: {}  lowercase={}  role_tokens={}",
                vocabSize,
                tokenizer.lowercase(),
                tokenizer.hasChatRoleTokens());

        // --- датасет: все книги в один DataLoader ---
        Files.createDirectories(checkpointsDir);
        DataLoader dataLoader = new DataLoader(tokenizer, llm.maxSeqLen, llm.batchSize);
        DataLoader trainLoader = dataLoader;
        DataLoader evalLoader = null;
        boolean sftDialogHoldout = false;

        if (isSftMode()) {
            double valFracEarly = readValFraction();
            long valSeedEarly = readValSeed();
            if (SftCorpus.dialogSplitFromEnv() && valFracEarly > 0d) {
                evalLoader = new DataLoader(tokenizer, llm.maxSeqLen, llm.batchSize);
                int windows =
                        SftCorpus.loadDialogHoldout(
                                dataLoader, evalLoader, tokenizer, books, valFracEarly, valSeedEarly);
                log.info("[SFT] окон обучения: {} (~{} батчей/эпоха)", windows, dataLoader.numBatches());
                if (windows == 0) {
                    throw new IllegalStateException("Нет SFT-окон — проверьте JSONL и шаблон реплик");
                }
                sftDialogHoldout = true;
                trainLoader = dataLoader;
                if (evalLoader.numSequences() == 0) {
                    evalLoader = null;
                    log.info("[DATA] hold-out по диалогам не создан — полный корпус в train");
                } else {
                    log.info(
                            "[DATA] hold-out validation (диалоги): fraction={}, seed={}, train_windows={}, val_windows={}",
                            String.format(Locale.ROOT, "%.4f", valFracEarly),
                            valSeedEarly,
                            trainLoader.numSequences(),
                            evalLoader.numSequences());
                }
            } else {
                int windows = SftCorpus.loadInto(dataLoader, tokenizer, books);
                log.info("[SFT] окон обучения: {} (~{} батчей/эпоха)", windows, dataLoader.numBatches());
                if (windows == 0) {
                    throw new IllegalStateException("Нет SFT-окон — проверьте JSONL и шаблон реплик");
                }
            }
        } else {
            int threads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
            log.info("[DATA] кодирование: {} потоков (последовательно для экономии памяти)", threads);

            long totalChars = 0;
            int skipped = 0;
            List<int[]> docs = new ArrayList<>(books.size());

            try (ExecutorService pool = Executors.newFixedThreadPool(threads)) {
                for (Path p : books) {
                    int[] tokens;
                    try {
                        Future<int[]> future = pool.submit((Callable<int[]>) () -> {
                            String text = readUtf8(p);
                            return tokenizer.encode(text, true);
                        });
                        tokens = future.get();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        throw e;
                    } catch (Exception e) {
                        log.warn("[DATA] ошибка кодирования {}: {}", p.getFileName(), e.getMessage());
                        skipped++;
                        continue;
                    }
                    // <bos>+<eos> без текста — пустой файл, в поток не идёт
                    if (tokens.length <= 2) {
                        skipped++;
                        continue;
                    }
                    totalChars += p.toFile().length();
                    log.debug("[DATA]   {} → {} токенов", p.getFileName(), tokens.length);
                    docs.add(tokens);
                }
            }
            log.info("[DATA] итого: {} символов, {} документов закодировано, {} пропущено (пустые/ошибка)",
                    String.format("%,d", totalChars), docs.size(), skipped);
            if (docs.isEmpty()) {
                throw new IllegalStateException("Нет документов для обучения в " + books.size() + " файлах");
            }

            double valFracDocs = readValFraction();
            long valSeedDocs = readValSeed();
            DataLoader valPacked = null;
            if (valFracDocs > 0d) {
                valPacked = new DataLoader(tokenizer, llm.maxSeqLen, llm.batchSize);
            }
            PackedDocsStats st =
                    packDocumentsIntoLoaders(docs, dataLoader, valPacked, llm.maxSeqLen, valFracDocs, valSeedDocs);
            docs.clear();
            log.info(
                    "[DATA] упаковка документов через <eos>: train_docs={} val_docs={} train_tokens={} val_tokens={}"
                            + " (хвост потока < {} токенов теряется один раз, не на документ)",
                    st.trainDocs,
                    st.valDocs,
                    String.format("%,d", st.trainTokens),
                    String.format("%,d", st.valTokens),
                    llm.maxSeqLen + 1);
            int nSeq = dataLoader.numSequences();
            log.info("[DATA] всего последовательностей: {} (~{} батчей/эпоха)",
                    nSeq, dataLoader.numBatches());
            if (nSeq == 0) {
                throw new IllegalStateException(
                        "Нет последовательностей — корпус короче одного окна (нужно >" + llm.maxSeqLen + " токенов)");
            }
            if (valPacked != null) {
                if (valPacked.numSequences() >= llm.batchSize) {
                    evalLoader = valPacked;
                    sftDialogHoldout = true; // hold-out уже сформирован (по документам), общий split ниже не нужен
                    log.info(
                            "[DATA] hold-out validation (по документам): fraction={}, seed={}, train_windows={}, val_windows={}",
                            String.format(Locale.ROOT, "%.4f", valFracDocs),
                            valSeedDocs,
                            trainLoader.numSequences(),
                            evalLoader.numSequences());
                } else {
                    log.info(
                            "[DATA] hold-out по документам не создан (val_windows={} < batch {}) — eval на train-потоке",
                            valPacked.numSequences(),
                            llm.batchSize);
                }
            }
        }

        double valFrac = readValFraction();
        long valSeed = readValSeed();
        if (sftDialogHoldout) {
            // train/eval уже заполнены (SFT по диалогам или LM по документам)
        } else if (valFrac > 0d) {
            DataLoader.TrainValSplit split = DataLoader.splitTrainValidation(dataLoader, valFrac, valSeed);
            trainLoader = split.train;
            evalLoader = split.validation;
            if (evalLoader != null) {
                log.info(
                        "[DATA] hold-out validation: fraction={}, seed={}, train_windows={}, val_windows={}",
                        String.format(Locale.ROOT, "%.4f", valFrac),
                        valSeed,
                        trainLoader.numSequences(),
                        evalLoader.numSequences());
            } else {
                log.info("[DATA] hold-out не создан (мало окон или доля) — полный корпус в train, eval на train-потоке");
            }
        } else {
            log.info("[DATA] JGPT_VAL_FRACTION не задан — метрики eval на train-потоке (как без hold-out)");
        }

        // --- модель ---
        boolean gpuResident = LLMConfig.canonicalGpuTrain();
        GPTModel model = new GPTModel(vocabSize, llm.maxSeqLen, llm.dModel,
                llm.numHeads, llm.numLayers, llm.dIntermediate, gpuResident);

        // Resume: самый свежий по globalStep среди checkpoint_final/step_N/epoch_N/best (после жёсткого
        // убийства процесса checkpoint_final может отсутствовать или быть старее step_N). Веса берём из
        // парного model_<name>.bin, чтобы Adam и веса были с одного и того же шага.
        Optional<Path> resumeCkpt = findResumeCheckpoint(checkpointsDir);
        Path weightsToLoad = null;
        if (resumeCkpt.isPresent()) {
            Path paired = pairedModelWeights(resumeCkpt.get());
            Path modelFinal = checkpointsDir.resolve("model_final.bin");
            if (Files.isRegularFile(paired)) {
                weightsToLoad = paired;
            } else if (Files.isRegularFile(modelFinal)) {
                log.warn(
                        "[CKPT] нет парных весов {} для {} — беру model_final.bin (веса и Adam могут быть с разных шагов)",
                        paired.getFileName(),
                        resumeCkpt.get().getFileName());
                weightsToLoad = modelFinal;
            } else {
                log.error("[CKPT] найден {}, но ни парных весов, ни model_final.bin нет — веса случайные!",
                        resumeCkpt.get().getFileName());
            }
        } else {
            Path modelFinal = checkpointsDir.resolve("model_final.bin");
            if (Files.isRegularFile(modelFinal)) {
                weightsToLoad = modelFinal;
            }
        }
        if (weightsToLoad != null) {
            log.info("[CKPT] продолжение: загрузка весов из {}", weightsToLoad.getFileName());
            model.loadWeights(weightsToLoad.toString());
        }

        // --- тренировка ---
        boolean finetune = isFinetuneMode();
        TrainingConfig trainConfig = llm.toTrainingConfig(checkpointsDir.toString(), vocabSize);
        LLMTrainer trainer = new LLMTrainer(model, trainConfig, trainLoader, evalLoader);

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

        if (TensorOpsGPU.isGpuAvailable()) {
            long usedMb = TensorOpsGPU.getGpuMemoryAllocated() / (1024 * 1024);
            long totalMb = TensorOpsGPU.getGpuMemoryReserved() / (1024 * 1024);
            log.info("[VRAM] до старта обучения: занято {} МиБ / {} МиБ (свободно {} МиБ)",
                    usedMb, totalMb, totalMb - usedMb);
        }

        AtomicBoolean exitCheckpointDone = new AtomicBoolean();
        TrainingStopFile.installOsInterrupt(trainer::requestSupervisedStop);
        log.info(
                "[STOP] мягкая остановка: создайте файл {} (Windows: .\\scripts\\windows\\jgpt-stop-train.cmd)",
                TrainingStopFile.resolveFromEnv());

        // Backup: JVM shutdown (не срабатывает, если Windows убил процесс через «завершить пакет?»).
        // ВАЖНО: если прогресса нет (OOM при первом шаге), НЕ перезаписываем checkpoint_final.bin
        // чтобы не затереть эпоху из предыдущего запуска.
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (!exitCheckpointDone.compareAndSet(false, true)) {
                return;
            }
            log.info("[SHUTDOWN] Получен сигнал остановки — сохраняем checkpoint...");
            System.out.println("[SHUTDOWN] Получен сигнал остановки — сохраняем checkpoint...");
            System.out.flush();
            try {
                if (trainer.getGlobalStep() > trainer.getShutdownProgressBaselineStep()) {
                    trainer.saveCheckpoint("final");
                    trainer.awaitPendingCheckpointWrites();
                    log.info("[SHUTDOWN] checkpoint сохранён. Возобновление: тот же скрипт без --fresh");
                    System.out.println("[SHUTDOWN] checkpoint сохранён");
                    System.out.flush();
                } else {
                    trainer.saveCheckpoint("emergency");
                    log.warn(
                            "[SHUTDOWN] Нет прогресса (шаг {} ≤ базы {}) — checkpoint_final.bin НЕ перезаписан. "
                                    + "Аварийный: checkpoint_emergency.bin",
                            trainer.getGlobalStep(),
                            trainer.getShutdownProgressBaselineStep());
                }
            } catch (Exception e) {
                log.warn("[SHUTDOWN] Не удалось сохранить checkpoint: {}", e.getMessage());
            }
        }, "shutdown-ckpt"));

        if (CheckpointPruner.pruningEnabled()) {
            try {
                int ks = CheckpointPruner.keepStepSnapshots();
                if (ks > 0) {
                    CheckpointPruner.pruneStepTriples(checkpointsDir, ks);
                }
                int ke = CheckpointPruner.keepEpochSnapshots();
                if (ke > 0) {
                    CheckpointPruner.pruneEpochTriples(checkpointsDir, ke);
                }
            } catch (IOException e) {
                log.warn("[CKPT] не удалось подчистить старые снимки перед обучением: {}", e.getMessage());
            }
        }

        log.info("=".repeat(60));
        try {
            try {
                trainer.train();
            } catch (TrainingPlanExhaustedException e) {
                log.error("[ALL-BOOKS] {}", e.getMessage());
                System.exit(2);
            }

            if (exitCheckpointDone.compareAndSet(false, true)) {
                trainer.saveCheckpoint("final");
            }
            trainer.awaitPendingCheckpointWrites();
            if (trainer.exitedDueToSupervisorRequest()) {
                log.info(
                        "[SHUTDOWN] checkpoint сохранён. Возобновление: тот же скрипт без --fresh. Лучший eval loss: {}",
                        String.format("%.4f", trainer.getBestLoss()));
            } else {
                log.info("[ALL-BOOKS] обучение завершено. Лучший eval loss: {}",
                        String.format("%.4f", trainer.getBestLoss()));
            }
        } finally {
            if (TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.synchronizeStream();
                TensorOpsGPU.drainDeferredGpuBuffers();
                TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
            }
        }
    }


    /** Статистика упаковки документов в LM-окна. */
    record PackedDocsStats(int trainDocs, int valDocs, long trainTokens, long valTokens) {}

    /**
     * Упаковывает документы ({@code <bos> … <eos>} каждый) в непрерывные потоки train/val и режет их на окна
     * {@code maxSeqLen+1} через {@link DataLoader#loadTokens(int[])}. Короткие документы и хвосты не теряются:
     * граница документа — это {@code <eos><bos>} внутри окна, как в GPT-2.
     *
     * <p>Hold-out — по документам (не по окнам): {@code valFraction} документов после детерминированного
     * перемешивания по {@code seed}. Если val-документов не хватает даже на один батч окон, всё уходит в train.
     *
     * @param val {@code null} — без hold-out
     */
    static PackedDocsStats packDocumentsIntoLoaders(
            List<int[]> docs, DataLoader train, DataLoader val, int maxSeqLen, double valFraction, long seed) {
        int n = docs.size();
        boolean[] isVal = new boolean[n];
        int valDocs = 0;
        long valTokens = 0;
        if (val != null && valFraction > 0d && n >= 2) {
            List<Integer> order = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                order.add(i);
            }
            java.util.Collections.shuffle(order, new java.util.Random(seed));
            int nVal = (int) Math.round(n * valFraction);
            nVal = Math.max(0, Math.min(nVal, n - 1));
            for (int k = 0; k < nVal; k++) {
                isVal[order.get(k)] = true;
                valTokens += docs.get(order.get(k)).length;
            }
            valDocs = nVal;
            long minValTokens = (long) train.getBatchSize() * maxSeqLen + 1;
            if (valTokens < minValTokens) {
                log.warn(
                        "[DATA] val-документов ({}, {} токенов) не хватает на батч окон ({}) — всё в train",
                        valDocs,
                        valTokens,
                        minValTokens);
                java.util.Arrays.fill(isVal, false);
                valDocs = 0;
                valTokens = 0;
            }
        }
        long trainTokens = 0;
        for (int i = 0; i < n; i++) {
            if (!isVal[i]) {
                trainTokens += docs.get(i).length;
            }
        }
        train.loadTokens(concatDocs(docs, isVal, false, trainTokens));
        if (valDocs > 0) {
            val.loadTokens(concatDocs(docs, isVal, true, valTokens));
        }
        return new PackedDocsStats(n - valDocs, valDocs, trainTokens, valTokens);
    }

    private static int[] concatDocs(List<int[]> docs, boolean[] isVal, boolean takeVal, long total) {
        if (total > Integer.MAX_VALUE - 8) {
            throw new IllegalStateException("корпус слишком большой для одного int[] потока: " + total + " токенов");
        }
        int[] stream = new int[(int) total];
        int off = 0;
        for (int i = 0; i < docs.size(); i++) {
            if (isVal[i] != takeVal) {
                continue;
            }
            int[] d = docs.get(i);
            System.arraycopy(d, 0, stream, off, d.length);
            off += d.length;
        }
        return stream;
    }

    /**
     * Ищет чекпоинт для resume: среди {@code checkpoint_final.bin}, {@code checkpoint_step_N.bin},
     * {@code checkpoint_epoch_N.bin}, {@code checkpoint_best.bin} берётся тот, у кого больший {@code globalStep}
     * в заголовке (при равенстве — {@code final}). {@code checkpoint_emergency.bin} — только если других нет.
     * Файлы, у которых нет парных {@code model_<name>.bin}, пропускаются (кроме случая, когда есть model_final).
     */
    static Optional<Path> findResumeCheckpoint(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return Optional.empty();

        List<Path> candidates;
        try (Stream<Path> s = Files.list(dir)) {
            candidates =
                    s.filter(Files::isRegularFile)
                            .filter(p -> {
                                String n = p.getFileName().toString();
                                if (!n.startsWith("checkpoint_") || !n.endsWith(".bin")) {
                                    return false;
                                }
                                return n.equals("checkpoint_final.bin")
                                        || n.equals("checkpoint_best.bin")
                                        || n.startsWith("checkpoint_step_")
                                        || n.startsWith("checkpoint_epoch_");
                            })
                            .collect(Collectors.toCollection(ArrayList::new));
        }
        boolean hasModelFinal = Files.isRegularFile(dir.resolve("model_final.bin"));
        Path best = null;
        int bestStep = Integer.MIN_VALUE;
        for (Path p : candidates) {
            if (!Files.isRegularFile(pairedModelWeights(p)) && !hasModelFinal) {
                continue;
            }
            int step = LLMTrainer.peekCheckpointGlobalStep(p);
            if (step < 0) {
                continue;
            }
            boolean isFinal = p.getFileName().toString().equals("checkpoint_final.bin");
            if (step > bestStep || (step == bestStep && isFinal)) {
                best = p;
                bestStep = step;
            }
        }
        if (best != null) {
            if (candidates.size() > 1) {
                log.info("[CKPT] resume: выбран {} (globalStep={}) из {} кандидатов",
                        best.getFileName(), bestStep, candidates.size());
            }
            return Optional.of(best);
        }
        Path emergency = dir.resolve("checkpoint_emergency.bin");
        if (Files.isRegularFile(emergency) && LLMTrainer.peekCheckpointGlobalStep(emergency) > 0) {
            log.warn("[CKPT] resume только из checkpoint_emergency.bin (других чекпоинтов нет)");
            return Optional.of(emergency);
        }
        return Optional.empty();
    }

    /** {@code checkpoint_<name>.bin} → {@code model_<name>.bin} в том же каталоге. */
    static Path pairedModelWeights(Path checkpoint) {
        String n = checkpoint.getFileName().toString();
        return checkpoint.resolveSibling("model_" + n.substring("checkpoint_".length()));
    }

    static Path resolveCheckpointsDir(Path root) {
        String sub = System.getenv("JGPT_CHECKPOINT_SUBDIR");
        if (sub == null || sub.isBlank()) {
            sub = "all_books";
        }
        sub = sub.trim();
        if (sub.isEmpty() || sub.contains("/") || sub.contains("\\") || sub.contains("..")) {
            throw new IllegalArgumentException(
                    "JGPT_CHECKPOINT_SUBDIR must be a single directory name under checkpoints/");
        }
        return root.resolve(CHECKPOINTS_DIR).resolve(sub);
    }

    static Path resolveTokenizerPath(Path root) {
        String e = System.getenv("JGPT_TOKENIZER_PATH");
        if (e == null || e.isBlank()) {
            return root.resolve(CHECKPOINTS_DIR).resolve("tokenizer_global.bin");
        }
        Path p = Path.of(e.trim());
        if (!p.isAbsolute()) {
            p = root.resolve(p);
        }
        return p.toAbsolutePath().normalize();
    }

    static boolean isSftMode() {
        String e = System.getenv("JGPT_SFT");
        if (e == null || e.isBlank()) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
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
        return listFilesSorted(dir, ".txt");
    }

    private static List<Path> listJsonlFilesSorted(Path dir) throws IOException {
        return listFilesSorted(dir, ".jsonl");
    }

    private static List<Path> listFilesSorted(Path dir, String suffix) throws IOException {
        if (!Files.isDirectory(dir)) return List.of();
        try (Stream<Path> s = Files.walk(dir)) {
            return s.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().endsWith(suffix))
                    .sorted(Comparator.comparing(Path::toString))
                    .collect(Collectors.toCollection(ArrayList::new));
        }
    }

    /**
     * Доля окон под hold-out validation: env {@code JGPT_VAL_FRACTION} в {@code (0, 0.5)}; {@code 0} или не задано —
     * без отдельного val.
     */
    static double readValFraction() {
        String e = System.getenv("JGPT_VAL_FRACTION");
        if (e == null || e.isBlank()) {
            return 0d;
        }
        try {
            double v = Double.parseDouble(e.trim().replace(',', '.'));
            if (v <= 0d || v >= 0.5d) {
                log.warn("[CFG] JGPT_VAL_FRACTION={} вне (0; 0.5) — hold-out отключён", e.trim());
                return 0d;
            }
            return v;
        } catch (NumberFormatException _) {
            log.warn("[CFG] JGPT_VAL_FRACTION: не число — hold-out отключён");
            return 0d;
        }
    }

    /** Seed для {@link DataLoader#splitTrainValidation}; env {@code JGPT_VAL_SEED}, иначе 42. */
    static long readValSeed() {
        String e = System.getenv("JGPT_VAL_SEED");
        if (e == null || e.isBlank()) {
            return 42L;
        }
        try {
            return Long.parseLong(e.trim());
        } catch (NumberFormatException _) {
            log.warn("[CFG] JGPT_VAL_SEED: не число — используем 42");
            return 42L;
        }
    }
}
