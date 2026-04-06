package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.training.LLMTrainer;
import com.veles.llm.jgpt.training.PresetConfig;
import com.veles.llm.jgpt.training.PresetDecider;
import com.veles.llm.jgpt.training.PresetDecider.Action;
import com.veles.llm.jgpt.training.TrainingEventCallback;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Java-аналог {@code jgpt-smart.sh}: один JVM, метрики через {@link TrainingEventCallback}, пресеты —
 * {@link PresetConfig}.
 *
 * <p>Запуск:
 *
 * <pre>{@code mvn -q exec:java -Dexec.mainClass=com.veles.llm.jgpt.app.SmartTrainingSupervisor}</pre>
 *
 * Аргументы: {@code [--boo DIR] [--data-dir DIR] [имя_пресета]}
 */
public final class SmartTrainingSupervisor {

    private static final Logger log = LoggerFactory.getLogger(SmartTrainingSupervisor.class);

    private static final Path STATE_DIR = Path.of("state");
    private static final Path PRESET_IDX_FILE = STATE_DIR.resolve("current_preset_idx");

    private SmartTrainingSupervisor() {}

    public static void main(String[] args) throws Exception {
        TensorOpsGPU.requireCuda("SmartTrainingSupervisor");

        String booRoot = ".";
        String dataDirArg = null;
        String startPresetName = null;
        for (int i = 0; i < args.length; i++) {
            if ("--boo".equals(args[i]) && i + 1 < args.length) {
                booRoot = args[++i];
            } else if ("--data-dir".equals(args[i]) && i + 1 < args.length) {
                dataDirArg = args[++i];
            } else if (!args[i].startsWith("--")) {
                startPresetName = args[i];
            }
        }

        Path root = Path.of(booRoot).toAbsolutePath().normalize();
        Path dataDir =
                dataDirArg != null
                        ? Path.of(dataDirArg).toAbsolutePath().normalize()
                        : root.resolve("data").resolve("books");

        List<Path> books = AllBooksTrain.listTxtFilesSortedPublic(dataDir);
        if (books.isEmpty()) {
            log.error("В каталоге нет .txt файлов: {}", dataDir);
            System.exit(1);
        }

        List<PresetConfig> chain = PresetConfig.SMART_PRESET_CHAIN;
        int idx = resolveStartIndex(startPresetName);
        if (idx < 0) {
            log.error("Неизвестный пресет: {}", startPresetName);
            System.exit(1);
        }

        PresetDecider decider = new PresetDecider();
        int downgradeCount = 0;
        int upgradeCount = 0;

        ExecutorService trainExec =
                Executors.newSingleThreadExecutor(
                        r -> {
                            Thread t = new Thread(r, "jgpt-smart-train");
                            t.setDaemon(false);
                            return t;
                        });

        try {
            outerLoop:
            while (true) {
                PresetConfig preset = chain.get(idx);
                decider.resetForNewPreset();
                writePresetState(idx, preset.name());

                log.info("════════════════════════════════════════════════════════════");
                log.info(" JGPT Smart (Java) | пресет: {} (idx={})", preset.name(), idx);
                log.info(" Downgrade: {} | Upgrade: {}", downgradeCount, upgradeCount);
                log.info("════════════════════════════════════════════════════════════");

                AtomicReference<LLMTrainer> trainerRef = new AtomicReference<>();
                AtomicReference<Action> armedSwitch = new AtomicReference<>(Action.NONE);
                TrainingEventCallback callback = new SupervisorCallback(decider);

                Future<?> trainFut =
                        trainExec.submit(
                                () -> {
                                    try {
                                        AllBooksTrain.runWithPreset(
                                                root,
                                                dataDir,
                                                books,
                                                preset,
                                                callback,
                                                preset.createLossScaler(),
                                                trainerRef);
                                    } catch (OutOfMemoryError oom) {
                                        decider.onOutOfMemoryOrCudaError();
                                        throw oom;
                                    } catch (Throwable t) {
                                        if (isLikelyCudaOrNativeFatal(t)) {
                                            decider.onOutOfMemoryOrCudaError();
                                        }
                                        throw new RuntimeException(t);
                                    }
                                });

                while (!trainFut.isDone()) {
                    Action a = decider.pollAction(idx, chain.size());
                    if (a != Action.NONE) {
                        armedSwitch.compareAndSet(Action.NONE, a);
                        LLMTrainer tr = trainerRef.get();
                        if (tr != null) {
                            tr.requestSupervisedStop();
                        }
                    }
                    try {
                        Thread.sleep(500L);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        LLMTrainer tr = trainerRef.get();
                        if (tr != null) {
                            tr.requestSupervisedStop();
                        }
                        try {
                            trainFut.get();
                        } catch (Exception ignored) {
                        }
                        throw ie;
                    }
                }

                try {
                    trainFut.get();
                } catch (ExecutionException ee) {
                    Throwable c = ee.getCause() != null ? ee.getCause() : ee;
                    log.warn("Сессия пресета завершилась с ошибкой: {}", c.getMessage());
                    if (isCudaContextLikelyCorrupted(c)) {
                        log.error(
                                "[SMART] Фатальная CUDA-ошибка — GPU-контекст повреждён, перезапустите JVM.");
                        log.error(
                                "[SMART] Продолжение в том же процессе небезопасно. Возобновление: "
                                        + "снова запустите SmartTrainingSupervisor (или ./scripts/jgpt-smart.sh).");
                        System.exit(2);
                    }
                    if (isCheckpointShapeMismatch(c)) {
                        log.error(
                                "[SMART] Несовпадение формы чекпоинта и модели — смена пресета не поможет. "
                                        + "Проверьте JGPT_MAX_SEQ_LEN / архитектуру и чекпоинт в checkpoints/all_books.");
                        System.exit(3);
                    }
                    if (idx + 1 >= chain.size()) {
                        log.error("Достигнут последний пресет — нужна ручная диагностика.");
                        System.exit(1);
                    }
                    downgradeCount++;
                    idx++;
                    Thread.sleep(3000L);
                    continue;
                }

                LLMTrainer tr = trainerRef.get();
                Action applied = armedSwitch.get();
                if (tr != null
                        && tr.exitedDueToSupervisorRequest()
                        && applied != Action.NONE) {
                    if (applied == Action.DOWNGRADE) {
                        if (idx + 1 >= chain.size()) {
                            log.error("Последний пресет и downgrade — остановка.");
                            System.exit(1);
                        }
                        downgradeCount++;
                        idx++;
                    } else if (applied == Action.UPGRADE && idx > 0) {
                        upgradeCount++;
                        idx--;
                    }
                    Thread.sleep(3000L);
                    continue;
                }

                log.info("[SMART] Обучение завершено штатно (пресет={}).", preset.name());
                break outerLoop;
            }
        } finally {
            trainExec.shutdown();
        }

        log.info(
                "[SMART] Итого: Downgrade={} | Upgrade={} | финальный idx={}",
                downgradeCount,
                upgradeCount,
                idx);
    }

    /**
     * После illegal access / асинхронных ошибок драйвера дальнейшие вызовы CUDA в этом процессе ненадёжны
     * (в отличие от отдельного JVM на каждый запуск в bash-обёртке).
     */
    private static boolean isCudaContextLikelyCorrupted(Throwable t) {
        for (Throwable x = t; x != null; x = x.getCause()) {
            if (x instanceof OutOfMemoryError) {
                return false;
            }
            String m = x.getMessage();
            if (m == null) {
                continue;
            }
            String u = m.toLowerCase();
            if (u.contains("illegal memory access")
                    || u.contains("code 700")
                    || u.contains("cuda error")
                    || u.contains("cudagraphlaunch")
                    || (u.contains("cublas") && u.contains("error"))) {
                return true;
            }
        }
        return false;
    }

    private static boolean isCheckpointShapeMismatch(Throwable t) {
        for (Throwable x = t; x != null; x = x.getCause()) {
            String m = x.getMessage();
            if (m != null && m.contains("shape mismatch")) {
                return true;
            }
        }
        return false;
    }

    private static boolean isLikelyCudaOrNativeFatal(Throwable t) {
        if (t == null) {
            return false;
        }
        if (t instanceof OutOfMemoryError) {
            return true;
        }
        for (Throwable x = t; x != null; x = x.getCause()) {
            String m = x.getMessage();
            if (m != null) {
                String u = m.toLowerCase();
                if (u.contains("cudamalloc")
                        || u.contains("out of memory")
                        || u.contains("outofmemoryerror")
                        || u.contains("illegal memory access")
                        || u.contains("code 700")) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int resolveStartIndex(String startPresetName) throws Exception {
        Files.createDirectories(STATE_DIR);
        if (startPresetName != null && !startPresetName.isBlank()) {
            int i = PresetConfig.indexOfName(startPresetName.trim());
            return i;
        }
        if (Files.isRegularFile(PRESET_IDX_FILE)) {
            try {
                int fromFile = Integer.parseInt(Files.readString(PRESET_IDX_FILE).trim());
                if (fromFile >= 0 && fromFile < PresetConfig.SMART_PRESET_CHAIN.size()) {
                    return fromFile;
                }
            } catch (NumberFormatException ignored) {
            }
        }
        return 1;
    }

    private static void writePresetState(int idx, String presetName) throws Exception {
        Files.createDirectories(STATE_DIR);
        Files.writeString(PRESET_IDX_FILE, String.valueOf(idx));
        Path curEnv = STATE_DIR.resolve("current.env");
        Path target = Path.of("..", "env", presetName + ".env");
        try {
            Files.deleteIfExists(curEnv);
        } catch (Exception ignored) {
        }
        try {
            Files.createSymbolicLink(curEnv, target);
        } catch (UnsupportedOperationException | java.nio.file.FileSystemException e) {
            log.debug("Не удалось обновить symlink state/current.env: {}", e.getMessage());
        }
    }

    private static final class SupervisorCallback implements TrainingEventCallback {

        private final PresetDecider decider;

        SupervisorCallback(PresetDecider decider) {
            this.decider = decider;
        }

        @Override
        public void onOptimizerStepCompleted(int globalStep, int epochOneBased) {
            decider.onOptimizerStepCompleted();
        }

        @Override
        public void onEvalCompleted(
                int epochOneBased,
                float evalLoss,
                float bestLossAfterEval,
                boolean improvedBest) {
            decider.onEvalCompleted(improvedBest);
        }

        @Override
        public void onOverflowStepSkipped(
                int plannedStep, int consecutiveSkipsForPlannedStep, float scaleAfterSkip) {
            decider.onOverflowStepSkipped(consecutiveSkipsForPlannedStep);
        }

        @Override
        public void onOutOfMemoryOrCudaError(Throwable error) {
            decider.onOutOfMemoryOrCudaError();
        }
    }
}
