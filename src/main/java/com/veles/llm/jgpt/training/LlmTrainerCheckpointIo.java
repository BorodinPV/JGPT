package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.model.GPTModel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
/** Сохранение/загрузка чекпоинтов и весов на диск. */
final class LlmTrainerCheckpointIo {

    private static final Logger log = LoggerFactory.getLogger(LlmTrainerCheckpointIo.class);

    static final String CHECKPOINT_FORMAT_V2 = "veles.ckpt.v2";
    static final String CHECKPOINT_FORMAT_V3 = "veles.ckpt.v3";
    static final String CHECKPOINT_FORMAT_V4 = "veles.ckpt.v4";
    /** v4 + состояние динамического loss scale (scale, шагов без overflow). */
    static final String CHECKPOINT_FORMAT_V5 = "veles.ckpt.v5";

    private LlmTrainerCheckpointIo() {}

    /**
     * Атомарная замена файла: запись во временный {@code *.tmp} рядом, затем {@code ATOMIC_MOVE}. Если процесс
     * убьют посреди записи, прежний файл остаётся целым (иначе после «Terminate batch job» на диске оставался
     * обрезанный чекпоинт и терялся и он, и предыдущий).
     */
    static void writeFileAtomically(Path target, IoWriter writer) throws IOException {
        Path dir = target.toAbsolutePath().getParent();
        if (dir != null) {
            Files.createDirectories(dir);
        }
        Path tmp = target.resolveSibling(target.getFileName() + ".tmp");
        try (DataOutputStream out =
                new DataOutputStream(new BufferedOutputStream(new FileOutputStream(tmp.toFile()), 1 << 20))) {
            writer.write(out);
        } catch (IOException | RuntimeException e) {
            try {
                Files.deleteIfExists(tmp);
            } catch (IOException _) {
                // best-effort
            }
            throw e;
        }
        try {
            Files.move(
                    tmp,
                    target,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE,
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (java.nio.file.AtomicMoveNotSupportedException _) {
            Files.move(tmp, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
    }

    @FunctionalInterface
    interface IoWriter {
        void write(DataOutputStream out) throws IOException;
    }

    /**
     * Читает {@code globalStep} из заголовка чекпоинта (v2..v5) без загрузки Adam. {@code -1} — не удалось
     * (legacy-формат или повреждённый файл).
     */
    public static int peekGlobalStep(Path checkpoint) {
        try (DataInputStream dis =
                new DataInputStream(new BufferedInputStream(new FileInputStream(checkpoint.toFile()), 4096))) {
            String tag = dis.readUTF();
            if (CHECKPOINT_FORMAT_V5.equals(tag)
                    || CHECKPOINT_FORMAT_V4.equals(tag)
                    || CHECKPOINT_FORMAT_V3.equals(tag)
                    || CHECKPOINT_FORMAT_V2.equals(tag)) {
                return dis.readInt();
            }
            return -1;
        } catch (IOException | RuntimeException _) {
            return -1;
        }
    }

    static void writeFloatArrayBigEndian(DataOutputStream out, float[] buf) throws IOException {
        if (buf.length == 0) {
            return;
        }
        ByteBuffer bb = ByteBuffer.allocate(buf.length * 4).order(ByteOrder.BIG_ENDIAN);
        bb.asFloatBuffer().put(buf);
        out.write(bb.array());
    }

    static void saveCheckpoint(LLMTrainer t, String name) throws IOException {
        Path dir = Path.of(t.config.checkpointDir);
        Files.createDirectories(dir);
        String path = t.config.checkpointDir + "/checkpoint_" + name + ".bin";

        try {
            Path stateDir = Path.of("state");
            Files.createDirectories(stateDir);
            Files.writeString(stateDir.resolve("last_step.txt"), String.valueOf(t.globalStep));
        } catch (IOException _) {
            // best-effort last_step.txt
        }

        if (t.config.fullGpuTrainStep && t.model.isGpuResident()) {
            t.model.syncWeightsFromGpu(t.model.gpuTensorByTrainableParameter());
        }

        if (!name.startsWith("epoch_")) {
            t.pendingCheckpointDataLoaderIndex = t.dataLoader.getCurrentIndex();
        }

        writeFileAtomically(
                Path.of(path),
                out -> {
                    out.writeUTF(CHECKPOINT_FORMAT_V5);
                    out.writeInt(t.globalStep);
                    out.writeFloat(t.bestLoss);
                    int ep = Math.clamp(t.pendingCheckpointEpochIndex, 0, t.config.epochs);
                    out.writeInt(ep);
                    int nSeq = t.dataLoader.numSequences();
                    int seqIdx = Math.clamp(t.pendingCheckpointDataLoaderIndex, 0, nSeq);
                    out.writeInt(seqIdx);
                    if (t.dynamicLossScaler != null) {
                        out.writeFloat(t.dynamicLossScaler.getScale());
                        out.writeInt(t.dynamicLossScaler.getConsecutiveNonOverflowSteps());
                    } else {
                        out.writeFloat(0f);
                        out.writeInt(0);
                    }
                    t.optimizer.setStep(t.globalStep);
                    t.optimizer.writeMomentBuffers(out, t.parameters);
                });
        log.info(
                "{} checkpoint(v5+Adam+epoch+pos+scale): {} (resumeEpochIndex={}/{}, seqIndex={})",
                com.veles.llm.jgpt.util.LogFmt.badge("CKPT"),
                path,
                Math.clamp(t.pendingCheckpointEpochIndex, 0, t.config.epochs),
                t.config.epochs,
                Math.clamp(t.pendingCheckpointDataLoaderIndex, 0, t.dataLoader.numSequences()));

        if (t.checkpointAsyncIo && t.checkpointIoExecutor != null) {
            List<Tensor> params = t.model.getParameters();
            List<float[]> weightSnap = new ArrayList<>(params.size());
            for (Tensor p : params) {
                weightSnap.add(p.internalBuffer().clone());
            }
            t.checkpointIoTail =
                    t.checkpointIoTail.thenRunAsync(
                            () -> {
                                try {
                                    writeModelWeightsFromSnapshot(t, name, weightSnap);
                                } catch (IOException e) {
                                    log.error("Асинхронная запись весов чекпоинта не удалась: {}", name, e);
                                    return;
                                }
                                try {
                                    CheckpointPruner.pruneAfterSave(dir, name);
                                } catch (IOException e) {
                                    log.warn(
                                            "{} не удалось удалить устаревшие чекпоинты: {}",
                                            com.veles.llm.jgpt.util.LogFmt.badge("CKPT"),
                                            e.toString());
                                }
                            },
                            t.checkpointIoExecutor);
            log.info("{} веса checkpoint '{}' поставлены в очередь асинхронной записи", com.veles.llm.jgpt.util.LogFmt.badge("CKPT"), name);
        } else {
            saveModelWeights(t, name);
            try {
                CheckpointPruner.pruneAfterSave(dir, name);
            } catch (IOException e) {
                log.warn("{} не удалось удалить устаревшие чекпоинты: {}", com.veles.llm.jgpt.util.LogFmt.badge("CKPT"), e.toString());
            }
        }
    }

    static void saveModelWeights(LLMTrainer t, String name) throws IOException {
        Path dir = Path.of(t.config.checkpointDir);
        Files.createDirectories(dir);
        String modelPath = dir.resolve("model_" + name + ".bin").toString();

        if (t.config.fullGpuTrainStep && t.model.isGpuResident()) {
            t.model.syncWeightsFromGpu(t.model.gpuTensorByTrainableParameter());
        }

        List<Tensor> params = t.model.getParameters();
        writeFileAtomically(
                Path.of(modelPath),
                out -> {
                    out.writeUTF(GPTModel.MODEL_WEIGHTS_FORMAT_V1);
                    out.writeInt(params.size());
                    for (Tensor param : params) {
                        int[] shape = param.getShape();
                        out.writeInt(shape.length);
                        for (int d : shape) {
                            out.writeInt(d);
                        }
                        writeFloatArrayBigEndian(out, param.internalBuffer());
                    }
                });
        log.info("{} веса модели записаны: {}", com.veles.llm.jgpt.util.LogFmt.badge("CKPT"), modelPath);

        saveTokenizerAtomically(t, dir.resolve("tokenizer_" + name + ".bin"));
    }

    private static void saveTokenizerAtomically(LLMTrainer t, Path tokPath) throws IOException {
        Path tmp = tokPath.resolveSibling(tokPath.getFileName() + ".tmp");
        t.dataLoader.getTokenizer().save(tmp.toString());
        try {
            Files.move(
                    tmp,
                    tokPath,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE,
                    java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        } catch (java.nio.file.AtomicMoveNotSupportedException _) {
            Files.move(tmp, tokPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        log.info("{} токенизатор записан: {}", com.veles.llm.jgpt.util.LogFmt.badge("CKPT"), tokPath);
    }

    static void writeModelWeightsFromSnapshot(LLMTrainer t, String name, List<float[]> weightSnap) throws IOException {
        Path dir = Path.of(t.config.checkpointDir);
        Files.createDirectories(dir);
        String modelPath = dir.resolve("model_" + name + ".bin").toString();
        List<Tensor> params = t.model.getParameters();
        if (params.size() != weightSnap.size()) {
            throw new IllegalStateException("weight snapshot size mismatch");
        }
        writeFileAtomically(
                Path.of(modelPath),
                out -> {
                    out.writeUTF(GPTModel.MODEL_WEIGHTS_FORMAT_V1);
                    out.writeInt(params.size());
                    for (int i = 0; i < params.size(); i++) {
                        int[] shape = params.get(i).getShape();
                        out.writeInt(shape.length);
                        for (int d : shape) {
                            out.writeInt(d);
                        }
                        writeFloatArrayBigEndian(out, weightSnap.get(i));
                    }
                });
        log.info("{} веса модели записаны (асинхронный снимок): {}", com.veles.llm.jgpt.util.LogFmt.badge("CKPT"), modelPath);
        saveTokenizerAtomically(t, dir.resolve("tokenizer_" + name + ".bin"));
    }

    static void awaitPendingCheckpointWrites(LLMTrainer t) {
        if (t.checkpointIoExecutor == null) {
            return;
        }
        try {
            t.checkpointIoTail.get();
        } catch (InterruptedException _) {
            Thread.currentThread().interrupt();
            log.warn("Ожидание фоновой записи чекпоинта прервано");
        } catch (Exception e) {
            log.warn("Ожидание фоновой записи чекпоинта: {}", e.toString());
        }
    }

    static void loadCheckpoint(LLMTrainer t, String path) throws IOException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(path))) {
            bis.mark(1 << 20);
            DataInputStream dis = new DataInputStream(bis);
            String tag;
            try {
                tag = dis.readUTF();
            } catch (IOException _) {
                bis.reset();
                loadLegacyCheckpoint(t, path);
                return;
            }
            boolean v5 = CHECKPOINT_FORMAT_V5.equals(tag);
            if (v5 || CHECKPOINT_FORMAT_V4.equals(tag)) {
                t.globalStep = dis.readInt();
                t.bestLoss = dis.readFloat();
                if (t.bestLoss == 0f) {
                    log.warn(
                            "Чекпоинт: в файле bestLoss=0 (часто артефакт eval без батчей в старых прогонах) — сброс к «ещё не зафиксирован»");
                    t.bestLoss = Float.MAX_VALUE;
                }
                int ep = dis.readInt();
                t.loadedResumeEpochIndex = Math.clamp(ep, 0, t.config.epochs);
                t.loadedResumeDataLoaderIndex = Math.max(0, dis.readInt());
                t.resumeReplayCheckpointShuffles = true;
                String scaleInfo = "";
                if (v5) {
                    float savedScale = dis.readFloat();
                    int savedGood = dis.readInt();
                    if (t.dynamicLossScaler != null && savedScale > 0f) {
                        t.dynamicLossScaler.restoreState(savedScale, savedGood);
                        scaleInfo =
                                String.format(
                                        java.util.Locale.ROOT,
                                        ", loss scale %.4g× (%d стабильных шагов)",
                                        t.dynamicLossScaler.getScale(),
                                        savedGood);
                    }
                }
                t.optimizer.setStep(t.globalStep);
                t.optimizer.readMomentBuffers(dis, t.parameters);
                log.info(
                        "Чекпоинт загружен ({} + Adam + эпоха + позиция): {} (шаг {}, resumeEpochIndex={}/{}, seqIndex={}, лучший оценочный loss {}{})",
                        v5 ? "v5" : "v4",
                        path,
                        t.globalStep,
                        t.loadedResumeEpochIndex,
                        t.config.epochs,
                        t.loadedResumeDataLoaderIndex,
                        LlmTrainerTrainingFormat.formatEvalBestLossForLog(t.bestLoss),
                        scaleInfo);
                t.syncShutdownProgressBaselineFromGlobalStep();
                return;
            }
            if (CHECKPOINT_FORMAT_V3.equals(tag)) {
                t.globalStep = dis.readInt();
                t.bestLoss = dis.readFloat();
                if (t.bestLoss == 0f) {
                    log.warn(
                            "Чекпоинт: в файле bestLoss=0 (часто артефакт eval без батчей в старых прогонах) — сброс к «ещё не зафиксирован»");
                    t.bestLoss = Float.MAX_VALUE;
                }
                int ep = dis.readInt();
                t.loadedResumeEpochIndex = Math.clamp(ep, 0, t.config.epochs);
                t.loadedResumeDataLoaderIndex = 0;
                t.resumeReplayCheckpointShuffles = true;
                t.optimizer.setStep(t.globalStep);
                t.optimizer.readMomentBuffers(dis, t.parameters);
                log.info(
                        "Чекпоинт загружен (v3 + Adam + эпоха): {} (шаг {}, resumeEpochIndex={}/{}, позиция в эпохе не хранилась — 0; лучший оценочный loss {})",
                        path,
                        t.globalStep,
                        t.loadedResumeEpochIndex,
                        t.config.epochs,
                        LlmTrainerTrainingFormat.formatEvalBestLossForLog(t.bestLoss));
                t.syncShutdownProgressBaselineFromGlobalStep();
                return;
            }
            if (CHECKPOINT_FORMAT_V2.equals(tag)) {
                t.globalStep = dis.readInt();
                t.bestLoss = dis.readFloat();
                if (t.bestLoss == 0f) {
                    log.warn(
                            "Чекпоинт: в файле bestLoss=0 (часто артефакт eval без батчей в старых прогонах) — сброс к «ещё не зафиксирован»");
                    t.bestLoss = Float.MAX_VALUE;
                }
                t.loadedResumeEpochIndex = 0;
                t.loadedResumeDataLoaderIndex = 0;
                t.resumeReplayCheckpointShuffles = false;
                t.optimizer.setStep(t.globalStep);
                t.optimizer.readMomentBuffers(dis, t.parameters);
                log.info(
                        "Чекпоинт загружен (v2 + Adam): {} (шаг {}, лучший оценочный loss {}; эпоха в файле не хранилась — старт с 1-й)",
                        path,
                        t.globalStep,
                        LlmTrainerTrainingFormat.formatEvalBestLossForLog(t.bestLoss));
                t.syncShutdownProgressBaselineFromGlobalStep();
                return;
            }
            bis.reset();
        }
        loadLegacyCheckpoint(t, path);
    }

    private static void loadLegacyCheckpoint(LLMTrainer t, String path) throws IOException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            t.globalStep = in.readInt();
            t.bestLoss = in.readFloat();
            if (t.bestLoss == 0f) {
                log.warn("Чекпоинт (legacy): bestLoss=0 — сброс к «ещё не зафиксирован»");
                t.bestLoss = Float.MAX_VALUE;
            }
            t.loadedResumeEpochIndex = 0;
            t.loadedResumeDataLoaderIndex = 0;
            t.resumeReplayCheckpointShuffles = false;
            t.optimizer.setStep(t.globalStep);
            log.info(
                    "Чекпоинт загружен (старый формат, без буферов Adam m/v): {} (шаг {}, лучший loss {})",
                    path,
                    t.globalStep,
                    LlmTrainerTrainingFormat.formatEvalBestLossForLog(t.bestLoss));
            t.syncShutdownProgressBaselineFromGlobalStep();
        }
    }
}
