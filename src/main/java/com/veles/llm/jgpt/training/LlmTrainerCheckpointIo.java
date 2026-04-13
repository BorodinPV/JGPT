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

    private LlmTrainerCheckpointIo() {}

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
        } catch (IOException ignored) {
        }

        if (t.config.fullGpuTrainStep && t.model.isGpuResident()) {
            t.model.syncWeightsFromGpu(t.model.gpuTensorByTrainableParameter());
        }

        if (!name.startsWith("epoch_")) {
            t.pendingCheckpointDataLoaderIndex = t.dataLoader.getCurrentIndex();
        }

        try (DataOutputStream out =
                new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeUTF(CHECKPOINT_FORMAT_V4);
            out.writeInt(t.globalStep);
            out.writeFloat(t.bestLoss);
            int ep = Math.max(0, Math.min(t.pendingCheckpointEpochIndex, t.config.epochs));
            out.writeInt(ep);
            int nSeq = t.dataLoader.numSequences();
            int seqIdx = Math.max(0, Math.min(t.pendingCheckpointDataLoaderIndex, nSeq));
            out.writeInt(seqIdx);
            t.optimizer.setStep(t.globalStep);
            t.optimizer.writeMomentBuffers(out, t.parameters);
        }
        log.info(
                "{} checkpoint(v4+Adam+epoch+pos): {} (resumeEpochIndex={}/{}, seqIndex={})",
                com.veles.llm.jgpt.util.LogFmt.badge("CKPT"),
                path,
                Math.max(0, Math.min(t.pendingCheckpointEpochIndex, t.config.epochs)),
                t.config.epochs,
                Math.max(0, Math.min(t.pendingCheckpointDataLoaderIndex, t.dataLoader.numSequences())));

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

        try (DataOutputStream out =
                new DataOutputStream(new BufferedOutputStream(new FileOutputStream(modelPath)))) {
            List<Tensor> params = t.model.getParameters();
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
        }
        log.info("{} веса модели записаны: {}", com.veles.llm.jgpt.util.LogFmt.badge("CKPT"), modelPath);

        String tokPath = dir.resolve("tokenizer_" + name + ".bin").toString();
        t.dataLoader.getTokenizer().save(tokPath);
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
        try (DataOutputStream out =
                new DataOutputStream(new BufferedOutputStream(new FileOutputStream(modelPath)))) {
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
        }
        log.info("{} веса модели записаны (асинхронный снимок): {}", com.veles.llm.jgpt.util.LogFmt.badge("CKPT"), modelPath);
        String tokPath = dir.resolve("tokenizer_" + name + ".bin").toString();
        t.dataLoader.getTokenizer().save(tokPath);
        log.info("{} токенизатор записан: {}", com.veles.llm.jgpt.util.LogFmt.badge("CKPT"), tokPath);
    }

    static void awaitPendingCheckpointWrites(LLMTrainer t) {
        if (t.checkpointIoExecutor == null) {
            return;
        }
        try {
            t.checkpointIoTail.get();
        } catch (Exception e) {
            log.warn("Ожидание фоновой записи чекпоинта: {}", e.toString());
        }
    }

    static void loadCheckpoint(LLMTrainer t, String path) throws IOException, ClassNotFoundException {
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(path))) {
            bis.mark(1 << 20);
            DataInputStream dis = new DataInputStream(bis);
            String tag;
            try {
                tag = dis.readUTF();
            } catch (IOException e) {
                bis.reset();
                loadLegacyCheckpoint(t, path);
                return;
            }
            if (CHECKPOINT_FORMAT_V4.equals(tag)) {
                t.globalStep = dis.readInt();
                t.bestLoss = dis.readFloat();
                if (t.bestLoss == 0f) {
                    log.warn(
                            "Чекпоинт: в файле bestLoss=0 (часто артефакт eval без батчей в старых прогонах) — сброс к «ещё не зафиксирован»");
                    t.bestLoss = Float.MAX_VALUE;
                }
                int ep = dis.readInt();
                t.loadedResumeEpochIndex = Math.max(0, Math.min(ep, t.config.epochs));
                t.loadedResumeDataLoaderIndex = Math.max(0, dis.readInt());
                t.resumeReplayCheckpointShuffles = true;
                t.optimizer.setStep(t.globalStep);
                t.optimizer.readMomentBuffers(dis, t.parameters);
                log.info(
                        "Чекпоинт загружен (v4 + Adam + эпоха + позиция): {} (шаг {}, resumeEpochIndex={}/{}, seqIndex={}, лучший оценочный loss {})",
                        path,
                        t.globalStep,
                        t.loadedResumeEpochIndex,
                        t.config.epochs,
                        t.loadedResumeDataLoaderIndex,
                        LlmTrainerTrainingFormat.formatEvalBestLossForLog(t.bestLoss));
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
                t.loadedResumeEpochIndex = Math.max(0, Math.min(ep, t.config.epochs));
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

    private static void loadLegacyCheckpoint(LLMTrainer t, String path) throws IOException, ClassNotFoundException {
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
