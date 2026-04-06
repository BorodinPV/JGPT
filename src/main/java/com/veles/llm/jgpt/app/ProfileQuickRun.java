package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.training.LLMConfig;
import com.veles.llm.jgpt.training.LLMTrainer;
import com.veles.llm.jgpt.training.TrainingConfig;

import java.nio.file.Files;
import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Короткий прогон {@link LLMTrainer#train()} для проверки {@code JGPT_PROFILE=1} без полного multi-book.
 * Запуск: {@code JGPT_PROFILE=1 mvn -q compile exec:java -Dexec.mainClass=com.veles.llm.jgpt.app.ProfileQuickRun}
 */
public final class ProfileQuickRun {

    private static final Logger log = LoggerFactory.getLogger(ProfileQuickRun.class);

    public static void main(String[] args) throws Exception {
        TensorOpsGPU.requireCuda("ProfileQuickRun");
        log.info("Короткий прогон обучения: 1 эпоха, конфиг nano(), синтетический английский текст");
        log.info("Для разбивки по фазам задайте: JGPT_PROFILE=1 и при необходимости JGPT_PROFILE_STEPS=15");
        log.info("{}", "=".repeat(60));

        LLMConfig nano = LLMConfig.nano();
        LLMConfig llm =
                new LLMConfig(
                        nano.name,
                        nano.vocabSize,
                        nano.maxSeqLen,
                        nano.dModel,
                        nano.numHeads,
                        nano.numLayers,
                        nano.dIntermediate,
                        nano.batchSize,
                        nano.accumulationSteps,
                        nano.learningRate,
                        1);
        llm = LLMConfig.applyAccumulationStepsOverrideFromEnv(LLMConfig.applyBatchSizeOverrideFromEnv(llm));
        StringBuilder sb = new StringBuilder();
        String line =
                "the cat sat on the mat. the dog ran in the park. "
                        + "the bird flew over the tree. the fish swam in the lake. ";
        for (int i = 0; i < 200; i++) {
            sb.append(line);
        }
        String text = sb.toString();

        BPETokenizer tokenizer = BPETokenizer.train(java.util.List.of(text), llm.vocabSize);
        int vocabSize = tokenizer.getVocabSize();

        if (LLMConfig.gpuResidentTrainingExplicitlyOn() && !TensorOpsGPU.isGpuAvailable()) {
            log.warn(
                    "JGPT_TRAIN_GPU_RESIDENT задан, но CUDA недоступна — прогон без GPU-резидентных весов.");
        }
        TrainingConfig config = llm.toTrainingConfig("checkpoints-profile-smoke", vocabSize);

        GPTModel model =
                new GPTModel(
                        vocabSize,
                        llm.maxSeqLen,
                        llm.dModel,
                        llm.numHeads,
                        llm.numLayers,
                        llm.dIntermediate,
                        config.useGpuResident);

        DataLoader dataLoader = new DataLoader(tokenizer, config.maxSeqLen, config.batchSize);
        dataLoader.loadText(text);

        Files.createDirectories(Path.of(config.checkpointDir));
        LLMTrainer trainer = new LLMTrainer(model, config, dataLoader);
        trainer.train();
    }
}
