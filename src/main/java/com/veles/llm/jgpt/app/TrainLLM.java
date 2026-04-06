package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.data.TextDataset;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.training.LLMConfig;
import com.veles.llm.jgpt.training.LLMTrainer;
import com.veles.llm.jgpt.training.TrainingConfig;

import java.nio.file.Files;
import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Скрипт обучения LLM: {@link LLMConfig}, {@link TextDataset}, {@link LLMTrainer}, генерация после
 * эпохи.
 */
public final class TrainLLM {

    private static final Logger log = LoggerFactory.getLogger(TrainLLM.class);

    public static void main(String[] args) throws Exception {
        TensorOpsGPU.requireCuda("TrainLLM");
        log.info("Скрипт обучения LLM");
        log.info("{}", "=".repeat(60));

        LLMConfig llm = LLMConfig.applyAccumulationStepsOverrideFromEnv(
                LLMConfig.applyBatchSizeOverrideFromEnv(LLMConfig.smart50M()));
        log.info("Конфигурация модели: {}", llm);

        log.info("Загрузка текстов...");
        TextDataset dataset = new TextDataset();

        // Пути относительно текущей рабочей директории (cwd), не относительно .jar
        String[] bookPaths = {"data/book.txt", "data/books/book.txt"};
        boolean loaded = false;
        Exception lastException = null;
        for (String path : bookPaths) {
            try {
                dataset.loadFile(path);
                loaded = true;
                break;
            } catch (Exception e) {
                lastException = e;
            }
        }
        if (!loaded) {
            log.warn("Не найден ни один из файлов: {}", String.join(", ", bookPaths));
            if (lastException != null) {
                log.warn("Последняя ошибка при чтении: {}", lastException.getMessage());
            }
            log.warn(
                    "Запускайте из корня проекта (где есть каталог data/) или положите текст в data/book.txt.");
            log.info("Используется синтетический текст для демонстрации.");
            dataset.loadText(generateSyntheticText(100_000));
        }

        log.info("  всего символов: {}", String.format("%,d", dataset.totalCharacters()));
        log.info("  фрагментов текста: {}", dataset.size());

        String checkpointDir = "checkpoints";
        Path tokenizerFinalPath = Path.of(checkpointDir).resolve("tokenizer_final.bin");

        BPETokenizer tokenizer;
        if (Files.isRegularFile(tokenizerFinalPath)) {
            log.info("Загрузка токенизатора из чекпоинта...");
            tokenizer = BPETokenizer.load(tokenizerFinalPath.toString());
            log.info(
                    "  словарь: {} токенов (файл {})",
                    tokenizer.getVocabSize(),
                    tokenizerFinalPath);
        } else {
            log.info("Обучение BPE-токенизатора на тексте...");
            tokenizer = BPETokenizer.train(dataset.getTexts(), llm.vocabSize);
            log.info(
                    "  итоговый размер словаря: {} (целевой бюджет слияний был {})",
                    tokenizer.getVocabSize(),
                    llm.vocabSize);
        }

        int vocabSize = tokenizer.getVocabSize();

        log.info("Создание модели...");
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

        log.info("  точное число параметров: {}", String.format("%,d", model.countParameters()));

        Path modelFinalPath = Path.of(checkpointDir).resolve("model_final.bin");
        if (Files.isRegularFile(modelFinalPath)) {
            log.info("Продолжение: загрузка весов из {}...", modelFinalPath);
            model.loadWeights(modelFinalPath.toString());
        }

        log.info("Подготовка загрузчика данных (DataLoader)...");
        DataLoader dataLoader = new DataLoader(tokenizer, llm.maxSeqLen, llm.batchSize);
        for (String text : dataset.getTexts()) {
            dataLoader.loadText(text);
        }

        int nSeq = dataLoader.numSequences();
        log.info(
                "  последовательностей: {} (около {} батчей на эпоху)",
                nSeq,
                dataLoader.numBatches());

        TrainingConfig trainConfig = llm.toTrainingConfig(checkpointDir, vocabSize);
        LLMTrainer trainer = new LLMTrainer(model, trainConfig, dataLoader);

        Path checkpointFinalPath = Path.of(checkpointDir).resolve("checkpoint_final.bin");
        if (Files.isRegularFile(checkpointFinalPath)) {
            log.info("Продолжение: загрузка состояния обучения из {}...", checkpointFinalPath);
            trainer.loadCheckpoint(checkpointFinalPath.toString());
        }

        log.info("Запуск цикла обучения...");
        log.info("{}", "=".repeat(60));
        trainer.train();

        log.info("{}", "=".repeat(60));
        log.info("Обучение завершено.");
        log.info("Лучший оценочный loss: {}", String.format("%.4f", trainer.getBestLoss()));

        trainer.saveCheckpoint("final");

        log.info("Пример генерации текста...");
        log.info("{}", "=".repeat(60));

        String[] prompts = {
            "кот",
            "собака",
            "птица",
            "в доме",
            "он сказал",
            "который"
        };
        for (String prompt : prompts) {
            String generated = LlmTextGeneration.generateText(model, tokenizer, prompt, 50, 1.0f, 40);
            log.info("Промпт: \"{}\" → сгенерировано: \"{}\"", prompt, generated);
        }

        log.info("Готово.");
    }

    private static String generateSyntheticText(int charCount) {
        String[] sentences = {
            "the cat sat on the mat",
            "the dog ran in the park",
            "the bird flew over the tree",
            "the fish swam in the lake",
            "the sun shines bright today",
            "the moon glows at night",
            "the cat and the dog are friends",
            "the park is full of children",
            "the tree has many green leaves",
            "the lake is calm and peaceful"
        };

        StringBuilder sb = new StringBuilder();
        java.util.Random rnd = new java.util.Random(42);
        while (sb.length() < charCount) {
            sb.append(sentences[rnd.nextInt(sentences.length)]).append(". ");
        }
        return sb.substring(0, Math.min(charCount, sb.length()));
    }
}
