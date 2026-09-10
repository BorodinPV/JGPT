package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.SftExampleEncoder;
import com.veles.llm.jgpt.model.DecodeSampling;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.training.LLMConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Arrays;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Инференс после обучения {@link AllBooksTrain}: загрузка весов и BPE, ввод промптов в консоли (или один запуск с
 * {@code --prompt}).
 *
 * <p>Геометрия модели как при обучении на базе {@link LLMConfig#canonical()} с учётом {@code JGPT_MAX_SEQ_LEN} и
 * {@code JGPT_PRESET_NUM_LAYERS} из окружения; при необходимости переопределите явно {@code --seq-len} и {@code --layers}.
 *
 * <p>Пример:
 *
 * <pre>{@code
 * cd JGPT && export JGPT_CUDA_LIB=$PWD/build/libjgpt_cuda.so
 * mvn -q compile exec:java -Dexec.mainClass=com.veles.llm.jgpt.app.InferChat \
 *   -Dexec.args='--boo . --layers 12 --seq-len 1024'
 * }</pre>
 */
public final class InferChat {

    private static final Logger log = LoggerFactory.getLogger(InferChat.class);

    private static final String PROMPT_EQ = "--prompt=";

    private InferChat() {
        // no instances
    }

    public static void main(String[] args) throws Exception {
        TensorOpsGPU.requireCuda("InferChat");

        String boo = ".";
        String modelRel = "checkpoints/all_books/model_final.bin";
        String tokenizerRel = null;
        int maxNewTokens = 128;
        float temperature = 0.8f;
        int topK = 50;
        float topP = 1f;
        float repetitionPenalty = 1f;
        int noRepeatNgramSize = 0;
        int seqLenOverride = -1;
        int layersOverride = -1;
        String singlePrompt = null;

        ArrayDeque<String> argv = new ArrayDeque<>(Arrays.asList(args));
        while (!argv.isEmpty()) {
            String a = argv.removeFirst();
            switch (a) {
                case "--help", "-h" -> {
                    printUsage();
                    return;
                }
                case "--boo" -> boo = requireArg(argv, a);
                case "--model" -> modelRel = requireArg(argv, a);
                case "--tokenizer" -> tokenizerRel = requireArg(argv, a);
                case "--max-new-tokens" -> maxNewTokens = Math.max(1, Integer.parseInt(requireArg(argv, a)));
                case "--temperature" -> temperature = Float.parseFloat(requireArg(argv, a));
                case "--top-k" -> topK = Math.max(0, Integer.parseInt(requireArg(argv, a)));
                case "--top-p" -> topP = Float.parseFloat(requireArg(argv, a));
                case "--repetition-penalty" -> repetitionPenalty = Float.parseFloat(requireArg(argv, a));
                case "--no-repeat-ngram-size" ->
                        noRepeatNgramSize = Math.max(0, Integer.parseInt(requireArg(argv, a)));
                case "--seq-len" -> seqLenOverride = Math.max(1, Integer.parseInt(requireArg(argv, a)));
                case "--layers" -> layersOverride = Math.max(1, Integer.parseInt(requireArg(argv, a)));
                case "--prompt" -> singlePrompt = requireArg(argv, a);
                default -> {
                    if (a.startsWith(PROMPT_EQ) && a.length() > PROMPT_EQ.length()) {
                        singlePrompt = a.substring(PROMPT_EQ.length());
                    } else {
                        log.error("Неизвестный или неполный аргумент: {} (см. --help)", a);
                        printUsage();
                        System.exit(2);
                    }
                }
            }
        }

        Path root = Path.of(boo).toAbsolutePath().normalize();
        Path modelPath = root.resolve(modelRel).normalize();
        Path tokPath = resolveTokenizer(root, tokenizerRel);

        if (!Files.isRegularFile(modelPath)) {
            log.error("Нет файла весов: {}", modelPath);
            System.exit(1);
        }
        if (!Files.isRegularFile(tokPath)) {
            log.error("Нет файла токенизатора: {}", tokPath);
            System.exit(1);
        }

        LLMConfig cfg = geometryFromEnvAndOverrides(seqLenOverride, layersOverride);
        BPETokenizer tokenizer = BPETokenizer.load(tokPath.toString());
        int vocab = tokenizer.getVocabSize();

        log.info(
                "[INF] boo={} model={} tokenizer={} vocab={} seq={} layers={} d_model={} heads={}",
                root,
                modelPath.getFileName(),
                tokPath.getFileName(),
                vocab,
                cfg.maxSeqLen,
                cfg.numLayers,
                cfg.dModel,
                cfg.numHeads);

        boolean gpuResident = LLMConfig.canonicalGpuTrain();
        GPTModel model =
                new GPTModel(
                        vocab,
                        cfg.maxSeqLen,
                        cfg.dModel,
                        cfg.numHeads,
                        cfg.numLayers,
                        cfg.dIntermediate,
                        gpuResident);
        DecodeSampling sampling =
                new DecodeSampling(temperature, topK, topP, repetitionPenalty, noRepeatNgramSize);
        model.loadWeights(modelPath.toString());

        try {
            if (singlePrompt != null) {
                String out =
                        LlmTextGeneration.generateText(
                                model,
                                tokenizer,
                                applySftChatTemplate(singlePrompt),
                                maxNewTokens,
                                sampling);
                log.info("{}", out);
                return;
            }

            java.io.Console console = System.console();
            if (console == null) {
                log.error("Нет консоли (System.console() == null). Задайте --prompt \"...\" или запустите из терминала.");
                System.exit(1);
            }

            console.printf(
                    "JGPT InferChat — max_new_tokens=%d temperature=%.2f top_k=%d top_p=%.2f"
                            + " repetition_penalty=%.2f no_repeat_ngram=%d%n"
                            + "Пустая строка — выход. Команды: quit | exit%n",
                    maxNewTokens,
                    sampling.temperature,
                    sampling.topK,
                    sampling.topP,
                    sampling.repetitionPenalty,
                    sampling.noRepeatNgramSize);
            while (true) {
                console.printf("> ");
                console.flush();
                String line = console.readLine();
                if (line == null) {
                    break;
                }
                String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    break;
                }
                if ("quit".equalsIgnoreCase(trimmed) || "exit".equalsIgnoreCase(trimmed)) {
                    break;
                }
                try {
                    String out =
                            LlmTextGeneration.generateText(
                                    model,
                                    tokenizer,
                                    applySftChatTemplate(trimmed),
                                    maxNewTokens,
                                    sampling);
                    console.printf("%s%n", out);
                    console.flush();
                } catch (Exception e) {
                    log.warn("Генерация: {}", e.getMessage());
                }
            }
            console.printf("Выход.%n");
        } finally {
            if (TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.synchronizeStream();
                TensorOpsGPU.drainDeferredGpuBuffers();
                TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
            }
        }
    }

    static boolean sftChatTemplateFromEnv() {
        return SftExampleEncoder.chatTemplateFromEnv();
    }

    static String applySftChatTemplate(String prompt) {
        return SftExampleEncoder.applyChatTemplateIfEnabled(prompt);
    }

    private static LLMConfig geometryFromEnvAndOverrides(int seqLenOverride, int layersOverride) {
        LLMConfig base =
                LLMConfig.applyPresetNumLayersOverrideFromEnv(
                        LLMConfig.applyVocabSizeOverrideFromEnv(
                                LLMConfig.applySeqLenOverrideFromEnv(LLMConfig.canonical())));
        int seq = seqLenOverride > 0 ? seqLenOverride : base.maxSeqLen;
        int layers = layersOverride > 0 ? layersOverride : base.numLayers;
        if (seq == base.maxSeqLen && layers == base.numLayers) {
            return base;
        }
        return new LLMConfig(
                base.name,
                base.vocabSize,
                seq,
                base.dModel,
                base.numHeads,
                layers,
                base.dIntermediate,
                base.batchSize,
                base.accumulationSteps,
                base.learningRate,
                base.epochs);
    }

    private static Path resolveTokenizer(Path root, String tokenizerRel) {
        if (tokenizerRel != null && !tokenizerRel.isBlank()) {
            return root.resolve(tokenizerRel).normalize();
        }
        Path global = root.resolve("checkpoints").resolve("tokenizer_global.bin");
        if (Files.isRegularFile(global)) {
            return global;
        }
        Path finalTok = root.resolve("checkpoints").resolve("all_books").resolve("tokenizer_final.bin");
        if (Files.isRegularFile(finalTok)) {
            return finalTok;
        }
        return global;
    }

    private static String requireArg(ArrayDeque<String> argv, String flag) {
        String v = argv.pollFirst();
        if (v == null) {
            log.error("Нет значения для {} (см. --help)", flag);
            printUsage();
            System.exit(2);
        }
        return v;
    }

    private static void printUsage() {
        log.info(
                """
                InferChat — промпты к обученной модели (CUDA обязательна).

                Аргументы:
                  --boo DIR              корень проекта (по умолчанию .)
                  --model PATH           веса относительно boo (по умолчанию checkpoints/all_books/model_final.bin)
                  --tokenizer PATH       BPE; по умолчанию checkpoints/tokenizer_global.bin или all_books/tokenizer_final.bin
                  --seq-len N            max контекст (иначе env JGPT_MAX_SEQ_LEN / canonical 1024)
                  --layers N             число слоёв (иначе env JGPT_PRESET_NUM_LAYERS / canonical 12)
                  --max-new-tokens N     длина продолжения (по умолчанию 128)
                  --temperature F        (по умолчанию 0.8)
                  --top-k N              (по умолчанию 50; 0 — выкл.)
                  --top-p F              nucleus, 1 = выкл. (по умолчанию 1)
                  --repetition-penalty F HF-штраф, 1 = выкл. (по умолчанию 1)
                  --no-repeat-ngram-size N  запрет повторных n-грамм, 0 = выкл.
                  --prompt TEXT          один промпт и выход (без интерактива)
                  {}TEXT          то же одним аргументом (удобно для mvn -Dexec.args без кавычек к пробелам)
                  -h, --help             эта справка

                Окружение: JGPT_CUDA_LIB, JGPT_MAX_SEQ_LEN, JGPT_PRESET_NUM_LAYERS, JGPT_VOCAB_SIZE,
                JGPT_SFT / JGPT_SFT_CHAT_TEMPLATE (обёртка Пользователь:/Ассистент:), JGPT_GENERATE_GPU_KV, …
                """,
                PROMPT_EQ);
    }
}
