package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.training.LLMConfig;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Инференс после обучения {@link AllBooksTrain}: загрузка весов и BPE, ввод промптов в консоли (или один запуск с
 * {@code --prompt}).
 *
 * <p>Геометрия модели как при обучении на базе {@link LLMConfig#smart50M()} с учётом {@code JGPT_MAX_SEQ_LEN} и
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

    private InferChat() {}

    public static void main(String[] args) throws Exception {
        TensorOpsGPU.requireCuda("InferChat");

        String boo = ".";
        String modelRel = "checkpoints/all_books/model_final.bin";
        String tokenizerRel = null;
        int maxNewTokens = 128;
        float temperature = 0.8f;
        int topK = 50;
        int seqLenOverride = -1;
        int layersOverride = -1;
        String singlePrompt = null;

        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if ("--help".equals(a) || "-h".equals(a)) {
                printUsage();
                return;
            }
            if ("--boo".equals(a) && i + 1 < args.length) {
                boo = args[++i];
            } else if ("--model".equals(a) && i + 1 < args.length) {
                modelRel = args[++i];
            } else if ("--tokenizer".equals(a) && i + 1 < args.length) {
                tokenizerRel = args[++i];
            } else if ("--max-new-tokens".equals(a) && i + 1 < args.length) {
                maxNewTokens = Math.max(1, Integer.parseInt(args[++i]));
            } else if ("--temperature".equals(a) && i + 1 < args.length) {
                temperature = Float.parseFloat(args[++i]);
            } else if ("--top-k".equals(a) && i + 1 < args.length) {
                topK = Math.max(1, Integer.parseInt(args[++i]));
            } else if ("--seq-len".equals(a) && i + 1 < args.length) {
                seqLenOverride = Math.max(1, Integer.parseInt(args[++i]));
            } else if ("--layers".equals(a) && i + 1 < args.length) {
                layersOverride = Math.max(1, Integer.parseInt(args[++i]));
            } else if ("--prompt".equals(a) && i + 1 < args.length) {
                singlePrompt = args[++i];
            } else if (a.startsWith("--prompt=") && a.length() > "--prompt=".length()) {
                singlePrompt = a.substring("--prompt=".length());
            } else {
                log.error("Неизвестный или неполный аргумент: {} (см. --help)", a);
                printUsage();
                System.exit(2);
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

        boolean gpuResident = LLMConfig.effectiveGpuResidentTraining();
        GPTModel model =
                new GPTModel(
                        vocab,
                        cfg.maxSeqLen,
                        cfg.dModel,
                        cfg.numHeads,
                        cfg.numLayers,
                        cfg.dIntermediate,
                        gpuResident);
        model.loadWeights(modelPath.toString());

        try {
            if (singlePrompt != null) {
                String out =
                        LlmTextGeneration.generateText(
                                model, tokenizer, singlePrompt, maxNewTokens, temperature, topK);
                System.out.println(out);
                return;
            }

            java.io.Console console = System.console();
            if (console == null) {
                log.error("Нет консоли (System.console() == null). Задайте --prompt \"...\" или запустите из терминала.");
                System.exit(1);
            }

            console.printf(
                    "JGPT InferChat — max_new_tokens=%d temperature=%.2f top_k=%d%n"
                            + "Пустая строка — выход. Команды: quit | exit%n",
                    maxNewTokens,
                    temperature,
                    topK);
            BufferedReader stdin =
                    new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            while (true) {
                console.printf("> ");
                console.flush();
                String line = stdin.readLine();
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
                                    model, tokenizer, trimmed, maxNewTokens, temperature, topK);
                    System.out.println(out);
                    System.out.println();
                } catch (Exception e) {
                    log.warn("Генерация: {}", e.getMessage());
                }
            }
            log.info("Выход.");
        } finally {
            if (TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.synchronizeStream();
                TensorOpsGPU.drainDeferredGpuBuffers();
                TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
            }
        }
    }

    private static LLMConfig geometryFromEnvAndOverrides(int seqLenOverride, int layersOverride) {
        LLMConfig base =
                LLMConfig.applyPresetNumLayersOverrideFromEnv(
                        LLMConfig.applySeqLenOverrideFromEnv(LLMConfig.smart50M()));
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

    private static void printUsage() {
        System.err.println(
                """
                InferChat — промпты к обученной модели (CUDA обязательна).

                Аргументы:
                  --boo DIR              корень проекта (по умолчанию .)
                  --model PATH           веса относительно boo (по умолчанию checkpoints/all_books/model_final.bin)
                  --tokenizer PATH       BPE; по умолчанию checkpoints/tokenizer_global.bin или all_books/tokenizer_final.bin
                  --seq-len N            max контекст (иначе env JGPT_MAX_SEQ_LEN / smart50M)
                  --layers N             число слоёв (иначе env JGPT_PRESET_NUM_LAYERS / smart50M)
                  --max-new-tokens N     длина продолжения (по умолчанию 128)
                  --temperature F        (по умолчанию 0.8)
                  --top-k N              (по умолчанию 50)
                  --prompt TEXT          один промпт и выход (без интерактива)
                  --prompt=TEXT          то же одним аргументом (удобно для mvn -Dexec.args без кавычек к пробелам)
                  -h, --help             эта справка

                Окружение: JGPT_CUDA_LIB, JGPT_MAX_SEQ_LEN, JGPT_PRESET_NUM_LAYERS, JGPT_GENERATE_GPU_KV, … как при train.
                """);
    }
}
