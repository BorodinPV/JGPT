package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.app.LlmTextGeneration;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.util.LogFmt;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Eval loss на батчах и промежуточная генерация во время обучения. */
final class LlmTrainerEvalAndSample {

    private static final Logger log = LoggerFactory.getLogger(LlmTrainerEvalAndSample.class);

    private static final int SAMPLE_MAX_NEW_TOKENS = 64;
    private static final float SAMPLE_TEMP = 0.9f;
    private static final int SAMPLE_TOP_K = 40;

    private static final String[] AUTO_PROMPTS_RU = {
        "мороз и солнце день чудесный",
        "весна пришла в город сегодня",
        "он сказал ей одно слово",
        "книга лежала на столе тихо",
        "ветер гонит тучи прочь сейчас",
        "старинная улица встретила утро зарёю",
        "тишина стояла в комнате ночью",
        "дети бежали навстречу лету радостно"
    };

    private LlmTrainerEvalAndSample() {}

    static String pickSamplePrompt(LLMTrainer t, int epochOneBased) {
        String env = System.getenv("JGPT_SAMPLE_PROMPT");
        if (env != null && !env.isBlank()) {
            String[] parts = env.split("\\|");
            return parts[(t.globalStep + epochOneBased) % parts.length].trim();
        }
        return AUTO_PROMPTS_RU[(t.globalStep + epochOneBased) % AUTO_PROMPTS_RU.length];
    }

    static void maybeAutoSample(LLMTrainer t, int epochOneBased) {
        if (t.config.interactiveSampleEverySteps <= 0) {
            return;
        }
        if (t.globalStep % t.config.interactiveSampleEverySteps != 0) {
            return;
        }
        String prompt = pickSamplePrompt(t, epochOneBased);
        log.info(
                "{} промежуточная генерация: эпоха {}/{}, шаг {}",
                LogFmt.badge("SAMPLE"),
                epochOneBased,
                t.config.epochs,
                t.globalStep);
        log.info("{} промпт: {}", LogFmt.badge("SAMPLE"), prompt);
        try {
            t.model.zeroGradParameters();
            String out =
                    LlmTextGeneration.generateText(
                            t.model,
                            t.dataLoader.getTokenizer(),
                            prompt,
                            SAMPLE_MAX_NEW_TOKENS,
                            SAMPLE_TEMP,
                            SAMPLE_TOP_K);
            log.info("{} сгенерировано: {}", LogFmt.badge("SAMPLE"), out);
            if (t.trainingStatsWriter != null) {
                t.trainingStatsWriter.onSample(t.globalStep, out);
            }
        } catch (Exception e) {
            log.warn("{} генерация не удалась: {}", LogFmt.badge("SAMPLE"), e.getMessage());
        } finally {
            t.synchronizeTrainingPipelineAfterGpuAuxiliaryInfer("sample");
        }
    }

    static float evaluate(LLMTrainer t) {
        DataLoader evalLoader = t.evalDataLoader != null ? t.evalDataLoader : t.dataLoader;
        int saved = evalLoader.getCurrentIndex();
        float total = 0f;
        int n = 0;
        int maxBatches = Math.min(64, evalLoader.numBatches());
        boolean deviceLogitsEval = false;
        for (int i = 0; i < maxBatches && evalLoader.hasMore(); i++) {
            DataLoader.Batch batch = evalLoader.nextBatch();
            int[] inSh = batch.input.getShape();
            int batchSize = inSh[0];
            int seqLen = inSh[1];
            if (i == 0) {
                deviceLogitsEval =
                        t.config.useGpuResident && t.model.canInferLogitsOnDevice(batchSize, seqLen);
            }
            if (deviceLogitsEval) {
                t.model.forward(batch.input, false, true, true);
                GpuFloatBuffer logitsGpu = t.model.deviceLogitsBuffer();
                total +=
                        LlmTrainerCrossEntropy.evaluateCrossEntropyLossDevice(
                                t, batch.target, logitsGpu, batchSize, seqLen, t.config.vocabSize);
            } else {
                Tensor logits = t.model.forward(batch.input, false, t.config.useGpuResident);
                total += LlmTrainerCrossEntropy.evaluateCrossEntropyLoss(t, logits, batch.target);
            }
            n++;
        }
        evalLoader.setCurrentIndex(saved);
        // Free VRAM allocated during eval (logits grad buffers)
        t.model.clearDeviceLogitsBuffers();
        if (deviceLogitsEval && n > 0 && TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
        }
        if (n == 0) {
            log.warn(
                    "{} ни одного eval-батча (hasMore={}) — не обновляем best/patience early-stop",
                    LogFmt.badge("EVAL"),
                    evalLoader.hasMore());
            return Float.NaN;
        }
        float loss = total / n;
        float perplexity = (float) Math.exp(loss);
        log.info(
                "{} перплексия: {} ({})",
                LogFmt.badge("EVAL"),
                String.format("%.2f", perplexity),
                t.evalDataLoader != null ? "hold-out val" : "train stream");
        return loss;
    }
}
