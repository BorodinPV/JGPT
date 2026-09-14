package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.app.LlmTextGeneration;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.data.SftExampleEncoder;
import com.veles.llm.jgpt.util.LogFmt;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Eval CE (global token-mean) и промежуточная генерация во время обучения. */
final class LlmTrainerEvalAndSample {

    private static final Logger log = LoggerFactory.getLogger(LlmTrainerEvalAndSample.class);

    private static final int SAMPLE_MAX_NEW_TOKENS = 64;
    private static final float SAMPLE_TEMP = 0.9f;
    private static final int SAMPLE_TOP_K = 50;
    private static final String SAMPLE_BADGE = "SAMPLE";

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

    private static final String[] AUTO_PROMPTS_SFT = {
        "столица Франции",
        "сколько будет 2+2",
        "ответь да или нет: небо голубое"
    };

    private LlmTrainerEvalAndSample() {}

    static String pickSamplePrompt(LLMTrainer t, int epochOneBased) {
        String env = System.getenv("JGPT_SAMPLE_PROMPT");
        if (env != null && !env.isBlank()) {
            String[] parts = env.split("\\|");
            return parts[(t.globalStep + epochOneBased) % parts.length].trim();
        }
        String[] bank =
                SftExampleEncoder.chatTemplateFromEnv() ? AUTO_PROMPTS_SFT : AUTO_PROMPTS_RU;
        return bank[(t.globalStep + epochOneBased) % bank.length];
    }

    static void maybeAutoSample(LLMTrainer t, int epochOneBased) {
        if (t.config.interactiveSampleEverySteps <= 0) {
            return;
        }
        if (t.globalStep % t.config.interactiveSampleEverySteps != 0) {
            return;
        }
        String prompt =
                SftExampleEncoder.applyChatTemplateIfEnabled(
                        t.dataLoader.getTokenizer(), pickSamplePrompt(t, epochOneBased));
        log.info(
                "{} промежуточная генерация: эпоха {}/{}, шаг {}",
                LogFmt.badge(SAMPLE_BADGE),
                epochOneBased,
                t.config.epochs,
                t.globalStep);
        log.info("{} промпт: {}", LogFmt.badge(SAMPLE_BADGE), prompt);
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
            log.info("{} сгенерировано: {}", LogFmt.badge(SAMPLE_BADGE), out);
            if (t.trainingStatsWriter != null) {
                t.trainingStatsWriter.onSample(t.globalStep, out);
            }
        } catch (Exception e) {
            log.warn("{} генерация не удалась: {}", LogFmt.badge(SAMPLE_BADGE), e.getMessage());
        } finally {
            t.synchronizeTrainingPipelineAfterGpuAuxiliaryInfer("sample");
        }
    }

    static float evaluate(LLMTrainer t) {
        DataLoader evalLoader = t.evalDataLoader != null ? t.evalDataLoader : t.dataLoader;
        int saved = evalLoader.getCurrentIndex();
        TokenMeanAcc acc = new TokenMeanAcc();
        int n = 0;
        int maxBatches = Math.min(64, evalLoader.numBatches());
        boolean deviceLogitsEval = false;
        for (int i = 0; i < maxBatches && evalLoader.hasMore(); i++) {
            DataLoader.Batch batch = evalLoader.nextBatch();
            int[] inSh = batch.input.getShape();
            int batchSize = inSh[0];
            int seqLen = inSh[1];
            int nrows = batchSize * seqLen;
            if (i == 0) {
                deviceLogitsEval =
                        t.config.useGpuResident && t.model.canInferLogitsOnDevice(batchSize, seqLen);
            }
            float batchMean;
            if (deviceLogitsEval) {
                t.model.forward(batch.input, false, true, true);
                GpuFloatBuffer logitsGpu = t.model.deviceLogitsBuffer();
                batchMean =
                        LlmTrainerCrossEntropy.evaluateCrossEntropyLossDevice(
                                t, batch.target, logitsGpu, batchSize, seqLen, t.config.vocabSize);
            } else {
                Tensor logits = t.model.forward(batch.input, false, t.config.useGpuResident);
                batchMean = LlmTrainerCrossEntropy.evaluateCrossEntropyLoss(t, logits, batch.target);
            }
            LlmTrainerCrossEntropy.fillCeTargetsHostSanitized(t, batch.target, nrows, t.config.vocabSize);
            acc.addBatchMean(batchMean, LlmTrainerCrossEntropy.validCountFromScratch(t, nrows));
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
        float loss = acc.mean();
        if (!Float.isFinite(loss)) {
            log.warn(
                    "{} eval без валидных токенов — не обновляем best/patience early-stop",
                    LogFmt.badge("EVAL"));
            return Float.NaN;
        }
        float perplexity = (float) Math.exp(loss);
        log.info(
                "{} перплексия: {} ({})",
                LogFmt.badge("EVAL"),
                String.format("%.2f", perplexity),
                t.evalDataLoader != null ? "hold-out val" : "train stream");
        return loss;
    }

    /**
     * Склеивает per-batch token-mean CE в global token-mean: {@code Σ(mean_b × N_b) / Σ N_b}.
     * Mean-of-batch-means дал бы равный вес короткому и длинному диалогу.
     */
    static final class TokenMeanAcc {
        private double weightedSum;
        private long validTokens;

        void addBatchMean(float batchTokenMean, int nValid) {
            if (nValid <= 0 || !Float.isFinite(batchTokenMean)) {
                return;
            }
            weightedSum += (double) batchTokenMean * (double) nValid;
            validTokens += nValid;
        }

        long validTokens() {
            return validTokens;
        }

        float mean() {
            if (validTokens <= 0L) {
                return Float.NaN;
            }
            return (float) (weightedSum / (double) validTokens);
        }
    }
}
