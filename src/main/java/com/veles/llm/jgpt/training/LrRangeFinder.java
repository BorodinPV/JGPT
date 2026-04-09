package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * LR range test (Leslie Smith): экспоненциальный рост LR между {@code lrStart} и {@code lrEnd}, один
 * шаг AdamW на батч (как в {@link LLMTrainer}), веса при необходимости восстанавливаются из снимка.
 * Результат — кривая loss vs LR и эвристика {@link Result#suggestedLearningRate} (~0.1× LR при
 * минимуме loss). При включённом FP16 matmul CE и цепочка unscale→clip→Adam совпадают с
 * {@link LLMTrainer} (динамический {@link DynamicLossScaler}).
 *
 * <p>Для очень больших моделей полный {@linkplain #run(GPTModel, DataLoader, int, float, float,
 * float, boolean) backup весов} может занимать гигабайты heap; тогда задайте {@code backupWeights =
 * false} и восстановите веса из чекпоинта после прогона.
 */
public final class LrRangeFinder {

    private LrRangeFinder() {}

    public static final class Result {
        public final float[] learningRates;
        public final float[] losses;
        /** Часто берут как стартовый LR обучения (порядок ниже минимума loss). */
        public final float suggestedLearningRate;

        public Result(float[] learningRates, float[] losses, float suggestedLearningRate) {
            this.learningRates = learningRates;
            this.losses = losses;
            this.suggestedLearningRate = suggestedLearningRate;
        }
    }

    /**
     * @param numSteps число шагов (батчей), не меньше 2
     * @param lrStart начальный LR (например 1e-7)
     * @param lrEnd конечный LR (например 1e-1)
     */
    public static Result run(
            GPTModel model,
            DataLoader dataLoader,
            int numSteps,
            float lrStart,
            float lrEnd,
            float maxGradNorm) {
        return run(model, dataLoader, numSteps, lrStart, lrEnd, maxGradNorm, true, AdamOptimizer.defaultLLM());
    }

    /**
     * @param backupWeights если {@code false}, веса после прогона не восстанавливаются (экономия
     *     памяти на больших моделях)
     */
    public static Result run(
            GPTModel model,
            DataLoader dataLoader,
            int numSteps,
            float lrStart,
            float lrEnd,
            float maxGradNorm,
            boolean backupWeights) {
        return run(model, dataLoader, numSteps, lrStart, lrEnd, maxGradNorm, backupWeights, AdamOptimizer.defaultLLM());
    }

    /** AdamW и {@link TrainingConfig#maxGradNorm} из конфигурации (как при обычном обучении). */
    public static Result run(
            GPTModel model,
            DataLoader dataLoader,
            TrainingConfig config,
            int numSteps,
            float lrStart,
            float lrEnd) {
        return run(
                model,
                dataLoader,
                numSteps,
                lrStart,
                lrEnd,
                config.maxGradNorm,
                true,
                AdamOptimizer.fromConfig(config));
    }

    public static Result run(
            GPTModel model,
            DataLoader dataLoader,
            TrainingConfig config,
            int numSteps,
            float lrStart,
            float lrEnd,
            boolean backupWeights) {
        return run(
                model,
                dataLoader,
                numSteps,
                lrStart,
                lrEnd,
                config.maxGradNorm,
                backupWeights,
                AdamOptimizer.fromConfig(config));
    }

    private static Result run(
            GPTModel model,
            DataLoader dataLoader,
            int numSteps,
            float lrStart,
            float lrEnd,
            float maxGradNorm,
            boolean backupWeights,
            AdamOptimizer optimizer) {
        TensorOpsGPU.requireCuda("LrRangeFinder.run");
        if (numSteps < 2) {
            throw new IllegalArgumentException("numSteps must be >= 2");
        }
        if (lrStart <= 0f || lrEnd <= 0f || lrEnd < lrStart) {
            throw new IllegalArgumentException("need 0 < lrStart <= lrEnd");
        }

        List<Tensor> parameters = new ArrayList<>(model.getParameters());
        float[][] backup = backupWeights ? backupParameters(parameters) : null;

        float[] lrs = new float[numSteps];
        float[] losses = new float[numSteps];

        optimizer.reset();
        dataLoader.shuffle();
        dataLoader.reset();

        List<Tensor> toClip = new ArrayList<>(parameters.size() + 2);
        DynamicLossScaler lossScaler = DynamicLossScaler.fromEnvironmentIfFp16();
        boolean fp16 = TensorOpsGPU.useFp16Matmul();

        try {
            for (int i = 0; i < numSteps; i++) {
                if (!dataLoader.hasMore()) {
                    dataLoader.reset();
                    dataLoader.shuffle();
                }
                DataLoader.Batch batch = dataLoader.nextBatch();

                float t = (float) i / (float) (numSteps - 1);
                float lr = lrStart * (float) Math.pow(lrEnd / lrStart, t);
                lrs[i] = lr;

                float lossScale = fp16 ? lossScaler.getScale() : 1f;

                Tensor logits = model.forward(batch.input, true);
                float loss = applyCrossEntropyLossAndGrad(logits, batch.target, 1f, lossScale);
                losses[i] = loss;

                model.backward(logits, true);

                boolean hasOverflow = gradientOverflow(logits, loss, parameters);
                if (fp16) {
                    Fp16Metrics.global().recordStep();
                    if (hasOverflow) {
                        Fp16Metrics.global().recordOverflow();
                    }
                    if (!lossScaler.step(hasOverflow)) {
                        zeroGrads(parameters, logits);
                        losses[i] = Float.NaN;
                        continue;
                    }
                } else if (hasOverflow) {
                    zeroGrads(parameters, logits);
                    losses[i] = Float.NaN;
                    continue;
                }

                float unscale = fp16 ? lossScaler.getScale() : 1f;

                toClip.clear();
                for (Tensor p : parameters) {
                    if (p.hasGrad()) {
                        toClip.add(p);
                    }
                }
                if (logits.hasGrad()) {
                    toClip.add(logits);
                }
                if (!toClip.isEmpty()) {
                    DynamicLossScaler.unscaleGradients(toClip, unscale);
                    AdamOptimizer.clipGradientsGlobal(toClip, maxGradNorm);
                }

                optimizer.setLearningRate(lr);
                optimizer.beginStep();
                optimizer.stepAllWithParamGrad(parameters);

                zeroGrads(parameters, logits);
            }
        } finally {
            if (backup != null) {
                restoreParameters(parameters, backup);
            }
            if (TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.synchronizeStream();
                TensorOpsGPU.drainDeferredGpuBuffers();
                TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
            }
        }

        float suggested = suggestLr(lrs, losses, lrStart, lrEnd);
        return new Result(lrs, losses, suggested);
    }

    private static float suggestLr(float[] lrs, float[] losses, float lrStart, float lrEnd) {
        int n = losses.length;
        int minIdx = -1;
        for (int i = 0; i < n; i++) {
            if (!Float.isFinite(losses[i])) {
                continue;
            }
            if (minIdx < 0 || losses[i] < losses[minIdx]) {
                minIdx = i;
            }
        }
        if (minIdx < 0) {
            return lrStart;
        }
        float atMin = lrs[minIdx];
        float suggested = atMin * 0.1f;
        if (suggested < lrStart) {
            suggested = lrStart;
        }
        if (suggested > lrEnd * 0.5f) {
            suggested = lrEnd * 0.5f;
        }
        return suggested;
    }

    private static float[][] backupParameters(List<Tensor> params) {
        float[][] bufs = new float[params.size()][];
        for (int i = 0; i < params.size(); i++) {
            float[] src = params.get(i).internalBuffer();
            bufs[i] = Arrays.copyOf(src, src.length);
        }
        return bufs;
    }

    private static void restoreParameters(List<Tensor> params, float[][] backup) {
        for (int i = 0; i < params.size(); i++) {
            System.arraycopy(backup[i], 0, params.get(i).internalBuffer(), 0, backup[i].length);
        }
    }

    private static boolean gradientOverflow(Tensor logits, float loss, List<Tensor> parameters) {
        if (!Float.isFinite(loss)) {
            return true;
        }
        for (Tensor p : parameters) {
            if (p.hasGrad() && !floatGradBufferFinite(p)) {
                return true;
            }
        }
        return logits.hasGrad() && !floatGradBufferFinite(logits);
    }

    private static boolean floatGradBufferFinite(Tensor t) {
        float[] g = t.gradBuffer();
        for (float v : g) {
            if (!Float.isFinite(v)) {
                return false;
            }
        }
        return true;
    }

    private static void scaleTensorGrad(Tensor t, float scale) {
        if (!t.hasGrad()) {
            return;
        }
        float[] g = t.gradBuffer();
        if (g.length <= 0) {
            return;
        }
        TensorOpsGPU.scaleInPlaceGPU(g, g.length, scale);
    }

    /**
     * CE (mean по токенам) + ∂L/∂logits: fused JNI на GPU. {@code lossScale} как в {@link LLMTrainer}.
     */
    private static float applyCrossEntropyLossAndGrad(
            Tensor logits, Tensor target, float gradScale, float lossScale) {
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];
        int totalTokens = batch * seqLen;
        if (!logits.hasGrad()) {
            logits.zeroGrad();
        }
        float[] logitData = logits.internalBuffer();
        float[] targetData = target.internalBuffer();
        float[] gradData = logits.gradBuffer();

        if (!TensorOpsGPU.shouldUseGpuCrossEntropy(logitData.length)) {
            return 0f;
        }
        float gradScaleOverTotal = gradScale * lossScale / (float) totalTokens;
        return TensorOpsGPU.crossEntropySoftmaxGradLossGpu(
                logitData, targetData, gradData, batch, seqLen, vocabSize, gradScaleOverTotal);
    }

    private static void zeroGrads(List<Tensor> params, Tensor logits) {
        for (Tensor p : params) {
            if (p.hasGrad()) {
                p.zeroGrad();
            }
        }
        if (logits.hasGrad()) {
            logits.zeroGrad();
        }
    }

}
