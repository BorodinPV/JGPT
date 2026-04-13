package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.cuda.TensorCudaLibrary;
import com.veles.llm.jgpt.util.DebugGpuTrain;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Clip градиентов, overflow FP16, шаг Adam (хост и full GPU). */
final class LlmTrainerOptimizerStep {

    private static final Logger log = LoggerFactory.getLogger(LlmTrainerOptimizerStep.class);

    private LlmTrainerOptimizerStep() {}

    static boolean clipAndOptimizerStep(LLMTrainer t, Tensor logits, float avgMicroLoss) {
        if (t.model.isGpuResident()
                && TensorOpsGPU.isGpuAvailable()
                && GpuPendingGradients.allDirtyTargetsHaveGpuTensor(t.model.gpuTensorByTrainableParameter())) {
            return clipAndOptimizerStepGpuResidentMergeFirst(t, logits, avgMicroLoss);
        }
        GpuPendingGradients.flushAllToHost();
        if (t.fp16Matmul) {
            Fp16Metrics.global().recordStep();
        }
        boolean hasOverflow = checkGradientOverflow(t, logits, avgMicroLoss);
        if (t.fp16Matmul && hasOverflow) {
            Fp16Metrics.global().recordOverflow();
        }

        float scaleUsedInForward = t.fp16Matmul && t.dynamicLossScaler != null ? t.dynamicLossScaler.getScale() : 1f;
        if (t.fp16Matmul) {
            if (!t.dynamicLossScaler.step(hasOverflow)) {
                zeroGradients(t, logits);
                clearGpuParamGradsAfterOverflowSkip(t);
                markGpuTrainableParamGradsMaybeDirtyAfterHostOptimizerPath(t);
                logGradientOverflowSkipped(t, t.dynamicLossScaler.getScale());
                LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip();
                return false;
            }
        } else if (hasOverflow) {
            zeroGradients(t, logits);
            clearGpuParamGradsAfterOverflowSkip(t);
            markGpuTrainableParamGradsMaybeDirtyAfterHostOptimizerPath(t);
            LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip();
            return false;
        }

        List<Tensor> gradsToUnscale = collectGradTensorsWithLossScale(t, logits);
        if (t.fp16Matmul) {
            DynamicLossScaler.unscaleGradients(gradsToUnscale, scaleUsedInForward);
        }

        float gradNorm = 0f;
        if (!gradsToUnscale.isEmpty()) {
            gradNorm = AdamOptimizer.clipGradientsGlobal(gradsToUnscale, t.config.maxGradNorm);
        }
        t.lastGlobalGradNorm = gradNorm;

        optimizerStep(t);
        t.model.onParametersUpdated();

        zeroGradients(t, logits);
        markGpuTrainableParamGradsMaybeDirtyAfterHostOptimizerPath(t);
        return true;
    }

    static boolean clipAndOptimizerStepGpuResidentMergeFirst(LLMTrainer t, Tensor logits, float avgMicroLoss) {
        if (t.fp16Matmul) {
            Fp16Metrics.global().recordStep();
        }
        Map<Tensor, GpuTensor> paramMap = t.model.gpuTensorByTrainableParameter();
        TensorOpsGPU.synchronizeStream();
        boolean lossNonFinite = !Float.isFinite(avgMicroLoss);
        boolean pendingNonFinite = GpuPendingGradients.anyNonFinitePending();
        boolean hasOverflow = lossNonFinite || pendingNonFinite;
        if (lossNonFinite || pendingNonFinite) {
            GpuPendingGradients.discardDirtyPending();
        } else {
            GpuPendingGradients.flushMergeToGpuGrads(paramMap);
        }
        if (!hasOverflow) {
            String info = checkGpuParamGradsNonFiniteFused(t, paramMap);
            if (info != null) {
                hasOverflow = true;
            }
        }
        boolean logitsGradNonFinite =
                !hasOverflow && logits.hasGrad() && !floatArrayIsFinite(logits.gradBuffer());
        if (logitsGradNonFinite) {
            hasOverflow = true;
        }

        if (t.fp16Matmul && hasOverflow) {
            Fp16Metrics.global().recordOverflow();
        }
        float scaleUsedInForward = t.fp16Matmul && t.dynamicLossScaler != null ? t.dynamicLossScaler.getScale() : 1f;
        if (t.fp16Matmul) {
            if (!t.dynamicLossScaler.step(hasOverflow)) {
                zeroGradients(t, logits);
                clearGpuParamGradsAfterOverflowSkip(t);
                zeroGpuGradsMarkingParamGradsClean(t, paramMap);
                logGradientOverflowSkipped(t, t.dynamicLossScaler.getScale());
                LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip();
                return false;
            }
        } else if (hasOverflow) {
            zeroGradients(t, logits);
            clearGpuParamGradsAfterOverflowSkip(t);
            zeroGpuGradsMarkingParamGradsClean(t, paramMap);
            LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip();
            return false;
        }

        float lossScaleForUnscale = scaleUsedInForward;
        if (lossScaleForUnscale > 1f) {
            DynamicLossScaler.unscaleGpuDeviceGrads(paramMap, lossScaleForUnscale);
            if (logits.hasGrad()) {
                t.logitsOnlyScratch.clear();
                t.logitsOnlyScratch.add(logits);
                t.dynamicLossScaler.unscaleGradients(t.logitsOnlyScratch);
            }
        }

        double sumSq = sumSquaresGpuParamGrads(t, paramMap);
        if (logits.hasGrad()) {
            float[] lg = logits.gradBuffer();
            if (lg.length > 0) {
                sumSq += TensorOpsGPU.sumSquaresGPU(lg, lg.length);
            }
        }
        float totalNorm = (float) Math.sqrt(sumSq);
        t.lastGlobalGradNorm = totalNorm;
        if (totalNorm > t.config.maxGradNorm && t.config.maxGradNorm > 0f) {
            float clipCoeff = t.config.maxGradNorm / totalNorm;
            for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
                GpuTensor gt = e.getValue();
                if (gt.hasGradBuffer()) {
                    TensorOpsGPU.scaleInPlaceGpuDevice(gt.gradBuffer(), e.getKey().size(), clipCoeff);
                }
            }
            if (logits.hasGrad()) {
                float[] lg = logits.gradBuffer();
                if (lg.length > 0) {
                    TensorOpsGPU.scaleInPlaceGPU(lg, lg.length, clipCoeff);
                }
            }
        }

        for (Tensor p : t.parameters) {
            GpuTensor gt = paramMap.get(p);
            if (gt != null && p.hasGrad() && gt.hasGradBuffer()) {
                gt.gradBuffer().copyTo(p.gradBuffer(), 0, p.size());
            }
        }

        optimizerStep(t);
        t.model.onParametersUpdated();

        zeroGpuGradsMarkingParamGradsClean(t, paramMap);
        zeroGradients(t, logits);
        return true;
    }

    static void logGradientOverflowSkipped(LLMTrainer t, float scaleAfterSkip) {
        int planned = t.globalStep + 1;
        if (planned != t.overflowLogPlannedStepKey) {
            t.overflowLogPlannedStepKey = planned;
            t.overflowSkipRepeatCount = 0;
        }
        t.overflowSkipRepeatCount++;
        String scaleStr = String.format(Locale.ROOT, "%.4g", scaleAfterSkip);
        if (t.overflowSkipRepeatCount == 1) {
            log.warn(
                    "Шаг {} пропущен: переполнение градиентов/loss, масштаб loss (после сброса) {}×",
                    planned,
                    scaleStr);
        } else {
            log.debug(
                    "Шаг {} пропущен снова ({} подряд; scale {})",
                    planned,
                    t.overflowSkipRepeatCount,
                    scaleStr);
        }
        t.trainingEventCallback.onOverflowStepSkipped(planned, t.overflowSkipRepeatCount, scaleAfterSkip);
        if (t.trainingStatsWriter != null) {
            t.trainingStatsWriter.onOverflow(planned);
        }
    }

    static boolean checkGradientOverflow(LLMTrainer t, Tensor logits, float avgMicroLoss) {
        if (!Float.isFinite(avgMicroLoss)) {
            return true;
        }
        for (int pi = 0; pi < t.parameters.size(); pi++) {
            Tensor p = t.parameters.get(pi);
            if (p.hasGrad() && !floatArrayIsFinite(p.gradBuffer())) {
                return true;
            }
        }
        if (logits.hasGrad() && !floatArrayIsFinite(logits.gradBuffer())) {
            return true;
        }
        if (t.config.fullGpuTrainStep && t.model.isGpuResident()) {
            Map<Tensor, GpuTensor> gpuParams = t.model.gpuTensorByTrainableParameter();
            if (checkGpuParamGradsNonFiniteFused(t, gpuParams) != null) {
                return true;
            }
        }
        return false;
    }

    static List<Tensor> collectGradTensorsWithLossScale(LLMTrainer t, Tensor logits) {
        List<Tensor> list = new ArrayList<>(t.parameters.size() + 1);
        for (Tensor p : t.parameters) {
            if (p.hasGrad()) {
                list.add(p);
            }
        }
        if (logits.hasGrad()) {
            list.add(logits);
        }
        return list;
    }

    static boolean floatArrayIsFinite(float[] a) {
        for (float v : a) {
            if (!Float.isFinite(v)) {
                return false;
            }
        }
        return true;
    }

    static void scaleGradients(List<Tensor> tensors, float scale) {
        for (Tensor tensor : tensors) {
            scaleTensorGrad(tensor, scale);
        }
    }

    static void scaleGpuGradients(Map<Tensor, GpuTensor> paramMap, float scale) {
        if (scale == 1f) {
            return;
        }
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt.hasGradBuffer()) {
                TensorOpsGPU.scaleInPlaceGpuDevice(gt.gradBuffer(), e.getKey().size(), scale);
            }
        }
        GpuPendingGradients.scaleAll(scale);
    }

    private static void scaleTensorGrad(Tensor tensor, float scale) {
        if (!tensor.hasGrad()) {
            return;
        }
        float[] g = tensor.gradBuffer();
        if (TensorOpsGPU.shouldUseGpuOptimizer(g.length)) {
            TensorOpsGPU.scaleInPlaceGPU(g, g.length, scale);
            return;
        }
        for (int i = 0; i < g.length; i++) {
            g[i] *= scale;
        }
    }

    static void zeroGradients(LLMTrainer t, Tensor logits) {
        if (!t.config.fullGpuTrainStep || !t.model.isGpuResident()) {
            for (Tensor p : t.parameters) {
                if (p.hasGrad()) {
                    p.zeroGrad();
                }
            }
        }
        t.model.clearSampledTrainLossGrad();
        if (logits.hasGrad()) {
            logits.zeroGrad();
        }
    }

    static void clearGpuParamGradsAfterOverflowSkip(LLMTrainer t) {
        if (t.config.accumulationSteps > 1 && t.model.isGpuResident()) {
            t.model.zeroGpuTrainableParameterGrads();
        }
    }

    static void optimizerStep(LLMTrainer t) {
        int stepForLr = t.globalStep + 1;
        t.optimizer.setLearningRate(t.learningRateForStep(stepForLr));
        t.optimizer.beginStep();
        t.optimizer.stepAllWithParamGrad(t.parameters);
    }

    static boolean clipAndOptimizerStepFullGpu(LLMTrainer t, Tensor logits, float avgMicroLoss) {
        Map<Tensor, GpuTensor> paramMap = t.model.gpuTensorByTrainableParameter();
        int stepForLr = t.globalStep + 1;

        if (t.fp16Matmul) {
            Fp16Metrics.global().recordStep();
        }
        TensorOpsGPU.synchronizeStream();
        boolean lossNonFinite = !Float.isFinite(avgMicroLoss);
        boolean pendingNonFinite = GpuPendingGradients.anyNonFinitePending();
        String pendingDebug = pendingNonFinite ? GpuPendingGradients.firstNonFinitePendingDebugInfo() : "";
        boolean hasOverflow = lossNonFinite || pendingNonFinite;
        if (lossNonFinite || pendingNonFinite) {
            GpuPendingGradients.discardDirtyPending();
        } else {
            GpuPendingGradients.flushMergeToGpuGrads(paramMap);
        }
        String firstNonFiniteGrad = null;
        if (!hasOverflow) {
            firstNonFiniteGrad = checkGpuParamGradsNonFiniteFused(t, paramMap);
            if (firstNonFiniteGrad != null) {
                hasOverflow = true;
            }
        }
        boolean logitsGradNonFinite = !hasOverflow && logits.hasGrad() && !floatArrayIsFinite(logits.gradBuffer());
        if (logitsGradNonFinite) {
            hasOverflow = true;
        }

        if (DebugGpuTrain.isEnabled() && (hasOverflow || t.globalStep >= 10)) {
            String primary;
            if (lossNonFinite) {
                primary = "avgMicroLoss";
            } else if (pendingNonFinite) {
                primary = "pendingNonFinite";
            } else if (firstNonFiniteGrad != null) {
                primary = "deviceParamGrad";
            } else if (logitsGradNonFinite) {
                primary = "logitsHostGrad";
            } else if (!hasOverflow) {
                primary = "preUnscaleOk";
            } else {
                primary = "unknown";
            }
            LlmTrainerDebugLog.b39372(
                    "H_ovf",
                    "LlmTrainerOptimizerStep.clipAndOptimizerStepFullGpu",
                    "state",
                    "{\"plannedStep\":"
                            + stepForLr
                            + ",\"globalStep\":"
                            + t.globalStep
                            + ",\"hasOverflow\":"
                            + hasOverflow
                            + ",\"primary\":\""
                            + LlmTrainerDebugLog.jsonEsc(primary)
                            + "\",\"avgMicroLoss\":"
                            + avgMicroLoss
                            + ",\"lossScale\":"
                            + (t.fp16Matmul && t.dynamicLossScaler != null ? t.dynamicLossScaler.getScale() : 1f)
                            + ",\"pending\":"
                            + pendingNonFinite
                            + ",\"logitsHostBad\":"
                            + logitsGradNonFinite
                            + ",\"devGradHint\":\""
                            + LlmTrainerDebugLog.jsonEsc(firstNonFiniteGrad)
                            + "\",\"pendingKey\":\""
                            + LlmTrainerDebugLog.jsonEsc(pendingDebug)
                            + "\",\"cudaLib\":\""
                            + LlmTrainerDebugLog.jsonEsc(
                                    TensorCudaLibrary.getLastLoadedPath() != null
                                            ? TensorCudaLibrary.getLastLoadedPath()
                                            : "")
                            + "\"}");
        }

        if (t.fp16Matmul && hasOverflow) {
            Fp16Metrics.global().recordOverflow();
            t.dynamicLossScaler.step(true);
            zeroGradients(t, logits);
            clearGpuParamGradsAfterOverflowSkip(t);
            zeroGpuGradsMarkingParamGradsClean(t, paramMap);
            logGradientOverflowSkipped(t, t.dynamicLossScaler.getScale());
            LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip();
            return false;
        }
        if (!t.fp16Matmul && hasOverflow) {
            zeroGradients(t, logits);
            clearGpuParamGradsAfterOverflowSkip(t);
            zeroGpuGradsMarkingParamGradsClean(t, paramMap);
            LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip();
            return false;
        }

        float lossScaleForUnscale = t.fp16Matmul ? t.dynamicLossScaler.getScale() : 1f;
        if (lossScaleForUnscale > 1f) {
            DynamicLossScaler.unscaleGpuDeviceGrads(paramMap, lossScaleForUnscale);
            if (logits.hasGrad()) {
                t.logitsOnlyScratch.clear();
                t.logitsOnlyScratch.add(logits);
                t.dynamicLossScaler.unscaleGradients(t.logitsOnlyScratch);
            }
        }

        double sumSq = sumSquaresGpuParamGrads(t, paramMap);
        if (logits.hasGrad()) {
            float[] lg = logits.gradBuffer();
            if (lg.length > 0) {
                sumSq += TensorOpsGPU.sumSquaresGPU(lg, lg.length);
            }
        }
        float totalNorm = (float) Math.sqrt(sumSq);
        t.lastGlobalGradNorm = totalNorm;
        if (!Float.isFinite(totalNorm)) {
            if (DebugGpuTrain.isEnabled()) {
                LlmTrainerDebugLog.b39372(
                        "H_norm",
                        "LlmTrainerOptimizerStep.clipAndOptimizerStepFullGpu",
                        "non_finite_total_norm",
                        "{\"plannedStep\":"
                                + stepForLr
                                + ",\"globalStep\":"
                                + t.globalStep
                                + ",\"sumSq\":"
                                + sumSq
                                + ",\"totalNorm\":"
                                + totalNorm
                                + ",\"lossScale\":"
                                + (t.fp16Matmul && t.dynamicLossScaler != null ? t.dynamicLossScaler.getScale() : 1f)
                                + "}");
            }
            if (t.fp16Matmul) {
                Fp16Metrics.global().recordOverflow();
                t.dynamicLossScaler.step(true);
            }
            zeroGradients(t, logits);
            clearGpuParamGradsAfterOverflowSkip(t);
            zeroGpuGradsMarkingParamGradsClean(t, paramMap);
            logGradientOverflowSkipped(
                    t, t.fp16Matmul && t.dynamicLossScaler != null ? t.dynamicLossScaler.getScale() : 1f);
            LlmTrainerGpuUtils.synchronizeGpuAfterOverflowSkip();
            return false;
        }
        float clipCoeff = 1f;
        if (totalNorm > t.config.maxGradNorm && t.config.maxGradNorm > 0f) {
            clipCoeff = t.config.maxGradNorm / totalNorm;
            for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
                GpuTensor gt = e.getValue();
                if (gt.hasGradBuffer()) {
                    TensorOpsGPU.scaleInPlaceGpuDevice(gt.gradBuffer(), e.getKey().size(), clipCoeff);
                }
            }
            if (logits.hasGrad()) {
                float[] lg = logits.gradBuffer();
                if (lg.length > 0) {
                    TensorOpsGPU.scaleInPlaceGPU(lg, lg.length, clipCoeff);
                }
            }
        }
        if (t.fp16Matmul) {
            t.dynamicLossScaler.step(false);
        }
        t.optimizer.setLearningRate(t.learningRateForStep(stepForLr));
        t.optimizer.beginStep();
        t.optimizer.stepAllGpuDevice(paramMap);
        t.model.onGpuParametersUpdated();

        zeroGpuGradsMarkingParamGradsClean(t, paramMap);
        zeroGradients(t, logits);
        return true;
    }

    private static String firstNonFiniteGpuParamInfo(LLMTrainer t, Map<Tensor, GpuTensor> paramMap, boolean grad) {
        for (int i = 0; i < t.parameters.size(); i++) {
            Tensor cpu = t.parameters.get(i);
            GpuTensor gt = paramMap.get(cpu);
            if (gt == null) {
                continue;
            }
            if (grad) {
                if (gt.hasGradBuffer() && TensorOpsGPU.anyNonFiniteGpuDevice(gt.gradBuffer(), cpu.size())) {
                    return "param#" + i + "/grad/size=" + cpu.size();
                }
            } else if (TensorOpsGPU.anyNonFiniteGpuDevice(gt.dataBuffer(), cpu.size())) {
                return "param#" + i + "/weight/size=" + cpu.size();
            }
        }
        return null;
    }

    private static String checkGpuParamGradsNonFiniteFused(LLMTrainer t, Map<Tensor, GpuTensor> paramMap) {
        int n = paramMap.size();
        if (n == 0) {
            return null;
        }
        if (t.nonFiniteParamBufsScratch.length < n) {
            t.nonFiniteParamBufsScratch = new GpuFloatBuffer[n];
            t.nonFiniteParamLensScratch = new int[n];
        }
        int count = 0;
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt != null && gt.hasGradBuffer()) {
                t.nonFiniteParamBufsScratch[count] = gt.gradBuffer();
                t.nonFiniteParamLensScratch[count] = e.getKey().size();
                count++;
            }
        }
        if (count == 0) {
            return null;
        }
        if (!TensorOpsGPU.anyNonFiniteGpuDeviceMulti(
                t.nonFiniteParamBufsScratch, t.nonFiniteParamLensScratch, count)) {
            return null;
        }
        return firstNonFiniteGpuParamInfo(t, paramMap, true);
    }

    static void markGpuTrainableParamGradsMaybeDirtyAfterHostOptimizerPath(LLMTrainer t) {
        if (t.model.isGpuResident()) {
            t.gpuTrainableParamGradsKnownClean = false;
        }
    }

    private static double sumSquaresGpuParamGrads(LLMTrainer t, Map<Tensor, GpuTensor> paramMap) {
        int n = paramMap.size();
        if (n == 0) {
            return 0.0;
        }
        if (t.sumSqPtrsScratch.length < n) {
            t.sumSqPtrsScratch = new long[n];
            t.sumSqLensScratch = new int[n];
        }
        int count = 0;
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt != null && !gt.isClosed() && gt.hasGradBuffer()) {
                t.sumSqPtrsScratch[count] = gt.gradBuffer().devicePointer();
                t.sumSqLensScratch[count] = e.getKey().size();
                count++;
            }
        }
        if (count == 0) {
            return 0.0;
        }
        return TensorOpsGPU.sumSquaresGPUDeviceFused(t.sumSqPtrsScratch, t.sumSqLensScratch, count);
    }

    static void zeroGpuGradsMarkingParamGradsClean(LLMTrainer t, Map<Tensor, GpuTensor> paramMap) {
        zeroGpuGrads(paramMap);
        if (t.model.isGpuResident()) {
            t.gpuTrainableParamGradsKnownClean = true;
        }
    }

    private static void zeroGpuGrads(Map<Tensor, GpuTensor> paramMap) {
        for (GpuTensor gt : paramMap.values()) {
            if (gt.hasGradBuffer()) {
                gt.zeroGrad();
            }
        }
    }
}
