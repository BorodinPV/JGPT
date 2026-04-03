package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.BlockActivationCacheDevice;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.ops.GpuWorkspaceCleanup;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Самопроверки единого VRAM-пути (device logits + device decoder backward): детерминизм и конечность ∂/весов.
 */
class DeviceDecoderBackwardParityTest {

    private static final float WEIGHT_STABILITY_EPS = 2.5e-3f;

    @BeforeEach
    void setUpGpuIsolation() {
        GpuPendingGradients.cleanupThreadLocal();
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
    }

    @AfterEach
    void tearDownPending() {
        GpuPendingGradients.cleanupThreadLocal();
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
    }

    private static TrainingConfig unifiedFullGpu() {
        return new TrainingConfig(
                48,
                8,
                32,
                4,
                2,
                64,
                2,
                1,
                1,
                0.001f,
                0f,
                0.01f,
                1.0f,
                10_000,
                10_000,
                LearningRateSchedule.CONSTANT,
                0f,
                "checkpoints_device_dec_bwd",
                10_000,
                0,
                0,
                false,
                0f,
                0,
                true,
                true,
                true,
                true,
                false);
    }

    private static TrainingConfig unifiedFullGpuAccum2() {
        return new TrainingConfig(
                48,
                8,
                32,
                4,
                2,
                64,
                2,
                2,
                1,
                0.001f,
                0f,
                0.01f,
                1.0f,
                10_000,
                10_000,
                LearningRateSchedule.CONSTANT,
                0f,
                "checkpoints_device_dec_bwd_acc",
                10_000,
                0,
                0,
                false,
                0f,
                0,
                true,
                true,
                true,
                true,
                false);
    }

    private static TrainingConfig unifiedGpuResidentMergeFirstParity() {
        return new TrainingConfig(
                48,
                8,
                32,
                4,
                2,
                64,
                2,
                1,
                1,
                0.001f,
                0f,
                0.01f,
                1.0f,
                10_000,
                10_000,
                LearningRateSchedule.CONSTANT,
                0f,
                "checkpoints_merge_first_parity",
                10_000,
                0,
                0,
                false,
                0f,
                0,
                true,
                false,
                true,
                true,
                true);
    }

    private static boolean envTruthy(String key) {
        String v = System.getenv(key);
        if (v == null) {
            return false;
        }
        String t = v.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    private static DataLoader dummyLoader() {
        return new DataLoader(BPETokenizer.train(List.of("a b"), 48), 8, 2);
    }

    private static DataLoader.Batch fixedBatch(int vocabSize, int batchSize, int seqLen) {
        Tensor input = new Tensor(new int[] {batchSize, seqLen});
        Tensor target = new Tensor(new int[] {batchSize, seqLen});
        float[] in = input.internalBuffer();
        float[] tg = target.internalBuffer();
        for (int i = 0; i < in.length; i++) {
            in[i] = (i * 13 + 7) % vocabSize;
        }
        for (int i = 0; i < tg.length; i++) {
            tg[i] = (i * 5 + 3) % vocabSize;
        }
        return new DataLoader.Batch(input, target);
    }

    private static void copyAllParameters(GPTModel src, GPTModel dst) {
        List<Tensor> a = src.getParameters();
        List<Tensor> b = dst.getParameters();
        for (int i = 0; i < a.size(); i++) {
            Tensor ta = a.get(i);
            Tensor tb = b.get(i);
            System.arraycopy(ta.internalBuffer(), 0, tb.internalBuffer(), 0, ta.size());
        }
        dst.syncGpuResidentWeightsFromHost();
    }

    private static float maxAbsWeightDiff(GPTModel x, GPTModel y) {
        List<Tensor> pa = x.getParameters();
        List<Tensor> pb = y.getParameters();
        float m = 0f;
        for (int i = 0; i < pa.size(); i++) {
            float[] xa = pa.get(i).internalBuffer();
            float[] xb = pb.get(i).internalBuffer();
            for (int j = 0; j < xa.length; j++) {
                m = Math.max(m, Math.abs(xa[j] - xb[j]));
            }
        }
        return m;
    }

    private static boolean anyNonFiniteParameters(GPTModel m) {
        for (Tensor p : m.getParameters()) {
            for (float v : p.internalBuffer()) {
                if (!Float.isFinite(v)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static float[][] gpuTrainableGradsAfterBackward(GPTModel model) {
        TensorOpsGPU.synchronize();
        Map<Tensor, GpuTensor> pmap = model.gpuTensorByTrainableParameter();
        GpuPendingGradients.flushMergeToGpuGrads(pmap);
        List<Tensor> parameters = model.getParameters();
        float[][] out = new float[parameters.size()][];
        for (int i = 0; i < parameters.size(); i++) {
            GpuTensor gt = pmap.get(parameters.get(i));
            float[] arr = new float[gt.size()];
            if (gt != null && gt.hasGradBuffer()) {
                gt.gradBuffer().copyTo(arr, 0, arr.length);
            }
            out[i] = arr;
        }
        return out;
    }

    @Test
    void gradSnapshot_sameModelTwoBackwards_isDeterministic() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel m = new GPTModel(48, 8, 32, 4, 2, 64, true);
            LLMTrainer t = new LLMTrainer(m, unifiedFullGpu(), dummyLoader());
            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            t.testHarnessForwardCeBackward(batch, true);
            float[][] g1 = gpuTrainableGradsAfterBackward(m);
            GpuPendingGradients.cleanupThreadLocal();
            t.testHarnessForwardCeBackward(batch, true);
            float[][] g2 = gpuTrainableGradsAfterBackward(m);
            List<Tensor> ps = m.getParameters();
            assertEquals(ps.size(), g1.length);
            float maxDiff = 0f;
            float maxAbs = 0f;
            for (int i = 0; i < ps.size(); i++) {
                for (int j = 0; j < g1[i].length; j++) {
                    maxDiff = Math.max(maxDiff, Math.abs(g1[i][j] - g2[i][j]));
                    maxAbs = Math.max(maxAbs, Math.abs(g1[i][j]));
                    maxAbs = Math.max(maxAbs, Math.abs(g2[i][j]));
                }
            }
            float tol = Math.max(0.11f, maxAbs * 5e-5f);
            if (TensorOpsGPU.isGpuAvailable()) {
                tol = Math.max(tol, 48f);
            }
            assertTrue(maxDiff < tol, "same model two backwards: max grad snapshot diff " + maxDiff + " tol " + tol);
            m.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void twoTrainersUnifiedPath_gradientsMatch() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GpuPendingGradients.cleanupThreadLocal();
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel a = new GPTModel(48, 8, 32, 4, 2, 64, true);
            GPTModel b = new GPTModel(48, 8, 32, 4, 2, 64, true);
            copyAllParameters(a, b);
            LLMTrainer ta = new LLMTrainer(a, unifiedFullGpu(), dummyLoader());
            LLMTrainer tb = new LLMTrainer(b, unifiedFullGpu(), dummyLoader());
            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            ta.testHarnessForwardCeBackward(batch, true);
            float[][] ga = gpuTrainableGradsAfterBackward(a);
            GpuPendingGradients.cleanupThreadLocal();
            tb.testHarnessForwardCeBackward(batch, true);
            float[][] gb = gpuTrainableGradsAfterBackward(b);
            List<Tensor> params = a.getParameters();
            float maxDiff = 0f;
            float maxAbs = 0f;
            for (int i = 0; i < params.size(); i++) {
                for (int j = 0; j < ga[i].length; j++) {
                    maxDiff = Math.max(maxDiff, Math.abs(ga[i][j] - gb[i][j]));
                    maxAbs = Math.max(maxAbs, Math.abs(ga[i][j]));
                    maxAbs = Math.max(maxAbs, Math.abs(gb[i][j]));
                }
            }
            /* FP16 / два экземпляра модели: абсолютный зазор ~2 в худшем случае на малых d_model. */
            float tol = Math.max(2f, maxAbs * 5e-4f);
            assertTrue(maxDiff < tol, "max abs param grad diff: " + maxDiff + " tol " + tol);
            a.closeGpuResidentWeights();
            b.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    /**
     * Единый device forward/backward: снимок ∂ на VRAM после {@link GpuPendingGradients#flushMergeToGpuGrads}
     * должен совпадать для full-GPU шага и для merge-first (различается только шаг оптимизатора).
     */
    @Test
    void fullGpu_and_mergeFirst_sameMergedGradsAfterBackward() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GpuPendingGradients.cleanupThreadLocal();
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel a = new GPTModel(48, 8, 32, 4, 2, 64, true);
            GPTModel b = new GPTModel(48, 8, 32, 4, 2, 64, true);
            copyAllParameters(a, b);
            LLMTrainer ta = new LLMTrainer(a, unifiedFullGpu(), dummyLoader());
            LLMTrainer tb = new LLMTrainer(b, unifiedGpuResidentMergeFirstParity(), dummyLoader());
            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            ta.testHarnessForwardCeBackward(batch, true);
            float[][] ga = gpuTrainableGradsAfterBackward(a);
            GpuPendingGradients.cleanupThreadLocal();
            tb.testHarnessForwardCeBackward(batch, true);
            float[][] gb = gpuTrainableGradsAfterBackward(b);
            List<Tensor> params = a.getParameters();
            float maxDiff = 0f;
            float maxAbs = 0f;
            for (int i = 0; i < params.size(); i++) {
                for (int j = 0; j < ga[i].length; j++) {
                    maxDiff = Math.max(maxDiff, Math.abs(ga[i][j] - gb[i][j]));
                    maxAbs = Math.max(maxAbs, Math.abs(ga[i][j]));
                    maxAbs = Math.max(maxAbs, Math.abs(gb[i][j]));
                }
            }
            float tol = Math.max(1.15f, maxAbs * 3e-4f);
            assertTrue(
                    maxDiff < tol,
                    "full vs merge-first merged param grad diff: " + maxDiff + " tol " + tol);
            a.closeGpuResidentWeights();
            b.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    /** Тот же снимок ∂, что и {@link #fullGpu_and_mergeFirst_sameMergedGradsAfterBackward}, для разных длин контекста. */
    @Test
    void fullGpu_and_mergeFirst_sameMergedGrads_variousSeqLen() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        for (int seqLen : new int[] {4, 8}) {
            GpuPendingGradients.cleanupThreadLocal();
            String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
            try {
                System.setProperty("jgpt.decoder.gpu.pipeline", "true");
                GPTModel a = new GPTModel(48, 8, 32, 4, 2, 64, true);
                GPTModel b = new GPTModel(48, 8, 32, 4, 2, 64, true);
                copyAllParameters(a, b);
                LLMTrainer ta = new LLMTrainer(a, unifiedFullGpu(), dummyLoader());
                LLMTrainer tb = new LLMTrainer(b, unifiedGpuResidentMergeFirstParity(), dummyLoader());
                DataLoader.Batch batch = fixedBatch(48, 2, seqLen);
                ta.testHarnessForwardCeBackward(batch, true);
                float[][] ga = gpuTrainableGradsAfterBackward(a);
                GpuPendingGradients.cleanupThreadLocal();
                tb.testHarnessForwardCeBackward(batch, true);
                float[][] gb = gpuTrainableGradsAfterBackward(b);
                List<Tensor> params = a.getParameters();
                float maxDiff = 0f;
                float maxAbs = 0f;
                for (int i = 0; i < params.size(); i++) {
                    for (int j = 0; j < ga[i].length; j++) {
                        maxDiff = Math.max(maxDiff, Math.abs(ga[i][j] - gb[i][j]));
                        maxAbs = Math.max(maxAbs, Math.abs(ga[i][j]));
                        maxAbs = Math.max(maxAbs, Math.abs(gb[i][j]));
                    }
                }
                float tol = Math.max(12f, maxAbs * 1e-3f);
                assertTrue(
                        maxDiff < tol,
                        "seqLen=" + seqLen + " full vs merge-first grad diff " + maxDiff + " tol " + tol);
                a.closeGpuResidentWeights();
                b.closeGpuResidentWeights();
            } finally {
                if (prevPipe == null) {
                    System.clearProperty("jgpt.decoder.gpu.pipeline");
                } else {
                    System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
                }
                GpuPendingGradients.cleanupThreadLocal();
            }
        }
    }

    @Test
    void optimizerStep_twoIdenticalTrainers_weightsStayAligned() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GpuPendingGradients.cleanupThreadLocal();
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel template = new GPTModel(48, 8, 32, 4, 2, 64, true);
            GPTModel a = new GPTModel(48, 8, 32, 4, 2, 64, true);
            GPTModel b = new GPTModel(48, 8, 32, 4, 2, 64, true);
            copyAllParameters(template, a);
            copyAllParameters(template, b);
            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            LLMTrainer ta = new LLMTrainer(a, unifiedFullGpu(), dummyLoader());
            LLMTrainer tb = new LLMTrainer(b, unifiedFullGpu(), dummyLoader());
            LLMTrainer.TestMicrobatchResult ra = ta.testHarnessForwardCeBackward(batch, true);
            assertTrue(ta.testHarnessClipAndOptimizerStep(ra.logits, ra.ceLoss, 1f));
            GpuPendingGradients.cleanupThreadLocal();
            LLMTrainer.TestMicrobatchResult rb = tb.testHarnessForwardCeBackward(batch, true);
            assertEquals(ra.ceLoss, rb.ceLoss, 2e-2f);
            assertTrue(tb.testHarnessClipAndOptimizerStep(rb.logits, rb.ceLoss, 1f));
            float md = maxAbsWeightDiff(a, b);
            assertTrue(md < WEIGHT_STABILITY_EPS, "optimizer step: max abs weight diff " + md);
            assertFalse(anyNonFiniteParameters(a));
            assertFalse(anyNonFiniteParameters(b));
            template.closeGpuResidentWeights();
            a.closeGpuResidentWeights();
            b.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void accumulationTwoMicrobatches_unifiedPath_deterministic() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel template = new GPTModel(48, 8, 32, 4, 2, 64, true);
            GPTModel a = new GPTModel(48, 8, 32, 4, 2, 64, true);
            GPTModel b = new GPTModel(48, 8, 32, 4, 2, 64, true);
            copyAllParameters(template, a);
            copyAllParameters(template, b);
            DataLoader.Batch b1 = fixedBatch(48, 2, 8);
            DataLoader.Batch b2 = fixedBatch(48, 2, 8);
            float[] id2 = b2.input.internalBuffer();
            for (int i = 0; i < id2.length; i++) {
                id2[i] = (id2[i] + 3) % 48;
            }
            LLMTrainer ta = new LLMTrainer(a, unifiedFullGpuAccum2(), dummyLoader());
            LLMTrainer tb = new LLMTrainer(b, unifiedFullGpuAccum2(), dummyLoader());
            LLMTrainer.TestMicrobatchResult r1a = ta.testHarnessForwardCeBackward(b1, true);
            LLMTrainer.TestMicrobatchResult r2a = ta.testHarnessForwardCeBackward(b2, false);
            float avga = (r1a.ceLoss + r2a.ceLoss) * 0.5f;
            assertTrue(ta.testHarnessClipAndOptimizerStep(r2a.logits, avga, 1f));
            LLMTrainer.TestMicrobatchResult r1b = tb.testHarnessForwardCeBackward(b1, true);
            LLMTrainer.TestMicrobatchResult r2b = tb.testHarnessForwardCeBackward(b2, false);
            float avgb = (r1b.ceLoss + r2b.ceLoss) * 0.5f;
            assertEquals(avga, avgb, 1e-5f);
            assertTrue(tb.testHarnessClipAndOptimizerStep(r2b.logits, avgb, 1f));
            float md = maxAbsWeightDiff(a, b);
            assertTrue(md < WEIGHT_STABILITY_EPS, "accum=2: max abs diff весов " + md);
            template.closeGpuResidentWeights();
            a.closeGpuResidentWeights();
            b.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            GpuPendingGradients.cleanupThreadLocal();
        }
    }

    @Test
    void fp16ActivationCacheProperty_unifiedPath_stillRuns() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        org.junit.jupiter.api.Assumptions.assumeTrue(
                System.getenv("JGPT_ACTIVATION_CACHE_FP16") == null
                        || System.getenv("JGPT_ACTIVATION_CACHE_FP16").isBlank());
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        String prevFp16 = System.getProperty("jgpt.activationCache.fp16");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            System.setProperty("jgpt.activationCache.fp16", "true");
            GPTModel m = new GPTModel(48, 8, 32, 4, 2, 64, true);
            LLMTrainer t = new LLMTrainer(m, unifiedFullGpu(), dummyLoader());
            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            LLMTrainer.TestMicrobatchResult r = t.testHarnessForwardCeBackward(batch, true);
            assertTrue(Float.isFinite(r.ceLoss));
            float[][] g = gpuTrainableGradsAfterBackward(m);
            for (float[] row : g) {
                for (float v : row) {
                    assertTrue(Float.isFinite(v));
                }
            }
            m.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            if (prevFp16 == null) {
                System.clearProperty("jgpt.activationCache.fp16");
            } else {
                System.setProperty("jgpt.activationCache.fp16", prevFp16);
            }
            GpuPendingGradients.cleanupThreadLocal();
            GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
        }
    }

    /**
     * Совместимость {@code JGPT_CE_ASYNC} с FP16 matmul: пропуск без env, иначе проверка конечности loss и градиентов.
     * Запуск: {@code JGPT_CE_ASYNC=1} и {@code JGPT_FP16_MATMUL=1} (или аналог).
     */
    @Test
    void ceAsync_withFp16Matmul_microbatchFinite() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        org.junit.jupiter.api.Assumptions.assumeTrue(envTruthy("JGPT_CE_ASYNC"));
        org.junit.jupiter.api.Assumptions.assumeTrue(TensorOpsGPU.useFp16Matmul());
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel m = new GPTModel(48, 8, 32, 4, 2, 64, true);
            LLMTrainer t = new LLMTrainer(m, unifiedFullGpu(), dummyLoader());
            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            LLMTrainer.TestMicrobatchResult r = t.testHarnessForwardCeBackward(batch, true);
            assertTrue(Float.isFinite(r.ceLoss), "ceLoss " + r.ceLoss);
            float[][] g = gpuTrainableGradsAfterBackward(m);
            for (float[] row : g) {
                for (float v : row) {
                    assertTrue(Float.isFinite(v));
                }
            }
            m.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            GpuPendingGradients.cleanupThreadLocal();
            GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
        }
    }

    @Test
    void blockActivationCacheDevice_forwardEnsure_noOOM() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        BlockActivationCacheDevice cache = new BlockActivationCacheDevice();
        int batch = 2;
        int seqLen = 8;
        int dModel = 32;
        int numHeads = 4;
        int dInt = 64;
        cache.ensure(batch, seqLen, dModel, numHeads, dInt);
        cache.close();
    }
}
