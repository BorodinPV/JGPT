package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuPendingGradients;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.data.DataLoader;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.ops.GpuDropout;
import com.veles.llm.jgpt.ops.GpuWorkspaceCleanup;

import java.util.List;
import java.util.Map;
import java.util.Random;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Gradient check VRAM-пути с dropout: производная по направлению из backward должна совпадать с центральной
 * разностью loss при замороженной маске. Пробег без dropout даёт базовое отношение analytic/numeric (учитывает
 * возможный loss-scale пайплайна), пробег с dropout должен дать то же отношение — иначе маска forward и
 * backward расходится (seed/слой/сайт) или residual маскируется дважды.
 */
class GpuDropoutGradCheckTest {

    @BeforeEach
    void setUp() {
        GpuPendingGradients.cleanupThreadLocal();
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
    }

    @AfterEach
    void tearDown() {
        GpuDropout.unfreezeStepSeedForTests();
        GpuDropout.configure(0f, 0f);
        GpuPendingGradients.cleanupThreadLocal();
        GpuWorkspaceCleanup.releaseAllGpuWorkspacesThreadLocal();
    }

    private static TrainingConfig unifiedFullGpuNoDropout() {
        return new TrainingConfig(
                48, 8, 32, 4, 2, 64, 2, 1, 1, 0.001f, 0f, 0.01f, 1.0f, 10_000, 10_000,
                LearningRateSchedule.CONSTANT, 0f, "checkpoints_gpu_dropout_gradcheck", 10_000, 0, 0, false, 0f, 0,
                true, true, true, true, false);
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

    private static float[][] gpuGrads(GPTModel model) {
        TensorOpsGPU.synchronize();
        Map<Tensor, GpuTensor> pmap = model.gpuTensorByTrainableParameter();
        GpuPendingGradients.flushMergeToGpuGrads(pmap);
        List<Tensor> ps = model.getParameters();
        float[][] out = new float[ps.size()][];
        for (int i = 0; i < ps.size(); i++) {
            GpuTensor gt = pmap.get(ps.get(i));
            float[] arr = new float[ps.get(i).size()];
            if (gt != null && gt.hasGradBuffer()) {
                gt.gradBuffer().copyTo(arr, 0, arr.length);
            }
            out[i] = arr;
        }
        return out;
    }

    private static float[][] snapshot(GPTModel m) {
        List<Tensor> ps = m.getParameters();
        float[][] s = new float[ps.size()][];
        for (int i = 0; i < ps.size(); i++) {
            s[i] = ps.get(i).internalBuffer().clone();
        }
        return s;
    }

    private static void setParams(GPTModel m, float[][] base, float[][] dir, float eps) {
        List<Tensor> ps = m.getParameters();
        for (int i = 0; i < ps.size(); i++) {
            float[] dst = ps.get(i).internalBuffer();
            for (int j = 0; j < dst.length; j++) {
                dst[j] = base[i][j] + eps * dir[i][j];
            }
        }
        m.syncGpuResidentWeightsFromHost();
    }

    private static float[][] randomDirection(float[][] base, long seed) {
        Random r = new Random(seed);
        float[][] d = new float[base.length][];
        for (int i = 0; i < base.length; i++) {
            d[i] = new float[base[i].length];
            for (int j = 0; j < d[i].length; j++) {
                d[i][j] = (float) r.nextGaussian();
            }
        }
        return d;
    }

    /** @return {analytic g·d, numeric (L+ - L-)/2ε, L0} */
    private static double[] directionalDerivative(
            LLMTrainer t, GPTModel m, DataLoader.Batch batch, float[][] base, float[][] dir, float eps) {
        setParams(m, base, dir, 0f);
        GpuPendingGradients.cleanupThreadLocal();
        LLMTrainer.TestMicrobatchResult r0 = t.testHarnessForwardCeBackward(batch, true);
        float[][] g = gpuGrads(m);
        double analytic = 0;
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[i].length; j++) {
                analytic += (double) g[i][j] * dir[i][j];
            }
        }
        setParams(m, base, dir, eps);
        GpuPendingGradients.cleanupThreadLocal();
        float lPlus = t.testHarnessForwardCeBackward(batch, true).ceLoss;
        setParams(m, base, dir, -eps);
        GpuPendingGradients.cleanupThreadLocal();
        float lMinus = t.testHarnessForwardCeBackward(batch, true).ceLoss;
        setParams(m, base, dir, 0f);
        double numeric = ((double) lPlus - lMinus) / (2.0 * eps);
        return new double[] {analytic, numeric, r0.ceLoss};
    }

    @Test
    void directionalDerivative_withDropout_matchesFiniteDifference() {
        assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        GPTModel m = null;
        LLMTrainer t = null;
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            m = new GPTModel(48, 8, 32, 4, 2, 64, true);
            t = new LLMTrainer(m, unifiedFullGpuNoDropout(), dummyLoader());
            DataLoader.Batch batch = fixedBatch(48, 2, 8);
            float[][] base = snapshot(m);
            float[][] dir = randomDirection(base, 7L);
            float eps = 2e-4f;

            GpuDropout.configure(0f, 0f);
            double[] noDrop = directionalDerivative(t, m, batch, base, dir, eps);
            double ratioNoDrop = noDrop[0] / noDrop[1];
            assertTrue(
                    Math.abs(noDrop[1]) > 1e-3,
                    "baseline numeric derivative too small to compare: " + noDrop[1]);
            assertTrue(
                    Math.abs(ratioNoDrop - 1.0) < 0.05,
                    "baseline (no dropout) analytic/numeric = " + ratioNoDrop + " analytic=" + noDrop[0] + " numeric=" + noDrop[1]);

            GpuDropout.configure(0.25f, 0.2f);
            GpuDropout.freezeStepSeedForTests(0x5EEDL);
            double[] drop = directionalDerivative(t, m, batch, base, dir, eps);
            double ratioDrop = drop[0] / drop[1];
            assertTrue(
                    Math.abs(drop[2] - noDrop[2]) > 1e-6,
                    "dropout должен менять forward loss (маска не применилась): " + drop[2] + " vs " + noDrop[2]);
            assertTrue(
                    Math.abs(ratioDrop - ratioNoDrop) < 0.05,
                    "with dropout analytic/numeric = " + ratioDrop + " (baseline " + ratioNoDrop + ") analytic=" + drop[0]
                            + " numeric=" + drop[1]);

            // Другой seed → другая маска → другой loss; при той же маске loss воспроизводим.
            GpuDropout.freezeStepSeedForTests(0xBEEFL);
            setParams(m, base, dir, 0f);
            GpuPendingGradients.cleanupThreadLocal();
            float lOther = t.testHarnessForwardCeBackward(batch, true).ceLoss;
            GpuDropout.freezeStepSeedForTests(0x5EEDL);
            GpuPendingGradients.cleanupThreadLocal();
            float lSame = t.testHarnessForwardCeBackward(batch, true).ceLoss;
            assertTrue(Math.abs(lSame - drop[2]) < 1e-4, "same seed must reproduce loss: " + lSame + " vs " + drop[2]);
            assertTrue(Math.abs(lOther - drop[2]) > 1e-6, "different seed must change loss: " + lOther + " vs " + drop[2]);
        } finally {
            GpuDropout.unfreezeStepSeedForTests();
            GpuDropout.configure(0f, 0f);
            if (t != null) {
                t.releaseGpuResourcesAfterBook();
            } else if (m != null) {
                m.closeGpuResidentWeights();
            }
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
        }
    }
}
