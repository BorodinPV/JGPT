package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Parity: fused JNI vs разнесённые launch’ы ({@link TensorOpsGPU#rmsNormMatmulLmHeadGpuDevice},
 * {@link TensorOpsGPU#rmsNormMatmulFfnW1W3GpuDevice}).
 */
class FusedKernelParityRoadmapTest {

    @Test
    void rmsNormMatmulLmHeadMatchesSplitOps() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");

        int rows = 5;
        int dModel = 32;
        int vocab = 48;
        float eps = 1e-6f;
        long plane = (long) rows * dModel;
        long wElems = (long) dModel * vocab;
        long logitElems = (long) rows * vocab;

        float[] hx = new float[(int) plane];
        float[] hg = new float[dModel];
        float[] hw = new float[(int) wElems];
        for (int i = 0; i < hx.length; i++) {
            hx[i] = (i % 17) * 0.01f - 0.08f;
        }
        for (int i = 0; i < hg.length; i++) {
            hg[i] = 1f + (i % 5) * 0.01f;
        }
        for (int i = 0; i < hw.length; i++) {
            hw[i] = (i % 13) * 0.005f - 0.02f;
        }

        try (GpuFloatBuffer dX = GpuFloatBuffer.allocate(hx.length);
                GpuFloatBuffer dG = GpuFloatBuffer.allocate(hg.length);
                GpuFloatBuffer dW = GpuFloatBuffer.allocate(hw.length);
                GpuFloatBuffer normSplit = GpuFloatBuffer.allocate((int) plane);
                GpuFloatBuffer normFused = GpuFloatBuffer.allocate((int) plane);
                GpuFloatBuffer logitsSplit = GpuFloatBuffer.allocate((int) logitElems);
                GpuFloatBuffer logitsFused = GpuFloatBuffer.allocate((int) logitElems)) {
            dX.copyFrom(hx, 0, hx.length);
            dG.copyFrom(hg, 0, hg.length);
            dW.copyFrom(hw, 0, hw.length);

            TensorOpsGPU.rmsNormGpuDevice(dX, dG, eps, normSplit, rows, dModel);
            TensorOpsGPU.matmulGpuDevice(normSplit, dW, logitsSplit, rows, dModel, vocab);

            boolean usedFused = true;
            try {
                TensorOpsGPU.rmsNormMatmulLmHeadGpuDevice(
                        dX, dG, eps, normFused, dW, logitsFused, rows, dModel, vocab, TensorOpsGPU.useFp16Matmul());
            } catch (Throwable t) {
                usedFused = false;
                TensorOpsGPU.rmsNormGpuDevice(dX, dG, eps, normFused, rows, dModel);
                TensorOpsGPU.matmulGpuDevice(normFused, dW, logitsFused, rows, dModel, vocab);
            }
            TensorOpsGPU.synchronizeStream();

            float[] hostSplit = new float[(int) logitElems];
            float[] hostFused = new float[(int) logitElems];
            logitsSplit.copyTo(hostSplit, 0, hostSplit.length);
            logitsFused.copyTo(hostFused, 0, hostFused.length);

            float maxAbs = 0f;
            for (int i = 0; i < hostSplit.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(hostSplit[i] - hostFused[i]));
            }
            float tol = 1e-5f;
            assertTrue(
                    maxAbs < tol,
                    "max|split−fused|="
                            + maxAbs
                            + " (tol "
                            + tol
                            + ", fusedJNI="
                            + usedFused
                            + ", fp16Matmul="
                            + TensorOpsGPU.useFp16Matmul()
                            + ")");
        }
    }

    @Test
    void ffnW1W3StridedBatchedMatchesTwoMatmuls() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");

        int rows = 7;
        int dModel = 24;
        int dInt = 40;
        long plane = (long) rows * dModel;
        long wElems = (long) dModel * dInt;
        long midElems = (long) rows * dInt;

        float[] hx = new float[(int) plane];
        float[] hw1 = new float[(int) wElems];
        float[] hw3 = new float[(int) wElems];
        for (int i = 0; i < hx.length; i++) {
            hx[i] = (i % 11) * 0.013f - 0.05f;
        }
        for (int i = 0; i < hw1.length; i++) {
            hw1[i] = (i % 7) * 0.007f - 0.02f;
        }
        for (int i = 0; i < hw3.length; i++) {
            hw3[i] = (i % 9) * 0.011f - 0.03f;
        }

        try (GpuFloatBuffer dX = GpuFloatBuffer.allocate(hx.length);
                GpuFloatBuffer dW1 = GpuFloatBuffer.allocate(hw1.length);
                GpuFloatBuffer dW3 = GpuFloatBuffer.allocate(hw3.length);
                GpuFloatBuffer h1Split = GpuFloatBuffer.allocate((int) midElems);
                GpuFloatBuffer gateSplit = GpuFloatBuffer.allocate((int) midElems);
                GpuFloatBuffer h1Fused = GpuFloatBuffer.allocate((int) midElems);
                GpuFloatBuffer gateFused = GpuFloatBuffer.allocate((int) midElems)) {
            dX.copyFrom(hx, 0, hx.length);
            dW1.copyFrom(hw1, 0, hw1.length);
            dW3.copyFrom(hw3, 0, hw3.length);

            TensorOpsGPU.matmulGpuDeviceEx(dX, dW1, h1Split, rows, dModel, dInt, false, false);
            TensorOpsGPU.matmulGpuDeviceEx(dX, dW3, gateSplit, rows, dModel, dInt, false, false);
            TensorOpsGPU.matmulGpuDeviceFfnW1W3Projections(dX, dW1, dW3, h1Fused, gateFused, rows, dModel, dInt);
            TensorOpsGPU.synchronizeStream();

            float[] hs1 = new float[(int) midElems];
            float[] gs1 = new float[(int) midElems];
            float[] hf1 = new float[(int) midElems];
            float[] gf1 = new float[(int) midElems];
            h1Split.copyTo(hs1, 0, hs1.length);
            gateSplit.copyTo(gs1, 0, gs1.length);
            h1Fused.copyTo(hf1, 0, hf1.length);
            gateFused.copyTo(gf1, 0, gf1.length);

            float maxAbs = 0f;
            for (int i = 0; i < hs1.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(hs1[i] - hf1[i]));
                maxAbs = Math.max(maxAbs, Math.abs(gs1[i] - gf1[i]));
            }
            float tol = 1e-5f;
            assertTrue(maxAbs < tol, "max|2×sgemm−strided|=" + maxAbs);
        }
    }

    @Test
    void rmsNormMatmulFfnW1W3MatchesRmsThenStridedBatched() {
        Assumptions.assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA");

        int rows = 6;
        int dModel = 28;
        int dInt = 36;
        float eps = 1e-6f;
        long plane = (long) rows * dModel;
        long wElems = (long) dModel * dInt;
        long midElems = (long) rows * dInt;

        float[] hx = new float[(int) plane];
        float[] hg = new float[dModel];
        float[] hw1 = new float[(int) wElems];
        float[] hw3 = new float[(int) wElems];
        for (int i = 0; i < hx.length; i++) {
            hx[i] = (i % 19) * 0.012f - 0.06f;
        }
        for (int i = 0; i < hg.length; i++) {
            hg[i] = 0.9f + (i % 4) * 0.02f;
        }
        for (int i = 0; i < hw1.length; i++) {
            hw1[i] = (i % 11) * 0.006f - 0.03f;
        }
        for (int i = 0; i < hw3.length; i++) {
            hw3[i] = (i % 13) * 0.008f - 0.025f;
        }

        try (GpuFloatBuffer dX = GpuFloatBuffer.allocate(hx.length);
                GpuFloatBuffer dG = GpuFloatBuffer.allocate(hg.length);
                GpuFloatBuffer dW1 = GpuFloatBuffer.allocate(hw1.length);
                GpuFloatBuffer dW3 = GpuFloatBuffer.allocate(hw3.length);
                GpuFloatBuffer normSplit = GpuFloatBuffer.allocate((int) plane);
                GpuFloatBuffer normFused = GpuFloatBuffer.allocate((int) plane);
                GpuFloatBuffer h1Split = GpuFloatBuffer.allocate((int) midElems);
                GpuFloatBuffer gateSplit = GpuFloatBuffer.allocate((int) midElems);
                GpuFloatBuffer h1Fused = GpuFloatBuffer.allocate((int) midElems);
                GpuFloatBuffer gateFused = GpuFloatBuffer.allocate((int) midElems)) {
            dX.copyFrom(hx, 0, hx.length);
            dG.copyFrom(hg, 0, hg.length);
            dW1.copyFrom(hw1, 0, hw1.length);
            dW3.copyFrom(hw3, 0, hw3.length);

            TensorOpsGPU.rmsNormGpuDevice(dX, dG, eps, normSplit, rows, dModel);
            TensorOpsGPU.matmulGpuDeviceFfnW1W3Projections(
                    normSplit, dW1, dW3, h1Split, gateSplit, rows, dModel, dInt);

            boolean usedFused = true;
            try {
                TensorOpsGPU.rmsNormMatmulFfnW1W3GpuDevice(
                        dX,
                        dG,
                        eps,
                        normFused,
                        dW1,
                        dW3,
                        h1Fused,
                        gateFused,
                        rows,
                        dModel,
                        dInt,
                        TensorOpsGPU.useFp16Matmul());
            } catch (Throwable t) {
                usedFused = false;
                TensorOpsGPU.rmsNormGpuDevice(dX, dG, eps, normFused, rows, dModel);
                TensorOpsGPU.matmulGpuDeviceFfnW1W3Projections(
                        normFused, dW1, dW3, h1Fused, gateFused, rows, dModel, dInt);
            }
            TensorOpsGPU.synchronizeStream();

            float[] ns = new float[(int) plane];
            float[] nf = new float[(int) plane];
            normSplit.copyTo(ns, 0, ns.length);
            normFused.copyTo(nf, 0, nf.length);
            float[] hs = new float[(int) midElems];
            float[] gs = new float[(int) midElems];
            float[] hf = new float[(int) midElems];
            float[] gf = new float[(int) midElems];
            h1Split.copyTo(hs, 0, hs.length);
            gateSplit.copyTo(gs, 0, gs.length);
            h1Fused.copyTo(hf, 0, hf.length);
            gateFused.copyTo(gf, 0, gf.length);

            float maxAbs = 0f;
            for (int i = 0; i < ns.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(ns[i] - nf[i]));
            }
            for (int i = 0; i < hs.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(hs[i] - hf[i]));
                maxAbs = Math.max(maxAbs, Math.abs(gs[i] - gf[i]));
            }
            float tol = 2e-5f;
            assertTrue(
                    maxAbs < tol,
                    "max|split−fused|="
                            + maxAbs
                            + " (tol "
                            + tol
                            + ", fusedJNI="
                            + usedFused
                            + ", fp16Matmul="
                            + TensorOpsGPU.useFp16Matmul()
                            + ")");
        }
    }
}
