package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

class LinearGpuTest {

    private static void cpuBackward(
            float[] x,
            float[] w,
            float[] go,
            int m,
            int in,
            int out,
            float[] gX,
            float[] gW,
            float[] gB) {
        Arrays.fill(gX, 0f);
        Arrays.fill(gW, 0f);
        Arrays.fill(gB, 0f);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < in; j++) {
                float s = 0f;
                for (int k = 0; k < out; k++) {
                    s += go[i * out + k] * w[j * out + k];
                }
                gX[i * in + j] = s;
            }
        }
        for (int ii = 0; ii < in; ii++) {
            for (int jj = 0; jj < out; jj++) {
                float s = 0f;
                for (int r = 0; r < m; r++) {
                    s += x[r * in + ii] * go[r * out + jj];
                }
                gW[ii * out + jj] = s;
            }
        }
        for (int jj = 0; jj < out; jj++) {
            float s = 0f;
            for (int r = 0; r < m; r++) {
                s += go[r * out + jj];
            }
            gB[jj] = s;
        }
    }

    @Test
    void forward_matchesCpu() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int m = 7;
        int in = 5;
        int out = 4;
        Random r = new Random(42);
        float[] xData = new float[m * in];
        float[] wData = new float[in * out];
        float[] bData = new float[out];
        for (int i = 0; i < xData.length; i++) {
            xData[i] = r.nextFloat() * 2f - 1f;
        }
        for (int i = 0; i < wData.length; i++) {
            wData[i] = r.nextFloat() * 0.5f - 0.25f;
        }
        for (int i = 0; i < bData.length; i++) {
            bData[i] = r.nextFloat() * 0.3f;
        }
        Tensor xCpu = Tensor.fromArray(xData, new int[]{m, in});
        Tensor wCpu = Tensor.fromArray(wData, new int[]{in, out});
        Tensor bCpu = Tensor.fromArray(bData, new int[]{out});
        Tensor yRef = Linear.forwardCpu(xCpu, wCpu, bCpu);

        try (GpuTensor wG = GpuTensor.fromHostTensor(wCpu);
                GpuTensor bG = GpuTensor.fromHostTensor(bCpu);
                GpuTensor xG = GpuTensor.fromHostTensor(xCpu);
                Linear layer = new Linear(wG, bG)) {
            GpuTensor yG = layer.forwardGpu(xG);
            Tensor yHost = yG.toHostTensor();
            assertArrayEquals(yRef.internalBuffer(), yHost.internalBuffer(), 2e-3f);
        }
    }

    @Test
    void backward_matchesCpuReference() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int m = 6;
        int in = 4;
        int out = 3;
        Random r = new Random(7);
        float[] xData = new float[m * in];
        float[] wData = new float[in * out];
        float[] bData = new float[out];
        float[] goData = new float[m * out];
        for (int i = 0; i < xData.length; i++) {
            xData[i] = r.nextFloat() - 0.5f;
        }
        for (int i = 0; i < wData.length; i++) {
            wData[i] = r.nextFloat() * 0.4f - 0.2f;
        }
        for (int i = 0; i < bData.length; i++) {
            bData[i] = r.nextFloat() * 0.1f;
        }
        for (int i = 0; i < goData.length; i++) {
            goData[i] = r.nextFloat() - 0.5f;
        }

        float[] gX = new float[m * in];
        float[] gW = new float[in * out];
        float[] gB = new float[out];
        cpuBackward(xData, wData, goData, m, in, out, gX, gW, gB);

        Tensor xCpu = Tensor.fromArray(xData, new int[]{m, in});
        Tensor wCpu = Tensor.fromArray(wData, new int[]{in, out});
        Tensor bCpu = Tensor.fromArray(bData, new int[]{out});
        Tensor goCpu = Tensor.fromArray(goData, new int[]{m, out});

        try (GpuTensor wG = GpuTensor.fromHostTensor(wCpu);
                GpuTensor bG = GpuTensor.fromHostTensor(bCpu);
                GpuTensor xG = GpuTensor.fromHostTensor(xCpu);
                GpuTensor goG = GpuTensor.fromHostTensor(goCpu);
                Linear layer = new Linear(wG, bG)) {
            GpuTensor gIn = layer.backwardGpu(xG, goG);
            Tensor gInHost = gIn.toHostTensor();
            float[] gwHost = new float[in * out];
            wG.gradBuffer().copyTo(gwHost, 0, gwHost.length);
            float[] gbHost = new float[out];
            bG.gradBuffer().copyTo(gbHost, 0, out);

            assertArrayEquals(gX, gInHost.internalBuffer(), 2e-3f);
            assertArrayEquals(gW, gwHost, 2e-3f);
            assertArrayEquals(gB, gbHost, 2e-3f);
        }
    }

    @Test
    void backward_accumulate_sumsMicrobatchGrads() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int m1 = 3;
        int m2 = 2;
        int in = 4;
        int out = 3;
        Random r = new Random(11);
        float[] x1 = new float[m1 * in];
        float[] x2 = new float[m2 * in];
        float[] wData = new float[in * out];
        float[] bData = new float[out];
        float[] go1 = new float[m1 * out];
        float[] go2 = new float[m2 * out];
        for (int i = 0; i < x1.length; i++) {
            x1[i] = r.nextFloat() - 0.5f;
        }
        for (int i = 0; i < x2.length; i++) {
            x2[i] = r.nextFloat() - 0.5f;
        }
        for (int i = 0; i < wData.length; i++) {
            wData[i] = r.nextFloat() * 0.4f - 0.2f;
        }
        for (int i = 0; i < bData.length; i++) {
            bData[i] = r.nextFloat() * 0.1f;
        }
        for (int i = 0; i < go1.length; i++) {
            go1[i] = r.nextFloat() - 0.5f;
        }
        for (int i = 0; i < go2.length; i++) {
            go2[i] = r.nextFloat() - 0.5f;
        }

        float[] gW1 = new float[in * out];
        float[] gW2 = new float[in * out];
        float[] gB1 = new float[out];
        float[] gB2 = new float[out];
        float[] gx1 = new float[m1 * in];
        float[] gx2 = new float[m2 * in];
        cpuBackward(x1, wData, go1, m1, in, out, gx1, gW1, gB1);
        cpuBackward(x2, wData, go2, m2, in, out, gx2, gW2, gB2);
        float[] gWRef = new float[in * out];
        float[] gBRef = new float[out];
        for (int i = 0; i < gWRef.length; i++) {
            gWRef[i] = gW1[i] + gW2[i];
        }
        for (int j = 0; j < out; j++) {
            gBRef[j] = gB1[j] + gB2[j];
        }

        Tensor wCpu = Tensor.fromArray(wData, new int[] {in, out});
        Tensor bCpu = Tensor.fromArray(bData, new int[] {out});
        Tensor x1t = Tensor.fromArray(x1, new int[] {m1, in});
        Tensor x2t = Tensor.fromArray(x2, new int[] {m2, in});
        Tensor go1t = Tensor.fromArray(go1, new int[] {m1, out});
        Tensor go2t = Tensor.fromArray(go2, new int[] {m2, out});

        try (GpuTensor wG = GpuTensor.fromHostTensor(wCpu);
                GpuTensor bG = GpuTensor.fromHostTensor(bCpu);
                GpuTensor x1g = GpuTensor.fromHostTensor(x1t);
                GpuTensor x2g = GpuTensor.fromHostTensor(x2t);
                GpuTensor go1g = GpuTensor.fromHostTensor(go1t);
                GpuTensor go2g = GpuTensor.fromHostTensor(go2t);
                Linear layer = new Linear(wG, bG)) {
            layer.backwardGpu(x1g, go1g, false);
            layer.backwardGpu(x2g, go2g, true);
            float[] gwHost = new float[in * out];
            wG.gradBuffer().copyTo(gwHost, 0, gwHost.length);
            float[] gbHost = new float[out];
            bG.gradBuffer().copyTo(gbHost, 0, out);
            assertArrayEquals(gWRef, gwHost, 2e-3f);
            assertArrayEquals(gBRef, gbHost, 2e-3f);
        }
    }

    @Test
    void dimensionsEnforced() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        try (GpuTensor w = GpuTensor.allocate(new int[]{3, 2});
                GpuTensor b = GpuTensor.allocate(new int[]{2});
                GpuTensor xBad = GpuTensor.allocate(new int[]{4, 2});
                Linear layer = new Linear(w, b)) {
            assertThrows(IllegalArgumentException.class, () -> layer.forwardGpu(xBad));
        }
    }
}
