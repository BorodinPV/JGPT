package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.training.AdamOptimizer;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class AdamGpuTensorStepTest {

    @Test
    void stepGpu_matchesCpuAdam() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 48;
        float[] p0 = new float[n];
        float[] g0 = new float[n];
        for (int i = 0; i < n; i++) {
            p0[i] = (float) Math.sin(i * 0.17) * 0.5f;
            g0[i] = (float) Math.cos(i * 0.13) * 0.3f;
        }

        Tensor pCpu = Tensor.fromArray(p0, new int[]{n});
        pCpu.zeroGrad();
        System.arraycopy(g0, 0, pCpu.gradBuffer(), 0, n);

        AdamOptimizer optCpu = AdamOptimizer.forTesting();
        optCpu.beginStep();
        optCpu.stepWithParamGrad(pCpu);

        GpuTensor pGpu = GpuTensor.allocate(new int[]{n});
        pGpu.uploadFrom(p0, 0, n);
        pGpu.zeroGrad();
        pGpu.gradBuffer().copyFrom(g0, 0, n);
        GpuTensor mGpu = GpuTensor.allocate(new int[]{n});
        GpuTensor vGpu = GpuTensor.allocate(new int[]{n});

        AdamOptimizer optGpu = AdamOptimizer.forTesting();
        optGpu.beginStep();
        optGpu.stepGpu(pGpu, mGpu, vGpu);

        float[] fromGpu = new float[n];
        pGpu.downloadTo(fromGpu, 0, n);
        assertArrayEquals(pCpu.internalBuffer(), fromGpu, 1e-5f);
    }

    @Test
    void stepAllGpu_twoTensors_matchesSequentialCpu() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n1 = 8;
        int n2 = 12;
        Tensor a = Tensor.fromArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}, new int[]{n1});
        Tensor b = new Tensor(new int[]{n2});
        Arrays.fill(b.internalBuffer(), 1f);
        a.zeroGrad();
        b.zeroGrad();
        for (int i = 0; i < n1; i++) {
            a.gradBuffer()[i] = 0.1f * (i + 1);
        }
        for (int i = 0; i < n2; i++) {
            b.gradBuffer()[i] = -0.05f * i;
        }

        Tensor aRef = Tensor.fromArray(a.internalBuffer().clone(), new int[]{n1});
        Tensor bRef = Tensor.fromArray(b.internalBuffer().clone(), new int[]{n2});
        aRef.zeroGrad();
        bRef.zeroGrad();
        System.arraycopy(a.gradBuffer(), 0, aRef.gradBuffer(), 0, n1);
        System.arraycopy(b.gradBuffer(), 0, bRef.gradBuffer(), 0, n2);

        AdamOptimizer optCpu = AdamOptimizer.forTesting();
        optCpu.beginStep();
        optCpu.stepWithParamGrad(aRef);
        optCpu.stepWithParamGrad(bRef);

        GpuTensor aG = GpuTensor.allocate(new int[]{n1});
        GpuTensor bG = GpuTensor.allocate(new int[]{n2});
        aG.uploadFrom(a.internalBuffer(), 0, n1);
        bG.uploadFrom(b.internalBuffer(), 0, n2);
        aG.zeroGrad();
        bG.zeroGrad();
        aG.gradBuffer().copyFrom(a.gradBuffer(), 0, n1);
        bG.gradBuffer().copyFrom(b.gradBuffer(), 0, n2);
        GpuTensor m1 = GpuTensor.allocate(new int[]{n1});
        GpuTensor v1 = GpuTensor.allocate(new int[]{n1});
        GpuTensor m2 = GpuTensor.allocate(new int[]{n2});
        GpuTensor v2 = GpuTensor.allocate(new int[]{n2});

        AdamOptimizer optGpu = AdamOptimizer.forTesting();
        optGpu.beginStep();
        optGpu.stepAllGpu(List.of(aG, bG), List.of(m1, m2), List.of(v1, v2));

        float[] aOut = new float[n1];
        float[] bOut = new float[n2];
        aG.downloadTo(aOut, 0, n1);
        bG.downloadTo(bOut, 0, n2);
        assertArrayEquals(aRef.internalBuffer(), aOut, 1e-5f);
        assertArrayEquals(bRef.internalBuffer(), bOut, 1e-5f);
    }

    @Test
    void stepAllGpu_empty_noop() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        AdamOptimizer opt = AdamOptimizer.forTesting();
        opt.beginStep();
        opt.stepAllGpu(List.of(), List.of(), List.of());
        assertTrue(true);
    }

    @Test
    void stepAllGpu_threeTensors_matchesSequentialCpu() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n1 = 4;
        int n2 = 7;
        int n3 = 5;
        Tensor a = new Tensor(new int[]{n1});
        Tensor b = new Tensor(new int[]{n2});
        Tensor c = new Tensor(new int[]{n3});
        for (int i = 0; i < n1; i++) {
            a.internalBuffer()[i] = 0.2f * i;
        }
        Arrays.fill(b.internalBuffer(), 0.5f);
        for (int i = 0; i < n3; i++) {
            c.internalBuffer()[i] = -0.1f * i;
        }
        a.zeroGrad();
        b.zeroGrad();
        c.zeroGrad();
        for (int i = 0; i < n1; i++) {
            a.gradBuffer()[i] = 0.15f * (i + 1);
        }
        for (int i = 0; i < n2; i++) {
            b.gradBuffer()[i] = 0.02f * i - 0.1f;
        }
        for (int i = 0; i < n3; i++) {
            c.gradBuffer()[i] = 0.3f;
        }

        Tensor ar = Tensor.fromArray(a.internalBuffer().clone(), new int[]{n1});
        Tensor br = Tensor.fromArray(b.internalBuffer().clone(), new int[]{n2});
        Tensor cr = Tensor.fromArray(c.internalBuffer().clone(), new int[]{n3});
        ar.zeroGrad();
        br.zeroGrad();
        cr.zeroGrad();
        System.arraycopy(a.gradBuffer(), 0, ar.gradBuffer(), 0, n1);
        System.arraycopy(b.gradBuffer(), 0, br.gradBuffer(), 0, n2);
        System.arraycopy(c.gradBuffer(), 0, cr.gradBuffer(), 0, n3);

        AdamOptimizer optCpu = AdamOptimizer.forTesting();
        optCpu.beginStep();
        optCpu.stepWithParamGrad(ar);
        optCpu.stepWithParamGrad(br);
        optCpu.stepWithParamGrad(cr);

        GpuTensor aG = GpuTensor.allocate(new int[]{n1});
        GpuTensor bG = GpuTensor.allocate(new int[]{n2});
        GpuTensor cG = GpuTensor.allocate(new int[]{n3});
        aG.uploadFrom(a.internalBuffer(), 0, n1);
        bG.uploadFrom(b.internalBuffer(), 0, n2);
        cG.uploadFrom(c.internalBuffer(), 0, n3);
        aG.zeroGrad();
        bG.zeroGrad();
        cG.zeroGrad();
        aG.gradBuffer().copyFrom(a.gradBuffer(), 0, n1);
        bG.gradBuffer().copyFrom(b.gradBuffer(), 0, n2);
        cG.gradBuffer().copyFrom(c.gradBuffer(), 0, n3);
        GpuTensor m1 = GpuTensor.allocate(new int[]{n1});
        GpuTensor v1 = GpuTensor.allocate(new int[]{n1});
        GpuTensor m2 = GpuTensor.allocate(new int[]{n2});
        GpuTensor v2 = GpuTensor.allocate(new int[]{n2});
        GpuTensor m3 = GpuTensor.allocate(new int[]{n3});
        GpuTensor v3 = GpuTensor.allocate(new int[]{n3});

        AdamOptimizer optGpu = AdamOptimizer.forTesting();
        optGpu.beginStep();
        optGpu.stepAllGpu(List.of(aG, bG, cG), List.of(m1, m2, m3), List.of(v1, v2, v3));

        float[] o1 = new float[n1];
        float[] o2 = new float[n2];
        float[] o3 = new float[n3];
        aG.downloadTo(o1, 0, n1);
        bG.downloadTo(o2, 0, n2);
        cG.downloadTo(o3, 0, n3);
        assertArrayEquals(ar.internalBuffer(), o1, 1e-5f);
        assertArrayEquals(br.internalBuffer(), o2, 1e-5f);
        assertArrayEquals(cr.internalBuffer(), o3, 1e-5f);
    }
}
