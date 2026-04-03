package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class AdamGpuMomentsCheckpointTest {

    @Test
    void momentsCheckpointRoundTrip_withoutPreSyncedHostMoments() throws IOException {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        Tensor p = new Tensor(new int[] {8});
        float[] pb = p.internalBuffer();
        for (int i = 0; i < pb.length; i++) {
            pb[i] = 0.05f * (i + 1);
        }
        List<Tensor> params = List.of(p);
        Map<Tensor, GpuTensor> paramMap = new IdentityHashMap<>();
        AdamOptimizer opt = AdamOptimizer.forTesting();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        try (GpuTensor g = GpuTensor.fromHostTensor(p)) {
            paramMap.put(p, g);
            float[] grad = new float[8];
            for (int step = 0; step < 4; step++) {
                opt.beginStep();
                for (int i = 0; i < grad.length; i++) {
                    grad[i] = 0.01f * (step + 1) * (i % 3 - 1);
                }
                g.zeroGrad();
                g.gradBuffer().copyFrom(grad, 0, grad.length);
                opt.stepAllGpuDevice(paramMap);
            }
            opt.writeMomentBuffers(new DataOutputStream(bos), params);
        }

        AdamOptimizer opt2 = AdamOptimizer.forTesting();
        opt2.setStep(opt.getStep());
        opt2.readMomentBuffers(new DataInputStream(new ByteArrayInputStream(bos.toByteArray())), params);

        ByteArrayOutputStream bos2 = new ByteArrayOutputStream();
        opt2.writeMomentBuffers(new DataOutputStream(bos2), params);

        assertArrayEquals(bos.toByteArray(), bos2.toByteArray());
    }
}
