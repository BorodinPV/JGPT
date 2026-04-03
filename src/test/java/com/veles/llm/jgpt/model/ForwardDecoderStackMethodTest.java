package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import org.junit.jupiter.api.Test;

/** Регрессия после выноса {@link GPTModel#forwardDecoderStack}. */
class ForwardDecoderStackMethodTest {

    @Test
    void forward_trainingFalse_logitsShape_gpuResident() {
        assumeTrue(TensorOpsGPU.isGpuAvailable(), "CUDA required for resident decoder path in this test");
        String prop = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            // Изолировать от других тестов, выставляющих pipeline=true на малой геометрии
            System.setProperty("jgpt.decoder.gpu.pipeline", "false");
            GPTModel m = new GPTModel(64, 16, 32, 4, 1, 64, true);
            try {
                Tensor in = new Tensor(new int[] {2, 2});
                float[] id = in.internalBuffer();
                for (int i = 0; i < id.length; i++) {
                    id[i] = (i * 7 + 3) % 64;
                }
                Tensor logits = m.forward(in, false, false);
                assertEquals(2, logits.getShape()[0]);
                assertEquals(2, logits.getShape()[1]);
                assertEquals(64, logits.getShape()[2]);
            } finally {
                m.closeGpuResidentWeights();
            }
        } finally {
            if (prop == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prop);
            }
        }
    }
}
