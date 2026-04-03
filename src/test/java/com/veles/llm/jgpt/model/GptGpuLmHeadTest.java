package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;
import org.junit.jupiter.api.Test;

class GptGpuLmHeadTest {

    @Test
    void forwardGpuLmHead_matchesCpuPipeline() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int vocab = 17;
        int maxSeq = 8;
        int dModel = 16;
        int heads = 4;
        int layers = 1;
        int dFf = 32;
        GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);
        assertTrue(model.isGpuResident());

        int batch = 1;
        int seq = 3;
        Tensor x = new Tensor(new int[]{batch, seq, dModel});
        float[] buf = x.internalBuffer();
        for (int i = 0; i < buf.length; i++) {
            buf[i] = (float) Math.sin(i * 0.13) * 0.7f;
        }

        Tensor xNorm = TensorOps.rmsNorm(x, model.getLayerNormFinal(), 1e-6f);
        Tensor hiddenFlat = Tensor.wrap(xNorm.internalBuffer(), new int[]{batch * seq, dModel});
        Tensor logitsRef = TensorOps.matmul(hiddenFlat, model.getLmHead());
        Tensor logitsRef3d = Tensor.wrap(logitsRef.internalBuffer(), new int[]{batch, seq, vocab});

        Tensor logitsGpu = model.forwardGpuLmHead(x);

        float[] a = logitsRef3d.internalBuffer();
        float[] b = logitsGpu.internalBuffer();
        float maxAbs = 0f;
        for (int i = 0; i < a.length; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
        }
        assertTrue(maxAbs < 5e-2f, "max abs diff " + maxAbs);
        model.closeGpuResidentWeights();
    }

    @Test
    void forwardGpuLmHead_requiresResident() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        GPTModel model = new GPTModel(8, 4, 8, 2, 1, 16, false);
        Tensor x = new Tensor(new int[]{1, 1, 8});
        assertThrows(IllegalStateException.class, () -> model.forwardGpuLmHead(x));
    }

    @Test
    void forwardThreeArg_training_gpuHead_backwardRuns() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            int vocab = 32;
            int maxSeq = 4;
            int dModel = 16;
            int heads = 4;
            int layers = 1;
            int dFf = 32;
            int batch = 1;
            int seq = 3;
            GPTModel model = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);
            model.setDeviceLogitsEnabled(true);
            model.setDeviceDecoderBackward(true);
            Tensor input = new Tensor(new int[]{batch, seq});
            float[] tok = input.internalBuffer();
            for (int i = 0; i < tok.length; i++) {
                tok[i] = i % vocab;
            }
            Tensor logits = model.forward(input, true, true);
            logits.zeroGrad();
            float[] gb = logits.gradBuffer();
            Arrays.fill(gb, 1.0e-4f / gb.length);
            model.backward(logits);
            model.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
        }
    }
}
