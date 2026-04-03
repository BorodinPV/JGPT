package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Сравнение logits: обычный CPU-стек декодера и сквозной GPU-пайплайн (уровень C). Требуется CUDA и {@code
 * gpuResident}.
 */
class DecoderGpuPipelineParityTest {

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
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

    @Test
    void forward_logitsMatchCpuDecoderVsGpuPipeline() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        if (System.getenv("JGPT_DECODER_GPU_PIPELINE") != null) {
            return;
        }
        String prop = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            int vocab = 64;
            int maxSeq = 16;
            int dModel = 32;
            int heads = 4;
            int layers = 2;
            int dFf = 64;
            int batch = 2;
            int seqLen = 2;

            System.setProperty("jgpt.decoder.gpu.pipeline", "false");
            GPTModel cpuDecoder = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);
            assertTrue(cpuDecoder.isGpuResident());
            assertTrue(!cpuDecoder.isDecoderGpuPipeline());

            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel gpuPipeline = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);
            assertTrue(gpuPipeline.isDecoderGpuPipeline());

            copyAllParameters(cpuDecoder, gpuPipeline);

            Tensor input = new Tensor(new int[] {batch, seqLen});
            float[] id = input.internalBuffer();
            for (int i = 0; i < id.length; i++) {
                id[i] = (i * 7 + 3) % vocab;
            }

            Tensor logitsRef = cpuDecoder.forward(input, false, true);
            Tensor logitsPipe = gpuPipeline.forward(input, false, true);

            float d = maxAbsDiff(logitsRef.internalBuffer(), logitsPipe.internalBuffer());
            assertTrue(d < 8e-3f, "CPU decoder vs GPU pipeline max abs diff " + d);

            gpuPipeline.closeGpuResidentWeights();
            cpuDecoder.closeGpuResidentWeights();
        } finally {
            if (prop == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prop);
            }
        }
    }
}
