package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.ops.TensorOps;
import org.junit.jupiter.api.Test;

/**
 * Один CUDA graph на слой (MHA+FFN): сравнение {@link GPTModel#forwardGpuDecoderInfer} с graph и без.
 */
class DecoderLayerCudaGraphTest {

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    @Test
    void inferDecoder_graphMatchesEager() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        if (System.getenv("JGPT_DECODER_GPU_PIPELINE") != null
                || System.getenv("JGPT_DECODER_LAYER_CUDA_GRAPH") != null) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        String prevGraph = System.getProperty("jgpt.decoder.layer.cudaGraph");
        try {
            int vocab = 48;
            int maxSeq = 8;
            int dModel = 24;
            int heads = 4;
            int layers = 2;
            int dFf = 48;
            int batch = 1;
            int seqLen = 4;
            int plane = batch * seqLen * dModel;

            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            System.setProperty("jgpt.decoder.layer.cudaGraph", "false");
            GPTModel eager = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);

            System.setProperty("jgpt.decoder.layer.cudaGraph", "true");
            GPTModel graphed = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);

            var ep = eager.getParameters();
            var gp = graphed.getParameters();
            for (int i = 0; i < ep.size(); i++) {
                System.arraycopy(ep.get(i).internalBuffer(), 0, gp.get(i).internalBuffer(), 0, ep.get(i).size());
            }
            graphed.syncGpuResidentWeightsFromHost();
            eager.syncGpuResidentWeightsFromHost();

            assertTrue(graphed.isDecoderLayerCudaGraphRequested());
            assertTrue(graphed.isDecoderLayerCudaGraphActive());

            float[] hx = new float[plane];
            for (int i = 0; i < hx.length; i++) {
                hx[i] = (float) Math.sin(i * 0.09) * 0.2f;
            }
            try (GpuFloatBuffer xE = GpuFloatBuffer.allocate(plane);
                    GpuFloatBuffer xG = GpuFloatBuffer.allocate(plane)) {
                xE.copyFrom(hx, 0, plane);
                xG.copyFrom(hx, 0, plane);

                GpuFloatBuffer oE = eager.forwardGpuDecoderInfer(xE, null, batch, seqLen);
                GpuFloatBuffer oG1 = graphed.forwardGpuDecoderInfer(xG, null, batch, seqLen);
                GpuFloatBuffer oG2 = graphed.forwardGpuDecoderInfer(xG, null, batch, seqLen);

                float[] bE = new float[plane];
                float[] b1 = new float[plane];
                float[] b2 = new float[plane];
                oE.copyTo(bE, 0, plane);
                oG1.copyTo(b1, 0, plane);
                oG2.copyTo(b2, 0, plane);

                float tol = 2e-3f;
                assertTrue(maxAbsDiff(bE, b1) < tol, "eager vs graph[1] " + maxAbsDiff(bE, b1));
                assertTrue(maxAbsDiff(b1, b2) < 1e-5f, "graph replay " + maxAbsDiff(b1, b2));
            }

            graphed.closeGpuResidentWeights();
            eager.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            if (prevGraph == null) {
                System.clearProperty("jgpt.decoder.layer.cudaGraph");
            } else {
                System.setProperty("jgpt.decoder.layer.cudaGraph", prevGraph);
            }
        }
    }

    /**
     * С маской H2D с хоста нельзя вызывать внутри cuda graph capture; проверяем стабильность двух replay подряд.
     */
    @Test
    void inferDecoder_graphReplayStable_withCausalMask() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        if (System.getenv("JGPT_DECODER_GPU_PIPELINE") != null
                || System.getenv("JGPT_DECODER_LAYER_CUDA_GRAPH") != null) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        String prevGraph = System.getProperty("jgpt.decoder.layer.cudaGraph");
        try {
            int vocab = 48;
            int maxSeq = 8;
            int dModel = 24;
            int heads = 4;
            int layers = 2;
            int dFf = 48;
            int batch = 1;
            int seqLen = 4;
            int plane = batch * seqLen * dModel;
            Tensor mask = TensorOps.createCausalMask(seqLen);

            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            System.setProperty("jgpt.decoder.layer.cudaGraph", "true");
            GPTModel graphed = new GPTModel(vocab, maxSeq, dModel, heads, layers, dFf, true);
            graphed.syncGpuResidentWeightsFromHost();

            float[] hx = new float[plane];
            for (int i = 0; i < hx.length; i++) {
                hx[i] = (float) Math.sin(i * 0.11) * 0.15f;
            }
            try (GpuFloatBuffer xG = GpuFloatBuffer.allocate(plane)) {
                xG.copyFrom(hx, 0, plane);

                GpuFloatBuffer oG1 = graphed.forwardGpuDecoderInfer(xG, mask, batch, seqLen);
                GpuFloatBuffer oG2 = graphed.forwardGpuDecoderInfer(xG, mask, batch, seqLen);

                float[] b1 = new float[plane];
                float[] b2 = new float[plane];
                oG1.copyTo(b1, 0, plane);
                oG2.copyTo(b2, 0, plane);
                assertTrue(maxAbsDiff(b1, b2) < 1e-5f, "graph replay with mask " + maxAbsDiff(b1, b2));
            }

            graphed.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
            if (prevGraph == null) {
                System.clearProperty("jgpt.decoder.layer.cudaGraph");
            } else {
                System.setProperty("jgpt.decoder.layer.cudaGraph", prevGraph);
            }
        }
    }
}
