package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;

/**
 * Копии на GPU весов финального RMSNorm (γ) и LM head. Синхронизируются с CPU через {@link #syncFromHost()}.
 */
final class GptGpuWeights implements AutoCloseable {

    private boolean closed;

    private final GpuTensor lmHead;
    private final GpuTensor layerNormGamma;
    private final Tensor cpuLmHead;
    private final Tensor cpuGamma;

    private GptGpuWeights(GpuTensor lmHead, GpuTensor gamma, Tensor cpuLmHead, Tensor cpuGamma) {
        this.lmHead = lmHead;
        this.layerNormGamma = gamma;
        this.cpuLmHead = cpuLmHead;
        this.cpuGamma = cpuGamma;
    }

    static GptGpuWeights upload(Tensor lmHead, Tensor layerNormGamma) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        GpuTensor gLm = GpuTensor.fromHostTensor(lmHead);
        GpuTensor gGamma = GpuTensor.fromHostTensor(layerNormGamma);
        return new GptGpuWeights(gLm, gGamma, lmHead, layerNormGamma);
    }

    GpuTensor lmHeadGpu() {
        return lmHead;
    }

    GpuTensor layerNormGammaGpu() {
        return layerNormGamma;
    }

    void collectMapping(Tensor cpuLm, Tensor cpuG, java.util.Map<Tensor, GpuTensor> map) {
        map.put(cpuLm, lmHead);
        map.put(cpuG, layerNormGamma);
    }

    void syncFromHost() {
        if (closed) {
            return;
        }
        lmHead.uploadFrom(cpuLmHead.internalBuffer(), 0, cpuLmHead.size());
        layerNormGamma.uploadFrom(cpuGamma.internalBuffer(), 0, cpuGamma.size());
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        lmHead.close();
        layerNormGamma.close();
    }
}
