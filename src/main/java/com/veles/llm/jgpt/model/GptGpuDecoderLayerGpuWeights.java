package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.ops.TensorOps;

/**
 * VRAM-копии весов одного декодер-блока: pre-norm + Q/K/V/O и FFN (SwiGLU + второй RMSNorm γ).
 * Синхронизация с CPU — {@link #syncFromHost()}.
 */
final class GptGpuDecoderLayerGpuWeights implements AutoCloseable {

    private boolean closed;

    private final GpuTensor norm1Gamma;
    private final GpuTensor wq;
    private final GpuTensor wk;
    private final GpuTensor wv;
    private final GpuTensor wo;

    private final GpuTensor norm2Gamma;
    private final GpuTensor w1;
    private final GpuTensor w2;
    private final GpuTensor w3;

    private final Tensor cpuNorm1;
    private final Tensor cpuWq;
    private final Tensor cpuWk;
    private final Tensor cpuWv;
    private final Tensor cpuWo;
    private final Tensor cpuNorm2;
    private final Tensor cpuW1;
    private final Tensor cpuW2;
    private final Tensor cpuW3;

    private GptGpuDecoderLayerGpuWeights(
            GpuTensor norm1Gamma,
            GpuTensor wq,
            GpuTensor wk,
            GpuTensor wv,
            GpuTensor wo,
            GpuTensor norm2Gamma,
            GpuTensor w1,
            GpuTensor w2,
            GpuTensor w3,
            Tensor cpuNorm1,
            Tensor cpuWq,
            Tensor cpuWk,
            Tensor cpuWv,
            Tensor cpuWo,
            Tensor cpuNorm2,
            Tensor cpuW1,
            Tensor cpuW2,
            Tensor cpuW3) {
        this.norm1Gamma = norm1Gamma;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.norm2Gamma = norm2Gamma;
        this.w1 = w1;
        this.w2 = w2;
        this.w3 = w3;
        this.cpuNorm1 = cpuNorm1;
        this.cpuWq = cpuWq;
        this.cpuWk = cpuWk;
        this.cpuWv = cpuWv;
        this.cpuWo = cpuWo;
        this.cpuNorm2 = cpuNorm2;
        this.cpuW1 = cpuW1;
        this.cpuW2 = cpuW2;
        this.cpuW3 = cpuW3;
    }

    static GptGpuDecoderLayerGpuWeights upload(
            Tensor norm1,
            Tensor wq,
            Tensor wk,
            Tensor wv,
            Tensor wo,
            Tensor norm2,
            Tensor w1,
            Tensor w2,
            Tensor w3) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        return new GptGpuDecoderLayerGpuWeights(
                GpuTensor.fromHostTensor(norm1),
                GpuTensor.fromHostTensor(wq),
                GpuTensor.fromHostTensor(wk),
                GpuTensor.fromHostTensor(wv),
                GpuTensor.fromHostTensor(wo),
                GpuTensor.fromHostTensor(norm2),
                GpuTensor.fromHostTensor(w1),
                GpuTensor.fromHostTensor(w2),
                GpuTensor.fromHostTensor(w3),
                norm1,
                wq,
                wk,
                wv,
                wo,
                norm2,
                w1,
                w2,
                w3);
    }

    TensorOps.GpuFfnResidentBuffers ffnBuffers() {
        return new TensorOps.GpuFfnResidentBuffers(
                norm2Gamma.dataBuffer(), w1.dataBuffer(), w2.dataBuffer(), w3.dataBuffer());
    }

    TensorOps.GpuAttnResidentBuffers attnBuffers() {
        return new TensorOps.GpuAttnResidentBuffers(
                norm1Gamma.dataBuffer(),
                wq.dataBuffer(),
                wk.dataBuffer(),
                wv.dataBuffer(),
                wo.dataBuffer());
    }

    /**
     * Добавляет пары (CPU {@link Tensor} → GPU {@link GpuTensor}) в переданную карту.
     *
     * @param map изменяемая карта; не передавать immutable-view (будет {@link
     *     UnsupportedOperationException}).
     */
    void collectMapping(java.util.Map<Tensor, GpuTensor> map) {
        map.put(cpuNorm1, norm1Gamma);
        map.put(cpuWq, wq);
        map.put(cpuWk, wk);
        map.put(cpuWv, wv);
        map.put(cpuWo, wo);
        map.put(cpuNorm2, norm2Gamma);
        map.put(cpuW1, w1);
        map.put(cpuW2, w2);
        map.put(cpuW3, w3);
    }

    void syncFromHost() {
        if (closed) {
            return;
        }
        norm1Gamma.uploadFrom(cpuNorm1.internalBuffer(), 0, cpuNorm1.size());
        wq.uploadFrom(cpuWq.internalBuffer(), 0, cpuWq.size());
        wk.uploadFrom(cpuWk.internalBuffer(), 0, cpuWk.size());
        wv.uploadFrom(cpuWv.internalBuffer(), 0, cpuWv.size());
        wo.uploadFrom(cpuWo.internalBuffer(), 0, cpuWo.size());
        norm2Gamma.uploadFrom(cpuNorm2.internalBuffer(), 0, cpuNorm2.size());
        w1.uploadFrom(cpuW1.internalBuffer(), 0, cpuW1.size());
        w2.uploadFrom(cpuW2.internalBuffer(), 0, cpuW2.size());
        w3.uploadFrom(cpuW3.internalBuffer(), 0, cpuW3.size());
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        norm1Gamma.close();
        wq.close();
        wk.close();
        wv.close();
        wo.close();
        norm2Gamma.close();
        w1.close();
        w2.close();
        w3.close();
    }
}
