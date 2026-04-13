package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.Arrays;
import java.util.List;

/**
 * Позиционные эмбеддинги; используется только из {@link GPTModel}.
 */
final class PositionEmbedding {
    private final Tensor weights;
    /** Копия {@link #weights} на VRAM при GPU-резидентной модели; backward scatter ∂ без alloc всей таблицы. */
    private final GpuTensor weightsGpu;

    PositionEmbedding(int maxSeqLen, int dModel, float scale, boolean gpuResidentEmbedding) {
        this.weights = TensorOps.randomTensor(new int[]{maxSeqLen, dModel}, scale);
        if (gpuResidentEmbedding) {
            this.weightsGpu = GpuTensor.fromHostTensor(weights);
        } else {
            this.weightsGpu = null;
        }
    }

    void syncGpuWeightsFromHost() {
        if (weightsGpu != null) {
            weightsGpu.uploadFrom(weights.internalBuffer(), 0, weights.size());
        }
    }

    void closeGpuWeights() {
        if (weightsGpu != null) {
            weightsGpu.close();
        }
    }

    boolean hasWeightsGpu() {
        return weightsGpu != null;
    }

    Tensor hostWeights() {
        return weights;
    }

    GpuTensor deviceWeightsOrNull() {
        return weightsGpu;
    }

    GpuFloatBuffer positionWeightsDataBuffer() {
        if (weightsGpu == null) {
            throw new IllegalStateException("weightsGpu is null");
        }
        return weightsGpu.dataBuffer();
    }

    /**
     * In-place: {@code x[b,s,:] += table[posRowStart+s,:]}. Таблица позиций — {@code [maxSeq, dModel]} на host или
     * VRAM.
     */
    void addToActivationsInPlace(Tensor x, int seqLen, int posRowStart) {
        int[] sh = x.getShape();
        if (sh.length != 3 || sh[1] != seqLen) {
            throw new IllegalArgumentException(
                    "x must be [batch, seqLen=" + seqLen + ", d_model], got " + Arrays.toString(sh));
        }
        int batch = sh[0];
        int dm = sh[2];
        int maxPos = weights.getShape()[0];
        int wDm = weights.getShape()[1];
        if (dm != wDm) {
            throw new IllegalArgumentException("d_model mismatch: x has " + dm + ", table has " + wDm);
        }
        if (posRowStart < 0 || posRowStart + seqLen > maxPos) {
            throw new IllegalArgumentException(
                    "position rows ["
                            + posRowStart
                            + ", "
                            + (posRowStart + seqLen)
                            + ") out of table rows [0,"
                            + maxPos
                            + ")");
        }
        int n = batch * seqLen * dm;
        if (n == 0) {
            return;
        }
        TensorOpsGPU.requireCuda("PositionEmbedding.addToActivationsInPlace");
        float[] xb = x.internalBuffer();
        if (weightsGpu != null) {
            TensorOpsGPU.addPositionEmbeddingGPUDeviceWeights(
                    xb, weightsGpu.devicePointer(), batch, seqLen, dm, posRowStart);
        } else {
            float[] slice = new float[seqLen * dm];
            float[] w = weights.internalBuffer();
            int rowStride = weights.stridesInternal()[0];
            for (int s = 0; s < seqLen; s++) {
                System.arraycopy(w, (posRowStart + s) * rowStride, slice, s * dm, dm);
            }
            TensorOpsGPU.addPositionEmbeddingInPlaceHostSlice(xb, slice, batch, seqLen, dm);
        }
    }

    void collectParameters(List<Tensor> out) {
        out.add(weights);
    }

    /** Сумма градиентов по батчу для позиций 0..seqLen-1. */
    void backwardAccumulate(Tensor gradCombined, int seqLen) {
        int dm = weights.getShape()[1];
        int[] gs = gradCombined.getShape();
        int batch = gs[0];
        int wPlane = seqLen * dm;
        if (!weights.hasGrad()) {
            weights.zeroGrad();
        }
        float[] gw = weights.gradBuffer();
        float[] g = gradCombined.gradBuffer();
        int n = batch * seqLen * dm;
        if (n <= 0) {
            return;
        }
        if (weightsGpu != null) {
            TensorOpsGPU.requireCuda("PositionEmbedding.backwardAccumulate");
            if (!weightsGpu.hasGradBuffer()) {
                weightsGpu.zeroGrad();
            }
            weightsGpu.gradBuffer().copyFrom(gw, 0, wPlane);
            TensorOpsGPU.embeddingPositionBackwardGPUDeviceGradWeights(
                    g, batch, seqLen, dm, weightsGpu.gradDevicePointer());
            weightsGpu.gradBuffer().copyTo(gw, 0, wPlane);
            weightsGpu.zeroGrad();
            return;
        }
        TensorOpsGPU.requireCuda("PositionEmbedding.backwardAccumulate");
        TensorOpsGPU.embeddingPositionBackwardGPU(g, gw, batch, seqLen, dm);
    }

    /** Как {@link #backwardAccumulate}, но градиент объединённого входа уже на device. */
    void backwardAccumulateFromDeviceGrad(GpuFloatBuffer gradDevice, int batch, int seqLen) {
        int dm = weights.getShape()[1];
        if (!weights.hasGrad()) {
            weights.zeroGrad();
        }
        if (weightsGpu == null) {
            throw new IllegalStateException(
                    "backwardAccumulateFromDeviceGrad requires GPU-resident position embedding");
        }
        TensorOpsGPU.requireCuda("PositionEmbedding.backwardAccumulateFromDeviceGrad");
        if (!weightsGpu.hasGradBuffer()) {
            weightsGpu.zeroGrad();
        }
        /* ∂ позиций 0..seqLen-1 накопление в VRAM; буфер размера maxSeq×d строки s≥seqLen не трогаются ядром. */
        TensorOpsGPU.embeddingPositionBackwardGPUDeviceGradWeightsDeviceGrad(
                gradDevice.devicePointer(), batch, seqLen, dm, weightsGpu.gradDevicePointer());
    }

    Tensor forward(int seqLen) {
        int dModel = weights.getShape()[1];
        Tensor out = new Tensor(new int[]{1, seqLen, dModel});
        float[] w = weights.internalBuffer();
        float[] o = out.internalBuffer();
        for (int s = 0; s < seqLen; s++) {
            System.arraycopy(w, s * dModel, o, s * dModel, dModel);
        }
        return out;
    }

    /** Одна строка таблицы позиций: {@code [1, 1, d_model]}. */
    Tensor forwardOne(int position) {
        int maxPos = weights.getShape()[0];
        int dModel = weights.getShape()[1];
        if (position < 0 || position >= maxPos) {
            throw new IllegalArgumentException("position " + position + " out of range [0," + maxPos + ")");
        }
        Tensor out = new Tensor(new int[]{1, 1, dModel});
        float[] w = weights.internalBuffer();
        System.arraycopy(w, position * dModel, out.internalBuffer(), 0, dModel);
        return out;
    }

    /** Строки {@code [start, start+len)} → {@code [1, len, d_model]}. */
    Tensor forwardRange(int start, int len) {
        int maxPos = weights.getShape()[0];
        int dModel = weights.getShape()[1];
        if (start < 0 || len < 0 || start + len > maxPos) {
            throw new IllegalArgumentException(
                    "range [" + start + ", " + (start + len) + ") out of [0," + maxPos + ")");
        }
        Tensor out = new Tensor(new int[]{1, len, dModel});
        float[] w = weights.internalBuffer();
        float[] o = out.internalBuffer();
        for (int s = 0; s < len; s++) {
            System.arraycopy(w, (start + s) * dModel, o, s * dModel, dModel);
        }
        return out;
    }
}
