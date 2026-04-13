package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;

/**
 * Таблица эмбеддингов токенов; используется только из {@link GPTModel}.
 */
final class TokenEmbedding {
    private final Tensor weights;
    /** Копия {@link #weights} на VRAM при GPU-резидентной модели; gather без H2D весов на каждый forward. */
    private final GpuTensor weightsGpu;

    TokenEmbedding(int vocabSize, int dModel, float scale, boolean gpuResidentEmbedding) {
        this.weights = TensorOps.randomTensor(new int[]{vocabSize, dModel}, scale);
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

    /**
     * Gather в {@code out} на device; веса — {@link #weightsGpu}. Форма {@code out} — {@code [batch, seq, dModel]}.
     */
    void forwardGatherToGpuTensor(Tensor tokens, GpuTensor out) {
        if (weightsGpu == null) {
            throw new IllegalStateException("weightsGpu is null");
        }
        int[] shape = tokens.getShape();
        int batch = shape[0];
        int seqLen = shape[1];
        int dModel = weights.getShape()[1];
        int vocabSize = weights.getShape()[0];
        int[] os = out.getShape();
        if (os.length != 3 || os[0] != batch || os[1] != seqLen || os[2] != dModel) {
            throw new IllegalArgumentException(
                    "out must be [batch, seq, d_model], got " + Arrays.toString(os));
        }
        int nTok = batch * seqLen;
        for (int i = 0; i < nTok; i++) {
            int tokenIdx = (int) tokens.getLinear(i);
            if (tokenIdx < 0 || tokenIdx >= vocabSize) {
                throw new IllegalArgumentException("token index out of range: " + tokenIdx);
            }
        }
        ByteBuffer tokenByteBuf = tokens.directByteBuffer();
        if (tokenByteBuf != null && tokenByteBuf.isDirect()) {
            TensorOpsGPU.embeddingTokenForwardGpuDirectDeviceWeightsToDevice(
                    tokenByteBuf, weightsGpu.devicePointer(), out.dataBuffer(), batch, seqLen, dModel, vocabSize);
        } else {
            float[] tokenData = tokens.internalBuffer();
            TensorOpsGPU.embeddingTokenForwardGpuDeviceWeightsToDevice(
                    tokenData, weightsGpu.devicePointer(), out.dataBuffer(), batch, seqLen, dModel, vocabSize);
        }
    }

    void collectParameters(List<Tensor> out) {
        out.add(weights);
    }

    /** Градиент по таблице эмбеддингов: scatter по индексам токенов. */
    void backwardScatter(Tensor tokens, Tensor gradOut) {
        int[] shape = tokens.getShape();
        int batch = shape[0];
        int seqLen = shape[1];
        int dm = weights.getShape()[1];
        int vocab = weights.getShape()[0];
        int wSize = vocab * dm;
        if (!weights.hasGrad()) {
            weights.zeroGrad();
        }
        float[] gw = weights.gradBuffer();
        float[] tok = tokens.internalBuffer();
        float[] g = gradOut.gradBuffer();
        int n = batch * seqLen * dm;
        if (n <= 0) {
            return;
        }
        if (weightsGpu != null) {
            TensorOpsGPU.requireCuda("TokenEmbedding.backwardScatter");
            if (!weightsGpu.hasGradBuffer()) {
                weightsGpu.zeroGrad();
            }
            weightsGpu.gradBuffer().copyFrom(gw, 0, wSize);
            TensorOpsGPU.embeddingTokenBackwardGPUDeviceGradWeights(
                    tok, g, batch, seqLen, dm, vocab, weightsGpu.gradDevicePointer());
            weightsGpu.gradBuffer().copyTo(gw, 0, wSize);
            weightsGpu.zeroGrad();
            return;
        }
        TensorOpsGPU.requireCuda("TokenEmbedding.backwardScatter");
        TensorOpsGPU.embeddingTokenBackwardGPU(tok, g, gw, batch, seqLen, dm, vocab);
    }

    /**
     * Как {@link #backwardScatter}, но градиент на входе эмбеддинга уже на device (без D2H всего
     * {@code [B,S,d]}).
     */
    void backwardScatterFromDeviceGrad(Tensor tokens, GpuFloatBuffer gradDevice, int batch, int seqLen) {
        int dm = weights.getShape()[1];
        int vocab = weights.getShape()[0];
        if (!weights.hasGrad()) {
            weights.zeroGrad();
        }
        float[] tok = tokens.internalBuffer();
        if (weightsGpu == null) {
            throw new IllegalStateException(
                    "backwardScatterFromDeviceGrad requires GPU-resident token embedding tables");
        }
        TensorOpsGPU.requireCuda("TokenEmbedding.backwardScatterFromDeviceGrad");
        if (!weightsGpu.hasGradBuffer()) {
            weightsGpu.zeroGrad();
        }
        /* Накопление только в VRAM (atomicAdd в CUDA); хостовый буфер весов не обновляется. Между микробатчами ∂W
         * на GPU не обнуляем (полный GPU-шаг оптимизатора читает только grad на VRAM). */
        TensorOpsGPU.embeddingTokenBackwardGPUDeviceGradWeightsDeviceGrad(
                tok,
                gradDevice.devicePointer(),
                batch,
                seqLen,
                dm,
                vocab,
                weightsGpu.gradDevicePointer());
    }

    Tensor forward(Tensor tokens) {
        int[] shape = tokens.getShape();
        int batch = shape[0];
        int seqLen = shape[1];
        int dModel = weights.getShape()[1];
        int vocabSize = weights.getShape()[0];

        Tensor output = new Tensor(new int[]{batch, seqLen, dModel});
        float[] outData = output.internalBuffer();
        float[] weightData = weights.internalBuffer();

        int nTok = batch * seqLen;
        for (int i = 0; i < nTok; i++) {
            int tokenIdx = (int) tokens.getLinear(i);
            if (tokenIdx < 0 || tokenIdx >= vocabSize) {
                throw new IllegalArgumentException("token index out of range: " + tokenIdx);
            }
        }

        int n = batch * seqLen * dModel;
        if (n <= 0) {
            return output;
        }
        TensorOpsGPU.requireCuda("TokenEmbedding.forward");
        ByteBuffer tokenByteBuf = tokens.directByteBuffer();
        if (weightsGpu != null) {
            long dW = weightsGpu.devicePointer();
            if (tokenByteBuf != null && tokenByteBuf.isDirect()) {
                TensorOpsGPU.embeddingTokenForwardGpuDirectDeviceWeights(
                        tokenByteBuf, dW, outData, batch, seqLen, dModel, vocabSize);
                return output;
            }
            float[] tokenData = tokens.internalBuffer();
            TensorOpsGPU.embeddingTokenForwardGpuDeviceWeights(
                    tokenData, dW, outData, batch, seqLen, dModel, vocabSize);
            return output;
        }
        if (tokenByteBuf != null && tokenByteBuf.isDirect()) {
            TensorOpsGPU.embeddingTokenForwardGpuDirect(
                    tokenByteBuf, weightData, outData, batch, seqLen, dModel, vocabSize);
            return output;
        }
        float[] tokenData = tokens.internalBuffer();
        TensorOpsGPU.embeddingTokenForwardGpu(tokenData, weightData, outData, batch, seqLen, dModel, vocabSize);
        return output;
    }
}
