package com.veles.llm.jgpt;

import java.nio.ByteBuffer;

/**
 * Обёртки embedding / position с выбором FP16-gather по {@link TensorOpsGPU#useFp16Matmul()}; JNI — в {@link
 * TensorOpsGPU}.
 */
final class TensorOpsGpuEmbeddingHost {

    private TensorOpsGpuEmbeddingHost() {}

    static void embeddingTokenForwardGpu(
            float[] tokens,
            float[] weights,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGPU.embeddingTokenForwardGPU(
                tokens, weights, out, batch, seqLen, dModel, vocabSize, TensorOpsGPU.useFp16Matmul());
    }

    static void embeddingTokenForwardGpuDirect(
            ByteBuffer directTokens,
            float[] weights,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGPU.embeddingTokenForwardGPUDirect(
                directTokens,
                weights,
                out,
                batch,
                seqLen,
                dModel,
                vocabSize,
                TensorOpsGPU.useFp16Matmul());
    }

    static void embeddingTokenForwardGpuDeviceWeights(
            float[] tokens,
            long weightsDevicePtr,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGPU.embeddingTokenForwardGPUDeviceWeights(
                tokens,
                weightsDevicePtr,
                out,
                batch,
                seqLen,
                dModel,
                vocabSize,
                TensorOpsGPU.useFp16Matmul());
    }

    static void embeddingTokenForwardGpuDirectDeviceWeights(
            ByteBuffer directTokens,
            long weightsDevicePtr,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGPU.embeddingTokenForwardGPUDirectDeviceWeights(
                directTokens,
                weightsDevicePtr,
                out,
                batch,
                seqLen,
                dModel,
                vocabSize,
                TensorOpsGPU.useFp16Matmul());
    }

    static void embeddingTokenForwardGpuDeviceWeightsToDevice(
            float[] tokens,
            long weightsDevicePtr,
            GpuFloatBuffer outDevice,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        long needOut = (long) batch * seqLen * dModel;
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(outDevice, "outDevice"), needOut, "outDevice");
        TensorOpsGPU.embeddingTokenForwardGPUDeviceWeightsToDevice(
                tokens,
                weightsDevicePtr,
                outDevice.devicePointer(),
                batch,
                seqLen,
                dModel,
                vocabSize,
                TensorOpsGPU.useFp16Matmul());
    }

    static void embeddingTokenForwardGpuDirectDeviceWeightsToDevice(
            ByteBuffer directTokens,
            long weightsDevicePtr,
            GpuFloatBuffer outDevice,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        long needOut = (long) batch * seqLen * dModel;
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(outDevice, "outDevice"), needOut, "outDevice");
        TensorOpsGPU.embeddingTokenForwardGPUDirectDeviceWeightsToDevice(
                directTokens,
                weightsDevicePtr,
                outDevice.devicePointer(),
                batch,
                seqLen,
                dModel,
                vocabSize,
                TensorOpsGPU.useFp16Matmul());
    }

    static void addPositionEmbeddingGpuDevice(
            GpuFloatBuffer xData, GpuFloatBuffer posWeightsData, int batch, int seqLen, int dModel) {
        addPositionEmbeddingGpuDevice(xData, posWeightsData, batch, seqLen, dModel, 0);
    }

    static void addPositionEmbeddingGpuDevice(
            GpuFloatBuffer xData,
            GpuFloatBuffer posWeightsData,
            int batch,
            int seqLen,
            int dModel,
            int posRowStart) {
        if (batch <= 0 || seqLen <= 0 || dModel <= 0) {
            throw new IllegalArgumentException("batch, seqLen, dModel must be positive");
        }
        if (posRowStart < 0) {
            throw new IllegalArgumentException("posRowStart must be >= 0");
        }
        long needX = (long) batch * seqLen * dModel;
        long needPos = (long) (posRowStart + seqLen) * dModel;
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(xData, "xData"), needX, "xData");
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(posWeightsData, "posWeightsData"), needPos, "posWeightsData");
        TensorOpsGPU.addPositionEmbeddingGPUDeviceBuffersWithOffset(
                xData.devicePointer(), posWeightsData.devicePointer(), batch, seqLen, dModel, posRowStart);
    }

    static void addPositionEmbeddingInPlaceHostSlice(
            float[] xBatchSeqD, float[] posRowsContiguous, int batch, int seqLen, int dModel) {
        TensorOpsGPU.requireCuda("addPositionEmbeddingInPlaceHostSlice");
        if (batch <= 0 || seqLen <= 0 || dModel <= 0) {
            return;
        }
        long needX = (long) batch * seqLen * dModel;
        long needSlice = (long) seqLen * dModel;
        if (xBatchSeqD.length < needX || posRowsContiguous.length < needSlice) {
            throw new IllegalArgumentException(
                    "buffers too small: need x>="
                            + needX
                            + ", posSlice>="
                            + needSlice
                            + " got x="
                            + xBatchSeqD.length
                            + " pos="
                            + posRowsContiguous.length);
        }
        TensorOpsGPU.addPositionEmbeddingGPUHostPosSlice(
                xBatchSeqD, posRowsContiguous, batch, seqLen, dModel);
    }

    static void embeddingTokenForwardGpuEx(
            float[] tokens,
            float[] weights,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            boolean useFp16Gather) {
        TensorOpsGPU.embeddingTokenForwardGPU(
                tokens, weights, out, batch, seqLen, dModel, vocabSize, useFp16Gather);
    }
}
