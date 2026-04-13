package com.veles.llm.jgpt;

/** FlashAttention-2 на {@link GpuFloatBuffer}; нативы в {@link TensorOpsGPU}. */
final class TensorOpsGpuFlashAttention {

    private TensorOpsGpuFlashAttention() {}

    static void flashAttentionForwardGpuDeviceResident(
            GpuFloatBuffer dQ,
            GpuFloatBuffer dK,
            GpuFloatBuffer dV,
            GpuFloatBuffer dOut,
            GpuFloatBuffer dLSE,
            int bh,
            int s,
            int dHead,
            float scale) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.flashAttentionForwardGpuDeviceResident");
        if (dHead != TensorOpsGPU.FLASH_ATTENTION_D_HEAD) {
            throw new IllegalArgumentException(
                    "FlashAttention forward requires d_head="
                            + TensorOpsGPU.FLASH_ATTENTION_D_HEAD
                            + ", got "
                            + dHead);
        }
        TensorOpsGPU.flashAttentionForwardGPUDeviceResident(
                dQ.devicePointer(),
                dK.devicePointer(),
                dV.devicePointer(),
                dOut.devicePointer(),
                dLSE.devicePointer(),
                bh,
                s,
                dHead,
                scale);
    }

    static void flashAttentionBackwardGpuDeviceResident(
            GpuFloatBuffer dQ,
            GpuFloatBuffer dK,
            GpuFloatBuffer dV,
            GpuFloatBuffer dO,
            GpuFloatBuffer dOGrad,
            GpuFloatBuffer dLSE,
            GpuFloatBuffer dGradQ,
            GpuFloatBuffer dGradK,
            GpuFloatBuffer dGradV,
            int bh,
            int s,
            int dHead,
            float scale) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.flashAttentionBackwardGpuDeviceResident");
        if (dHead != TensorOpsGPU.FLASH_ATTENTION_D_HEAD) {
            throw new IllegalArgumentException(
                    "FlashAttention backward requires d_head="
                            + TensorOpsGPU.FLASH_ATTENTION_D_HEAD
                            + ", got "
                            + dHead);
        }
        TensorOpsGPU.flashAttentionBackwardGPUDeviceResident(
                dQ.devicePointer(),
                dK.devicePointer(),
                dV.devicePointer(),
                dO.devicePointer(),
                dOGrad.devicePointer(),
                dLSE.devicePointer(),
                dGradQ.devicePointer(),
                dGradK.devicePointer(),
                dGradV.devicePointer(),
                bh,
                s,
                dHead,
                scale);
    }
}
