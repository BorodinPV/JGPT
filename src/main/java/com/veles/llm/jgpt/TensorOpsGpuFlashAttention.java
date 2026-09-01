package com.veles.llm.jgpt;

/** FlashAttention-2 на {@link GpuFloatBuffer}; нативы в {@link TensorOpsGPU}. */
final class TensorOpsGpuFlashAttention {

    private TensorOpsGpuFlashAttention() {}

    static int requireHeads(int bh, int numHeads) {
        int heads = numHeads <= 0 ? 1 : numHeads;
        if (bh <= 0 || bh % heads != 0) {
            throw new IllegalArgumentException("FlashAttention BH=" + bh + " not divisible by numHeads=" + heads);
        }
        return heads;
    }

    static void flashAttentionForwardGpuDeviceResident(
            GpuFloatBuffer dQ,
            GpuFloatBuffer dK,
            GpuFloatBuffer dV,
            GpuFloatBuffer dOut,
            GpuFloatBuffer dLSE,
            int bh,
            int s,
            int dHead,
            float scale,
            int numHeads) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.flashAttentionForwardGpuDeviceResident");
        if (dHead != TensorOpsGPU.FLASH_ATTENTION_D_HEAD) {
            throw new IllegalArgumentException(
                    "FlashAttention forward requires d_head="
                            + TensorOpsGPU.FLASH_ATTENTION_D_HEAD
                            + ", got "
                            + dHead);
        }
        int heads = requireHeads(bh, numHeads);
        TensorOpsGPU.flashAttentionForwardGPUDeviceResident(
                dQ.devicePointer(),
                dK.devicePointer(),
                dV.devicePointer(),
                dOut.devicePointer(),
                dLSE.devicePointer(),
                bh,
                s,
                dHead,
                scale,
                heads);
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
            float scale,
            int numHeads) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.flashAttentionBackwardGpuDeviceResident");
        if (dHead != TensorOpsGPU.FLASH_ATTENTION_D_HEAD) {
            throw new IllegalArgumentException(
                    "FlashAttention backward requires d_head="
                            + TensorOpsGPU.FLASH_ATTENTION_D_HEAD
                            + ", got "
                            + dHead);
        }
        int heads = requireHeads(bh, numHeads);
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
                scale,
                heads);
    }

    static boolean flashAttentionForwardGpuDeviceResidentHalf(
            long qHalf,
            long kHalf,
            long vHalf,
            long oHalf,
            GpuFloatBuffer dLSE,
            int bh,
            int s,
            int dHead,
            float scale,
            int numHeads) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.flashAttentionForwardGpuDeviceResidentHalf");
        if (dHead != TensorOpsGPU.FLASH_ATTENTION_D_HEAD || qHalf == 0L || kHalf == 0L || vHalf == 0L || oHalf == 0L) {
            return false;
        }
        int heads = requireHeads(bh, numHeads);
        return TensorOpsGPU.flashAttentionForwardGPUDeviceResidentHalf(
                qHalf, kHalf, vHalf, oHalf, dLSE.devicePointer(), bh, s, dHead, scale, heads);
    }

    static boolean flashAttentionBackwardGpuDeviceResidentHalf(
            long qHalf,
            long kHalf,
            long vHalf,
            long oHalf,
            long dOHalf,
            GpuFloatBuffer dLSE,
            long dQHalf,
            long dKHalf,
            long dVHalf,
            int bh,
            int s,
            int dHead,
            float scale,
            int numHeads) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.flashAttentionBackwardGpuDeviceResidentHalf");
        if (dHead != TensorOpsGPU.FLASH_ATTENTION_D_HEAD
                || qHalf == 0L
                || kHalf == 0L
                || vHalf == 0L
                || oHalf == 0L
                || dOHalf == 0L
                || dQHalf == 0L
                || dKHalf == 0L
                || dVHalf == 0L) {
            return false;
        }
        int heads = requireHeads(bh, numHeads);
        return TensorOpsGPU.flashAttentionBackwardGPUDeviceResidentHalf(
                qHalf,
                kHalf,
                vHalf,
                oHalf,
                dOHalf,
                dLSE.devicePointer(),
                dQHalf,
                dKHalf,
                dVHalf,
                bh,
                s,
                dHead,
                scale,
                heads);
    }
}
