package com.veles.llm.jgpt;

/** AdamW на {@link GpuFloatBuffer}; JNI в {@link TensorOpsGPU}. */
final class TensorOpsGpuDeviceAdamW {

    private TensorOpsGpuDeviceAdamW() {}

    static void adamWStepGpuDevice(
            GpuFloatBuffer param,
            GpuFloatBuffer grad,
            GpuFloatBuffer m,
            GpuFloatBuffer v,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            float weightDecay,
            float invBias1,
            float invBias2,
            int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(param, "param"), n, "param");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(grad, "grad"), n, "grad");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(m, "m"), n, "m");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(v, "v"), n, "v");
        TensorOpsGPU.adamWStepGPUDevice(
                param.devicePointer(),
                grad.devicePointer(),
                m.devicePointer(),
                v.devicePointer(),
                learningRate,
                beta1,
                beta2,
                epsilon,
                weightDecay,
                invBias1,
                invBias2,
                n);
    }
}
