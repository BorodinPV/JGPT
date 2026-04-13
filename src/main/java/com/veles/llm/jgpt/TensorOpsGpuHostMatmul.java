package com.veles.llm.jgpt;

/**
 * Host float[] GEMM: FP16 или FP32 по {@link TensorOpsGPU#useFp16Matmul()} (нативы объявлены в {@link TensorOpsGPU}).
 */
final class TensorOpsGpuHostMatmul {

    private TensorOpsGpuHostMatmul() {}

    @FunctionalInterface
    private interface HostSgemmMaybeFp16 {
        void apply(float[] a, float[] b, float[] c, int m, int k, int n);
    }

    @FunctionalInterface
    private interface HostBatchedGemmMaybeFp16 {
        void apply(float[] a, float[] b, float[] c, int m, int k, int n, int batchCount);
    }

    @FunctionalInterface
    private interface HostMatmulBiasReluMaybeFp16 {
        void apply(float[] a, float[] b, float[] bias, float[] c, int m, int k, int n);
    }

    private static final HostSgemmMaybeFp16 HOST_MATMUL =
            TensorOpsGPU.useFp16Matmul() ? TensorOpsGPU::matmulGPUFp16 : TensorOpsGPU::matmulGPU;

    private static final HostBatchedGemmMaybeFp16 HOST_BATCHED_MATMUL =
            TensorOpsGPU.useFp16Matmul()
                    ? TensorOpsGPU::matmulBatchedGPUFp16
                    : TensorOpsGPU::matmulBatchedGPU;

    private static final HostMatmulBiasReluMaybeFp16 HOST_MATMUL_BIAS_RELU =
            TensorOpsGPU.useFp16Matmul()
                    ? TensorOpsGPU::matmulAddReluGPUFp16
                    : TensorOpsGPU::matmulAddReluGPU;

    static void matmulGPUMaybeFp16(float[] a, float[] b, float[] c, int m, int k, int n) {
        HOST_MATMUL.apply(a, b, c, m, k, n);
    }

    static void matmulBatchedGPUMaybeFp16(
            float[] a, float[] b, float[] c, int m, int k, int n, int batchCount) {
        HOST_BATCHED_MATMUL.apply(a, b, c, m, k, n, batchCount);
    }

    static void matmulAddReluGPUMaybeFp16(
            float[] a, float[] b, float[] bias, float[] c, int m, int k, int n) {
        HOST_MATMUL_BIAS_RELU.apply(a, b, bias, c, m, k, n);
    }
}
