package com.veles.llm.jgpt;

/** Device GEMM / bias / QKV·FFN strided projections; JNI в {@link TensorOpsGPU}. */
final class TensorOpsGpuDeviceGemm {

    private TensorOpsGpuDeviceGemm() {}

    static void addBiasBroadcastGpuDevice(GpuFloatBuffer c, GpuFloatBuffer bias, int m, int n) {
        if (m <= 0 || n <= 0) {
            throw new IllegalArgumentException("M and N must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(c, "c"), (long) m * n, "c");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(bias, "bias"), n, "bias");
        TensorOpsGPU.addBiasBroadcastGPUDevice(c.devicePointer(), bias.devicePointer(), m, n);
    }

    static void sumColumnsGpuDevice(GpuFloatBuffer src, GpuFloatBuffer dst, int m, int n) {
        sumColumnsGpuDevice(src, dst, m, n, 0f);
    }

    static void sumColumnsGpuDevice(GpuFloatBuffer src, GpuFloatBuffer dst, int m, int n, float beta) {
        if (m <= 0 || n <= 0) {
            throw new IllegalArgumentException("M and N must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(
                TensorOpsGpuBufferChecks.requireGpu(src, "src"), (long) m * n, "src");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dst, "dst"), n, "dst");
        TensorOpsGPU.sumColumnsGPUDevice(src.devicePointer(), dst.devicePointer(), m, n, beta);
    }

    static void matmulGpuDevice(GpuFloatBuffer a, GpuFloatBuffer b, GpuFloatBuffer c, int m, int k, int n) {
        if (m <= 0 || k <= 0 || n <= 0) {
            throw new IllegalArgumentException("M, K, N must be positive");
        }
        long needA = (long) m * k;
        long needB = (long) k * n;
        long needC = (long) m * n;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(a, "a"), needA, "a");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(b, "b"), needB, "b");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(c, "c"), needC, "c");
        TensorOpsGPU.matmulGPUDevice(a.devicePointer(), b.devicePointer(), c.devicePointer(), m, k, n);
    }

    static void matmulGpuDeviceEx(
            GpuFloatBuffer a,
            GpuFloatBuffer b,
            GpuFloatBuffer c,
            int m,
            int k,
            int n,
            boolean transposeA,
            boolean transposeB) {
        matmulGpuDeviceEx(a, b, c, m, k, n, transposeA, transposeB, 0f);
    }

    static void matmulGpuDeviceEx(
            GpuFloatBuffer a,
            GpuFloatBuffer b,
            GpuFloatBuffer c,
            int m,
            int k,
            int n,
            boolean transposeA,
            boolean transposeB,
            float beta) {
        if (m <= 0 || k <= 0 || n <= 0) {
            throw new IllegalArgumentException("M, K, N must be positive");
        }
        long needA = (long) m * k;
        long needB = (long) k * n;
        long needC = (long) m * n;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(a, "a"), needA, "a");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(b, "b"), needB, "b");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(c, "c"), needC, "c");
        TensorOpsGPU.matmulGPUDeviceEx(
                a.devicePointer(),
                b.devicePointer(),
                c.devicePointer(),
                m,
                k,
                n,
                transposeA,
                transposeB,
                beta);
    }

    static void matmulGpuDeviceQkvProjections(
            GpuFloatBuffer xNorm,
            GpuFloatBuffer wq,
            GpuFloatBuffer wk,
            GpuFloatBuffer wv,
            GpuFloatBuffer q,
            GpuFloatBuffer k,
            GpuFloatBuffer v,
            int rows,
            int dModel) {
        if (rows <= 0 || dModel <= 0) {
            throw new IllegalArgumentException("rows and dModel must be positive");
        }
        int m = rows;
        int kDim = dModel;
        int nDim = dModel;
        long needX = (long) m * kDim;
        long needW = (long) kDim * nDim;
        long needOut = (long) m * nDim;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(xNorm, "xNorm"), needX, "xNorm");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(wq, "wq"), needW, "wq");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(wk, "wk"), needW, "wk");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(wv, "wv"), needW, "wv");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(q, "q"), needOut, "q");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(k, "k"), needOut, "k");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(v, "v"), needOut, "v");
        TensorOpsGPU.matmulGpuDeviceQkvProjections0(
                xNorm.devicePointer(),
                wq.devicePointer(),
                wk.devicePointer(),
                wv.devicePointer(),
                q.devicePointer(),
                k.devicePointer(),
                v.devicePointer(),
                m,
                kDim,
                nDim);
    }

    static void matmulGpuDeviceFfnW1W3Projections(
            GpuFloatBuffer xNorm,
            GpuFloatBuffer w1,
            GpuFloatBuffer w3,
            GpuFloatBuffer h1,
            GpuFloatBuffer gate,
            int rows,
            int dModel,
            int dIntermediate) {
        if (rows <= 0 || dModel <= 0 || dIntermediate <= 0) {
            throw new IllegalArgumentException("rows, dModel and dIntermediate must be positive");
        }
        int m = rows;
        int kDim = dModel;
        int nDim = dIntermediate;
        long needX = (long) m * kDim;
        long needW = (long) kDim * nDim;
        long needOut = (long) m * nDim;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(xNorm, "xNorm"), needX, "xNorm");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(w1, "w1"), needW, "w1");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(w3, "w3"), needW, "w3");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(h1, "h1"), needOut, "h1");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gate, "gate"), needOut, "gate");
        TensorOpsGPU.matmulGpuDeviceFfnW1W3Projections0(
                xNorm.devicePointer(),
                w1.devicePointer(),
                w3.devicePointer(),
                h1.devicePointer(),
                gate.devicePointer(),
                m,
                kDim,
                nDim);
    }
}
