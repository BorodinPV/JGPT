package com.veles.llm.jgpt;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import java.util.Map;
import java.util.Objects;

/** RMSNorm, поэлементные op и редукции на device; JNI в {@link TensorOpsGPU}. */
final class TensorOpsGpuDeviceUnary {

    private TensorOpsGpuDeviceUnary() {}

    static void rmsNormGpuDevice(
            GpuFloatBuffer x, GpuFloatBuffer gamma, float eps, GpuFloatBuffer out, int outer, int lastDim) {
        if (outer <= 0 || lastDim <= 0) {
            throw new IllegalArgumentException("outer and lastDim must be positive");
        }
        long plane = (long) outer * lastDim;
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(x, "x"), plane, "x");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gamma, "gamma"), lastDim, "gamma");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(out, "out"), plane, "out");
        TensorOpsGPU.rmsNormGPUDevice(
                x.devicePointer(),
                gamma.devicePointer(),
                eps,
                out.devicePointer(),
                outer,
                lastDim,
                TensorOpsGPU.useFp16Matmul());
    }

    static void sigmoidGpuDevice(GpuFloatBuffer src, GpuFloatBuffer dst, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(src, "src"), n, "src");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(dst, "dst"), n, "dst");
        TensorOpsGPU.sigmoidGPUDevice(src.devicePointer(), dst.devicePointer(), n);
    }

    static void multiplyGpuDevice(GpuFloatBuffer a, GpuFloatBuffer b, GpuFloatBuffer c, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(a, "a"), n, "a");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(b, "b"), n, "b");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(c, "c"), n, "c");
        TensorOpsGPU.multiplyGPUDevice(a.devicePointer(), b.devicePointer(), c.devicePointer(), n);
    }

    static void multiplyBackwardGpuDevice(
            GpuFloatBuffer gOut, GpuFloatBuffer a, GpuFloatBuffer b, GpuFloatBuffer gA, GpuFloatBuffer gB, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gOut, "gOut"), n, "gOut");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(a, "a"), n, "a");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(b, "b"), n, "b");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gA, "gA"), n, "gA");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gB, "gB"), n, "gB");
        TensorOpsGPU.multiplyBackwardGPUDevice(
                gOut.devicePointer(), a.devicePointer(), b.devicePointer(), gA.devicePointer(), gB.devicePointer(), n);
    }

    static void sigmoidBackwardGpuDevice(GpuFloatBuffer gOut, GpuFloatBuffer inp, GpuFloatBuffer gIn, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gOut, "gOut"), n, "gOut");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(inp, "inp"), n, "inp");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(gIn, "gIn"), n, "gIn");
        TensorOpsGPU.sigmoidBackwardGPUDevice(gOut.devicePointer(), inp.devicePointer(), gIn.devicePointer(), n);
    }

    static void accumulateAddGpuDevice(GpuFloatBuffer acc, GpuFloatBuffer delta, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(acc, "acc"), n, "acc");
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(delta, "delta"), n, "delta");
        TensorOpsGPU.accumulateAddGPUDevice(acc.devicePointer(), delta.devicePointer(), n);
    }

    static void scaleInPlaceGpuDevice(GpuFloatBuffer buf, int n, float scalar) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(buf, "buf"), n, "buf");
        TensorOpsGPU.scaleInPlaceGPUDevice(buf.devicePointer(), n, scalar);
    }

    static double sumSquaresGpuDevice(GpuFloatBuffer buf, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(buf, "buf"), n, "buf");
        return TensorOpsGPU.sumSquaresGPUDevice(buf.devicePointer(), n);
    }

    static double sumSquaresGpuDeviceParamGrads(Map<Tensor, GpuTensor> paramMap) {
        Objects.requireNonNull(paramMap, "paramMap");
        TensorOpsGPU.requireCuda("TensorOpsGPU.sumSquaresGpuDeviceParamGrads");
        int n = 0;
        for (GpuTensor gt : paramMap.values()) {
            if (gt != null && !gt.isClosed() && gt.hasGradBuffer()) {
                n++;
            }
        }
        if (n == 0) {
            return 0.0;
        }
        long[] ptrs = new long[n];
        int[] lens = new int[n];
        int i = 0;
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt == null || gt.isClosed() || !gt.hasGradBuffer()) {
                continue;
            }
            ptrs[i] = gt.gradBuffer().devicePointer();
            lens[i] = e.getKey().size();
            i++;
        }
        return TensorOpsGPU.sumSquaresGPUDeviceFused(ptrs, lens, n);
    }

    static boolean anyNonFiniteGpuDevice(GpuFloatBuffer buf, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        TensorOpsGpuBufferChecks.requireMinFloats(TensorOpsGpuBufferChecks.requireGpu(buf, "buf"), n, "buf");
        return TensorOpsGPU.anyNonFiniteGPUDevice(buf.devicePointer(), n);
    }

    static boolean anyNonFiniteGpuDeviceMulti(GpuFloatBuffer[] bufs, int[] lens, int nBufs) {
        if (nBufs <= 0 || bufs == null || lens == null) {
            return false;
        }
        long[] ptrs = new long[nBufs];
        int[] safeLens = new int[nBufs];
        int count = 0;
        for (int i = 0; i < nBufs; i++) {
            GpuFloatBuffer b = bufs[i];
            int ln = lens[i];
            if (b == null || b.isClosed() || ln <= 0) {
                continue;
            }
            ptrs[count] = b.devicePointer();
            safeLens[count] = ln;
            count++;
        }
        if (count == 0) {
            return false;
        }
        return TensorOpsGPU.anyNonFiniteGPUDeviceMulti(ptrs, safeLens, count);
    }
}
