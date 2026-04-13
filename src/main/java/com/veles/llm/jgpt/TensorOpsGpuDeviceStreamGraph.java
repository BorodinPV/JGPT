package com.veles.llm.jgpt;

/** Decoder graph / strided pack / D2D half↔float; JNI в {@link TensorOpsGPU}. */
final class TensorOpsGpuDeviceStreamGraph {

    private TensorOpsGpuDeviceStreamGraph() {}

    static void ensureStridedBatchedPackScratch(long rows, int dModel, int dIntermediate) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.ensureStridedBatchedPackScratch");
        if (rows <= 0 || dModel <= 0 || dIntermediate <= 0) {
            throw new IllegalArgumentException("rows, dModel, dIntermediate must be positive");
        }
        if (!TensorOpsGPU.ensureStridedBatchedPackScratch0(rows, dModel, dIntermediate)) {
            throw new IllegalStateException(
                    "ensureStridedBatchedPackScratch failed (native allocation); rows=" + rows);
        }
    }

    static long[] stridedBatchedPackNeed(long rows, int dModel, int dIntermediate) {
        if (rows <= 0 || dModel <= 0 || dIntermediate <= 0) {
            throw new IllegalArgumentException("rows, dModel, dIntermediate must be positive");
        }
        long knQkv = Math.multiplyExact((long) dModel, dModel);
        long mnQkv = Math.multiplyExact(rows, dModel);
        long knFfn = Math.multiplyExact((long) dModel, dIntermediate);
        long mnFfn = Math.multiplyExact(rows, dIntermediate);
        long lim = (long) Integer.MAX_VALUE / 4L;
        if (knQkv > lim || mnQkv > lim || knFfn > lim || mnFfn > lim) {
            throw new IllegalArgumentException("stridedBatchedPackNeed: size overflow");
        }
        long wNeed = Math.max(Math.multiplyExact(3L, knQkv), Math.multiplyExact(2L, knFfn));
        long cNeed = Math.max(Math.multiplyExact(3L, mnQkv), Math.multiplyExact(2L, mnFfn));
        return new long[] {wNeed, cNeed};
    }

    static void setStridedBatchedPackOverride(
            long wDevicePtr, long cDevicePtr, long capWElems, long capCElems) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.setStridedBatchedPackOverride");
        if (wDevicePtr == 0L || cDevicePtr == 0L || capWElems <= 0L || capCElems <= 0L) {
            throw new IllegalArgumentException("setStridedBatchedPackOverride: invalid pointer or capacity");
        }
        TensorOpsGPU.setStridedBatchedPackOverride0(wDevicePtr, cDevicePtr, capWElems, capCElems);
    }

    static void clearStridedBatchedPackOverride() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        TensorOpsGPU.clearStridedBatchedPackOverride0();
    }

    static void decoderGraphPrewarmDeviceOps(
            int batch, int seqLen, int dModel, int numHeads, int dIntermediate) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.decoderGraphPrewarmDeviceOps");
        if (batch <= 0 || seqLen <= 0 || dModel <= 0 || numHeads <= 0 || dIntermediate <= 0) {
            throw new IllegalArgumentException("batch, seqLen, dModel, numHeads, dIntermediate must be positive");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by numHeads");
        }
        TensorOpsGPU.decoderGraphPrewarmDeviceOps0(batch, seqLen, dModel, numHeads, dIntermediate);
    }

    static boolean cudaStreamBeginCapture() {
        TensorOpsGPU.requireCuda("TensorOpsGPU.cudaStreamBeginCapture");
        return TensorOpsGPU.cudaStreamBeginCapture0();
    }

    static long cudaStreamEndCaptureAndInstantiate() {
        return TensorOpsGPU.cudaStreamEndCaptureAndInstantiate0();
    }

    static void abortCudaStreamCaptureIfActive() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        TensorOpsGPU.abortCudaStreamCaptureIfActive0();
    }

    static boolean cudaGraphExecLaunch(long execPtr) {
        if (execPtr == 0L) {
            return true;
        }
        return TensorOpsGPU.cudaGraphExecLaunch0(execPtr);
    }

    static int decoderGraphExecLaunchLastCudaError() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return 0;
        }
        return TensorOpsGPU.decoderGraphExecLaunchLastCudaError0();
    }

    static long[] decoderGraphLaunchProbe(long execPtr) {
        if (!TensorOpsGPU.isGpuAvailable() || execPtr == 0L) {
            return new long[0];
        }
        long[] a = TensorOpsGPU.decoderGraphLaunchProbe0(execPtr);
        return a != null && a.length > 0 ? a : new long[0];
    }

    static void cudaGraphExecDestroy(long execPtr) {
        if (execPtr != 0L) {
            TensorOpsGPU.cudaGraphExecDestroy0(execPtr);
        }
    }

    static long[] decoderGraphDebugNativeAuxSnapshot() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return new long[] {0L, 0L, 0L, 0L};
        }
        long[] s = TensorOpsGPU.decoderGraphDebugNativeAuxSnapshot0();
        return s != null ? s : new long[] {0L, 0L, 0L, 0L};
    }

    static long decoderGraphNativeStabilityToken() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return 0L;
        }
        return TensorOpsGPU.decoderGraphNativeStabilityToken0();
    }

    static void convertFloatDeviceToHalfDevice(long srcFloatDevicePtr, long dstHalfDevicePtr, int numFloats) {
        TensorOpsGPU.requireCuda("convertFloatDeviceToHalfDevice");
        if (srcFloatDevicePtr == 0L || dstHalfDevicePtr == 0L || numFloats <= 0) {
            throw new IllegalArgumentException("convertFloatDeviceToHalfDevice: invalid args");
        }
        TensorOpsGPU.nativeConvertFloatDeviceToHalfDevice(srcFloatDevicePtr, dstHalfDevicePtr, numFloats);
    }

    static void convertHalfDeviceToFloatDevice(long srcHalfDevicePtr, long dstFloatDevicePtr, int numFloats) {
        TensorOpsGPU.requireCuda("convertHalfDeviceToFloatDevice");
        if (srcHalfDevicePtr == 0L || dstFloatDevicePtr == 0L || numFloats <= 0) {
            throw new IllegalArgumentException("convertHalfDeviceToFloatDevice: invalid args");
        }
        TensorOpsGPU.nativeConvertHalfDeviceToFloatDevice(srcHalfDevicePtr, dstFloatDevicePtr, numFloats);
    }
}
