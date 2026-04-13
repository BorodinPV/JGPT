package com.veles.llm.jgpt;

import java.util.Objects;

/**
 * Scratch для host→device путей ({@link TensorOpsGPU#splitHeadsFromHost}, {@link TensorOpsGPU#concatHeadsFromHost},
 * SDPA с {@code float[]} на хосте).
 */
final class TensorOpsGpuHostPathScratch {

    private static final ThreadLocal<SdpaHostPathScratch> TL_SDPA_HOST =
            ThreadLocal.withInitial(SdpaHostPathScratch::new);

    private static final ThreadLocal<HeadsHostPathScratch> TL_HEADS_HOST =
            ThreadLocal.withInitial(HeadsHostPathScratch::new);

    private TensorOpsGpuHostPathScratch() {}

    static void releaseThreadLocal() {
        TL_SDPA_HOST.get().release();
        TL_SDPA_HOST.remove();
        TL_HEADS_HOST.get().release();
        TL_HEADS_HOST.remove();
    }

    static void splitHeadsFromHost(float[] src, float[] dst, int batch, int seqLen, int dModel, int numHeads) {
        if (batch <= 0 || seqLen <= 0 || dModel <= 0 || numHeads <= 0 || dModel % numHeads != 0) {
            throw new IllegalArgumentException("splitHeadsFromHost: invalid shape");
        }
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(dst, "dst");
        TL_HEADS_HOST.get().splitHeads(src, dst, batch, seqLen, dModel, numHeads);
    }

    static void concatHeadsFromHost(float[] src, float[] dst, int batch, int numHeads, int seqLen, int dHead) {
        if (batch <= 0 || numHeads <= 0 || seqLen <= 0 || dHead <= 0) {
            throw new IllegalArgumentException("concatHeadsFromHost: invalid shape");
        }
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(dst, "dst");
        TL_HEADS_HOST.get().concatHeads(src, dst, batch, numHeads, seqLen, dHead);
    }

    static void scaledDotProductAttentionForwardFromHost(
            float[] q,
            float[] k,
            float[] v,
            float[] mask,
            float[] output,
            float[] probs,
            int batch,
            int seqLen,
            int dK,
            int dV,
            float scale,
            boolean useFp16Softmax) {
        TensorOpsGPU.requireCuda("TensorOpsGPU.scaledDotProductAttentionForwardFromHost");
        if (batch <= 0 || seqLen <= 0 || dK <= 0 || dV <= 0) {
            throw new IllegalArgumentException("batch, seqLen, dK, dV must be positive");
        }
        long qk = (long) batch * seqLen * dK;
        long vSz = (long) batch * seqLen * dV;
        long probSz = (long) batch * seqLen * seqLen;
        Objects.requireNonNull(q, "q");
        Objects.requireNonNull(k, "k");
        Objects.requireNonNull(v, "v");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(probs, "probs");
        if (q.length < qk || k.length < qk || v.length < vSz || output.length < vSz || probs.length < probSz) {
            throw new IllegalArgumentException("host array length too small for attention tensors");
        }
        TL_SDPA_HOST.get()
                .run(q, k, v, mask, output, probs, batch, seqLen, dK, dV, scale, useFp16Softmax);
    }

    private static final class HeadsHostPathScratch {
        private GpuFloatBuffer dSrc;
        private GpuFloatBuffer dDst;
        private long cap = -1;

        private void ensure(long need) {
            if (dSrc != null && cap >= need) {
                return;
            }
            if (dSrc != null) {
                dSrc.close();
                dDst.close();
            }
            dSrc = GpuFloatBuffer.allocate(need);
            dDst = GpuFloatBuffer.allocate(need);
            cap = need;
        }

        private void release() {
            if (dSrc != null) {
                dSrc.close();
                dDst.close();
                dSrc = null;
                dDst = null;
            }
            cap = -1;
        }

        private void splitHeads(float[] src, float[] dst, int batch, int seqLen, int dModel, int numHeads) {
            long total = (long) batch * seqLen * dModel;
            int ni = Math.toIntExact(total);
            ensure(total);
            dSrc.copyFrom(src, 0, ni);
            TensorOpsGPU.splitHeadsGPUDevice(
                    dSrc.devicePointer(), dDst.devicePointer(), batch, seqLen, dModel, numHeads);
            dDst.copyTo(dst, 0, ni);
        }

        private void concatHeads(float[] src, float[] dst, int batch, int numHeads, int seqLen, int dHead) {
            int dModel = numHeads * dHead;
            long total = (long) batch * seqLen * dModel;
            int ni = Math.toIntExact(total);
            ensure(total);
            dSrc.copyFrom(src, 0, ni);
            TensorOpsGPU.concatHeadsGPUDevice(
                    dSrc.devicePointer(), dDst.devicePointer(), batch, numHeads, seqLen, dHead);
            dDst.copyTo(dst, 0, ni);
        }
    }

    private static final class SdpaHostPathScratch {
        private GpuFloatBuffer dq;
        private GpuFloatBuffer dk;
        private GpuFloatBuffer dv;
        private GpuFloatBuffer dOut;
        private long capQk = -1;
        private long capV = -1;

        private void ensureQk(long need) {
            if (dq != null && capQk >= need) {
                return;
            }
            if (dq != null) {
                dq.close();
                dk.close();
            }
            dq = GpuFloatBuffer.allocate(need);
            dk = GpuFloatBuffer.allocate(need);
            capQk = need;
        }

        private void ensureV(long need) {
            if (dv != null && capV >= need) {
                return;
            }
            if (dv != null) {
                dv.close();
                dOut.close();
            }
            dv = GpuFloatBuffer.allocate(need);
            dOut = GpuFloatBuffer.allocate(need);
            capV = need;
        }

        private void release() {
            if (dq != null) {
                dq.close();
                dk.close();
                dq = null;
                dk = null;
            }
            if (dv != null) {
                dv.close();
                dOut.close();
                dv = null;
                dOut = null;
            }
            capQk = -1;
            capV = -1;
        }

        private void run(
                float[] q,
                float[] k,
                float[] v,
                float[] mask,
                float[] output,
                float[] probs,
                int batch,
                int seqLen,
                int dK,
                int dV,
                float scale,
                boolean useFp16Softmax) {
            long qk = (long) batch * seqLen * dK;
            long vSz = (long) batch * seqLen * dV;
            int qki = Math.toIntExact(qk);
            int vzi = Math.toIntExact(vSz);
            ensureQk(qk);
            ensureV(vSz);
            dq.copyFrom(q, 0, qki);
            dk.copyFrom(k, 0, qki);
            dv.copyFrom(v, 0, vzi);
            TensorOpsGPU.scaledDotProductAttentionForwardGPUDevice(
                    dq.devicePointer(),
                    dk.devicePointer(),
                    dv.devicePointer(),
                    mask,
                    dOut.devicePointer(),
                    probs,
                    batch,
                    seqLen,
                    dK,
                    dV,
                    scale,
                    useFp16Softmax);
            dOut.copyTo(output, 0, vzi);
        }
    }
}
