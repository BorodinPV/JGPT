package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;

import java.util.Arrays;

/** Утилиты среза [batch, seq, d] → [seq, d] и записи плоскости в 3D-тензор. */
final class GptTensorBatchPlanes {

    private GptTensorBatchPlanes() {}

    /** [batch, seq, d] → плоскость [seq, d]. */
    static Tensor sliceBatch3D(Tensor t, int batchIdx) {
        int[] shape = t.getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("expected 3D tensor");
        }
        Tensor slice = new Tensor(new int[] {shape[1], shape[2]});
        float[] src = t.internalBuffer();
        float[] dst = slice.internalBuffer();
        int[] s = t.stridesInternal();
        int srcBase = batchIdx * s[0];
        if (s[2] == 1) {
            for (int i = 0; i < shape[1]; i++) {
                System.arraycopy(src, srcBase + i * s[1], dst, i * shape[2], shape[2]);
            }
        } else {
            for (int i = 0; i < shape[1]; i++) {
                for (int j = 0; j < shape[2]; j++) {
                    dst[i * shape[2] + j] = src[srcBase + i * s[1] + j * s[2]];
                }
            }
        }
        return slice;
    }

    /** Копия [seq, last] в dest[batchIdx, :, :]. */
    static void copyBatchPlane(Tensor dest, Tensor src, int batchIdx) {
        int[] dShape = dest.getShape();
        int[] sShape = src.getShape();
        if (dShape.length != 3
                || sShape.length != 2
                || sShape[0] != dShape[1]
                || sShape[1] != dShape[2]) {
            throw new IllegalArgumentException(
                    "copyBatchPlane: dest " + Arrays.toString(dShape) + ", src " + Arrays.toString(sShape));
        }
        float[] d = dest.internalBuffer();
        float[] sbuf = src.internalBuffer();
        int[] ds = dest.stridesInternal();
        int[] ss = src.stridesInternal();
        int destBase = batchIdx * ds[0];
        if (ds[2] == 1 && ss[1] == 1) {
            for (int i = 0; i < sShape[0]; i++) {
                System.arraycopy(sbuf, i * ss[0], d, destBase + i * ds[1], sShape[1]);
            }
        } else {
            for (int i = 0; i < sShape[0]; i++) {
                for (int j = 0; j < sShape[1]; j++) {
                    d[destBase + i * ds[1] + j * ds[2]] = sbuf[i * ss[0] + j * ss[1]];
                }
            }
        }
    }
}
