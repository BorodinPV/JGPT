package com.veles.llm.jgpt;

import java.nio.ByteBuffer;
import java.util.Objects;

/** H2D загрузка id токенов из direct float-буфера; JNI в {@link TensorOpsGPU}. */
final class TensorOpsGpuHostUpload {

    private TensorOpsGpuHostUpload() {}

    static void uploadTokenIdsFromFloatDirectToGpuInt(
            ByteBuffer hostFloatsAsTokenIds, long byteOffset, int nTokens, GpuIntBuffer dst) {
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        Objects.requireNonNull(hostFloatsAsTokenIds, "hostFloatsAsTokenIds");
        Objects.requireNonNull(dst, "dst");
        if (!hostFloatsAsTokenIds.isDirect()) {
            throw new IllegalArgumentException("hostFloatsAsTokenIds must be direct");
        }
        if (nTokens <= 0) {
            return;
        }
        long needBytes = (long) nTokens * Integer.BYTES;
        if (byteOffset < 0 || byteOffset + needBytes > hostFloatsAsTokenIds.capacity()) {
            throw new IllegalArgumentException("host float buffer range invalid");
        }
        if (nTokens > dst.numInts()) {
            throw new IllegalArgumentException("GpuIntBuffer too small: need " + nTokens + ", have " + dst.numInts());
        }
        TensorOpsGPU.copyHostFloatBufferToGpuIntTokenIds(
                hostFloatsAsTokenIds, byteOffset, nTokens, dst.devicePointer());
    }
}
