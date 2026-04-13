package com.veles.llm.jgpt;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Лог VRAM перед крупными аллокациями на device (см. {@link TensorOpsGPU#logVramBeforeDeviceFloatAlloc}). */
final class TensorOpsGpuVramLog {

    private static final Logger LOG = LoggerFactory.getLogger(TensorOpsGpuVramLog.class);

    private TensorOpsGpuVramLog() {}

    static void logBeforeDeviceFloatAlloc(long numFloats, String context) {
        if (!TensorOpsGPU.isGpuAvailable() || numFloats <= 0L) {
            return;
        }
        final long thresholdFloats = 15_000_000L; /* ~57 MiB buffer */
        boolean allSizes = Boolean.getBoolean("jgpt.debug.vramBeforeAlloc");
        if (!allSizes && numFloats < thresholdFloats) {
            return;
        }
        long needBytes = Math.multiplyExact(numFloats, 4L);
        long needMiB = (needBytes + (1024L * 1024L - 1L)) / (1024L * 1024L);
        long allocMiB = TensorOpsGPU.getGpuMemoryAllocated() / (1024L * 1024L);
        long reservedMiB = TensorOpsGPU.getGpuMemoryReserved() / (1024L * 1024L);
        String where = (context != null && !context.isEmpty()) ? " " + context : "";
        LOG.info(
                "[VRAM] перед аллокацией{}: needFloats={} (~{} MiB), allocated={} MiB, reserved={} MiB, shapeHint={}",
                where,
                numFloats,
                needMiB,
                allocMiB,
                reservedMiB,
                shapeHintForFloatPlane(numFloats));
    }

    private static String shapeHintForFloatPlane(long n) {
        if (n <= 0L) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        long[][] rect = {
            {6144L, 4096L},
            {4096L, 6144L},
            {2048L, 12288L},
            {12288L, 2048L},
            {8192L, 3072L},
            {3072L, 8192L},
            {16384L, 1536L},
            {1536L, 16384L},
            {32768L, 768L},
            {768L, 32768L}
        };
        for (long[] p : rect) {
            if (Math.multiplyExact(p[0], p[1]) == n) {
                if (sb.length() > 0) {
                    sb.append("; ");
                }
                sb.append(p[0]).append('×').append(p[1]).append(" float");
            }
        }
        if (n % 1024L == 0L) {
            long rows = n / 1024L;
            if (sb.length() > 0) {
                sb.append("; ");
            }
            sb.append("плоскость 1024×").append(rows).append(" (часто dModel×tokens)");
        }
        return sb.length() == 0 ? "(нет совпадений с типовыми прямоугольниками)" : sb.toString();
    }
}
