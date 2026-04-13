package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;

import java.util.concurrent.atomic.AtomicBoolean;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * RMSNorm + matmul LM head на device; опционально fused JNI по {@code JGPT_FUSED_LM_HEAD}.
 */
final class GptLmHeadGpu {

    private static final Logger log = LoggerFactory.getLogger(GptLmHeadGpu.class);
    private static final AtomicBoolean FUSED_LM_HEAD_FAILURE_LOGGED = new AtomicBoolean();

    private GptLmHeadGpu() {}

    static boolean fusedLmHeadFromEnv() {
        String e = System.getenv("JGPT_FUSED_LM_HEAD");
        if (e == null) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    static void applyLmHeadSplitOps(
            GpuFloatBuffer xBeforeNorm,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer normScratch,
            GpuFloatBuffer w,
            GpuFloatBuffer logitsOut,
            int rows,
            int dModel,
            int vocab) {
        TensorOpsGPU.rmsNormGpuDevice(xBeforeNorm, gamma, eps, normScratch, rows, dModel);
        TensorOpsGPU.matmulGpuDevice(normScratch, w, logitsOut, rows, dModel, vocab);
    }

    /**
     * Предпочитает fused JNI при {@link #fusedLmHeadFromEnv()}; при {@link Throwable} — один раз в лог и откат на
     * раздельный путь.
     */
    static void applyFusedPreferredThenSplit(
            GpuFloatBuffer xBeforeNorm,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer normScratch,
            GpuFloatBuffer w,
            GpuFloatBuffer logitsOut,
            int rows,
            int dModel,
            int vocab) {
        if (!fusedLmHeadFromEnv()) {
            applyLmHeadSplitOps(xBeforeNorm, gamma, eps, normScratch, w, logitsOut, rows, dModel, vocab);
            return;
        }
        try {
            TensorOpsGPU.rmsNormMatmulLmHeadGpuDevice(
                    xBeforeNorm,
                    gamma,
                    eps,
                    normScratch,
                    w,
                    logitsOut,
                    rows,
                    dModel,
                    vocab,
                    TensorOpsGPU.useFp16Matmul());
        } catch (Throwable t) {
            if (FUSED_LM_HEAD_FAILURE_LOGGED.compareAndSet(false, true)) {
                log.warn(
                        "JGPT_FUSED_LM_HEAD: fused LM head недоступен или выбросил исключение; откат на RMSNorm+matmul ({})",
                        t.toString());
            }
            applyLmHeadSplitOps(xBeforeNorm, gamma, eps, normScratch, w, logitsOut, rows, dModel, vocab);
        }
    }
}
