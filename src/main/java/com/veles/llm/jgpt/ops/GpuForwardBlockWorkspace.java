package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;

/**
 * Thread-local scratch для fused forward FFN + RMSNorm на GPU.
 *
 * <p>Весовые буферы ({@code normGamma}, {@code w1}, {@code w2}, {@code w3}) зависят только от
 * {@code dModel} и {@code dInt}; при изменении только {@code rows} они переиспользуются без закрытия.
 * Буферы по батчу пересоздаются (или расширяются) при смене {@code rows} или размеров модели.
 *
 * <p>Единственный хостовый буфер — {@link #getHostOut()}: staging для D2H перед {@link
 * com.veles.llm.jgpt.core.Tensor#wrap(float[], int[])}.
 * При переиспользовании массива большей длины безопасно читать только первые {@code rows * dModel}
 * элементов — не использовать {@code hostOut.length} как логический размер тензора.
 *
 * <p>Device-буферы при переиспользовании не очищаются (см. {@link GpuBufferUtils}).
 *
 * <p>{@link #ensureFfnNormResidual} — только под внешним {@link #exclusiveUseLock()}; без вложенного
 * synchronized на том же мониторе.
 */
final class GpuForwardBlockWorkspace {

    private static final ThreadLocal<GpuForwardBlockWorkspace> LOCAL =
            ThreadLocal.withInitial(GpuForwardBlockWorkspace::new);

    private final Object exclusiveUse = new Object();

    /** Последний полный набор размеров для row-буферов. */
    private int cachedRows = -1;
    private int cachedDModel;
    private int cachedDInt;

    /** Размеры, для которых выделены весовые GPU-буферы. */
    private int weightDModel = -1;
    private int weightDInt = -1;

    private GpuFloatBuffer xRes1;
    private GpuFloatBuffer normGamma;
    private GpuFloatBuffer w1;
    private GpuFloatBuffer w2;
    private GpuFloatBuffer w3;
    private GpuFloatBuffer xNorm2;
    private GpuFloatBuffer h1;
    private GpuFloatBuffer gate;
    private GpuFloatBuffer sig;
    private GpuFloatBuffer gateSwish;
    private GpuFloatBuffer hAct;
    private GpuFloatBuffer ffnOut;
    private GpuFloatBuffer out;

    /** Staging для {@code copyTo} + wrap тензора на хосте. */
    private float[] hostOut;

    GpuForwardBlockWorkspace() {}

    Object exclusiveUseLock() {
        return exclusiveUse;
    }

    static GpuForwardBlockWorkspace local() {
        return LOCAL.get();
    }

    static void releaseThreadLocal() {
        GpuForwardBlockWorkspace w = LOCAL.get();
        if (w != null) {
            synchronized (w.exclusiveUse) {
                w.closeAllGpuBuffers();
                w.cachedRows = -1;
                w.weightDModel = -1;
                w.hostOut = null;
            }
        }
        LOCAL.remove();
    }

    void ensureFfnNormResidual(int rows, int dModel, int dInt) {
        if (rows <= 0 || dModel <= 0 || dInt <= 0) {
            throw new IllegalArgumentException(
                    "rows, dModel, dInt must be positive: rows="
                            + rows
                            + ", dModel="
                            + dModel
                            + ", dInt="
                            + dInt);
        }
        long rowPlaneLong = GpuBufferUtils.mulExact("rows*dModel", (long) rows, (long) dModel);
        int hostElements = GpuBufferUtils.intSize("hostOut", rowPlaneLong);

        if (rows == cachedRows && dModel == cachedDModel && dInt == cachedDInt) {
            hostOut = GpuBufferUtils.ensureHost(hostOut, hostElements);
            return;
        }

        boolean weightsChanged = (dModel != weightDModel || dInt != weightDInt);
        boolean rowDimsChanged = (rows != cachedRows || dModel != cachedDModel || dInt != cachedDInt);

        if (weightsChanged) {
            closeWeightBuffers();
        }

        long w1Size = GpuBufferUtils.mulExact("dModel*dInt", (long) dModel, (long) dInt);
        long w2Size = GpuBufferUtils.mulExact("dInt*dModel", (long) dInt, (long) dModel);

        normGamma = GpuBufferUtils.ensure(normGamma, dModel);
        w1 = GpuBufferUtils.ensure(w1, w1Size);
        w2 = GpuBufferUtils.ensure(w2, w2Size);
        w3 = GpuBufferUtils.ensure(w3, w1Size);

        if (weightsChanged) {
            weightDModel = dModel;
            weightDInt = dInt;
        }

        if (rowDimsChanged || weightsChanged) {
            closeRowBuffers();
            long rowDInt = GpuBufferUtils.mulExact("rows*dInt", (long) rows, (long) dInt);

            xRes1 = GpuBufferUtils.ensure(xRes1, rowPlaneLong);
            xNorm2 = GpuBufferUtils.ensure(xNorm2, rowPlaneLong);
            h1 = GpuBufferUtils.ensure(h1, rowDInt);
            gate = GpuBufferUtils.ensure(gate, rowDInt);
            sig = GpuBufferUtils.ensure(sig, rowDInt);
            gateSwish = GpuBufferUtils.ensure(gateSwish, rowDInt);
            hAct = GpuBufferUtils.ensure(hAct, rowDInt);
            ffnOut = GpuBufferUtils.ensure(ffnOut, rowPlaneLong);
            out = GpuBufferUtils.ensure(out, rowPlaneLong);

            cachedRows = rows;
            cachedDModel = dModel;
            cachedDInt = dInt;
        }

        hostOut = GpuBufferUtils.ensureHost(hostOut, hostElements);
    }

    private void closeAllGpuBuffers() {
        closeWeightBuffers();
        closeRowBuffers();
    }

    private void closeWeightBuffers() {
        normGamma = GpuBufferUtils.closeAndNull(normGamma);
        w1 = GpuBufferUtils.closeAndNull(w1);
        w2 = GpuBufferUtils.closeAndNull(w2);
        w3 = GpuBufferUtils.closeAndNull(w3);
    }

    private void closeRowBuffers() {
        xRes1 = GpuBufferUtils.closeAndNull(xRes1);
        xNorm2 = GpuBufferUtils.closeAndNull(xNorm2);
        h1 = GpuBufferUtils.closeAndNull(h1);
        gate = GpuBufferUtils.closeAndNull(gate);
        sig = GpuBufferUtils.closeAndNull(sig);
        gateSwish = GpuBufferUtils.closeAndNull(gateSwish);
        hAct = GpuBufferUtils.closeAndNull(hAct);
        ffnOut = GpuBufferUtils.closeAndNull(ffnOut);
        out = GpuBufferUtils.closeAndNull(out);
    }

    float[] getHostOut() {
        return hostOut;
    }

    GpuFloatBuffer getXRes1() {
        return xRes1;
    }

    GpuFloatBuffer getNormGamma() {
        return normGamma;
    }

    GpuFloatBuffer getW1() {
        return w1;
    }

    GpuFloatBuffer getW2() {
        return w2;
    }

    GpuFloatBuffer getW3() {
        return w3;
    }

    GpuFloatBuffer getXNorm2() {
        return xNorm2;
    }

    GpuFloatBuffer getH1() {
        return h1;
    }

    GpuFloatBuffer getGate() {
        return gate;
    }

    GpuFloatBuffer getSig() {
        return sig;
    }

    GpuFloatBuffer getGateSwish() {
        return gateSwish;
    }

    GpuFloatBuffer getHAct() {
        return hAct;
    }

    GpuFloatBuffer getFfnOut() {
        return ffnOut;
    }

    GpuFloatBuffer getOut() {
        return out;
    }
}
