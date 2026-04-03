package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.GpuFloatBuffer;

/**
 * Thread-local scratch для fused backward FFN + второй RMSNorm на GPU.
 *
 * <p>При смене {@code (rows, dModel, dInt)} все device-буферы закрываются одним проходом, затем
 * выделяются заново. Между вызовами с теми же размерами {@link #ensureFfnNorm2} сразу выходит.
 *
 * <p>{@link #ensureFfnNorm2} вызывать только под внешним {@link #exclusiveUseLock()} (см.
 * {@link TransformerBackward#tryFusedFfnNormResidualBackwardGpu}); внутренний mutex не дублируется.
 *
 * <p><b>Потокобезопасность:</b> один экземпляр на поток; не передавать между потоками.
 *
 * <p>Буферы при переиспользовании не очищаются автоматически (см. {@link GpuBufferUtils}).
 */
final class GpuBlockWorkspace {

    private static final ThreadLocal<GpuBlockWorkspace> LOCAL =
            ThreadLocal.withInitial(GpuBlockWorkspace::new);

    private final Object exclusiveUse = new Object();

    private int cachedRows = -1;
    private int cachedDModel;
    private int cachedDInt;

    private GpuFloatBuffer xFlat;
    private GpuFloatBuffer gradOut;
    private GpuFloatBuffer xRes1;
    private GpuFloatBuffer normGamma;
    private GpuFloatBuffer w1;
    private GpuFloatBuffer w2;
    private GpuFloatBuffer w3;
    private GpuFloatBuffer h1;
    private GpuFloatBuffer gate;
    private GpuFloatBuffer sig;
    private GpuFloatBuffer gateSwish;
    private GpuFloatBuffer hAct;
    private GpuFloatBuffer dHact;
    private GpuFloatBuffer dH1;
    private GpuFloatBuffer dGateSwish;
    private GpuFloatBuffer dGate;
    private GpuFloatBuffer dSig;
    private GpuFloatBuffer dXNorm2;
    private GpuFloatBuffer dXTmp;
    private GpuFloatBuffer dXRes1;
    private GpuFloatBuffer dNorm2;
    private GpuFloatBuffer dW1;
    private GpuFloatBuffer dW2;
    private GpuFloatBuffer dW3;

    GpuBlockWorkspace() {}

    Object exclusiveUseLock() {
        return exclusiveUse;
    }

    static GpuBlockWorkspace local() {
        return LOCAL.get();
    }

    static void releaseThreadLocal() {
        GpuBlockWorkspace w = LOCAL.get();
        if (w != null) {
            synchronized (w.exclusiveUse) {
                w.closeAllGpuBuffers();
                w.cachedRows = -1;
            }
        }
        LOCAL.remove();
    }

    void ensureFfnNorm2(int rows, int dModel, int dInt) {
        if (rows <= 0 || dModel <= 0 || dInt <= 0) {
            throw new IllegalArgumentException(
                    "rows, dModel, dInt must be positive: rows="
                            + rows
                            + ", dModel="
                            + dModel
                            + ", dInt="
                            + dInt);
        }
        if (rows == cachedRows && dModel == cachedDModel && dInt == cachedDInt) {
            return;
        }

        closeAllGpuBuffers();

        long rowPlane = GpuBufferUtils.mulExact("rows*dModel", (long) rows, (long) dModel);
        long w1Size = GpuBufferUtils.mulExact("dModel*dInt", (long) dModel, (long) dInt);
        long w2Size = GpuBufferUtils.mulExact("dInt*dModel", (long) dInt, (long) dModel);
        long rowDInt = GpuBufferUtils.mulExact("rows*dInt", (long) rows, (long) dInt);

        xFlat = GpuBufferUtils.ensure(xFlat, rowPlane);
        gradOut = GpuBufferUtils.ensure(gradOut, rowPlane);
        xRes1 = GpuBufferUtils.ensure(xRes1, rowPlane);
        normGamma = GpuBufferUtils.ensure(normGamma, dModel);
        w1 = GpuBufferUtils.ensure(w1, w1Size);
        w2 = GpuBufferUtils.ensure(w2, w2Size);
        w3 = GpuBufferUtils.ensure(w3, w1Size);
        h1 = GpuBufferUtils.ensure(h1, rowDInt);
        gate = GpuBufferUtils.ensure(gate, rowDInt);
        sig = GpuBufferUtils.ensure(sig, rowDInt);
        gateSwish = GpuBufferUtils.ensure(gateSwish, rowDInt);
        hAct = GpuBufferUtils.ensure(hAct, rowDInt);
        dHact = GpuBufferUtils.ensure(dHact, rowDInt);
        dH1 = GpuBufferUtils.ensure(dH1, rowDInt);
        dGateSwish = GpuBufferUtils.ensure(dGateSwish, rowDInt);
        dGate = GpuBufferUtils.ensure(dGate, rowDInt);
        dSig = GpuBufferUtils.ensure(dSig, rowDInt);
        dXNorm2 = GpuBufferUtils.ensure(dXNorm2, rowPlane);
        dXTmp = GpuBufferUtils.ensure(dXTmp, rowPlane);
        dXRes1 = GpuBufferUtils.ensure(dXRes1, rowPlane);
        dNorm2 = GpuBufferUtils.ensure(dNorm2, dModel);
        dW1 = GpuBufferUtils.ensure(dW1, w1Size);
        dW2 = GpuBufferUtils.ensure(dW2, w2Size);
        dW3 = GpuBufferUtils.ensure(dW3, w1Size);

        cachedRows = rows;
        cachedDModel = dModel;
        cachedDInt = dInt;
    }

    private void closeAllGpuBuffers() {
        GpuFloatBuffer[] buffers = {
            xFlat,
            gradOut,
            xRes1,
            normGamma,
            w1,
            w2,
            w3,
            h1,
            gate,
            sig,
            gateSwish,
            hAct,
            dHact,
            dH1,
            dGateSwish,
            dGate,
            dSig,
            dXNorm2,
            dXTmp,
            dXRes1,
            dNorm2,
            dW1,
            dW2,
            dW3
        };
        for (int i = 0; i < buffers.length; i++) {
            buffers[i] = GpuBufferUtils.closeAndNull(buffers[i]);
        }
        xFlat = buffers[0];
        gradOut = buffers[1];
        xRes1 = buffers[2];
        normGamma = buffers[3];
        w1 = buffers[4];
        w2 = buffers[5];
        w3 = buffers[6];
        h1 = buffers[7];
        gate = buffers[8];
        sig = buffers[9];
        gateSwish = buffers[10];
        hAct = buffers[11];
        dHact = buffers[12];
        dH1 = buffers[13];
        dGateSwish = buffers[14];
        dGate = buffers[15];
        dSig = buffers[16];
        dXNorm2 = buffers[17];
        dXTmp = buffers[18];
        dXRes1 = buffers[19];
        dNorm2 = buffers[20];
        dW1 = buffers[21];
        dW2 = buffers[22];
        dW3 = buffers[23];
    }

    GpuFloatBuffer getXFlat() {
        return xFlat;
    }

    GpuFloatBuffer getGradOut() {
        return gradOut;
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

    GpuFloatBuffer getDHact() {
        return dHact;
    }

    GpuFloatBuffer getDH1() {
        return dH1;
    }

    GpuFloatBuffer getDGateSwish() {
        return dGateSwish;
    }

    GpuFloatBuffer getDGate() {
        return dGate;
    }

    GpuFloatBuffer getDSig() {
        return dSig;
    }

    GpuFloatBuffer getDXNorm2() {
        return dXNorm2;
    }

    GpuFloatBuffer getDXTmp() {
        return dXTmp;
    }

    GpuFloatBuffer getDXRes1() {
        return dXRes1;
    }

    GpuFloatBuffer getDNorm2() {
        return dNorm2;
    }

    GpuFloatBuffer getDW1() {
        return dW1;
    }

    GpuFloatBuffer getDW2() {
        return dW2;
    }

    GpuFloatBuffer getDW3() {
        return dW3;
    }
}
