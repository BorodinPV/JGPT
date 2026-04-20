package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Обратное распространение для выбранных операций (накопление в буферы градиентов).
 *
 * <p><b>Контракт:</b> аргументы-«входящий градиент» ({@code gradC}, {@code gradOut}, …) читаются через
 * {@link #upstreamGrad(Tensor)}: либо {@link Tensor#gradBuffer()} (если {@link Tensor#hasGrad()}), либо
 * {@link Tensor#internalBuffer()} (например {@link Tensor#wrap(float[], int[])} с буфером ∂L/∂·). Целевые
 * тензоры ({@code gradA}, {@code gradX}, …) должны существовать; при первом вкладе буфер обнуляется внутри
 * {@link #accumulateGradientInto(Tensor, Tensor)}.
 */
public final class TensorOpsBackward {

    private TensorOpsBackward() {
    }

    /**
     * Backward для add: C = A + B
     * ∂L/∂A = ∂L/∂C, ∂L/∂B = ∂L/∂C
     */
    public static void addBackward(Tensor gradC, Tensor gradA, Tensor gradB) {
        Objects.requireNonNull(gradC, "gradC");
        Objects.requireNonNull(gradA, "gradA");
        Objects.requireNonNull(gradB, "gradB");
        float[] gc = upstreamGrad(gradC);
        float[] ga = ensureGradBuffer(gradA);
        float[] gb = ensureGradBuffer(gradB);
        if (gc.length != ga.length || gc.length != gb.length) {
            throw new IllegalArgumentException("shape mismatch for add backward");
        }
        if (gc.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.addBackward");
        TensorOpsGPU.accumulateAddGPU(ga, gc, gc.length);
        TensorOpsGPU.accumulateAddGPU(gb, gc, gc.length);
    }

    /**
     * Backward для matmul: C = A @ B
     * ∂L/∂A = ∂L/∂C @ Bᵀ, ∂L/∂B = Aᵀ @ ∂L/∂C.
     * Градиенты в {@code gradA}/{@code gradB} накапливаются (+=); при отсутствии буфера вызывается
     * {@link Tensor#zeroGrad()} внутри {@link #accumulateGradientInto(Tensor, Tensor)}.
     */
    public static void matmulBackward(
            Tensor gradC, Tensor a, Tensor b,
            Tensor gradA, Tensor gradB) {
        Objects.requireNonNull(gradC, "gradC");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(gradA, "gradA");
        Objects.requireNonNull(gradB, "gradB");
        int[] aShape = a.getShape();
        int[] bShape = b.getShape();
        int[] cShape = gradC.getShape();
        if (aShape.length != 2 || bShape.length != 2) {
            throw new IllegalArgumentException("matmul backward requires 2D A and B");
        }
        int m = aShape[0];
        int ka = aShape[1];
        int kb = bShape[0];
        int n = bShape[1];
        if (ka != kb) {
            throw new IllegalArgumentException("incompatible inner dims: " + ka + " != " + kb);
        }
        if (cShape.length != 2 || cShape[0] != m || cShape[1] != n) {
            throw new IllegalArgumentException(
                    "gradC shape must be [" + m + "," + n + "], got " + Arrays.toString(cShape));
        }

        Tensor gradCView = Tensor.wrap(upstreamGrad(gradC), gradC.getShape());
        Tensor bT = transpose(b);
        Tensor gradAUpdate = TensorOps.matmul(gradCView, bT);
        Tensor aT = transpose(a);
        Tensor gradBUpdate = TensorOps.matmul(aT, gradCView);
        accumulateGradientInto(gradA, gradAUpdate);
        accumulateGradientInto(gradB, gradBUpdate);
    }

    /** Накопление ∂L в буфер градиента {@code target} (+= {@code update}). */
    public static void accumulateGradientInto(Tensor target, Tensor update) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(update, "update");
        float[] tgt = ensureGradBuffer(target);
        float[] upd = update.internalBuffer();
        if (tgt.length != upd.length) {
            throw new IllegalArgumentException(
                    "gradient length mismatch: " + tgt.length + " vs " + upd.length);
        }
        if (tgt.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.accumulateGradientInto");
        TensorOpsGPU.accumulateAddGPU(tgt, upd, tgt.length);
    }

    /**
     * Backward для LayerNorm (нормализация по последней оси, как в {@link TensorOps#layerNorm}).
     * {@code gradOut}: ∂L/∂y — в {@link Tensor#gradBuffer()} (после {@link Tensor#zeroGrad()}) или в
     * {@link Tensor#internalBuffer()} (например {@link Tensor#wrap(float[], int[])}).
     * Градиенты накапливаются в {@code gradX}, {@code gradGamma}, {@code gradBeta} (+=).
     */
    public static void layerNormBackward(
            Tensor gradOut, Tensor x, Tensor gamma, Tensor beta, float eps,
            Tensor gradX, Tensor gradGamma, Tensor gradBeta) {
        Objects.requireNonNull(gradOut, "gradOut");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(gamma, "gamma");
        Objects.requireNonNull(beta, "beta");
        Objects.requireNonNull(gradX, "gradX");
        Objects.requireNonNull(gradGamma, "gradGamma");
        Objects.requireNonNull(gradBeta, "gradBeta");
        int[] shape = x.getShape();
        if (shape.length < 1) {
            throw new IllegalArgumentException("layerNorm backward requires at least 1D x");
        }
        int lastDim = shape[shape.length - 1];
        int outerSize = 1;
        for (int i = 0; i < shape.length - 1; i++) {
            outerSize *= shape[i];
        }
        TensorOps.layerNormValidateParams(gamma, beta, lastDim);

        float[] gOut = upstreamGrad(gradOut);
        float[] src = x.internalBuffer();
        float[] g = gamma.internalBuffer();

        float[] gx = ensureGradBuffer(gradX);
        float[] gg = ensureGradBuffer(gradGamma);
        float[] gb = ensureGradBuffer(gradBeta);

        int n = outerSize * lastDim;
        if (n <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.layerNormBackward");
        TensorOpsGPU.layerNormBackwardGPU(gOut, src, g, eps, gx, gg, gb, outerSize, lastDim);
    }

    /** ∂L/∂·: либо выделенный {@code grad}, либо данные {@link Tensor#wrap(float[], int[])}. */
    private static float[] upstreamGrad(Tensor t) {
        if (t.hasGrad()) {
            return t.gradBuffer();
        }
        return t.internalBuffer();
    }

    /** Обеспечивает наличие буфера градиента и возвращает его. */
    private static float[] ensureGradBuffer(Tensor t) {
        if (!t.hasGrad()) {
            t.zeroGrad();
        }
        return t.gradBuffer();
    }

    /**
     * RMSNorm (как {@link TensorOps#rmsNorm}): y_i = x_i / rms * gamma_i,
     * {@code rms = sqrt(mean(x²) + eps)}.
     */
    public static void rmsNormBackward(
            Tensor gradOut, Tensor x, Tensor gamma, float eps,
            Tensor gradX, Tensor gradGamma) {
        Objects.requireNonNull(gradOut, "gradOut");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(gamma, "gamma");
        Objects.requireNonNull(gradX, "gradX");
        Objects.requireNonNull(gradGamma, "gradGamma");
        int[] shape = x.getShape();
        if (shape.length < 1) {
            throw new IllegalArgumentException("rmsNorm backward requires at least 1D x");
        }
        int lastDim = shape[shape.length - 1];
        int outerSize = 1;
        for (int i = 0; i < shape.length - 1; i++) {
            outerSize *= shape[i];
        }
        int[] gShape = gamma.getShape();
        if (gShape.length != 1 || gShape[0] != lastDim) {
            throw new IllegalArgumentException(
                    "gamma must have shape [" + lastDim + "], got " + Arrays.toString(gShape));
        }

        float[] gOut = upstreamGrad(gradOut);
        float[] src = x.internalBuffer();
        float[] g = gamma.internalBuffer();

        float[] gx = ensureGradBuffer(gradX);
        float[] gg = ensureGradBuffer(gradGamma);

        int n = outerSize * lastDim;
        if (n <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.rmsNormBackward");
        TensorOpsGPU.rmsNormBackwardGPU(gOut, src, g, eps, gx, gg, outerSize, lastDim);
    }

    /**
     * C = A + B, градиенты накапливаются в gradA, gradB.
     */
    public static void subtractBackward(Tensor gradC, Tensor gradA, Tensor gradB) {
        Objects.requireNonNull(gradC, "gradC");
        Objects.requireNonNull(gradA, "gradA");
        Objects.requireNonNull(gradB, "gradB");
        float[] gc = upstreamGrad(gradC);
        float[] ga = ensureGradBuffer(gradA);
        float[] gb = ensureGradBuffer(gradB);
        if (gc.length != ga.length || gc.length != gb.length) {
            throw new IllegalArgumentException("shape mismatch for subtract backward");
        }
        if (gc.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.subtractBackward");
        TensorOpsGPU.accumulateAddGPU(ga, gc, gc.length);
        TensorOpsGPU.accumulateScaledAddGPU(gb, gc, -1f, gc.length);
    }

    /**
     * C = A ⊙ B (поэлементно), ∂L/∂A = ∂L/∂C ⊙ B, ∂L/∂B = ∂L/∂C ⊙ A.
     */
    public static void multiplyBackward(Tensor gradC, Tensor a, Tensor b, Tensor gradA, Tensor gradB) {
        Objects.requireNonNull(gradC, "gradC");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        Objects.requireNonNull(gradA, "gradA");
        Objects.requireNonNull(gradB, "gradB");
        float[] gc = upstreamGrad(gradC);
        float[] da = a.internalBuffer();
        float[] db = b.internalBuffer();
        float[] ga = ensureGradBuffer(gradA);
        float[] gb = ensureGradBuffer(gradB);
        if (gc.length != ga.length || gc.length != gb.length) {
            throw new IllegalArgumentException("shape mismatch for multiply backward");
        }
        if (gc.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.multiplyBackward");
        TensorOpsGPU.multiplyBackwardGPU(gc, da, db, ga, gb, gc.length);
    }

    /** C = A * scalar, ∂L/∂A = ∂L/∂C * scalar. */
    public static void multiplyScalarBackward(Tensor gradC, float scalar, Tensor gradA) {
        Objects.requireNonNull(gradC, "gradC");
        Objects.requireNonNull(gradA, "gradA");
        float[] gc = upstreamGrad(gradC);
        float[] ga = ensureGradBuffer(gradA);
        if (gc.length != ga.length) {
            throw new IllegalArgumentException("shape mismatch for multiplyScalar backward");
        }
        if (gc.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.multiplyScalarBackward");
        TensorOpsGPU.accumulateScaledAddGPU(ga, gc, scalar, gc.length);
    }

    /**
     * ReLU: ∂L/∂x = ∂L/∂y · 1{ x &gt; 0 }.
     */
    public static void reluBackward(Tensor gradOut, Tensor input, Tensor gradIn) {
        Objects.requireNonNull(gradOut, "gradOut");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(gradIn, "gradIn");
        float[] gOut = upstreamGrad(gradOut);
        float[] inp = input.internalBuffer();
        float[] gIn = ensureGradBuffer(gradIn);

        if (gOut.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.reluBackward");
        TensorOpsGPU.reluBackwardGPU(gOut, inp, gIn, gOut.length);
    }

    /** σ(x), ∂L/∂x = ∂L/∂y · σ(x)(1−σ(x)). */
    public static void sigmoidBackward(Tensor gradOut, Tensor input, Tensor gradIn) {
        Objects.requireNonNull(gradOut, "gradOut");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(gradIn, "gradIn");
        float[] gOut = upstreamGrad(gradOut);
        float[] inp = input.internalBuffer();
        float[] gIn = ensureGradBuffer(gradIn);
        if (gOut.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.sigmoidBackward");
        TensorOpsGPU.sigmoidBackwardGPU(gOut, inp, gIn, gOut.length);
    }

    /** GELU (tanh-аппроксимация как в {@link TensorOps#gelu}). */
    public static void geluBackward(Tensor gradOut, Tensor input, Tensor gradIn) {
        Objects.requireNonNull(gradOut, "gradOut");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(gradIn, "gradIn");
        float[] gOut = upstreamGrad(gradOut);
        float[] inp = input.internalBuffer();
        float[] gIn = ensureGradBuffer(gradIn);
        if (gOut.length <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.geluBackward");
        TensorOpsGPU.geluBackwardGPU(gOut, inp, gIn, gOut.length);
    }

    /**
     * Softmax по последней оси для 3D тензора: P = softmax(X), ∂L/∂X из ∂L/∂P.
     */
    public static void softmaxLastDimBackward3D(Tensor gradOut, Tensor probs, Tensor gradInput) {
        Objects.requireNonNull(gradOut, "gradOut");
        Objects.requireNonNull(probs, "probs");
        Objects.requireNonNull(gradInput, "gradInput");
        int[] shape = probs.getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("softmaxLastDimBackward3D requires 3D");
        }
        float[] go = upstreamGrad(gradOut);
        float[] p = probs.internalBuffer();
        float[] gi = ensureGradBuffer(gradInput);
        int batch = shape[0];
        int mid = shape[1];
        int inner = shape[2];
        int n = batch * mid * inner;
        if (n <= 0) {
            return;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.softmaxLastDimBackward3D");
        TensorOpsGPU.softmaxLastDimBackward3DGPU(go, p, gi, batch, mid, inner);
    }

    /**
     * CE + softmax: градиент ∂L/∂logits (среднее по токенам batch×seq, как в train step).
     * <p>Возвращается <b>новый</b> тензор той же формы, что {@code logits}; вклад лежит в
     * {@link Tensor#gradBuffer()} (внутри вызывается {@link Tensor#zeroGrad()}). Активации logits не
     * меняются. Типичное использование: скопировать градиент в {@code logits.zeroGrad(); logits.gradBuffer()}
     * перед {@code model.backward(logits)}, либо подставить этот тензор как носитель ∂L/∂logits в свой граф.
     */
    public static Tensor crossEntropySoftmaxBackward(Tensor logits, Tensor target) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(target, "target");
        int[] logitShape = logits.getShape();
        int batch = logitShape[0];
        int seqLen = logitShape[1];
        int vocabSize = logitShape[2];

        Tensor grad = new Tensor(logitShape);
        float[] gradData = ensureGradBuffer(grad);

        float[] logitData = logits.internalBuffer();
        float[] targetData = target.internalBuffer();

        int totalTokens = batch * seqLen;
        if (totalTokens <= 0 || !TensorOpsGPU.shouldUseGpuCrossEntropy(logitData.length)) {
            return grad;
        }
        TensorOpsGPU.requireCuda("TensorOpsBackward.crossEntropySoftmaxBackward");
        float[] logitCopy = Arrays.copyOf(logitData, logitData.length);
        TensorOpsGPU.crossEntropySoftmaxGradLossGpu(
                logitCopy,
                targetData,
                gradData,
                batch,
                seqLen,
                vocabSize,
                1.0f / (float) totalTokens);
        return grad;
    }

    /**
     * Транспонирование 2D тензора для backprop matmul.
     * <p>Выделяет новый row-major буфер и копирует элементы; отдельного «view» без копирования в этой
     * реализации нет (континуальный transpose не совпадает с раскладкой исходного массива).
     */
    public static Tensor transpose(Tensor a) {
        Objects.requireNonNull(a, "a");
        int[] shape = a.getShape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("transpose requires 2D tensor");
        }
        float[] src = a.internalBuffer();
        if (shape[0] == 0 || shape[1] == 0) {
            return new Tensor(new int[]{shape[1], shape[0]});
        }
        Tensor t3 = Tensor.wrap(src, new int[]{1, shape[0], shape[1]});
        Tensor r3 = TensorOps.transpose2DLast(t3);
        return Tensor.wrap(r3.internalBuffer(), new int[]{shape[1], shape[0]});
    }
}
