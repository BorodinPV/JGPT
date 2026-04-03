package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.ops.TensorOps;

import java.util.Arrays;
import java.util.Objects;

/**
 * Полносвязный слой на GPU: {@code y = x @ W + b}, где {@code x} — [M, in], {@code W} — [in, out] (row-major),
 * {@code b} — [out]. GEMM и broadcast bias на device; градиенты в буферах {@link GpuTensor#zeroGrad()}.
 */
public final class Linear implements AutoCloseable {

    private final int inFeatures;
    private final int outFeatures;
    private final GpuTensor weight;
    private final GpuTensor bias;

    /**
     * @param weight [in, out] row-major
     * @param bias   [out]
     */
    public Linear(GpuTensor weight, GpuTensor bias) {
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(bias, "bias");
        if (!TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException("GPU not available");
        }
        int[] ws = weight.getShape();
        int[] bs = bias.getShape();
        if (ws.length != 2 || bs.length != 1 || ws[1] != bs[0]) {
            throw new IllegalArgumentException(
                    "expected weight [in,out] and bias [out], got " + Arrays.toString(ws) + " and " + Arrays.toString(bs));
        }
        this.inFeatures = ws[0];
        this.outFeatures = ws[1];
        this.weight = weight;
        this.bias = bias;
    }

    public int inFeatures() {
        return inFeatures;
    }

    public int outFeatures() {
        return outFeatures;
    }

    public GpuTensor weight() {
        return weight;
    }

    public GpuTensor bias() {
        return bias;
    }

    /**
     * @param input [M, inFeatures]
     * @return новый тензор [M, outFeatures]
     */
    public GpuTensor forwardGpu(GpuTensor input) {
        Objects.requireNonNull(input, "input");
        int[] is = input.getShape();
        if (is.length != 2 || is[1] != inFeatures) {
            throw new IllegalArgumentException(
                    "input must be [M," + inFeatures + "], got " + Arrays.toString(is));
        }
        int m = is[0];
        GpuTensor out = GpuTensor.allocate(new int[]{m, outFeatures});
        TensorOpsGPU.matmulGpuDevice(input.dataBuffer(), weight.dataBuffer(), out.dataBuffer(), m, inFeatures, outFeatures);
        TensorOpsGPU.addBiasBroadcastGpuDevice(out.dataBuffer(), bias.dataBuffer(), m, outFeatures);
        return out;
    }

    /**
     * То же, что {@link #backwardGpu(GpuTensor, GpuTensor, boolean)} с {@code accumulate = false} (градиенты
     * весов/bias перезаписываются после внутреннего {@link GpuTensor#zeroGrad()}).
     */
    public GpuTensor backwardGpu(GpuTensor input, GpuTensor gradOutput) {
        return backwardGpu(input, gradOutput, false);
    }

    /**
     * Обратное распространение на GPU.
     *
     * @param accumulate если {@code false} — перед вычислением вызываются {@link GpuTensor#zeroGrad()} для {@link
     *     #weight} и {@link #bias}, градиенты параметров перезаписываются вкладом этого шага. Если {@code true} —
     *     вклад добавляется к уже лежащим в буферах градиентам (gradient accumulation); перед первым таким шагом
     *     буферы должны быть обнулены вручную.
     * @return gradInput [M, inFeatures] (новый {@link GpuTensor})
     */
    public GpuTensor backwardGpu(GpuTensor input, GpuTensor gradOutput, boolean accumulate) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(gradOutput, "gradOutput");
        int[] is = input.getShape();
        int[] gos = gradOutput.getShape();
        if (is.length != 2 || is[1] != inFeatures || gos.length != 2 || gos[0] != is[0] || gos[1] != outFeatures) {
            throw new IllegalArgumentException(
                    "input [M," + inFeatures + "], gradOutput [M," + outFeatures + "], got "
                            + Arrays.toString(is) + " and " + Arrays.toString(gos));
        }
        int m = is[0];

        if (!accumulate) {
            weight.zeroGrad();
            bias.zeroGrad();
        }
        float betaAcc = accumulate ? 1f : 0f;

        GpuTensor gradInput = GpuTensor.allocate(new int[]{m, inFeatures});

        TensorOpsGPU.matmulGpuDeviceEx(
                gradOutput.dataBuffer(),
                weight.dataBuffer(),
                gradInput.dataBuffer(),
                m,
                outFeatures,
                inFeatures,
                false,
                true);

        TensorOpsGPU.matmulGpuDeviceEx(
                input.dataBuffer(),
                gradOutput.dataBuffer(),
                weight.gradBuffer(),
                inFeatures,
                m,
                outFeatures,
                true,
                false,
                betaAcc);

        TensorOpsGPU.sumColumnsGpuDevice(gradOutput.dataBuffer(), bias.gradBuffer(), m, outFeatures, betaAcc);

        TensorOpsGPU.synchronize();

        return gradInput;
    }

    /**
     * Эталон для тестов/сравнения с {@link #forwardGpu}: {@link TensorOps#matmul(Tensor, Tensor)} (CUDA) и явное
     * прибавление bias по строкам на CPU (тонкий цикл без дублирования GEMM).
     */
    public static Tensor forwardCpu(Tensor x, Tensor w, Tensor b) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(w, "w");
        Objects.requireNonNull(b, "b");
        Tensor y = TensorOps.matmul(x, w);
        int[] ys = y.getShape();
        int mm = ys[0];
        int n = ys[1];
        float[] yd = y.internalBuffer();
        float[] bd = b.internalBuffer();
        for (int i = 0; i < mm; i++) {
            int row = i * n;
            for (int j = 0; j < n; j++) {
                yd[row + j] += bd[j];
            }
        }
        return y;
    }

    @Override
    public void close() {
        weight.close();
        bias.close();
    }
}
