package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Fp16Tensor;
import com.veles.llm.jgpt.core.Tensor;

/**
 * Операции над {@link Fp16Tensor}: вычисления выполняются через float32 ({@link Tensor} + {@link TensorOps}) и
 * результат снова приводится к FP16. Для GEMM на GPU с Tensor Cores см. {@code JGPT_FP16_MATMUL=1}
 * и {@link TensorOpsGPU#useFp16Matmul()}; мастер-веса FP32 в этом классе не реализованы.
 *
 * <p><b>Потокобезопасность:</b> только статические методы без собственного изменяемого состояния; безопасны при
 * одновременном использовании разными потоками, если сами {@link Tensor} / {@link Fp16Tensor} не разделяются без
 * синхронизации.
 */
public final class Fp16Ops {

    private Fp16Ops() {}

    public static Fp16Tensor randomTensor(int[] shape, float scale) {
        return Fp16Tensor.fromTensor(TensorOps.randomTensor(shape, scale));
    }

    /** Matmul 2D: как {@link TensorOps#matmul(Tensor, Tensor)}. */
    public static Fp16Tensor matmul(Fp16Tensor a, Fp16Tensor b) {
        Tensor c = TensorOps.matmul(a.toTensor(), b.toTensor());
        return Fp16Tensor.fromTensor(c);
    }

    /** Покомпонентное сложение; формы должны совпадать. */
    public static Fp16Tensor add(Fp16Tensor a, Fp16Tensor b) {
        Tensor c = TensorOps.add(a.toTensor(), b.toTensor());
        return Fp16Tensor.fromTensor(c);
    }

    /** Покомпонентное умножение. */
    public static Fp16Tensor multiply(Fp16Tensor a, Fp16Tensor b) {
        Tensor c = TensorOps.multiply(a.toTensor(), b.toTensor());
        return Fp16Tensor.fromTensor(c);
    }

    public static Fp16Tensor multiplyScalar(Fp16Tensor a, float scalar) {
        Tensor c = TensorOps.multiplyScalar(a.toTensor(), scalar);
        return Fp16Tensor.fromTensor(c);
    }

    public static Fp16Tensor rmsNorm(Fp16Tensor x, Tensor gamma, float eps) {
        Tensor c = TensorOps.rmsNorm(x.toTensor(), gamma, eps);
        return Fp16Tensor.fromTensor(c);
    }
}
