package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.ops.TensorOps;
import com.veles.llm.jgpt.ops.TensorOpsBackward;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Минимальный цикл обучения для проверки forward/backward (линейная регрессия y ≈ x·w + b).
 */
public final class SimpleTrainer {

    private static final Logger log = LoggerFactory.getLogger(SimpleTrainer.class);

    private SimpleTrainer() {
    }

    public static void main(String[] args) {
        trainLinear();
    }

    /**
     * Тренировка: y = x @ W + b (без ReLU), цель — восстановить коэффициенты у первых двух признаков.
     */
    public static void trainLinear() {
        TensorOpsGPU.requireCuda("SimpleTrainer.trainLinear");
        int inputDim = 10;
        int outputDim = 1;
        int batchSize = 32;

        Tensor W = randomTensor(new int[]{inputDim, outputDim}, 0.1f);
        Tensor b = new Tensor(new int[]{outputDim});

        float learningRate = 0.05f;

        for (int epoch = 0; epoch < 100; epoch++) {
            Tensor x = randomTensor(new int[]{batchSize, inputDim}, 1.0f);
            Tensor yTrue = new Tensor(new int[]{batchSize, outputDim});
            float[] yData = yTrue.internalBuffer();
            float[] xData = x.internalBuffer();
            for (int i = 0; i < batchSize; i++) {
                yData[i * outputDim] = 2f * xData[i * inputDim] + 3f * xData[i * inputDim + 1]
                        + (float) (Math.random() * 0.1f - 0.05f);
            }

            Tensor yLin = TensorOps.matmul(x, W);
            Tensor yPred = addBiasRows(yLin, b);

            Tensor diff = TensorOps.subtract(yPred, yTrue);
            Tensor squared = TensorOps.multiply(diff, diff);
            float loss = TensorOps.mean(squared);

            int n = batchSize * outputDim;
            float scale = 2f / n;

            Tensor gradY = new Tensor(new int[]{batchSize, outputDim});
            gradY.zeroGrad();
            float[] diffData = diff.internalBuffer();
            float[] gy = gradY.gradBuffer();
            for (int i = 0; i < gy.length; i++) {
                gy[i] = scale * diffData[i];
            }

            Tensor gradX = new Tensor(new int[]{batchSize, inputDim});
            gradX.zeroGrad();
            W.zeroGrad();
            b.zeroGrad();

            TensorOpsBackward.matmulBackward(gradY, x, W, gradX, W);

            float[] bg = b.gradBuffer();
            for (int j = 0; j < outputDim; j++) {
                float s = 0f;
                for (int i = 0; i < batchSize; i++) {
                    s += gy[i * outputDim + j];
                }
                bg[j] = s;
            }

            updateWeights(W, learningRate);
            updateWeights(b, learningRate);

            if (epoch % 10 == 0) {
                log.info("Эпоха {}: loss = {}", epoch, String.format("%.6f", loss));
            }
        }

        log.info("Обучение завершено.");
        log.info(
                "Оценка W[0,0]={} (ожидается ~2), W[1,0]={} (ожидается ~3), b[0]={} (ожидается ~0)",
                W.get(0, 0),
                W.get(1, 0),
                b.get(0));
    }

    /** yPred[i,j] = yLin[i,j] + b[j] */
    private static Tensor addBiasRows(Tensor yLin, Tensor b) {
        int[] shape = yLin.getShape();
        int rows = shape[0];
        int cols = shape[1];
        Tensor out = new Tensor(shape);
        float[] src = yLin.internalBuffer();
        float[] dst = out.internalBuffer();
        float[] bias = b.internalBuffer();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dst[i * cols + j] = src[i * cols + j] + bias[j];
            }
        }
        return out;
    }

    private static Tensor randomTensor(int[] shape, float std) {
        Tensor t = new Tensor(shape);
        float[] data = t.internalBuffer();
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Math.random() * 2 - 1) * std;
        }
        return t;
    }

    private static void updateWeights(Tensor w, float lr) {
        float[] data = w.internalBuffer();
        float[] grad = w.gradBuffer();
        for (int i = 0; i < data.length; i++) {
            data[i] -= lr * grad[i];
        }
    }
}
