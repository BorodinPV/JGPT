package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.core.Tensor;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

/**
 * Загрузка весов в порядке {@link GPTModel#getParameters()}.
 */
final class GptModelWeightsLoader {

    private static final int WEIGHT_SAVE_NON_LAYER_TENSOR_COUNT = 4;
    private static final int WEIGHT_SAVE_TENSORS_PER_DECODER_LAYER = 9;

    private GptModelWeightsLoader() {}

    static String weightTensorCountMismatchMessage(int numLayers, int savedN, int modelN) {
        StringBuilder sb = new StringBuilder(280);
        sb.append("saved parameter count ")
                .append(savedN)
                .append(" != model parameter count ")
                .append(modelN)
                .append(" (текущая геометрия: ")
                .append(numLayers)
                .append(" слоёв).");
        if (savedN >= WEIGHT_SAVE_NON_LAYER_TENSOR_COUNT
                && (savedN - WEIGHT_SAVE_NON_LAYER_TENSOR_COUNT) % WEIGHT_SAVE_TENSORS_PER_DECODER_LAYER == 0) {
            int fileLayers =
                    (savedN - WEIGHT_SAVE_NON_LAYER_TENSOR_COUNT) / WEIGHT_SAVE_TENSORS_PER_DECODER_LAYER;
            sb.append(" По числу тензоров файл похож на ")
                    .append(fileLayers)
                    .append("-слойный чекпоинт; задайте --layers ")
                    .append(fileLayers)
                    .append(" (и при необходимости --seq-len как при обучении), либо те же env, что при train.");
        }
        return sb.toString();
    }

    static void loadFromJavaSerialization(int numLayers, List<Tensor> params, ObjectInputStream in)
            throws IOException, ClassNotFoundException {
        int n = in.readInt();
        if (n != params.size()) {
            throw new IOException(weightTensorCountMismatchMessage(numLayers, n, params.size()));
        }
        for (int i = 0; i < n; i++) {
            int[] shape = (int[]) in.readObject();
            float[] data = (float[]) in.readObject();
            copyLoadedParamSlice(i, params, shape, data);
        }
    }

    static void loadFromBinaryV1(int numLayers, List<Tensor> params, DataInputStream in) throws IOException {
        int n = in.readInt();
        if (n != params.size()) {
            throw new IOException(weightTensorCountMismatchMessage(numLayers, n, params.size()));
        }
        for (int i = 0; i < n; i++) {
            int rank = in.readInt();
            int[] shape = new int[rank];
            for (int r = 0; r < rank; r++) {
                shape[r] = in.readInt();
            }
            int expected = 1;
            for (int d : shape) {
                expected *= d;
            }
            float[] data = new float[expected];
            if (expected > 0) {
                byte[] raw = new byte[expected * 4];
                in.readFully(raw);
                ByteBuffer.wrap(raw).order(ByteOrder.BIG_ENDIAN).asFloatBuffer().get(data);
            }
            copyLoadedParamSlice(i, params, shape, data);
        }
    }

    private static IOException shapeMismatchAtParam(int i, int[] savedShape, int[] modelShape) {
        String base =
                "shape mismatch at param "
                        + i
                        + ": saved "
                        + Arrays.toString(savedShape)
                        + " vs model "
                        + Arrays.toString(modelShape);
        if (i == 1
                && savedShape.length == 2
                && modelShape.length == 2
                && savedShape[1] == modelShape[1]
                && savedShape[0] != modelShape[0]) {
            return new IOException(
                    base
                            + ". Несовпадение maxSeqLen у позиционной таблицы: в чекпоинте "
                            + savedShape[0]
                            + " позиций, в текущей геометрии "
                            + modelShape[0]
                            + ". Задайте --seq-len "
                            + savedShape[0]
                            + " (и то же, что при обучении / JGPT_MAX_SEQ_LEN).");
        }
        if (i == 0
                && savedShape.length == 2
                && modelShape.length == 2
                && savedShape[1] == modelShape[1]
                && savedShape[0] != modelShape[0]) {
            return new IOException(
                    base
                            + ". Несовпадение vocab у эмбеддинга токенов: файл "
                            + savedShape[0]
                            + " vs модель "
                            + modelShape[0]
                            + " (токенизатор/пресет).");
        }
        return new IOException(base);
    }

    private static void copyLoadedParamSlice(int i, List<Tensor> params, int[] shape, float[] data)
            throws IOException {
        Tensor p = params.get(i);
        if (!Arrays.equals(shape, p.getShape())) {
            throw shapeMismatchAtParam(i, shape, p.getShape());
        }
        float[] dst = p.internalBuffer();
        if (data.length != dst.length) {
            throw new IOException("length mismatch at param " + i);
        }
        System.arraycopy(data, 0, dst, 0, dst.length);
    }
}
