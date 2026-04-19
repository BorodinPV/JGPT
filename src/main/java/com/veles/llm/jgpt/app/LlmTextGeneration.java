package com.veles.llm.jgpt.app;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.model.GPTModel;

/**
 * Авторегрессивная генерация текста (batch=1) для инференса и интерактива при обучении.
 * <p>
 * Если модель {@link GPTModel#isGpuResident()} и CUDA доступна — по умолчанию используется {@link
 * GPTModel#generateGpuKv} (KV на VRAM). Отключить: {@code JGPT_GENERATE_GPU_KV=0} или {@code false} в окружении —
 * тогда всегда {@link GPTModel#generate} (host {@link com.veles.llm.jgpt.model.KvCache}).
 * <p>
 * Границы GPU: {@link GPTModel#generateGpuKv} вызывает {@link TensorOpsGPU#synchronizeStream()} перед закрытием KV (не
 * {@code cudaDeviceSynchronize} по всему устройству). После генерации — {@link TensorOpsGPU#synchronizeStream()} (для
 * host-KV обязателен; для GPU-KV — дополнительная граница перед drain) и
 * {@link TensorOpsGPU#drainDeferredGpuBuffers()}, чтобы очереди отложенных освобождений VRAM не копились между вызовами.
 */
public final class LlmTextGeneration {

    private LlmTextGeneration() {}

    private static boolean useGpuKvFromEnv() {
        String e = System.getenv("JGPT_GENERATE_GPU_KV");
        if (e == null || e.isBlank()) {
            return true;
        }
        String t = e.trim();
        return !"0".equals(t) && !"false".equalsIgnoreCase(t);
    }

    public static String generateText(
            GPTModel model,
            BPETokenizer tokenizer,
            String prompt,
            int maxNewTokens,
            float temperature,
            int topK) {
        TensorOpsGPU.requireCuda("LlmTextGeneration.generateText");
        int[] inputTokens = tokenizer.encode(prompt, true);
        Tensor input = new Tensor(new int[]{1, inputTokens.length});
        float[] inputData = input.internalBuffer();
        for (int i = 0; i < inputTokens.length; i++) {
            inputData[i] = inputTokens[i];
        }

        boolean gpuKv =
                useGpuKvFromEnv()
                        && model.isGpuResident()
                        && TensorOpsGPU.isGpuAvailable();
        Tensor generated =
                gpuKv
                        ? model.generateGpuKv(input, maxNewTokens, temperature, topK)
                        : model.generate(input, maxNewTokens, temperature, topK);
        if (TensorOpsGPU.isGpuAvailable()) {
            TensorOpsGPU.synchronizeStream();
            TensorOpsGPU.drainDeferredGpuBuffers();
        }
        float[] buf = generated.internalBuffer();
        int n = buf.length;
        while (n > 0 && buf[n - 1] == 0f) {
            n--;
        }
        int[] tokens = new int[n];
        for (int i = 0; i < n; i++) {
            tokens[i] = (int) buf[i];
        }
        String text = tokenizer.decode(tokens);
        // Нормализуем пробелы: модель часто генерирует двойные/тройные пробелы
        // (особенно после знаков пунктуации)
        return text.replaceAll("\\s{2,}", " ").trim();
    }
}
