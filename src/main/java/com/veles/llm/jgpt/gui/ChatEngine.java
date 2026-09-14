package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.model.DecodeSampling;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.model.KvCacheGpu;

/**
 * Модель в процессе GUI: загрузка весов, потоковая генерация (KV на VRAM) и сэмплинг. Все обращения к CUDA идут с
 * одного рабочего потока (cuBLAS у {@link TensorOpsGPU} инициализируется per-thread).
 */
final class ChatEngine {

    /** Одна реплика диалога. */
    record Turn(boolean user, String text) {}

    private final ProjectPaths paths;
    private final ExecutorService worker =
            Executors.newSingleThreadExecutor(
                    r -> {
                        Thread t = new Thread(r, "jgpt-gui-model");
                        t.setDaemon(true);
                        return t;
                    });

    private volatile ModelCatalog.ModelEntry loaded;
    private GPTModel model;
    private BPETokenizer tokenizer;
    private EnvPreset.ModelGeometry geometry;
    private final AtomicBoolean cancel = new AtomicBoolean();

    ChatEngine(ProjectPaths paths) {
        this.paths = paths;
    }

    ModelCatalog.ModelEntry loaded() {
        return loaded;
    }

    boolean isLoaded() {
        return loaded != null;
    }

    boolean hasRoleTokens() {
        BPETokenizer t = tokenizer;
        return t != null && t.hasChatRoleTokens();
    }

    void shutdown() {
        cancel.set(true);
        worker.submit(this::unloadNow);
        worker.shutdown();
    }

    CompletableFuture<String> load(ModelCatalog.ModelEntry entry) {
        return CompletableFuture.supplyAsync(
                () -> {
                    try {
                        return loadNow(entry);
                    } catch (Exception e) {
                        throw new IllegalStateException(e.getMessage(), e);
                    }
                },
                worker);
    }

    CompletableFuture<Void> unload() {
        return CompletableFuture.runAsync(this::unloadNow, worker);
    }

    void cancelGeneration() {
        cancel.set(true);
    }

    /**
     * Сгенерировать ответ. {@code onToken} получает приращения текста (в рабочем потоке — оборачивайте в
     * {@code Platform.runLater}). Возвращает полный текст ответа.
     */
    CompletableFuture<String> generate(
            List<Turn> history, boolean chatTemplate, DecodeSampling sampling, int maxNewTokens, Consumer<String> onToken) {
        cancel.set(false);
        return CompletableFuture.supplyAsync(
                () -> generateNow(history, chatTemplate, sampling, maxNewTokens, onToken), worker);
    }

    // ── рабочий поток ────────────────────────────────────────────────────

    private String loadNow(ModelCatalog.ModelEntry entry) throws IOException, ClassNotFoundException {
        TensorOpsGPU.requireCuda("JGPT GUI chat");
        unloadNow();
        Path tok = entry.tokenizer(paths);
        if (!Files.isRegularFile(tok)) {
            throw new IOException("нет токенизатора: " + tok);
        }
        BPETokenizer t = BPETokenizer.load(tok.toString());
        EnvPreset.ModelGeometry g = entry.preset().geometry();
        int vocab = t.getVocabSize();
        GPTModel m = new GPTModel(vocab, g.maxSeqLen(), g.dModel(), g.numHeads(), g.numLayers(), g.dIntermediate(), true);
        try {
            m.loadWeights(entry.file().toString());
        } catch (IOException | RuntimeException e) {
            m.closeGpuResidentWeights();
            throw e;
        }
        m.setDropout(0f, 0f, 0f);
        model = m;
        tokenizer = t;
        geometry = g;
        loaded = entry;
        return g.describe() + "  vocab=" + vocab + "  role_tokens=" + t.hasChatRoleTokens();
    }

    private void unloadNow() {
        GPTModel m = model;
        model = null;
        tokenizer = null;
        loaded = null;
        if (m != null) {
            try {
                m.closeGpuResidentWeights();
            } catch (RuntimeException _) {
                // освобождение VRAM best-effort
            }
            if (TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.synchronizeStream();
                TensorOpsGPU.drainDeferredGpuBuffers();
            }
        }
    }

    private String generateNow(
            List<Turn> history,
            boolean chatTemplate,
            DecodeSampling sampling,
            int maxNewTokens,
            Consumer<String> onToken) {
        GPTModel m = model;
        BPETokenizer t = tokenizer;
        if (m == null || t == null) {
            throw new IllegalStateException("модель не загружена");
        }
        if (t.hasChatRoleTokens()) {
            m.setExtraGenerationStopTokens(t.userId(), t.assistantId());
        } else {
            m.setExtraGenerationStopTokens();
        }
        int maxSeq = geometry.maxSeqLen();
        int maxNew = Math.max(1, Math.min(maxNewTokens, maxSeq - 8));
        int[] prompt = buildPrompt(t, history, chatTemplate);
        int keep = Math.min(prompt.length, maxSeq - maxNew);
        if (keep < prompt.length) {
            int[] cut = new int[keep];
            cut[0] = prompt[0];
            System.arraycopy(prompt, prompt.length - keep + 1, cut, 1, keep - 1);
            prompt = cut;
        }

        Tensor input = new Tensor(new int[] {1, prompt.length});
        float[] in = input.internalBuffer();
        float[] context = new float[prompt.length + maxNew];
        for (int i = 0; i < prompt.length; i++) {
            in[i] = prompt[i];
            context[i] = prompt[i];
        }
        int dHead = geometry.dModel() / geometry.numHeads();
        List<Integer> generated = new ArrayList<>();
        int len = prompt.length;
        String emitted = "";
        try (KvCacheGpu cache = new KvCacheGpu(geometry.numLayers(), geometry.numHeads(), dHead, maxSeq)) {
            Tensor logits = m.forwardPrefill(input, cache, 0);
            float[] row = logits.internalBuffer();
            int vocab = t.getVocabSize();
            int offset = (prompt.length - 1) * vocab;
            Tensor one = new Tensor(new int[] {1, 1});
            for (int j = 0; j < maxNew && !cancel.get(); j++) {
                int next = m.sampleNextToken(row, offset, sampling, context, len);
                if (m.isGenerationStopToken(next)) {
                    break;
                }
                context[len++] = next;
                generated.add(next);
                String full = t.decode(toArray(generated));
                if (full.length() > emitted.length() && full.startsWith(emitted)) {
                    onToken.accept(full.substring(emitted.length()));
                    emitted = full;
                } else if (!full.equals(emitted)) {
                    onToken.accept("\u0000" + full);
                    emitted = full;
                }
                if (len >= maxSeq) {
                    break;
                }
                one.internalBuffer()[0] = next;
                logits = m.forwardDecode(one, cache, cache.length(), len - 1);
                row = logits.internalBuffer();
                offset = 0;
            }
        } finally {
            if (TensorOpsGPU.isGpuAvailable()) {
                TensorOpsGPU.synchronizeStream();
                TensorOpsGPU.drainDeferredGpuBuffers();
            }
        }
        return emitted.replaceAll("\\s{2,}", " ").trim();
    }

    /** {@code <bos>} + ({@code <user>} текст {@code <assistant>} текст …) как в SFT; без шаблона — просто продолжение. */
    private static int[] buildPrompt(BPETokenizer t, List<Turn> history, boolean chatTemplate) {
        List<Integer> ids = new ArrayList<>();
        ids.add(t.bosId());
        if (chatTemplate && t.hasChatRoleTokens()) {
            for (Turn turn : history) {
                ids.add(turn.user() ? t.userId() : t.assistantId());
                for (int id : t.encode(turn.text(), false, false)) {
                    ids.add(id);
                }
            }
            ids.add(t.assistantId());
        } else {
            StringBuilder sb = new StringBuilder();
            for (Turn turn : history) {
                if (!sb.isEmpty()) {
                    sb.append('\n');
                }
                sb.append(turn.text());
            }
            for (int id : t.encode(sb.toString(), false, false)) {
                ids.add(id);
            }
        }
        return toArray(ids);
    }

    private static int[] toArray(List<Integer> l) {
        int[] a = new int[l.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = l.get(i);
        }
        return a;
    }
}
