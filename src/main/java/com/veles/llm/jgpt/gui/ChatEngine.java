package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.data.BPETokenizer;
import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.model.KvCacheGpu;

/**
 * Модель в процессе GUI: загрузка весов, потоковая генерация (KV на VRAM) и сэмплинг. Все обращения к CUDA идут с
 * одного рабочего потока (cuBLAS у {@link TensorOpsGPU} инициализируется per-thread).
 */
final class ChatEngine {

    /** Параметры сэмплинга. */
    record Sampling(float temperature, int topK, float topP, float repetitionPenalty, int maxNewTokens) {}

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
    CompletableFuture<String> generate(List<Turn> history, boolean chatTemplate, Sampling s, Consumer<String> onToken) {
        cancel.set(false);
        return CompletableFuture.supplyAsync(() -> generateNow(history, chatTemplate, s, onToken), worker);
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

    private String generateNow(List<Turn> history, boolean chatTemplate, Sampling s, Consumer<String> onToken) {
        GPTModel m = model;
        BPETokenizer t = tokenizer;
        if (m == null || t == null) {
            throw new IllegalStateException("модель не загружена");
        }
        int maxSeq = geometry.maxSeqLen();
        int maxNew = Math.max(1, Math.min(s.maxNewTokens(), maxSeq - 8));
        int[] prompt = buildPrompt(t, history, chatTemplate);
        int keep = Math.min(prompt.length, maxSeq - maxNew);
        if (keep < prompt.length) {
            // контекст длиннее окна — оставляем хвост, но <bos> в начале сохраняем
            int[] cut = new int[keep];
            cut[0] = prompt[0];
            System.arraycopy(prompt, prompt.length - keep + 1, cut, 1, keep - 1);
            prompt = cut;
        }
        int[] stops = t.hasChatRoleTokens()
                ? new int[] {t.eosId(), t.userId(), t.assistantId()}
                : new int[] {t.eosId()};

        Tensor input = new Tensor(new int[] {1, prompt.length});
        float[] in = input.internalBuffer();
        for (int i = 0; i < prompt.length; i++) {
            in[i] = prompt[i];
        }
        int dHead = geometry.dModel() / geometry.numHeads();
        List<Integer> generated = new ArrayList<>();
        int[] context = Arrays.copyOf(prompt, prompt.length + maxNew);
        int len = prompt.length;
        String emitted = "";
        try (KvCacheGpu cache = new KvCacheGpu(geometry.numLayers(), geometry.numHeads(), dHead, maxSeq)) {
            Tensor logits = m.forwardPrefill(input, cache, 0);
            float[] row = logits.internalBuffer();
            int vocab = t.getVocabSize();
            int offset = (prompt.length - 1) * vocab;
            Tensor one = new Tensor(new int[] {1, 1});
            for (int j = 0; j < maxNew && !cancel.get(); j++) {
                int next = sample(row, offset, vocab, s, context, len);
                if (contains(stops, next)) {
                    break;
                }
                context[len++] = next;
                generated.add(next);
                String full = t.decode(toArray(generated));
                if (full.length() > emitted.length() && full.startsWith(emitted)) {
                    onToken.accept(full.substring(emitted.length()));
                    emitted = full;
                } else if (!full.equals(emitted)) {
                    onToken.accept("\u0000" + full); // декодер переписал хвост — заменить весь текст
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
        return emitted;
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

    private static boolean contains(int[] a, int v) {
        for (int x : a) {
            if (x == v) {
                return true;
            }
        }
        return false;
    }

    // ── сэмплинг ─────────────────────────────────────────────────────────

    private static int sample(float[] logits, int offset, int vocab, Sampling s, int[] context, int len) {
        float[] z = new float[vocab];
        System.arraycopy(logits, offset, z, 0, vocab);

        if (s.repetitionPenalty() > 1f) {
            int from = Math.max(0, len - 256);
            for (int i = from; i < len; i++) {
                int tok = context[i];
                if (tok >= 0 && tok < vocab) {
                    z[tok] = z[tok] > 0 ? z[tok] / s.repetitionPenalty() : z[tok] * s.repetitionPenalty();
                }
            }
        }
        if (s.temperature() <= 0f) {
            return argmax(z);
        }
        for (int i = 0; i < vocab; i++) {
            z[i] /= s.temperature();
        }
        Integer[] idx = new Integer[vocab];
        for (int i = 0; i < vocab; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Float.compare(z[b], z[a]));
        int n = vocab;
        if (s.topK() > 0) {
            n = Math.min(n, s.topK());
        }
        double max = z[idx[0]];
        double[] p = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            p[i] = Math.exp(z[idx[i]] - max);
            sum += p[i];
        }
        if (s.topP() < 1f) {
            double acc = 0;
            int cut = n;
            for (int i = 0; i < n; i++) {
                acc += p[i] / sum;
                if (acc >= s.topP()) {
                    cut = i + 1;
                    break;
                }
            }
            n = cut;
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += p[i];
            }
        }
        double r = ThreadLocalRandom.current().nextDouble() * sum;
        double acc = 0;
        for (int i = 0; i < n; i++) {
            acc += p[i];
            if (r <= acc) {
                return idx[i];
            }
        }
        return idx[n - 1];
    }

    private static int argmax(float[] z) {
        int best = 0;
        for (int i = 1; i < z.length; i++) {
            if (z[i] > z[best]) {
                best = i;
            }
        }
        return best;
    }
}
