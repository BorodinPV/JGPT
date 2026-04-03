package com.veles.llm.jgpt.data;

import com.veles.llm.jgpt.core.Tensor;

import java.io.BufferedReader;
import java.nio.FloatBuffer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * DataLoader для обучения LLM: текст → токены → последовательности длины {@code maxSeqLen+1} (causal LM).
 * <p>
 * Для фоновой подготовки следующего батча используйте {@link #buildBatchNoAdvance()} в отдельном потоке,
 * затем {@link #advanceAfterPreparedBatch()} на потоке обучения — эквивалентно {@link #nextBatch()}.
 * <p>
 * Батчевые {@link Tensor} и scratch-массивы id переиспользуются (двойная буферизация), чтобы не создавать новые
 * массивы на каждый вызов.
 */
public final class DataLoader {

    private static final Logger log = LoggerFactory.getLogger(DataLoader.class);

    private final BPETokenizer tokenizer;
    private final int maxSeqLen;
    private final int batchSize;
    private final List<int[]> sequences;
    private final Random random;
    private int currentIndex;
    /** {@code 0} — без ограничения; иначе не больше столько последовательностей (защита от OOM на огромных книгах). */
    private int maxSequences;

    /**
     * Переиспользование {@link Tensor} батча (два слота — безопасно при фоновом
     * {@link #buildBatchNoAdvance()} и последовательном {@link #nextBatch()}).
     */
    private Tensor batchInput0;
    private Tensor batchTarget0;
    private Tensor batchInput1;
    private Tensor batchTarget1;
    private int batchWriteSlot;

    private int[] scratchInputIds;
    private int[] scratchTargetIds;

    /**
     * {@code true} — батчевые {@link Tensor} на direct-буфере (см. {@link Tensor#allocateDirect(int[])}),
     * путь embedding на GPU без копии токенов в {@code float[]}. Включается: {@code JGPT_BATCH_DIRECT=1} или
     * {@code -Djgpt.batch.direct=true}.
     */
    private final boolean useDirectBatchBuffers;
    /** Page-locked host батч (CUDA); подразумевает direct-путь заполнения. */
    private final boolean usePinnedHostBatchBuffers;

    public DataLoader(BPETokenizer tokenizer, int maxSeqLen, int batchSize) {
        this(tokenizer, maxSeqLen, batchSize, resolveDirectBatchBuffers(), resolvePinnedHostBatchBuffers());
    }

    public DataLoader(BPETokenizer tokenizer, int maxSeqLen, int batchSize, boolean useDirectBatchBuffers) {
        this(tokenizer, maxSeqLen, batchSize, useDirectBatchBuffers, false);
    }

    public DataLoader(
            BPETokenizer tokenizer,
            int maxSeqLen,
            int batchSize,
            boolean useDirectBatchBuffers,
            boolean usePinnedHostBatchBuffers) {
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.sequences = new ArrayList<>();
        this.random = new Random(42);
        this.currentIndex = 0;
        this.maxSequences = 0;
        this.usePinnedHostBatchBuffers = usePinnedHostBatchBuffers;
        this.useDirectBatchBuffers = useDirectBatchBuffers || usePinnedHostBatchBuffers;
    }

    private static boolean resolvePinnedHostBatchBuffers() {
        if (Boolean.getBoolean("jgpt.batch.pinned")) {
            return true;
        }
        String e = System.getenv("JGPT_BATCH_PINNED");
        if (e != null) {
            String t = e.trim();
            return "1".equals(t) || "true".equalsIgnoreCase(t);
        }
        return false;
    }

    private static boolean resolveDirectBatchBuffers() {
        if (Boolean.getBoolean("jgpt.batch.direct")) {
            return true;
        }
        String e = System.getenv("JGPT_BATCH_DIRECT");
        if (e != null) {
            String t = e.trim();
            if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                return true;
            }
        }
        return false;
    }

    public boolean usesDirectBatchBuffers() {
        return useDirectBatchBuffers;
    }

    public boolean usesPinnedHostBatchBuffers() {
        return usePinnedHostBatchBuffers;
    }

    public void setMaxSequences(int max) {
        this.maxSequences = Math.max(0, max);
    }

    public void loadTextFile(String path) throws IOException {
        Path file = Path.of(path);
        String text =
                maxSequences > 0
                        ? readPrefixForMaxSequences(file)
                        : Files.readString(file);
        loadText(text);
    }

    /**
     * Читает только префикс файла, достаточный для {@link #maxSequences} окон.
     * Это не идеальная оценка, но она не держит в памяти всю книгу и останавливается,
     * как только токенов уже хватает с запасом.
     */
    private String readPrefixForMaxSequences(Path path) throws IOException {
        int targetTokens = maxSequences * (maxSeqLen + 1);
        int nextProbeChars = Math.max(32_768, targetTokens * 6);
        StringBuilder sb = new StringBuilder(nextProbeChars);
        char[] buf = new char[8192];
        boolean eof = false;

        try (BufferedReader reader = Files.newBufferedReader(path)) {
            while (!eof) {
                while (sb.length() < nextProbeChars) {
                    int toRead = Math.min(buf.length, nextProbeChars - sb.length());
                    int n = reader.read(buf, 0, toRead);
                    if (n < 0) {
                        eof = true;
                        break;
                    }
                    sb.append(buf, 0, n);
                }

                int[] probeTokens = tokenizer.encode(sb.toString(), true);
                if (probeTokens.length >= targetTokens + maxSeqLen || eof) {
                    log.info(
                            "  прочитан префикс {} символов (пробных токенов: {}) при maxSequences={}",
                            String.format("%,d", sb.length()),
                            String.format("%,d", probeTokens.length),
                            maxSequences);
                    return sb.toString();
                }

                nextProbeChars += Math.max(32_768, nextProbeChars / 2);
            }
        }

        return sb.toString();
    }

    public void loadText(String text) {
        if (tokenizer == null) {
            log.warn("Токенизатор не задан — загрузка текста пропущена.");
            return;
        }

        // 1. Токенизируем весь текст сразу
        int[] tokens = tokenizer.encode(text, true);
        log.info("  закодировано {} токенов из {} символов", tokens.length, text.length());

        // 2. Проверка длины
        if (tokens.length < maxSeqLen + 1) {
            log.warn(
                    "Текст слишком короткий: {} токенов (нужно минимум {})",
                    tokens.length,
                    maxSeqLen + 1);
            return;
        }

        // 3. Нарезаем на последовательности
        int count = 0;
        for (int i = 0; i + maxSeqLen < tokens.length; i += maxSeqLen) {
            if (maxSequences > 0 && count >= maxSequences) {
                log.warn(
                        "Достигнут лимит maxSequences={}: хвост книги в этот DataLoader не попал.",
                        maxSequences);
                break;
            }
            int[] seq = Arrays.copyOfRange(tokens, i, i + maxSeqLen + 1);
            sequences.add(seq);
            count++;
        }

        log.info("Собрано обучающих окон (последовательностей): {}", count);
    }

    public void shuffle() {
        Collections.shuffle(sequences, random);
        currentIndex = 0;
        log.info("Данные перемешаны (новый порядок батчей на эпоху).");
    }

    public boolean hasMore() {
        return currentIndex + batchSize <= sequences.size();
    }

    /**
     * Собирает батч с текущего {@code currentIndex} без сдвига указателя. Поток обучения не должен
     * вызывать другие методы, меняющие {@code currentIndex}, пока идёт подготовка (или используйте один
     * фоновый поток-консьюмер).
     *
     * @throws NoSuchElementException если {@link #hasMore()} ложно
     */
    public Batch buildBatchNoAdvance() {
        if (!hasMore()) {
            throw new NoSuchElementException("No more batches");
        }

        int flat = batchSize * maxSeqLen;
        if (scratchInputIds == null || scratchInputIds.length != flat) {
            scratchInputIds = new int[flat];
            scratchTargetIds = new int[flat];
        }

        for (int b = 0; b < batchSize; b++) {
            int[] seq = sequences.get(currentIndex + b);
            if (seq.length < maxSeqLen + 1) {
                throw new IllegalStateException("sequence length < maxSeqLen+1");
            }
            for (int i = 0; i < maxSeqLen; i++) {
                scratchInputIds[b * maxSeqLen + i] = seq[i];
                scratchTargetIds[b * maxSeqLen + i] = seq[i + 1];
            }
        }

        int slot = batchWriteSlot & 1;
        batchWriteSlot ^= 1;
        Tensor input;
        Tensor target;
        if (slot == 0) {
            if (batchInput0 == null) {
                int[] sh = new int[]{batchSize, maxSeqLen};
                batchInput0 = allocateBatchTensor(sh);
                batchTarget0 = allocateBatchTensor(sh);
            }
            input = batchInput0;
            target = batchTarget0;
        } else {
            if (batchInput1 == null) {
                int[] sh = new int[]{batchSize, maxSeqLen};
                batchInput1 = allocateBatchTensor(sh);
                batchTarget1 = allocateBatchTensor(sh);
            }
            input = batchInput1;
            target = batchTarget1;
        }

        if (useDirectBatchBuffers) {
            FloatBuffer fbIn = input.directFloatBuffer();
            FloatBuffer fbTg = target.directFloatBuffer();
            for (int i = 0; i < flat; i++) {
                fbIn.put(i, (float) scratchInputIds[i]);
                fbTg.put(i, (float) scratchTargetIds[i]);
            }
        } else {
            float[] inputData = input.internalBuffer();
            float[] targetData = target.internalBuffer();
            for (int i = 0; i < flat; i++) {
                inputData[i] = scratchInputIds[i];
                targetData[i] = scratchTargetIds[i];
            }
        }

        return new Batch(input, target);
    }

    private Tensor allocateBatchTensor(int[] shape) {
        if (usePinnedHostBatchBuffers) {
            return Tensor.allocatePinnedHost(shape);
        }
        if (useDirectBatchBuffers) {
            return Tensor.allocateDirect(shape);
        }
        return new Tensor(shape);
    }

    /**
     * Сдвигает указатель после батча, собранного {@link #buildBatchNoAdvance()}. Эквивалентно
     * второй половине {@link #nextBatch()}.
     */
    public void advanceAfterPreparedBatch() {
        if (currentIndex + batchSize > sequences.size()) {
            throw new IllegalStateException("advance past end of sequences");
        }
        currentIndex += batchSize;
    }

    public Batch nextBatch() {
        Batch b = buildBatchNoAdvance();
        advanceAfterPreparedBatch();
        return b;
    }

    public void reset() {
        currentIndex = 0;
    }

    /** Очистить последовательности (например перед загрузкой другой книги в тот же loader). */
    public void clear() {
        sequences.clear();
        currentIndex = 0;
        batchWriteSlot = 0;
        batchInput0 = null;
        batchTarget0 = null;
        batchInput1 = null;
        batchTarget1 = null;
    }

    /** Для сохранения позиции при eval между батчами обучения. */
    public int getCurrentIndex() {
        return currentIndex;
    }

    public void setCurrentIndex(int index) {
        if (index < 0 || index > sequences.size()) {
            throw new IllegalArgumentException("index out of range");
        }
        this.currentIndex = index;
    }

    public int numBatches() {
        return sequences.size() / batchSize;
    }

    public int numSequences() {
        return sequences.size();
    }

    public BPETokenizer getTokenizer() {
        return tokenizer;
    }

    public static final class Batch {
        public final Tensor input;
        public final Tensor target;

        public Batch(Tensor input, Tensor target) {
            this.input = input;
            this.target = target;
        }
    }
}
