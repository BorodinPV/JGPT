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
import java.util.Locale;
import java.util.Objects;
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
    /**
     * Параллельно {@link #sequences}: цели длины {@code maxSeqLen}, {@code -1} = не считать CE (SFT-маска).
     * {@code null} — обычный LM ({@code target[i] = seq[i+1]}).
     */
    private List<int[]> sftTargets;
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
        this.random = new Random(resolveTrainShuffleSeed());
        this.currentIndex = 0;
        this.maxSequences = 0;
        this.usePinnedHostBatchBuffers = usePinnedHostBatchBuffers;
        this.useDirectBatchBuffers = useDirectBatchBuffers || usePinnedHostBatchBuffers;
    }

    /**
     * Seed для {@link #shuffle()} на каждой эпохе (и для {@link Collections#shuffle} списка окон).
     * Env {@code JGPT_TRAIN_SHUFFLE_SEED}; не задано или неверный формат — {@code 42}.
     */
    private static long resolveTrainShuffleSeed() {
        String e = System.getenv("JGPT_TRAIN_SHUFFLE_SEED");
        if (e == null || e.isBlank()) {
            return 42L;
        }
        try {
            return Long.parseLong(e.trim());
        } catch (NumberFormatException _) {
            return 42L;
        }
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
            while (true) {
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
    }

    /**
     * Принимает уже закодированный массив токенов и нарезает его на окна обучения.
     * Используется при параллельном кодировании в {@code AllBooksTrain}.
     */
    public void loadTokens(int[] tokens) {
        if (sftTargets != null) {
            throw new IllegalStateException("cannot mix LM tokens with SFT windows in one DataLoader");
        }
        if (tokens.length < maxSeqLen + 1) {
            log.warn(
                    "Текст слишком короткий: {} токенов (нужно минимум {})",
                    tokens.length,
                    maxSeqLen + 1);
            return;
        }
        int count = 0;
        for (int i = 0; i + maxSeqLen < tokens.length; i += maxSeqLen) {
            if (maxSequences > 0 && count >= maxSequences) {
                log.warn(
                        "Достигнут лимит maxSequences={}: хвост книги в этот DataLoader не попал.",
                        maxSequences);
                break;
            }
            sequences.add(Arrays.copyOfRange(tokens, i, i + maxSeqLen + 1));
            count++;
        }
    }

    /**
     * Режет документы в LM-окна так же, как {@link #loadTokens(int[])} по их конкатенации ({@code <eos><bos>}
     * внутри окна), без второго гигантского {@code int[]} на весь корпус. После каждого взятого документа слот
     * в {@code docs} обнуляется, чтобы GC мог отдать память пока режутся остальные.
     *
     * @param isVal параллельно {@code docs}; {@code takeVal=false} — train ({@code !isVal[i]})
     * @return сумма длин взятых документов (как у concat-потока, включая хвост короче окна)
     */
    public long loadPackedDocuments(List<int[]> docs, boolean[] isVal, boolean takeVal) {
        if (sftTargets != null) {
            throw new IllegalStateException("cannot mix LM tokens with SFT windows in one DataLoader");
        }
        if (docs == null || isVal == null || docs.size() != isVal.length) {
            throw new IllegalArgumentException("docs/isVal size mismatch");
        }
        int[] carry = new int[maxSeqLen];
        int carryLen = 0;
        long tokens = 0;
        int windows = 0;
        for (int i = 0; i < docs.size(); i++) {
            if (isVal[i] != takeVal) {
                continue;
            }
            int[] d = docs.get(i);
            if (d == null) {
                continue;
            }
            tokens += d.length;
            carryLen = absorbPackedDoc(d, carry, carryLen);
            windows = sequences.size();
            docs.set(i, null);
            if (maxSequences > 0 && windows >= maxSequences) {
                log.warn(
                        "Достигнут лимит maxSequences={}: хвост потока в этот DataLoader не попал.",
                        maxSequences);
                break;
            }
        }
        return tokens;
    }

    /**
     * Дописывает документ к незакрытому префиксу потока ({@code carry}) и эмитит окна длины
     * {@code maxSeqLen+1} со stride {@code maxSeqLen}.
     */
    private int absorbPackedDoc(int[] doc, int[] carry, int carryLen) {
        int pos = 0;
        int n = doc.length;
        while (pos < n) {
            if (maxSequences > 0 && sequences.size() >= maxSequences) {
                return 0;
            }
            int have = n - pos;
            if (carryLen + have < maxSeqLen + 1) {
                System.arraycopy(doc, pos, carry, carryLen, have);
                return carryLen + have;
            }
            int room = maxSeqLen + 1 - carryLen;
            int[] win = new int[maxSeqLen + 1];
            System.arraycopy(carry, 0, win, 0, carryLen);
            System.arraycopy(doc, pos, win, carryLen, room);
            sequences.add(win);
            pos += room;
            carry[0] = win[maxSeqLen];
            carryLen = 1;
        }
        return carryLen;
    }

    /**
     * Одно SFT-окно: {@code tokens.length == maxSeqLen+1}, {@code targets.length == maxSeqLen},
     * {@code targets[i] == -1} — позиция не входит в CE.
     */
    public void loadSftWindow(int[] tokens, int[] targets) {
        if (tokens == null || tokens.length != maxSeqLen + 1) {
            throw new IllegalArgumentException("SFT tokens must have length maxSeqLen+1");
        }
        if (targets == null || targets.length != maxSeqLen) {
            throw new IllegalArgumentException("SFT targets must have length maxSeqLen");
        }
        if (sftTargets == null) {
            if (!sequences.isEmpty()) {
                throw new IllegalStateException("cannot mix SFT windows with LM sequences in one DataLoader");
            }
            sftTargets = new ArrayList<>();
        }
        sequences.add(tokens);
        sftTargets.add(targets);
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

    /** Как {@link #shuffle(boolean)} с одной строкой в лог. */
    public void shuffle() {
        shuffle(true);
    }

    /**
     * Перемешать последовательности и сбросить указатель батча.
     *
     * @param logInfo если {@code false} — без INFO (для серии shuffle при resume чекпоинта, см. {@code LLMTrainer}).
     */
    public void shuffle(boolean logInfo) {
        shufflePaired(sequences, sftTargets, random);
        currentIndex = 0;
        if (logInfo) {
            log.info("Данные перемешаны (новый порядок батчей на эпоху).");
        }
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
            int[] sftTgt = sftTargets != null ? sftTargets.get(currentIndex + b) : null;
            for (int i = 0; i < maxSeqLen; i++) {
                scratchInputIds[b * maxSeqLen + i] = seq[i];
                scratchTargetIds[b * maxSeqLen + i] = sftTgt != null ? sftTgt[i] : seq[i + 1];
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
        if (sftTargets != null) {
            sftTargets.clear();
            sftTargets = null;
        }
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

    public int getMaxSeqLen() {
        return maxSeqLen;
    }

    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Копия списка окон (те же {@code int[]} — без клонирования массивов), для разбиения train/val.
     */
    public List<int[]> copySequences() {
        return new ArrayList<>(sequences);
    }

    public boolean hasSftTargets() {
        return sftTargets != null;
    }

    List<int[]> copySftTargetsOrNull() {
        return sftTargets == null ? null : new ArrayList<>(sftTargets);
    }

    private static void shufflePaired(List<int[]> seqs, List<int[]> tgts, Random random) {
        if (tgts == null) {
            Collections.shuffle(seqs, random);
            return;
        }
        if (tgts.size() != seqs.size()) {
            throw new IllegalStateException("SFT targets size != sequences");
        }
        List<Integer> order = new ArrayList<>(seqs.size());
        for (int i = 0; i < seqs.size(); i++) {
            order.add(i);
        }
        Collections.shuffle(order, random);
        List<int[]> ns = new ArrayList<>(seqs.size());
        List<int[]> nt = new ArrayList<>(tgts.size());
        for (int i : order) {
            ns.add(seqs.get(i));
            nt.add(tgts.get(i));
        }
        seqs.clear();
        seqs.addAll(ns);
        tgts.clear();
        tgts.addAll(nt);
    }

    /**
     * Детерминированное разбиение на train и validation: перемешивание по {@code seed}, доля val —
     * {@code valFraction} от числа окон. Исходный loader очищается (освобождает ссылки на окна).
     *
     * <p>Если окон слишком мало для хотя бы одного полного val-батча ({@code >= batchSize}), возвращается
     * исходный loader как train и {@code validation == null}.
     *
     * @param valFraction доля окон под validation, {@code (0, 0.5]}
     */
    public static TrainValSplit splitTrainValidation(DataLoader source, double valFraction, long seed) {
        Objects.requireNonNull(source, "source");
        if (valFraction <= 0d || valFraction >= 0.5d) {
            return new TrainValSplit(source, null);
        }
        List<int[]> all = source.copySequences();
        List<int[]> allTgt = source.copySftTargetsOrNull();
        source.clear();
        int n = all.size();
        int bs = source.batchSize;
        if (n < 2 * bs) {
            log.warn(
                    "splitTrainValidation: окон {} < 2×batch ({}), hold-out отключён — всё в train",
                    n,
                    bs);
            DataLoader train = fromSequencesTemplate(source, all, allTgt);
            return new TrainValSplit(train, null);
        }
        List<Integer> order = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            order.add(i);
        }
        Collections.shuffle(order, new Random(seed));
        List<int[]> shuffled = new ArrayList<>(n);
        List<int[]> shuffledTgt = allTgt == null ? null : new ArrayList<>(n);
        for (int i : order) {
            shuffled.add(all.get(i));
            if (shuffledTgt != null) {
                shuffledTgt.add(allTgt.get(i));
            }
        }
        int nVal = (int) Math.round(n * valFraction);
        nVal = Math.min(nVal, n - bs);
        if (nVal < bs) {
            log.warn(
                    "splitTrainValidation: после доли {} val-окон {} < batch {}, hold-out отключён",
                    String.format(Locale.ROOT, "%.4f", valFraction),
                    nVal,
                    bs);
            DataLoader train = fromSequencesTemplate(source, shuffled, shuffledTgt);
            return new TrainValSplit(train, null);
        }
        List<int[]> valSeq = new ArrayList<>(shuffled.subList(0, nVal));
        List<int[]> trainSeq = new ArrayList<>(shuffled.subList(nVal, n));
        List<int[]> valTgt =
                shuffledTgt == null ? null : new ArrayList<>(shuffledTgt.subList(0, nVal));
        List<int[]> trainTgt =
                shuffledTgt == null ? null : new ArrayList<>(shuffledTgt.subList(nVal, n));
        DataLoader train = fromSequencesTemplate(source, trainSeq, trainTgt);
        DataLoader val = fromSequencesTemplate(source, valSeq, valTgt);
        return new TrainValSplit(train, val);
    }

    private static DataLoader fromSequencesTemplate(
            DataLoader template, List<int[]> seqs, List<int[]> sftTgts) {
        DataLoader d =
                new DataLoader(
                        template.getTokenizer(),
                        template.getMaxSeqLen(),
                        template.getBatchSize(),
                        template.usesDirectBatchBuffers(),
                        template.usesPinnedHostBatchBuffers());
        d.sequences.addAll(seqs);
        if (sftTgts != null) {
            d.sftTargets = new ArrayList<>(sftTgts);
        }
        return d;
    }

    /** Результат {@link #splitTrainValidation(DataLoader, double, long)}. */
    public static final class TrainValSplit {
        public final DataLoader train;
        /** {@code null}, если hold-out отключён или невозможен. */
        public final DataLoader validation;

        public TrainValSplit(DataLoader train, DataLoader validation) {
            this.train = Objects.requireNonNull(train, "train");
            this.validation = validation;
        }
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
