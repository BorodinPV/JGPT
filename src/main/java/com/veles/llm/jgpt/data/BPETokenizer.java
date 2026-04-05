package com.veles.llm.jgpt.data;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * BPE (Byte Pair Encoding): обучение на частотах пар символов, кодирование по списку merge.
 */
public final class BPETokenizer {

    private static final Logger log = LoggerFactory.getLogger(BPETokenizer.class);

    public static final String PAD_TOKEN = "<pad>";
    public static final String UNK_TOKEN = "<unk>";
    public static final String BOS_TOKEN = "<bos>";
    public static final String EOS_TOKEN = "<eos>";

    private static final String WORD_END = "</w>";
    /** Разделитель символов в ключе слова (не может совпасть с обычным символом текста). */
    private static final String SYM_SEP = "\u001F";
    /**
     * Unicode-aware token pattern:
     * - слова (буквы/цифры/подчёркивание) как цельный токен
     * - знаки пунктуации отдельно
     * - пробелы отдельно
     */
    private static final Pattern WORD_PATTERN = Pattern.compile("(?U)\\w+|[^\\w\\s]|\\s+");

    /** С этого числа уникальных словоформ считаем пары и применяем merge параллельно по корпусу. */
    private static final int PARALLEL_CORPUS_MIN = 256;

    /** Одна словоформа в обучении: последовательность символов + счётчик вхождений. */
    private static final class WordCount {
        final ArrayList<String> symbols;
        final int freq;

        WordCount(ArrayList<String> symbols, int freq) {
            this.symbols = symbols;
            this.freq = freq;
        }
    }

    private final Map<String, Integer> tokenToIdMap;
    private final Map<Integer, String> idToTokenMap;
    private final List<String[]> merges;
    private final Pattern wordPattern;
    private final int targetVocabSize;

    /**
     * Индекс слияния по паре «left\u0000right» → rank (0 = наивысший приоритет).
     * Позволяет заменить O(M×W) перебор всех слияний на O(W²) поиск минимального ранга.
     * Перестраивается из {@link #merges} при конструировании и после загрузки.
     */
    private Map<String, Integer> mergeRanks;

    public BPETokenizer(int vocabSize) {
        this.targetVocabSize = vocabSize;
        this.tokenToIdMap = new HashMap<>();
        this.idToTokenMap = new HashMap<>();
        this.merges = new ArrayList<>();
        this.wordPattern = WORD_PATTERN;
        this.mergeRanks = new HashMap<>();

        addSpecialToken(PAD_TOKEN, 0);
        addSpecialToken(UNK_TOKEN, 1);
        addSpecialToken(BOS_TOKEN, 2);
        addSpecialToken(EOS_TOKEN, 3);
    }

    private BPETokenizer(
            int targetVocabSize,
            Map<String, Integer> stoi,
            Map<Integer, String> itos,
            List<String[]> merges) {
        this.targetVocabSize = targetVocabSize;
        this.tokenToIdMap = new HashMap<>(stoi);
        this.idToTokenMap = new HashMap<>(itos);
        this.merges = new ArrayList<>(merges);
        this.wordPattern = WORD_PATTERN;
        this.mergeRanks = buildMergeRanks(this.merges);
    }

    private static Map<String, Integer> buildMergeRanks(List<String[]> merges) {
        Map<String, Integer> ranks = new HashMap<>(merges.size() * 2);
        for (int i = 0; i < merges.size(); i++) {
            ranks.put(merges.get(i)[0] + '\0' + merges.get(i)[1], i);
        }
        return ranks;
    }

    private void addSpecialToken(String token, int id) {
        tokenToIdMap.put(token, id);
        idToTokenMap.put(id, token);
    }

    private void addTokenIfRoom(String token, int vocabSize) {
        if (tokenToIdMap.containsKey(token) || tokenToIdMap.size() >= vocabSize) {
            return;
        }
        int id = tokenToIdMap.size();
        tokenToIdMap.put(token, id);
        idToTokenMap.put(id, token);
    }

    public static BPETokenizer train(List<String> texts, int vocabSize) {
        BPETokenizer tokenizer = new BPETokenizer(vocabSize);

        Map<String, Integer> wordFreqs = new HashMap<>();
        for (String text : texts) {
            String normalized = text.toLowerCase();
            String[] words =
                    tokenizer.wordPattern.matcher(normalized).results()
                            .map(MatchResult::group)
                            .toArray(String[]::new);

            for (String word : words) {
                if (word.isEmpty() || word.isBlank()) {
                    continue;
                }
                String wordKey = charsToWordKey(word);
                wordFreqs.merge(wordKey, 1, Integer::sum);
            }
        }

        log.info("Уникальных слов (после нормализации): {}", wordFreqs.size());

        Set<String> vocab = new HashSet<>();
        for (String word : wordFreqs.keySet()) {
            for (String t : word.split(SYM_SEP, -1)) {
                if (!t.isEmpty()) {
                    vocab.add(t);
                }
            }
        }

        int currentSize = tokenizer.tokenToIdMap.size();
        for (String token : vocab) {
            if (currentSize >= vocabSize) {
                break;
            }
            if (!tokenizer.tokenToIdMap.containsKey(token)) {
                int id = tokenizer.tokenToIdMap.size();
                tokenizer.tokenToIdMap.put(token, id);
                tokenizer.idToTokenMap.put(id, token);
                currentSize++;
            }
        }

        tokenizer.addTokenIfRoom(" ", vocabSize);
        tokenizer.addTokenIfRoom(WORD_END, vocabSize);

        log.info("Начальный размер словаря (символы/части слов): {}", tokenizer.tokenToIdMap.size());

        List<WordCount> corpus = tokenizer.buildCorpus(wordFreqs);
        Map<String, Integer> pairFreqs = tokenizer.getPairFrequencies(corpus);
        int numMerges = Math.max(0, vocabSize - tokenizer.tokenToIdMap.size());

        for (int mergeIdx = 0; mergeIdx < numMerges; ) {
            String bestPair = null;
            int maxFreq = -1;
            for (Map.Entry<String, Integer> entry : pairFreqs.entrySet()) {
                if (entry.getValue() > maxFreq) {
                    maxFreq = entry.getValue();
                    bestPair = entry.getKey();
                }
            }

            if (bestPair == null || maxFreq <= 1) {
                break;
            }

            String[] parts = bestPair.split(SYM_SEP, 2);
            if (parts.length != 2) {
                break;
            }
            String a = parts[0];
            String b = parts[1];
            if (a.isEmpty() || b.isEmpty()) {
                pairFreqs.remove(bestPair);
                if (pairFreqs.isEmpty()) {
                    break;
                }
                continue;
            }
            String merged = a + b;

            if (tokenizer.tokenToIdMap.size() >= vocabSize) {
                break;
            }

            int newId = tokenizer.tokenToIdMap.size();
            tokenizer.tokenToIdMap.put(merged, newId);
            tokenizer.idToTokenMap.put(newId, merged);
            tokenizer.merges.add(new String[] {a, b});
            tokenizer.mergeRanks.put(a + '\0' + b, tokenizer.merges.size() - 1);

            tokenizer.applyMergeToCorpus(corpus, a, b, merged);
            pairFreqs = tokenizer.getPairFrequencies(corpus);

            if (mergeIdx % 100 == 0) {
                log.info(
                        "  слияние {}/{}: «{}» + «{}» → «{}» (частота пары {})",
                        mergeIdx,
                        numMerges,
                        a,
                        b,
                        merged,
                        maxFreq);
            }
            mergeIdx++;
        }

        log.info("Итоговый размер словаря после BPE: {}", tokenizer.tokenToIdMap.size());
        return tokenizer;
    }

    private static String charsToWordKey(String word) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < word.length(); i++) {
            if (i > 0) {
                sb.append(SYM_SEP);
            }
            sb.append(word.charAt(i));
        }
        sb.append(SYM_SEP);
        sb.append(WORD_END);
        return sb.toString();
    }

    public int[] encode(String text, boolean addSpecialTokens) {
        String normalized = text.toLowerCase();
        String[] words =
                wordPattern.matcher(normalized).results()
                        .map(m -> m.group())
                        .toArray(String[]::new);

        List<Integer> tokens = new ArrayList<>();

        if (addSpecialTokens) {
            tokens.add(tokenToIdMap.get(BOS_TOKEN));
        }

        for (String word : words) {
            if (word.isEmpty()) {
                continue;
            }
            List<String> wordTokens = new ArrayList<>();
            for (int i = 0; i < word.length(); i++) {
                wordTokens.add(String.valueOf(word.charAt(i)));
            }
            wordTokens.add(WORD_END);

            applyMerges(wordTokens);

            for (String token : wordTokens) {
                Integer id = tokenToIdMap.get(token);
                if (id == null) {
                    tokens.add(tokenToIdMap.get(UNK_TOKEN));
                } else {
                    tokens.add(id);
                }
            }
        }

        if (addSpecialTokens) {
            tokens.add(tokenToIdMap.get(EOS_TOKEN));
        }

        int[] result = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            result[i] = tokens.get(i);
        }
        return result;
    }

    /** Применяет merge в том же порядке, что при обучении. */
    private void applyMerges(List<String> wordTokens) {
        // Быстрый BPE: ищем пару с минимальным рангом слияния и применяем,
        // пока есть применимые слияния. O(W²) на слово вместо O(M×W).
        while (wordTokens.size() > 1) {
            int minRank = Integer.MAX_VALUE;
            int minIdx = -1;
            for (int i = 0; i < wordTokens.size() - 1; i++) {
                Integer rank = mergeRanks.get(wordTokens.get(i) + '\0' + wordTokens.get(i + 1));
                if (rank != null && rank < minRank) {
                    minRank = rank;
                    minIdx = i;
                }
            }
            if (minIdx == -1) {
                break;
            }
            wordTokens.set(minIdx, wordTokens.get(minIdx) + wordTokens.get(minIdx + 1));
            wordTokens.remove(minIdx + 1);
        }
    }

    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        boolean pendingSpace = false;
        for (int tokenId : tokens) {
            String token = idToTokenMap.get(tokenId);
            if (token == null) {
                continue;
            }

            if (token.equals(PAD_TOKEN) || token.equals(BOS_TOKEN) || token.equals(EOS_TOKEN)) {
                continue;
            }
            if (token.equals(UNK_TOKEN)) {
                if (pendingSpace && sb.length() > 0) {
                    sb.append(' ');
                }
                sb.append('?');
                pendingSpace = false;
                continue;
            }

            boolean wordEnd = token.endsWith(WORD_END);
            String core = wordEnd ? token.substring(0, token.length() - WORD_END.length()) : token;
            // После </w> ждём пробел перед следующим «словом». Если следующий токен сам — пробельная
            // словоформа (пробелы как отдельный матч \\s+), в core уже есть пробел — не дублировать.
            if (pendingSpace && sb.length() > 0) {
                if (core.isEmpty() || !Character.isWhitespace(core.charAt(0))) {
                    sb.append(' ');
                }
            }
            sb.append(core);
            pendingSpace = wordEnd;
        }
        return sb.toString();
    }

    public void save(String path) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(tokenToIdMap);
            out.writeObject(idToTokenMap);
            out.writeObject(merges);
            out.writeInt(targetVocabSize);
        }
    }

    public static BPETokenizer load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            @SuppressWarnings("unchecked")
            Map<String, Integer> stoi = (Map<String, Integer>) in.readObject();
            @SuppressWarnings("unchecked")
            Map<Integer, String> itos = (Map<Integer, String>) in.readObject();
            @SuppressWarnings("unchecked")
            List<String[]> mergeList = (List<String[]>) in.readObject();
            int savedTarget = in.readInt();
            return new BPETokenizer(savedTarget, stoi, itos, mergeList);
        }
    }

    public int getVocabSize() {
        return tokenToIdMap.size();
    }

    public int getTargetVocabSize() {
        return targetVocabSize;
    }

    public int tokenToId(String token) {
        return tokenToIdMap.getOrDefault(token, tokenToIdMap.get(UNK_TOKEN));
    }

    public String idToToken(int id) {
        return idToTokenMap.getOrDefault(id, UNK_TOKEN);
    }

    private List<WordCount> buildCorpus(Map<String, Integer> wordFreqs) {
        List<WordCount> corpus = new ArrayList<>(wordFreqs.size());
        for (Map.Entry<String, Integer> e : wordFreqs.entrySet()) {
            corpus.add(wordKeyToWordCount(e.getKey(), e.getValue()));
        }
        return corpus;
    }

    private static WordCount wordKeyToWordCount(String wordKey, int freq) {
        String[] parts = wordKey.split(SYM_SEP, -1);
        ArrayList<String> sym = new ArrayList<>(parts.length);
        for (String p : parts) {
            if (!p.isEmpty()) {
                sym.add(p);
            }
        }
        return new WordCount(sym, freq);
    }

    /**
     * Частоты смежных пар по текущей сегментации корпуса. Параллельно при большом числе уникальных слов.
     */
    private Map<String, Integer> getPairFrequencies(List<WordCount> corpus) {
        if (corpus.size() >= PARALLEL_CORPUS_MIN) {
            ConcurrentHashMap<String, Integer> pairs = new ConcurrentHashMap<>();
            corpus.parallelStream()
                    .forEach(
                            wc -> {
                                ArrayList<String> syms = wc.symbols;
                                int n = syms.size();
                                int f = wc.freq;
                                for (int i = 0; i < n - 1; i++) {
                                    String pair = syms.get(i) + SYM_SEP + syms.get(i + 1);
                                    pairs.merge(pair, f, Integer::sum);
                                }
                            });
            return pairs;
        }
        Map<String, Integer> pairs = new HashMap<>();
        for (WordCount wc : corpus) {
            ArrayList<String> syms = wc.symbols;
            int n = syms.size();
            for (int i = 0; i < n - 1; i++) {
                String pair = syms.get(i) + SYM_SEP + syms.get(i + 1);
                pairs.merge(pair, wc.freq, Integer::sum);
            }
        }
        return pairs;
    }

    /** Применить merge (a,b)→merged ко всем словоформам: один проход слева направу без перекрытий. */
    private void applyMergeToCorpus(List<WordCount> corpus, String a, String b, String merged) {
        if (corpus.size() >= PARALLEL_CORPUS_MIN) {
            corpus.parallelStream().forEach(wc -> applyMergeToSymbols(wc.symbols, a, b, merged));
        } else {
            for (WordCount wc : corpus) {
                applyMergeToSymbols(wc.symbols, a, b, merged);
            }
        }
    }

    private static void applyMergeToSymbols(ArrayList<String> syms, String a, String b, String merged) {
        int w = 0;
        int r = 0;
        int n = syms.size();
        while (r < n) {
            if (r + 1 < n && a.equals(syms.get(r)) && b.equals(syms.get(r + 1))) {
                syms.set(w++, merged);
                r += 2;
            } else {
                syms.set(w++, syms.get(r++));
            }
        }
        if (w < n) {
            syms.subList(w, n).clear();
        }
    }
}
