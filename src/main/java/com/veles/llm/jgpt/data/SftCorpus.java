package com.veles.llm.jgpt.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Загрузка JSONL SFT: парсинг → токены с маской → упакованные окна. */
public final class SftCorpus {

    private static final Logger log = LoggerFactory.getLogger(SftCorpus.class);

    private SftCorpus() {}

    /**
     * {@code JGPT_SFT_SPLIT=dialog} — hold-out по диалогам (сначала split, потом pack). Иначе —
     * текущий split по уже упакованным окнам.
     */
    public static boolean dialogSplitFromEnv() {
        String e = System.getenv("JGPT_SFT_SPLIT");
        if (e == null || e.isBlank()) {
            return false;
        }
        return "dialog".equalsIgnoreCase(e.trim());
    }

    public static List<String> collectPlainTexts(List<Path> jsonlFiles) throws IOException {
        List<String> texts = new ArrayList<>();
        for (Path p : jsonlFiles) {
            try (BufferedReader br = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
                String line;
                while ((line = br.readLine()) != null) {
                    List<SftTurn> turns = SftJsonlParser.parseLine(line);
                    if (turns.size() < 2) {
                        continue;
                    }
                    texts.add(SftJsonlParser.plainText(turns));
                }
            }
        }
        return texts;
    }

    public static int loadInto(DataLoader loader, BPETokenizer tokenizer, List<Path> jsonlFiles)
            throws IOException {
        int parsed = 0;
        int skipped = 0;
        int withLoss = 0;
        for (Path p : jsonlFiles) {
            EncodeFileResult file = encodeFile(tokenizer, p);
            parsed += file.ok;
            skipped += file.skipped;
            List<SftWindowPacker.Window> windows =
                    SftWindowPacker.pack(file.encoded, loader.getMaxSeqLen(), tokenizer.padId());
            file.encoded.clear();
            int fileWindows = addWindows(loader, windows);
            skipped += windows.size() - fileWindows;
            withLoss += fileWindows;
            log.info("[SFT]   {} → {} диалогов, {} окон", p.getFileName(), file.ok, fileWindows);
        }
        log.info(
                "[SFT] диалогов={} окон={} (с лоссом), пропущено строк/окон={}",
                parsed,
                withLoss,
                skipped);
        return withLoss;
    }

    /**
     * Кодирует все диалоги, детерминированно откладывает долю в val, упаковывает train и val
     * отдельно (диалоги из val не попадают в train-окна).
     *
     * @return число train-окон с лоссом
     */
    public static int loadDialogHoldout(
            DataLoader train,
            DataLoader eval,
            BPETokenizer tokenizer,
            List<Path> jsonlFiles,
            double valFrac,
            long seed)
            throws IOException {
        List<SftExampleEncoder.Encoded> all = new ArrayList<>();
        int parsed = 0;
        int skipped = 0;
        for (Path p : jsonlFiles) {
            EncodeFileResult file = encodeFile(tokenizer, p);
            parsed += file.ok;
            skipped += file.skipped;
            all.addAll(file.encoded);
            log.info("[SFT]   {} → {} диалогов", p.getFileName(), file.ok);
        }
        List<SftExampleEncoder.Encoded> trainEx = new ArrayList<>();
        List<SftExampleEncoder.Encoded> valEx = new ArrayList<>();
        splitShuffled(all, valFrac, seed, trainEx, valEx);
        all.clear();
        int pad = tokenizer.padId();
        int maxSeq = train.getMaxSeqLen();
        List<SftWindowPacker.Window> trainWindows = SftWindowPacker.pack(trainEx, maxSeq, pad);
        List<SftWindowPacker.Window> valWindows = SftWindowPacker.pack(valEx, maxSeq, pad);
        trainEx.clear();
        valEx.clear();
        int valGood = countWithLoss(valWindows);
        int bs = train.getBatchSize();
        if (valGood < bs) {
            log.warn(
                    "[SFT] dialog-split: val-окон с лоссом {} < batch {} — hold-out отключён, всё в train",
                    valGood,
                    bs);
            int withLoss = addWindows(train, trainWindows) + addWindows(train, valWindows);
            log.info(
                    "[SFT] split=dialog seed={} диалогов={} train_окон={} val_окон=0 (отключён) skipped={}",
                    seed,
                    parsed,
                    withLoss,
                    skipped);
            return withLoss;
        }
        int trainWindowsN = addWindows(train, trainWindows);
        int valWindowsN = addWindows(eval, valWindows);
        skipped += (trainWindows.size() - trainWindowsN) + (valWindows.size() - valWindowsN);
        log.info(
                "[SFT] split=dialog fraction={} seed={} диалогов={} train_окон={} val_окон={} skipped={}",
                String.format(Locale.ROOT, "%.4f", valFrac),
                seed,
                parsed,
                trainWindowsN,
                valWindowsN,
                skipped);
        return trainWindowsN;
    }

    /**
     * Перемешивание индексов по {@code seed}, первые {@code round(n * valFrac)} — val, остальные
     * train. Если val пуст или забрал все элементы — всё уходит в train.
     */
    static <T> void splitShuffled(
            List<T> all, double valFrac, long seed, List<T> trainOut, List<T> valOut) {
        trainOut.clear();
        valOut.clear();
        int n = all.size();
        if (n == 0) {
            return;
        }
        if (n == 1 || valFrac <= 0d) {
            trainOut.addAll(all);
            return;
        }
        List<Integer> order = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            order.add(i);
        }
        Collections.shuffle(order, new Random(seed));
        int nVal = (int) Math.round(n * valFrac);
        nVal = Math.min(nVal, n - 1);
        if (nVal < 1) {
            trainOut.addAll(all);
            return;
        }
        for (int k = 0; k < n; k++) {
            T item = all.get(order.get(k));
            if (k < nVal) {
                valOut.add(item);
            } else {
                trainOut.add(item);
            }
        }
    }

    private static EncodeFileResult encodeFile(BPETokenizer tokenizer, Path p) throws IOException {
        EncodeFileResult r = new EncodeFileResult();
        try (BufferedReader br = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                List<SftTurn> turns;
                try {
                    turns = SftJsonlParser.parseLine(line);
                } catch (RuntimeException e) {
                    r.skipped++;
                    continue;
                }
                SftExampleEncoder.Encoded ex = SftExampleEncoder.encode(tokenizer, turns);
                if (ex == null) {
                    r.skipped++;
                    continue;
                }
                r.encoded.add(ex);
                r.ok++;
            }
        }
        return r;
    }

    private static int addWindows(DataLoader loader, List<SftWindowPacker.Window> windows) {
        int withLoss = 0;
        for (SftWindowPacker.Window w : windows) {
            if (!hasLoss(w)) {
                continue;
            }
            loader.loadSftWindow(w.tokens, w.targets);
            withLoss++;
        }
        return withLoss;
    }

    private static int countWithLoss(List<SftWindowPacker.Window> windows) {
        int n = 0;
        for (SftWindowPacker.Window w : windows) {
            if (hasLoss(w)) {
                n++;
            }
        }
        return n;
    }

    private static boolean hasLoss(SftWindowPacker.Window w) {
        for (int t : w.targets) {
            if (t >= 0) {
                return true;
            }
        }
        return false;
    }

    private static final class EncodeFileResult {
        final List<SftExampleEncoder.Encoded> encoded = new ArrayList<>();
        int ok;
        int skipped;
    }
}
