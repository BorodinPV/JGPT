package com.veles.llm.jgpt.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Загрузка JSONL SFT: парсинг → токены с маской → упакованные окна. */
public final class SftCorpus {

    private static final Logger log = LoggerFactory.getLogger(SftCorpus.class);

    private SftCorpus() {}

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
            int fileOk = 0;
            List<SftExampleEncoder.Encoded> encoded = new ArrayList<>();
            try (BufferedReader br = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
                String line;
                while ((line = br.readLine()) != null) {
                    List<SftTurn> turns;
                    try {
                        turns = SftJsonlParser.parseLine(line);
                    } catch (RuntimeException e) {
                        skipped++;
                        continue;
                    }
                    SftExampleEncoder.Encoded ex = SftExampleEncoder.encode(tokenizer, turns);
                    if (ex == null) {
                        skipped++;
                        continue;
                    }
                    encoded.add(ex);
                    parsed++;
                    fileOk++;
                }
            }
            List<SftWindowPacker.Window> windows =
                    SftWindowPacker.pack(encoded, loader.getMaxSeqLen(), tokenizer.padId());
            encoded.clear();
            int fileWindows = 0;
            for (SftWindowPacker.Window w : windows) {
                boolean any = false;
                for (int t : w.targets) {
                    if (t >= 0) {
                        any = true;
                        break;
                    }
                }
                if (!any) {
                    skipped++;
                    continue;
                }
                loader.loadSftWindow(w.tokens, w.targets);
                withLoss++;
                fileWindows++;
            }
            log.info("[SFT]   {} → {} диалогов, {} окон", p.getFileName(), fileOk, fileWindows);
        }
        log.info(
                "[SFT] диалогов={} окон={} (с лоссом), пропущено строк/окон={}",
                parsed,
                withLoss,
                skipped);
        return withLoss;
    }
}
