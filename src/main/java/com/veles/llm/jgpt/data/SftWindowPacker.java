package com.veles.llm.jgpt.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Упаковка SFT-диалогов в окна {@code maxSeqLen+1}: несколько коротких примеров в одно окно,
 * без разрезания диалога посередине. Цель позиции {@code i} — токен {@code i+1}, либо {@code -1}
 * (промпт / pad), чтобы CE его игнорировал.
 */
public final class SftWindowPacker {

    public static final int IGNORE_TARGET = -1;

    private SftWindowPacker() {}

    public static List<Window> pack(List<SftExampleEncoder.Encoded> examples, int maxSeqLen, int padId) {
        if (onePerWindowFromEnv()) {
            return packOnePerWindow(examples, maxSeqLen, padId);
        }
        if (maxSeqLen < 1) {
            throw new IllegalArgumentException("maxSeqLen");
        }
        int winLen = maxSeqLen + 1;
        List<Window> out = new ArrayList<>();
        int[] accTok = new int[winLen];
        boolean[] accSup = new boolean[winLen];
        int acc = 0;
        for (SftExampleEncoder.Encoded raw : examples) {
            if (raw == null) {
                continue;
            }
            SftExampleEncoder.Encoded ex = raw;
            if (ex.tokens.length > winLen) {
                ex = SftExampleEncoder.truncateTail(ex, winLen);
                if (ex == null) {
                    continue;
                }
            }
            if (acc > 0 && acc + ex.tokens.length > winLen) {
                out.add(flush(accTok, accSup, acc, winLen, maxSeqLen, padId));
                acc = 0;
            }
            System.arraycopy(ex.tokens, 0, accTok, acc, ex.tokens.length);
            System.arraycopy(ex.supervised, 0, accSup, acc, ex.tokens.length);
            acc += ex.tokens.length;
            if (acc == winLen) {
                out.add(flush(accTok, accSup, acc, winLen, maxSeqLen, padId));
                acc = 0;
            }
        }
        if (acc > 0) {
            out.add(flush(accTok, accSup, acc, winLen, maxSeqLen, padId));
        }
        return out;
    }

    /** {@code JGPT_SFT_PACK=one} — один диалог на окно (без склейки чужих Q&A). */
    public static boolean onePerWindowFromEnv() {
        String e = System.getenv("JGPT_SFT_PACK");
        if (e == null || e.isBlank()) {
            return false;
        }
        String t = e.trim();
        return "one".equalsIgnoreCase(t) || "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    public static List<Window> packOnePerWindow(
            List<SftExampleEncoder.Encoded> examples, int maxSeqLen, int padId) {
        if (maxSeqLen < 1) {
            throw new IllegalArgumentException("maxSeqLen");
        }
        int winLen = maxSeqLen + 1;
        List<Window> out = new ArrayList<>();
        for (SftExampleEncoder.Encoded raw : examples) {
            if (raw == null) {
                continue;
            }
            SftExampleEncoder.Encoded ex = raw;
            if (ex.tokens.length > winLen) {
                ex = SftExampleEncoder.truncateTail(ex, winLen);
                if (ex == null) {
                    continue;
                }
            }
            int[] accTok = new int[winLen];
            boolean[] accSup = new boolean[winLen];
            System.arraycopy(ex.tokens, 0, accTok, 0, ex.tokens.length);
            System.arraycopy(ex.supervised, 0, accSup, 0, ex.tokens.length);
            out.add(flush(accTok, accSup, ex.tokens.length, winLen, maxSeqLen, padId));
        }
        return out;
    }

    private static Window flush(
            int[] accTok, boolean[] accSup, int acc, int winLen, int maxSeqLen, int padId) {
        int[] tokens = new int[winLen];
        Arrays.fill(tokens, padId);
        System.arraycopy(accTok, 0, tokens, 0, acc);
        int[] targets = new int[maxSeqLen];
        Arrays.fill(targets, IGNORE_TARGET);
        int last = Math.min(acc, winLen);
        for (int i = 0; i < maxSeqLen && i + 1 < last; i++) {
            if (accSup[i + 1]) {
                targets[i] = tokens[i + 1];
            }
        }
        return new Window(tokens, targets);
    }

    public static final class Window {
        public final int[] tokens;
        public final int[] targets;

        Window(int[] tokens, int[] targets) {
            this.tokens = tokens;
            this.targets = targets;
        }
    }
}
