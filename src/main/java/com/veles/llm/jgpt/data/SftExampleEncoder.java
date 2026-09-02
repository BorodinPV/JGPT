package com.veles.llm.jgpt.data;

import java.util.ArrayList;
import java.util.List;

/**
 * Токенизация диалога: лосс только на токенах реплик ассистента (и {@code <eos>} после него).
 */
public final class SftExampleEncoder {

    public static final String USER_PREFIX = "Пользователь: ";
    public static final String ASSISTANT_PREFIX = "Ассистент: ";

    private SftExampleEncoder() {}

    public static Encoded encode(BPETokenizer tokenizer, List<SftTurn> turns) {
        if (turns == null || turns.size() < 2) {
            return null;
        }
        List<Integer> ids = new ArrayList<>();
        List<Boolean> sup = new ArrayList<>();
        ids.add(tokenizer.bosId());
        sup.add(Boolean.FALSE);
        boolean lastAssistant = false;
        for (int t = 0; t < turns.size(); t++) {
            SftTurn turn = turns.get(t);
            boolean assistant = turn.role == SftTurn.Role.ASSISTANT;
            lastAssistant = assistant;
            String prefix = assistant ? ASSISTANT_PREFIX : USER_PREFIX;
            String text = prefix + turn.content;
            if (t + 1 < turns.size()) {
                text = text + "\n";
            }
            int[] piece = tokenizer.encode(text, false);
            for (int id : piece) {
                ids.add(id);
                sup.add(assistant);
            }
        }
        ids.add(tokenizer.eosId());
        sup.add(lastAssistant);
        if (ids.size() < 3) {
            return null;
        }
        boolean any = false;
        for (Boolean b : sup) {
            if (b) {
                any = true;
                break;
            }
        }
        if (!any) {
            return null;
        }
        int[] tokens = new int[ids.size()];
        boolean[] supervised = new boolean[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            tokens[i] = ids.get(i);
            supervised[i] = sup.get(i);
        }
        return new Encoded(tokens, supervised);
    }

    public static Encoded truncateTail(Encoded src, int maxTokens) {
        if (src.tokens.length <= maxTokens) {
            return src;
        }
        int from = src.tokens.length - maxTokens;
        int[] tokens = new int[maxTokens];
        boolean[] supervised = new boolean[maxTokens];
        System.arraycopy(src.tokens, from, tokens, 0, maxTokens);
        System.arraycopy(src.supervised, from, supervised, 0, maxTokens);
        boolean any = false;
        for (boolean b : supervised) {
            if (b) {
                any = true;
                break;
            }
        }
        return any ? new Encoded(tokens, supervised) : null;
    }

    public static final class Encoded {
        public final int[] tokens;
        public final boolean[] supervised;

        Encoded(int[] tokens, boolean[] supervised) {
            this.tokens = tokens;
            this.supervised = supervised;
        }
    }
}
