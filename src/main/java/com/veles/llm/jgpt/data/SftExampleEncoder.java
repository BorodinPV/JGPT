package com.veles.llm.jgpt.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Токенизация диалога: лосс только на токенах реплик ассистента (и {@code <eos>} после него).
 */
public final class SftExampleEncoder {

    public static final String USER_PREFIX = "Пользователь: ";
    public static final String ASSISTANT_PREFIX = "Ассистент: ";

    private SftExampleEncoder() {}

    /**
     * {@code JGPT_SFT_CHAT_TEMPLATE} или, если не задан, {@code JGPT_SFT}: оборачивать промпт
     * префиксами {@link #USER_PREFIX}/{@link #ASSISTANT_PREFIX}.
     */
    public static boolean chatTemplateFromEnv() {
        String e = System.getenv("JGPT_SFT_CHAT_TEMPLATE");
        if (e == null || e.isBlank()) {
            e = System.getenv("JGPT_SFT");
        }
        if (e == null || e.isBlank()) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    /**
     * Пользовательский текст → шаблон чата. Если строка уже начинается с {@code Пользователь:},
     * при необходимости только дописывается {@link #ASSISTANT_PREFIX}.
     */
    public static String wrapUserChatPrompt(String prompt) {
        if (prompt == null) {
            return null;
        }
        String p = prompt.trim();
        if (p.regionMatches(true, 0, "Пользователь:", 0, "Пользователь:".length())
                || p.regionMatches(true, 0, "пользователь:", 0, "пользователь:".length())) {
            String lower = p.toLowerCase(Locale.ROOT);
            if (!p.contains("Ассистент:") && !lower.contains("ассистент:")) {
                return p + "\n" + ASSISTANT_PREFIX;
            }
            return p;
        }
        return USER_PREFIX + p + "\n" + ASSISTANT_PREFIX;
    }

    public static String applyChatTemplateIfEnabled(String prompt) {
        return applyChatTemplateIfEnabled(null, prompt);
    }

    public static String applyChatTemplateIfEnabled(BPETokenizer tokenizer, String prompt) {
        if (prompt == null || !chatTemplateFromEnv()) {
            return prompt;
        }
        if (tokenizer != null && tokenizer.hasChatRoleTokens()) {
            String p = prompt.trim();
            if (p.startsWith(BPETokenizer.USER_TOKEN)) {
                if (!p.contains(BPETokenizer.ASSISTANT_TOKEN)) {
                    return p + BPETokenizer.ASSISTANT_TOKEN;
                }
                return p;
            }
            return BPETokenizer.USER_TOKEN + p + BPETokenizer.ASSISTANT_TOKEN;
        }
        return wrapUserChatPrompt(prompt);
    }

    public static Encoded encode(BPETokenizer tokenizer, List<SftTurn> turns) {
        if (turns == null || turns.size() < 2) {
            return null;
        }
        if (tokenizer.hasChatRoleTokens()) {
            return encodeWithRoleTokens(tokenizer, turns);
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
        return toEncoded(ids, sup);
    }

    private static Encoded encodeWithRoleTokens(BPETokenizer tokenizer, List<SftTurn> turns) {
        List<Integer> ids = new ArrayList<>();
        List<Boolean> sup = new ArrayList<>();
        ids.add(tokenizer.bosId());
        sup.add(Boolean.FALSE);
        boolean lastAssistant = false;
        for (SftTurn turn : turns) {
            boolean assistant = turn.role == SftTurn.Role.ASSISTANT;
            lastAssistant = assistant;
            ids.add(assistant ? tokenizer.assistantId() : tokenizer.userId());
            sup.add(assistant);
            int[] piece = tokenizer.encode(turn.content, false, false);
            for (int id : piece) {
                ids.add(id);
                sup.add(assistant);
            }
        }
        ids.add(tokenizer.eosId());
        sup.add(lastAssistant);
        return toEncoded(ids, sup);
    }

    private static Encoded toEncoded(List<Integer> ids, List<Boolean> sup) {
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
