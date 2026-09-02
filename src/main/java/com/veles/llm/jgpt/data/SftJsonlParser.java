package com.veles.llm.jgpt.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Разбор одной JSONL-строки SFT: {@code messages}/{@code conversation}, Alpaca
 * ({@code instruction}/{@code input}/{@code output}), либо готовые реплики.
 */
public final class SftJsonlParser {

    private SftJsonlParser() {}

    public static List<SftTurn> parseLine(String line) {
        if (line == null || line.isBlank()) {
            return List.of();
        }
        Object rootObj = SftJson.parse(line);
        Map<String, Object> root = SftJson.asObject(rootObj);
        if (root == null) {
            return List.of();
        }
        String answerLang = SftJson.asString(root.get("answer_lang"));
        if (answerLang != null && !answerLang.isBlank() && !"ru".equalsIgnoreCase(answerLang.trim())) {
            return List.of();
        }
        List<SftTurn> fromMsgs = turnsFromArray(root.get("messages"));
        if (fromMsgs.size() >= 2) {
            return fromMsgs;
        }
        List<SftTurn> fromConv = turnsFromArray(root.get("conversation"));
        if (fromConv.size() >= 2) {
            return fromConv;
        }
        List<SftTurn> fromConversations = turnsFromArray(root.get("conversations"));
        if (fromConversations.size() >= 2) {
            return fromConversations;
        }
        return turnsFromAlpaca(root);
    }

    public static String plainText(List<SftTurn> turns) {
        StringBuilder sb = new StringBuilder();
        for (SftTurn t : turns) {
            if (!sb.isEmpty()) {
                sb.append('\n');
            }
            sb.append(t.role == SftTurn.Role.USER ? "Пользователь: " : "Ассистент: ");
            sb.append(t.content);
        }
        return sb.toString();
    }

    private static List<SftTurn> turnsFromAlpaca(Map<String, Object> root) {
        String inst = trimToNull(SftJson.asString(root.get("instruction")));
        String out = trimToNull(SftJson.asString(root.get("output")));
        String alt = trimToNull(SftJson.asString(root.get("alternative_output")));
        String label = SftJson.asString(root.get("label"));
        if (label != null) {
            String l = label.toLowerCase(Locale.ROOT);
            if (l.contains("bad_task")) {
                return List.of();
            }
            if (l.startsWith("bad")) {
                if (alt == null) {
                    return List.of();
                }
                out = alt;
            }
        }
        if (inst == null || out == null) {
            return List.of();
        }
        String inp = trimToNull(SftJson.asString(root.get("input")));
        String user = inp == null ? inst : inst + "\n" + inp;
        List<SftTurn> t = new ArrayList<>(2);
        t.add(new SftTurn(SftTurn.Role.USER, user));
        t.add(new SftTurn(SftTurn.Role.ASSISTANT, out));
        return t;
    }

    private static List<SftTurn> turnsFromArray(Object arrObj) {
        List<Object> arr = SftJson.asArray(arrObj);
        if (arr == null || arr.isEmpty()) {
            return List.of();
        }
        List<SftTurn> out = new ArrayList<>(arr.size());
        List<String> systems = new ArrayList<>();
        for (Object item : arr) {
            Map<String, Object> m = SftJson.asObject(item);
            if (m == null) {
                continue;
            }
            String roleRaw = firstNonBlank(SftJson.asString(m.get("role")), SftJson.asString(m.get("from")));
            String content =
                    firstNonBlank(SftJson.asString(m.get("content")), SftJson.asString(m.get("value")));
            content = trimToNull(content);
            if (content == null) {
                continue;
            }
            if (roleRaw != null && "system".equals(roleRaw.trim().toLowerCase(Locale.ROOT))) {
                systems.add(content);
                continue;
            }
            SftTurn.Role role = mapRole(roleRaw);
            if (role == null) {
                continue;
            }
            out.add(new SftTurn(role, content));
        }
        if (!systems.isEmpty() && !out.isEmpty() && out.get(0).role == SftTurn.Role.USER) {
            String sys = String.join("\n", systems);
            out.set(0, new SftTurn(SftTurn.Role.USER, sys + "\n" + out.get(0).content));
        }
        return dropLeadingAssistant(out);
    }

    private static List<SftTurn> dropLeadingAssistant(List<SftTurn> turns) {
        int i = 0;
        while (i < turns.size() && turns.get(i).role != SftTurn.Role.USER) {
            i++;
        }
        if (i == 0) {
            return turns;
        }
        if (i >= turns.size()) {
            return List.of();
        }
        return new ArrayList<>(turns.subList(i, turns.size()));
    }

    private static SftTurn.Role mapRole(String raw) {
        if (raw == null) {
            return null;
        }
        String r = raw.trim().toLowerCase(Locale.ROOT);
        return switch (r) {
            case "user", "human", "prompter" -> SftTurn.Role.USER;
            case "assistant", "bot", "gpt", "char", "character", "model" -> SftTurn.Role.ASSISTANT;
            default -> null;
        };
    }

    private static String firstNonBlank(String a, String b) {
        String ta = trimToNull(a);
        return ta != null ? ta : trimToNull(b);
    }

    private static String trimToNull(String s) {
        if (s == null) {
            return null;
        }
        String t = s.trim();
        return t.isEmpty() ? null : t;
    }
}
