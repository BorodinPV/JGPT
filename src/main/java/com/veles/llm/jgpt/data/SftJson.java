package com.veles.llm.jgpt.data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Минимальный JSON (object/array/string/number/bool/null) для JSONL SFT, без внешних библиотек. */
final class SftJson {

    private SftJson() {}

    static Object parse(String raw) {
        return new Parser(raw).parseValue();
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asObject(Object o) {
        return o instanceof Map<?, ?> m ? (Map<String, Object>) m : null;
    }

    @SuppressWarnings("unchecked")
    static List<Object> asArray(Object o) {
        return o instanceof List<?> l ? (List<Object>) l : null;
    }

    static String asString(Object o) {
        if (o == null) {
            return null;
        }
        if (o instanceof String s) {
            return s;
        }
        if (o instanceof Boolean || o instanceof Number) {
            return String.valueOf(o);
        }
        return null;
    }

    private static final class Parser {
        private final String s;
        private int i;

        Parser(String s) {
            this.s = s;
        }

        Object parseValue() {
            skipWs();
            if (i >= s.length()) {
                throw new IllegalArgumentException("empty JSON");
            }
            char c = s.charAt(i);
            if (c == '{') {
                return parseObject();
            }
            if (c == '[') {
                return parseArray();
            }
            if (c == '"') {
                return parseString();
            }
            if (c == 't' || c == 'f') {
                return parseBool();
            }
            if (c == 'n') {
                return parseNull();
            }
            return parseNumber();
        }

        private Map<String, Object> parseObject() {
            expect('{');
            Map<String, Object> m = new LinkedHashMap<>();
            skipWs();
            if (peek('}')) {
                i++;
                return m;
            }
            while (true) {
                skipWs();
                String key = parseString();
                skipWs();
                expect(':');
                Object val = parseValue();
                m.put(key, val);
                skipWs();
                if (peek('}')) {
                    i++;
                    return m;
                }
                expect(',');
            }
        }

        private List<Object> parseArray() {
            expect('[');
            List<Object> a = new ArrayList<>();
            skipWs();
            if (peek(']')) {
                i++;
                return a;
            }
            while (true) {
                a.add(parseValue());
                skipWs();
                if (peek(']')) {
                    i++;
                    return a;
                }
                expect(',');
            }
        }

        private String parseString() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (i < s.length()) {
                char c = s.charAt(i++);
                if (c == '"') {
                    return sb.toString();
                }
                if (c != '\\') {
                    sb.append(c);
                    continue;
                }
                if (i >= s.length()) {
                    throw new IllegalArgumentException("bad escape");
                }
                char e = s.charAt(i++);
                sb.append(
                        switch (e) {
                            case '"', '\\', '/' -> e;
                            case 'b' -> '\b';
                            case 'f' -> '\f';
                            case 'n' -> '\n';
                            case 'r' -> '\r';
                            case 't' -> '\t';
                            case 'u' -> parseHex4();
                            default -> e;
                        });
            }
            throw new IllegalArgumentException("unterminated string");
        }

        private char parseHex4() {
            if (i + 4 > s.length()) {
                throw new IllegalArgumentException("bad \\u");
            }
            int v = Integer.parseInt(s.substring(i, i + 4), 16);
            i += 4;
            return (char) v;
        }

        private Boolean parseBool() {
            if (s.startsWith("true", i)) {
                i += 4;
                return Boolean.TRUE;
            }
            if (s.startsWith("false", i)) {
                i += 5;
                return Boolean.FALSE;
            }
            throw new IllegalArgumentException("bad bool");
        }

        private Object parseNull() {
            if (s.startsWith("null", i)) {
                i += 4;
                return null;
            }
            throw new IllegalArgumentException("bad null");
        }

        private String parseNumber() {
            int start = i;
            if (peek('-')) {
                i++;
            }
            while (i < s.length()) {
                char c = s.charAt(i);
                if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                    i++;
                } else {
                    break;
                }
            }
            return s.substring(start, i);
        }

        private void skipWs() {
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                    i++;
                } else {
                    break;
                }
            }
        }

        private boolean peek(char c) {
            return i < s.length() && s.charAt(i) == c;
        }

        private void expect(char c) {
            skipWs();
            if (!peek(c)) {
                throw new IllegalArgumentException("expected '" + c + "' at " + i);
            }
            i++;
        }
    }
}
