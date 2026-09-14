package com.veles.llm.jgpt.gui;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Минимальный JSON-парсер для {@code state/stats.json} (object/array/string/number/bool/null). */
final class JsonLite {

    private final String s;
    private int i;

    private JsonLite(String s) {
        this.s = s;
    }

    static Object parse(String raw) {
        JsonLite p = new JsonLite(raw);
        Object v = p.value();
        p.ws();
        if (p.i != raw.length()) {
            throw new IllegalArgumentException("trailing data at " + p.i);
        }
        return v;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> obj(Object o) {
        return o instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of();
    }

    @SuppressWarnings("unchecked")
    static List<Object> arr(Object o) {
        return o instanceof List<?> l ? (List<Object>) l : List.of();
    }

    static String str(Map<String, Object> m, String key, String def) {
        Object v = m.get(key);
        return v == null ? def : String.valueOf(v);
    }

    static double num(Map<String, Object> m, String key, double def) {
        Object v = m.get(key);
        if (v instanceof Number n) {
            return n.doubleValue();
        }
        if (v instanceof String t) {
            try {
                return Double.parseDouble(t.replace(',', '.'));
            } catch (NumberFormatException _) {
                return def;
            }
        }
        return def;
    }

    static double[] nums(Map<String, Object> m, String key) {
        List<Object> l = arr(m.get(key));
        double[] out = new double[l.size()];
        for (int k = 0; k < out.length; k++) {
            out[k] = l.get(k) instanceof Number n ? n.doubleValue() : Double.NaN;
        }
        return out;
    }

    private Object value() {
        ws();
        if (i >= s.length()) {
            throw new IllegalArgumentException("unexpected end");
        }
        char c = s.charAt(i);
        switch (c) {
            case '{' -> {
                return object();
            }
            case '[' -> {
                return array();
            }
            case '"' -> {
                return string();
            }
            case 't' -> {
                expect("true");
                return Boolean.TRUE;
            }
            case 'f' -> {
                expect("false");
                return Boolean.FALSE;
            }
            case 'n' -> {
                expect("null");
                return null;
            }
            default -> {
                return number();
            }
        }
    }

    private Map<String, Object> object() {
        Map<String, Object> m = new LinkedHashMap<>();
        i++;
        ws();
        if (peek() == '}') {
            i++;
            return m;
        }
        while (true) {
            ws();
            String k = string();
            ws();
            if (peek() != ':') {
                throw new IllegalArgumentException("':' expected at " + i);
            }
            i++;
            m.put(k, value());
            ws();
            char c = peek();
            i++;
            if (c == '}') {
                return m;
            }
            if (c != ',') {
                throw new IllegalArgumentException("',' expected at " + (i - 1));
            }
        }
    }

    private List<Object> array() {
        List<Object> l = new ArrayList<>();
        i++;
        ws();
        if (peek() == ']') {
            i++;
            return l;
        }
        while (true) {
            l.add(value());
            ws();
            char c = peek();
            i++;
            if (c == ']') {
                return l;
            }
            if (c != ',') {
                throw new IllegalArgumentException("',' expected at " + (i - 1));
            }
        }
    }

    private String string() {
        if (peek() != '"') {
            throw new IllegalArgumentException("string expected at " + i);
        }
        i++;
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
            char e = s.charAt(i++);
            switch (e) {
                case 'n' -> sb.append('\n');
                case 't' -> sb.append('\t');
                case 'r' -> sb.append('\r');
                case 'b' -> sb.append('\b');
                case 'f' -> sb.append('\f');
                case 'u' -> {
                    sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                    i += 4;
                }
                default -> sb.append(e);
            }
        }
        throw new IllegalArgumentException("unterminated string");
    }

    private Number number() {
        int start = i;
        while (i < s.length()) {
            char c = s.charAt(i);
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                i++;
            } else {
                break;
            }
        }
        if (start == i) {
            throw new IllegalArgumentException("number expected at " + i);
        }
        return Double.parseDouble(s.substring(start, i));
    }

    private void expect(String word) {
        if (!s.startsWith(word, i)) {
            throw new IllegalArgumentException("'" + word + "' expected at " + i);
        }
        i += word.length();
    }

    private void ws() {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
            i++;
        }
    }

    private char peek() {
        if (i >= s.length()) {
            throw new IllegalArgumentException("unexpected end");
        }
        return s.charAt(i);
    }
}
