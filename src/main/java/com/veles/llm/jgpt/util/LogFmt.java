package com.veles.llm.jgpt.util;

/**
 * Компактные semantic badges для логов. Цвета включаются только в интерактивной консоли или при
 * {@code JGPT_LOG_COLOR=1}; отключаются через {@code NO_COLOR} или {@code JGPT_LOG_COLOR=0}.
 */
public final class LogFmt {

    private static final String RESET = "\u001B[0m";
    private static final String BLUE = "\u001B[34m";
    private static final String CYAN = "\u001B[36m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String RED = "\u001B[31m";
    private static final String MAGENTA = "\u001B[35m";
    private static final String GRAY = "\u001B[90m";

    private LogFmt() {}

    public static String badge(String label) {
        return colorize("[" + label + "]", colorForLabel(label));
    }

    public static String success(String text) {
        return colorize(text, GREEN);
    }

    public static String muted(String text) {
        return colorize(text, GRAY);
    }

    private static String colorForLabel(String label) {
        return switch (label) {
            case "CFG", "CKPT" -> CYAN;
            case "TRAIN", "BOOK", "DATA" -> BLUE;
            case "STEP", "EPOCH" -> GREEN;
            case "EVAL", "SAMPLE", "PERF" -> MAGENTA;
            case "FP16" -> YELLOW;
            case "STOP", "ERROR" -> RED;
            default -> GRAY;
        };
    }

    private static String colorize(String text, String color) {
        if (!colorsEnabled()) {
            return text;
        }
        return color + text + RESET;
    }

    private static boolean colorsEnabled() {
        String noColor = System.getenv("NO_COLOR");
        if (noColor != null && !noColor.isEmpty()) {
            return false;
        }
        String force = System.getenv("JGPT_LOG_COLOR");
        if (force != null) {
            if ("0".equals(force) || "false".equalsIgnoreCase(force)) {
                return false;
            }
            if ("1".equals(force) || "true".equalsIgnoreCase(force)) {
                return true;
            }
        }
        return System.console() != null;
    }
}
