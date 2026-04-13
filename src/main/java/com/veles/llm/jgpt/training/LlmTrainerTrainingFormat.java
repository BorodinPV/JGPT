package com.veles.llm.jgpt.training;

import java.util.Locale;

/** Форматирование для логов обучения и чекпоинтов. */
final class LlmTrainerTrainingFormat {

    private LlmTrainerTrainingFormat() {}

    static String formatEvalBestLossForLog(float v) {
        if (!Float.isFinite(v) || v == Float.MAX_VALUE) {
            return "нет (ещё не зафиксирован)";
        }
        return String.format(Locale.ROOT, "%.4f", v);
    }

    /** Человекочитаемая длительность (ru-RU); если эпоха короче минуты — с долями секунды. */
    static String formatEpochDuration(long nanos) {
        if (nanos <= 0L) {
            return "0 с";
        }
        long totalSec = nanos / 1_000_000_000L;
        double secExact = nanos / 1_000_000_000.0;
        long h = totalSec / 3600L;
        long m = (totalSec % 3600L) / 60L;
        long s = totalSec % 60L;
        Locale ru = Locale.forLanguageTag("ru-RU");
        if (h > 0L) {
            return String.format(ru, "%d ч %02d мин %02d с", h, m, s);
        }
        if (m > 0L) {
            return String.format(ru, "%d мин %02d с", m, s);
        }
        return String.format(ru, "%.1f с", secExact);
    }
}
