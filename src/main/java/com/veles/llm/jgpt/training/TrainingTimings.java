package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.util.LogFmt;

import java.util.Locale;

/**
 * Сводные тайминги шага обучения (мс в среднем на шаг оптимизатора). Включается env {@code JGPT_TIMINGS=1} или
 * {@code JGPT_TRAIN_PERF=1}.
 *
 * <p>От {@link TrainingProfiler} ( {@code JGPT_PROFILE=1} ) отличается: без детального профиля по шагам,
 * блок строк рядом с периодическим логом train loss.
 */
public final class TrainingTimings {

    /** Одна строка для основного лога + необязательный многострочный блок (сбрасывает окно один раз). */
    public record WindowLog(String inline, String detailMultiline, double tokensPerSec) {
        public boolean isEmpty() {
            return (inline == null || inline.isEmpty()) && (detailMultiline == null || detailMultiline.isEmpty());
        }
    }

    private final boolean enabled;

    private long winFwdNs;
    private long winLossCeNs;
    private long winBwdNs;
    private long winOptNs;
    private long winTokens;
    private int winOptimizerSteps;

    private TrainingTimings(boolean enabled) {
        this.enabled = enabled;
    }

    /** {@code null} — выключено. */
    public static TrainingTimings fromEnv() {
        if (TensorTrainingPerfEnv.enabled()) {
            return new TrainingTimings(true);
        }
        String e = System.getenv("JGPT_TIMINGS");
        if (e == null || e.isBlank() || "0".equals(e) || "false".equalsIgnoreCase(e)) {
            return null;
        }
        return new TrainingTimings(true);
    }

    public boolean isEnabled() {
        return enabled;
    }

    /** Добавить один завершённый шаг оптимизатора (суммы наносекунд по микробатчам + clip+opt). */
    public void recordOptimizerStep(
            long forwardNs,
            long lossAndCeGradNs,
            long backwardNs,
            long clipAndOptimizerNs,
            long tokensThisStep) {
        if (!enabled) {
            return;
        }
        winFwdNs += forwardNs;
        winLossCeNs += lossAndCeGradNs;
        winBwdNs += backwardNs;
        winOptNs += clipAndOptimizerNs;
        winTokens += tokensThisStep;
        winOptimizerSteps++;
    }

    /**
     * Сводка за окно между строками {@code train_loss}: кратко — в одну строку, подробно — несколько строк. Один вызов —
     * один сброс окна.
     */
    public WindowLog formatForLogAndReset() {
        if (!enabled || winOptimizerSteps <= 0) {
            return new WindowLog("", "", 0.0);
        }
        double ms = 1e-6;
        int steps = winOptimizerSteps;
        double n = steps;
        double f = winFwdNs * ms / n;
        double l = winLossCeNs * ms / n;
        double b = winBwdNs * ms / n;
        double o = winOptNs * ms / n;
        double t = f + l + b + o;
        double windowSec = (winFwdNs + winLossCeNs + winBwdNs + winOptNs) * 1e-9;
        double tokPerSec = windowSec > 0.0 ? (winTokens / windowSec) : 0.0;
        winFwdNs = 0;
        winLossCeNs = 0;
        winBwdNs = 0;
        winOptNs = 0;
        winTokens = 0;
        winOptimizerSteps = 0;
        String inline =
                String.format(
                        Locale.forLanguageTag("ru-RU"),
                        "%s за %d шаг.: ток/с≈%.0f  средн. Σ фаз≈%.1f мс/шаг",
                        LogFmt.badge("PERF"),
                        steps,
                        tokPerSec,
                        t);
        String detail =
                String.format(
                        Locale.forLanguageTag("ru-RU"),
                        "%s среднее за последнее окно логов (на один шаг оптимизатора)%n"
                                + "   прямой проход: %.1f мс%n"
                                + "   лосс и ∂CE: %.1f мс%n"
                                + "   обратный проход: %.1f мс%n"
                                + "   клип + Adam: %.1f мс%n"
                                + "   сумма: %.1f мс%n"
                                + "   токенов/с: %.0f",
                        LogFmt.badge("PERF"),
                        f, l, b, o, t, tokPerSec);
        return new WindowLog(inline, detail, tokPerSec);
    }
}
