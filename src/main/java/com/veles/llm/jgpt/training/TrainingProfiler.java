package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.model.GPTModel;
import com.veles.llm.jgpt.util.LogFmt;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Лёгкое профилирование шага обучения (наносекунды). Включается через env {@code JGPT_PROFILE=1} или
 * комбинированное {@code JGPT_TRAIN_PERF=1}.
 *
 * <p>Разбивка: forward (только {@link GPTModel#forward}), затем loss + градиент по CE в логитах, затем
 * {@link GPTModel#backward}, затем клип + Adam.
 */
public final class TrainingProfiler {

    private static final Logger log = LoggerFactory.getLogger(TrainingProfiler.class);

    private final boolean enabled;
    private final int maxDetailSteps;

    private long sumFwdNs;
    private long sumLossCeNs;
    private long sumBwdNs;
    private long sumOptNs;
    private long sumTokens;
    private long recordedSteps;
    private int detailStepsRemaining;
    private String detailWindowReason;

    private TrainingProfiler(boolean enabled, int maxDetailSteps) {
        this.enabled = enabled;
        this.maxDetailSteps = maxDetailSteps;
    }

    /** {@code null} — выключено. */
    public static TrainingProfiler fromEnv() {
        if (TensorTrainingPerfEnv.enabled()) {
            return new TrainingProfiler(true, profileDetailStepsFromEnv());
        }
        String e = System.getenv("JGPT_PROFILE");
        if (e == null || e.isBlank() || "0".equals(e) || "false".equalsIgnoreCase(e)) {
            return null;
        }
        return new TrainingProfiler(true, profileDetailStepsFromEnv());
    }

    private static int profileDetailStepsFromEnv() {
        int detail = 20;
        String ds = System.getenv("JGPT_PROFILE_STEPS");
        if (ds != null && !ds.isBlank()) {
            try {
                detail = Integer.parseInt(ds.trim());
            } catch (NumberFormatException ignored) {
                // keep default
            }
        }
        return Math.max(0, detail);
    }

    /** Сколько первых шагов оптимизатора логировать построчно ({@code JGPT_PROFILE_STEPS}). */
    public int maxDetailSteps() {
        return maxDetailSteps;
    }

    public void printBanner() {
        if (!enabled) {
            return;
        }
        log.info(
                "{} JGPT_PROFILE=1: замер фаз прямой проход / лосс+∂CE / обратный / клип+оптимизатор "
                        + "(подробная строка на первые {} шагов после старта / reset / eval)",
                LogFmt.badge("PERF"),
                maxDetailSteps);
    }

    /** Открыть подробное PERF-окно на следующие {@code JGPT_PROFILE_STEPS} шагов. */
    public void armDetailWindow(String reason) {
        if (!enabled || maxDetailSteps <= 0) {
            return;
        }
        detailStepsRemaining = maxDetailSteps;
        detailWindowReason = (reason == null || reason.isBlank()) ? "события" : reason;
        log.info(
                "{} подробный PERF включён на {} шаг(ов) после {}",
                LogFmt.badge("PERF"),
                maxDetailSteps,
                detailWindowReason);
    }

    /**
     * Один завершённый шаг оптимизатора (с учётом накопления микробатчей: суммы по микрошагам).
     */
    public void recordStep(
            long forwardNs,
            long lossAndCeGradNs,
            long backwardNs,
            long clipAndOptimizerNs,
            long tokensThisStep,
            int optimizerStep) {
        if (!enabled) {
            return;
        }
        sumFwdNs += forwardNs;
        sumLossCeNs += lossAndCeGradNs;
        sumBwdNs += backwardNs;
        sumOptNs += clipAndOptimizerNs;
        sumTokens += tokensThisStep;
        recordedSteps++;

        if (detailStepsRemaining > 0) {
            double ms = 1e-6;
            long total = forwardNs + lossAndCeGradNs + backwardNs + clipAndOptimizerNs;
            double tokPerSec = total > 0 ? (tokensThisStep * 1e9) / (double) total : 0.0;
            int detailIndex = maxDetailSteps - detailStepsRemaining + 1;
            log.info(
                    "{} шаг {} [{} {}/{}]: прямой={} мс  лосс+∂CE={} мс  обратн={} мс  клип+опт={} мс  сумма={} мс  ток/с≈{}",
                    LogFmt.badge("PERF"),
                    optimizerStep,
                    detailWindowReason,
                    detailIndex,
                    maxDetailSteps,
                    String.format("%.2f", forwardNs * ms),
                    String.format("%.2f", lossAndCeGradNs * ms),
                    String.format("%.2f", backwardNs * ms),
                    String.format("%.2f", clipAndOptimizerNs * ms),
                    String.format("%.2f", total * ms),
                    String.format("%.0f", tokPerSec));
            detailStepsRemaining--;
        }
    }

    public void printSummary() {
        if (!enabled || recordedSteps == 0) {
            return;
        }
        double ms = 1e-6;
        double n = recordedSteps;
        long sum = sumFwdNs + sumLossCeNs + sumBwdNs + sumOptNs;
        double tokPerSec = sum > 0 ? (sumTokens * 1e9) / (double) sum : 0.0;
        log.info(
                "{} среднее за {} шаг(ов) оптимизатора: прямой={} мс, лосс+∂CE={} мс, обратный={} мс, клип+опт={} мс, всего={} мс, ток/с≈{}",
                LogFmt.badge("PERF"),
                recordedSteps,
                String.format("%.2f", sumFwdNs * ms / n),
                String.format("%.2f", sumLossCeNs * ms / n),
                String.format("%.2f", sumBwdNs * ms / n),
                String.format("%.2f", sumOptNs * ms / n),
                String.format("%.2f", sum * ms / n),
                String.format("%.0f", tokPerSec));
    }
}
