package com.veles.llm.jgpt.training;

import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

import com.veles.llm.jgpt.util.LogFmt;

import org.slf4j.Logger;

/**
 * Счётчики шагов оптимизатора и overflow (NaN/Inf в loss/градиентах) при FP16 — для отладки mixed
 * precision. Заполняется из {@link LLMTrainer}, если {@link com.veles.llm.jgpt.TensorOpsGPU#useFp16Matmul()}
 * включён.
 */
public final class Fp16Metrics {

    private static final Fp16Metrics GLOBAL = new Fp16Metrics();

    private final AtomicLong overflowCount = new AtomicLong();
    private final AtomicLong totalSteps = new AtomicLong();

    public static Fp16Metrics global() {
        return GLOBAL;
    }

    public void recordOverflow() {
        overflowCount.incrementAndGet();
    }

    public void recordStep() {
        totalSteps.incrementAndGet();
    }

    public long getTotalSteps() {
        return totalSteps.get();
    }

    public long getOverflowCount() {
        return overflowCount.get();
    }

    /** Сброс (в тестах или между прогонами). */
    public void reset() {
        overflowCount.set(0L);
        totalSteps.set(0L);
    }

    public void printStats() {
        long total = totalSteps.get();
        long overflow = overflowCount.get();
        if (total == 0) {
            System.out.printf(Locale.ROOT, "Метрики FP16: 0 шагов, 0 переполнений%n");
            return;
        }
        System.out.printf(
                Locale.ROOT,
                "Метрики FP16: %d шагов, %d переполнений (%.2f%%)%n",
                total,
                overflow,
                100.0f * overflow / total);
    }

    /** Одна строка в лог; при {@code totalSteps == 0} — no-op. */
    public void logStats(Logger log) {
        long total = totalSteps.get();
        if (total == 0) {
            return;
        }
        long overflow = overflowCount.get();
        log.info(
                "{} метрики: {} шагов, {} переполнений ({}%)",
                LogFmt.badge("FP16"),
                total,
                overflow,
                String.format(Locale.ROOT, "%.2f", 100.0 * overflow / total));
    }
}
