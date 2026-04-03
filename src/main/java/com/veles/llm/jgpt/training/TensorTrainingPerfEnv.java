package com.veles.llm.jgpt.training;

/**
 * Одна переменная окружения для «полного» perf-лога цикла обучения.
 *
 * <p>{@code JGPT_TRAIN_PERF=1} эквивалентно одновременно {@code JGPT_PROFILE=1} и {@code JGPT_TIMINGS=1}
 * (число пошаговых строк профиля по-прежнему задаёт {@code JGPT_PROFILE_STEPS}).
 */
final class TensorTrainingPerfEnv {

    private TensorTrainingPerfEnv() {}

    static boolean enabled() {
        String e = System.getenv("JGPT_TRAIN_PERF");
        if (e == null || e.isBlank() || "0".equals(e) || "false".equalsIgnoreCase(e)) {
            return false;
        }
        return true;
    }
}
