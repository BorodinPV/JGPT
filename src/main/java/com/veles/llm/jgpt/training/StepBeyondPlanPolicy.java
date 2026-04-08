package com.veles.llm.jgpt.training;

import java.util.Locale;

/**
 * Политика при {@code globalStep >= totalTrainingSteps} после загрузки чекпоинта (смена пресета / длины плана).
 * Env: {@code JGPT_IF_STEP_BEYOND_PLAN=skip|restart_schedule|fail}.
 */
public enum StepBeyondPlanPolicy {
    /** Как раньше: лог «план выполнен» и выход из {@link LLMTrainer#train()} без шагов. */
    SKIP,
    /** Сброс {@code globalStep} и шага Adam в 0; веса, моменты Adam и лучший eval loss сохраняются. */
    RESTART_SCHEDULE,
    /** Завершить с ошибкой (для сценариев вроде {@code jgpt-smart}: не считать сегмент успешным). */
    FAIL;

    public static StepBeyondPlanPolicy fromEnv(String raw) {
        if (raw == null || raw.isBlank()) {
            return SKIP;
        }
        String s = raw.trim().toLowerCase(Locale.ROOT);
        return switch (s) {
            case "restart_schedule", "restart-schedule", "restart" -> RESTART_SCHEDULE;
            case "fail", "error" -> FAIL;
            case "skip", "none", "off", "0", "false" -> SKIP;
            default -> SKIP;
        };
    }

    /** Читает {@code JGPT_IF_STEP_BEYOND_PLAN}; при отсутствии — {@link #SKIP}. */
    public static StepBeyondPlanPolicy fromEnvironment() {
        try {
            String e = System.getenv("JGPT_IF_STEP_BEYOND_PLAN");
            return fromEnv(e);
        } catch (Exception ignored) {
            return SKIP;
        }
    }
}
