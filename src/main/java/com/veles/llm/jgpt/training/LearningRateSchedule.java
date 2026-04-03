package com.veles.llm.jgpt.training;

/**
 * Режим изменения learning rate после warmup (warmup общий: линейный 0 → {@code base}).
 */
public enum LearningRateSchedule {

    /** Косинусный спад от {@code base} к {@code base * minLrRatio}. */
    COSINE,

    /** После warmup LR остаётся равным {@code base}. */
    CONSTANT,

    /** Линейное снижение от {@code base} к {@code base * minLrRatio}. */
    LINEAR,

    /**
     * После warmup: {@code base * sqrt(warmupSteps / step)}, с нижней границей {@code base * minLrRatio}.
     */
    INVERSE_SQRT
}
