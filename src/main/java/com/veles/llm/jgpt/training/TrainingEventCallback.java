package com.veles.llm.jgpt.training;

/**
 * Колбэки для супервизора обучения: метрики без парсинга лога.
 */
public interface TrainingEventCallback {

    TrainingEventCallback NOOP =
            new TrainingEventCallback() {
                @Override
                public void onOptimizerStepCompleted(int globalStep, int epochOneBased) {
                    // no-op
                }

                @Override
                public void onEvalCompleted(
                        int epochOneBased,
                        float evalLoss,
                        float bestLossAfterEval,
                        boolean improvedBest) {
                    // no-op
                }

                @Override
                public void onOverflowStepSkipped(
                        int plannedStep, int consecutiveSkipsForPlannedStep, float scaleAfterSkip) {
                    // no-op
                }

                @Override
                public void onOutOfMemoryOrCudaError(Throwable error) {
                    // no-op
                }
            };

    /** Успешный шаг оптимизатора (после {@code globalStep++}). */
    void onOptimizerStepCompleted(int globalStep, int epochOneBased);

    /**
     * Завершён eval на валидации.
     *
     * @param bestLossAfterEval лучший eval loss после возможного обновления чекпоинта
     * @param improvedBest {@code true}, если этот eval улучшил best
     */
    void onEvalCompleted(
            int epochOneBased,
            float evalLoss,
            float bestLossAfterEval,
            boolean improvedBest);

    /** Пропуск шага из‑за overflow / нечисловых градиентов. */
    void onOverflowStepSkipped(
            int plannedStep, int consecutiveSkipsForPlannedStep, float scaleAfterSkip);

    /** OOM или ошибка CUDA во время шага (если перехвачено). */
    void onOutOfMemoryOrCudaError(Throwable error);
}
