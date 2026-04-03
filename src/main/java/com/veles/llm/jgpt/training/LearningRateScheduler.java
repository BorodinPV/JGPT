package com.veles.llm.jgpt.training;

/**
 * Вычисление LR по шагу оптимизатора (warmup + выбранное расписание).
 */
public final class LearningRateScheduler {

    private LearningRateScheduler() {}

    /**
     * @param stepForLr        номер шага 1…totalTrainingSteps
     * @param totalTrainingSteps всего шагов оптимизатора
     * @param warmupSteps      шагов warmup (0 — без warmup)
     * @param baseLr           максимальный LR после warmup (пик)
     * @param schedule         режим после warmup
     * @param minLrRatio       нижняя граница как доля от {@code baseLr} [0,1]
     */
    public static float learningRateAtStep(
            int stepForLr,
            int totalTrainingSteps,
            int warmupSteps,
            float baseLr,
            LearningRateSchedule schedule,
            float minLrRatio) {
        if (stepForLr < 1) {
            stepForLr = 1;
        }
        if (stepForLr > totalTrainingSteps) {
            stepForLr = totalTrainingSteps;
        }
        float minR = clampRatio(minLrRatio);

        if (warmupSteps > 0 && stepForLr <= warmupSteps) {
            return baseLr * (stepForLr / (float) warmupSteps);
        }

        float floor = baseLr * minR;

        switch (schedule) {
            case CONSTANT:
                return baseLr;
            case COSINE: {
                int cosineSpan = totalTrainingSteps - warmupSteps;
                if (cosineSpan <= 0) {
                    return baseLr;
                }
                float t = (stepForLr - warmupSteps) / (float) cosineSpan;
                t = Math.min(1f, Math.max(0f, t));
                float mult = minR + (1f - minR) * 0.5f * (1f + (float) Math.cos(Math.PI * t));
                return baseLr * mult;
            }
            case LINEAR: {
                int span = totalTrainingSteps - warmupSteps;
                if (span <= 0) {
                    return baseLr;
                }
                float t = (stepForLr - warmupSteps) / (float) span;
                t = Math.min(1f, Math.max(0f, t));
                float mult = 1f - t * (1f - minR);
                return baseLr * mult;
            }
            case INVERSE_SQRT: {
                int w = Math.max(warmupSteps, 1);
                double lr = baseLr * Math.sqrt((double) w / (double) stepForLr);
                return Math.max(floor, (float) lr);
            }
            default:
                throw new IllegalArgumentException("unknown schedule: " + schedule);
        }
    }

    private static float clampRatio(float minLrRatio) {
        if (minLrRatio < 0f) {
            return 0f;
        }
        if (minLrRatio > 1f) {
            return 1f;
        }
        return minLrRatio;
    }
}
