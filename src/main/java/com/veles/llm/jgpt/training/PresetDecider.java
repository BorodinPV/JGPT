package com.veles.llm.jgpt.training;

/**
 * Решения супервизора: пороги как в {@code jgpt-smart.sh}.
 */
public final class PresetDecider {

    public enum Action {
        /** Продолжать текущий пресет. */
        NONE,
        /** Перейти на более медленный пресет (индекс +1). */
        DOWNGRADE,
        /** Перейти на более быстрый пресет (индекс −1). */
        UPGRADE
    }

    private final int plateauThreshold;
    private final int fp16StuckThreshold;
    private final long hangMillis;
    private final int stableEvalsForUpgrade;

    private final Object lock = new Object();
    private int consecutiveEvalWithoutImprovement;
    private int cumulativeEvalImprovements;
    private long lastOptimizerStepWallMs;
    private boolean oomOrCudaFatal;

    public PresetDecider() {
        this(15, 8, 300_000L, 30);
    }

    public PresetDecider(
            int plateauThreshold,
            int fp16StuckThreshold,
            long hangMillis,
            int stableEvalsForUpgrade) {
        this.plateauThreshold = Math.max(1, plateauThreshold);
        this.fp16StuckThreshold = Math.max(1, fp16StuckThreshold);
        this.hangMillis = Math.max(1L, hangMillis);
        this.stableEvalsForUpgrade = Math.max(1, stableEvalsForUpgrade);
        this.lastOptimizerStepWallMs = System.currentTimeMillis();
    }

    /** Сброс счётчиков при смене пресета. */
    public void resetForNewPreset() {
        synchronized (lock) {
            consecutiveEvalWithoutImprovement = 0;
            cumulativeEvalImprovements = 0;
            lastOptimizerStepWallMs = System.currentTimeMillis();
            oomOrCudaFatal = false;
        }
    }

    public void onOptimizerStepCompleted() {
        synchronized (lock) {
            lastOptimizerStepWallMs = System.currentTimeMillis();
        }
    }

    public void onEvalCompleted(boolean improvedBest) {
        synchronized (lock) {
            if (improvedBest) {
                consecutiveEvalWithoutImprovement = 0;
                cumulativeEvalImprovements++;
            } else {
                consecutiveEvalWithoutImprovement++;
            }
        }
    }

    public void onOverflowStepSkipped(int consecutiveSkipsForPlannedStep) {
        synchronized (lock) {
            if (consecutiveSkipsForPlannedStep >= fp16StuckThreshold) {
                oomOrCudaFatal = true;
            }
        }
    }

    public void onOutOfMemoryOrCudaError() {
        synchronized (lock) {
            oomOrCudaFatal = true;
        }
    }

    /**
     * @param currentPresetIdx текущий индекс в {@link PresetConfig#SMART_PRESET_CHAIN}
     */
    public Action pollAction(int currentPresetIdx, int chainSize) {
        synchronized (lock) {
            if (oomOrCudaFatal) {
                return Action.DOWNGRADE;
            }
            long idle = System.currentTimeMillis() - lastOptimizerStepWallMs;
            if (idle > hangMillis) {
                return Action.DOWNGRADE;
            }
            if (consecutiveEvalWithoutImprovement >= plateauThreshold) {
                return Action.DOWNGRADE;
            }
            if (cumulativeEvalImprovements >= stableEvalsForUpgrade && currentPresetIdx > 0) {
                return Action.UPGRADE;
            }
            return Action.NONE;
        }
    }
}
