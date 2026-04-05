package com.veles.llm.jgpt.training;

import java.util.List;

/**
 * Типизированный пресет обучения (аналог {@code env/*.env} для {@link com.veles.llm.jgpt.app.SmartTrainingSupervisor}).
 *
 * <p>Все пресеты в цепочке используют одинаковый {@link #maxSeqLen}, чтобы чекпоинты оставались совместимыми.
 */
public record PresetConfig(
        String name,
        int maxSeqLen,
        int batchSize,
        int sampledCeCandidates,
        float fp16DynamicInitial,
        float fp16DynamicMax,
        int fp16DynamicGrowthInterval,
        int earlyStopEvalPatience,
        int interactiveEvery) {

    /**
     * Цепочка как в {@code jgpt-smart.sh} (от быстрого к безопасному), значения согласованы с {@code env/*.env}.
     */
    public static final List<PresetConfig> SMART_PRESET_CHAIN =
            List.of(
                    new PresetConfig("00-max-throughput", 1024, 4, 512, 65536f, 65536f, 50, 20, 0),
                    new PresetConfig("01-aggressive", 1024, 1, 512, 65536f, 65536f, 50, 20, 500),
                    new PresetConfig("02-stable", 1024, 1, 256, 16384f, 65536f, 60, 30, 500),
                    new PresetConfig("03-recovery", 1024, 1, 64, 4096f, 16384f, 100, 50, 0));

    /** Создаёт {@link DynamicLossScaler} для FP16 matmul согласно полям пресета. */
    public DynamicLossScaler createLossScaler() {
        return new DynamicLossScaler(
                fp16DynamicInitial, fp16DynamicGrowthInterval, 1f, fp16DynamicMax);
    }

    public static int indexOfName(String name) {
        List<PresetConfig> chain = SMART_PRESET_CHAIN;
        for (int i = 0; i < chain.size(); i++) {
            if (chain.get(i).name().equals(name)) {
                return i;
            }
        }
        return -1;
    }
}
