package com.veles.llm.jgpt.training;

public enum TrainLossMode {
    FULL,
    SAMPLED;

    public boolean usesSampledCandidates() {
        return this == SAMPLED;
    }
}
