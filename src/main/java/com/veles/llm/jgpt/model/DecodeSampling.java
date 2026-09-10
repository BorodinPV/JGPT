package com.veles.llm.jgpt.model;

/**
 * Параметры авторегрессивного сэмплинга. {@code topK=0} — без top-k; {@code topP>=1} — без nucleus;
 * {@code repetitionPenalty=1} и {@code noRepeatNgramSize<2} — выкл.
 */
public final class DecodeSampling {

    public final float temperature;
    public final int topK;
    public final float topP;
    public final float repetitionPenalty;
    public final int noRepeatNgramSize;

    public DecodeSampling(
            float temperature, int topK, float topP, float repetitionPenalty, int noRepeatNgramSize) {
        this.temperature = temperature;
        this.topK = Math.max(0, topK);
        this.topP = topP;
        this.repetitionPenalty = repetitionPenalty <= 0f ? 1f : repetitionPenalty;
        this.noRepeatNgramSize = Math.max(0, noRepeatNgramSize);
    }

    public static DecodeSampling of(float temperature, int topK) {
        return new DecodeSampling(temperature, topK, 1f, 1f, 0);
    }
}
