package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.ops.TransformerBackward;

/**
 * Активации forward для одного decoder-блока; поля SwiGLU заполняются при обучении для
 * {@link TransformerBackward#feedForwardSwiGLUBackward} без повторных matmul.
 *
 * <p>{@link #useFp16ActivationStorage}: при {@code true} сохранённые активации квантуются в FP16
 * (см. {@link ActivationTensorSlot}) для снижения памяти на длинных последовательностях; включается из
 * {@link GPTModel} (env {@code JGPT_ACTIVATION_CACHE_FP16} или {@code -Djgpt.activationCache.fp16=true}).
 * При unified-пути с {@link BlockActivationCacheDevice} те же флаги дают хранение слотов на VRAM в настоящем FP16
 * ({@link com.veles.llm.jgpt.GpuHalfBuffer}).
 *
 * <p><b>Потокобезопасность:</b> не потокобезопасен; один экземпляр — один поток обучения.
 */
public final class BlockActivationCache {

    public final ActivationTensorSlot xIn = new ActivationTensorSlot();
    public final ActivationTensorSlot xNorm1 = new ActivationTensorSlot();
    public final ActivationTensorSlot attnOut = new ActivationTensorSlot();
    public final ActivationTensorSlot xRes1 = new ActivationTensorSlot();
    public final ActivationTensorSlot xNorm2 = new ActivationTensorSlot();
    public final ActivationTensorSlot ffnOut = new ActivationTensorSlot();
    public final ActivationTensorSlot xOut = new ActivationTensorSlot();

    public final ActivationTensorSlot ffnH1 = new ActivationTensorSlot();
    public final ActivationTensorSlot ffnGate = new ActivationTensorSlot();
    public final ActivationTensorSlot ffnSig = new ActivationTensorSlot();
    public final ActivationTensorSlot ffnGateSwish = new ActivationTensorSlot();
    public final ActivationTensorSlot ffnHActivated = new ActivationTensorSlot();

    /** Attention cache: после RoPE/split, до выхода через Wo. */
    public final ActivationTensorSlot attnQHeads = new ActivationTensorSlot();
    public final ActivationTensorSlot attnKHeads = new ActivationTensorSlot();
    public final ActivationTensorSlot attnVHeads = new ActivationTensorSlot();
    public final ActivationTensorSlot attnProbs = new ActivationTensorSlot();
    public final ActivationTensorSlot attnConcat = new ActivationTensorSlot();

    /**
     * Если {@code true}, следующие {@link ActivationTensorSlot#store} / {@link
     * ActivationTensorSlot#finalizeAfterWrite} сохраняют в FP16. Устанавливается перед forward слоя.
     */
    public boolean useFp16ActivationStorage;

    /**
     * При {@code true} вместе с {@link #useFp16ActivationStorage} тензоры, которые fused GPU backward
     * позже читает с хоста (attention Q/K/V/probs/concat, SwiGLU промежуточные), остаются в FP32 — без
     * декомпрессии FP16→FP32 перед H2D. Включается в {@link GPTModel} при VRAM-резидентных весах.
     */
    public boolean preferFp32ForFusedGpuBackwardSlots;

    /** Эффективное FP16 для слотов, участвующих в fused backward (см. {@link #preferFp32ForFusedGpuBackwardSlots}). */
    public boolean fp16ForFusedGpuBackwardConsumptionSlots() {
        return useFp16ActivationStorage && !preferFp32ForFusedGpuBackwardSlots;
    }

    /** Сбрасывает все слоты; вызывать между шагами при необходимости освободить ссылки на активации. */
    public void clear() {
        xIn.clear();
        xNorm1.clear();
        attnOut.clear();
        xRes1.clear();
        xNorm2.clear();
        ffnOut.clear();
        xOut.clear();
        ffnH1.clear();
        ffnGate.clear();
        ffnSig.clear();
        ffnGateSwish.clear();
        ffnHActivated.clear();
        attnQHeads.clear();
        attnKHeads.clear();
        attnVHeads.clear();
        attnProbs.clear();
        attnConcat.clear();
    }

    @Override
    public String toString() {
        return String.format(
                "BlockActivationCache{useFp16ActivationStorage=%b, preferFp32ForFusedGpuBackwardSlots=%b}",
                useFp16ActivationStorage, preferFp32ForFusedGpuBackwardSlots);
    }
}
