package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.util.LogFmt;

import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Динамический loss scaling для mixed precision (FP16 compute / FP32 master weights): при overflow
 * градиентов уменьшает множитель, при длительной стабильности — осторожно увеличивает (см. NVIDIA Apex
 * / PyTorch AMP).
 */
public final class DynamicLossScaler {

    private static final Logger log = LoggerFactory.getLogger(DynamicLossScaler.class);

    /** Повторный WARN при застревании на minScale не чаще, чем раз в столько overflow-шагов. */
    private static final int MIN_SCALE_OVERFLOW_WARN_INTERVAL = 100;

    private float scale;
    /** Эффективный стартовый множитель (после clamp); {@link #resetToInitial()} и recovery возвращают к нему. */
    private final float baselineScale;
    private final int growthInterval;
    private final float minScale;
    private final float maxScale;
    /**
     * После стольких подряд overflow на {@link #minScale} — сброс {@link #scale} к {@link #baselineScale} (0 —
     * выкл.). См. {@code JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK}.
     */
    private final int recoveryAfterMinStreak;
    private int consecutiveNonOverflowSteps;
    /** Подряд идущие overflow при scale == minScale (счётчик сбрасывается после шага без overflow). */
    private int consecutiveOverflowsAtMinScale;

    /** То же, что {@link #DynamicLossScaler(float, int, float, float)} с min=1 и max=65536. */
    public DynamicLossScaler(float initialScale, int growthInterval) {
        this(initialScale, growthInterval, 1f, 65536f);
    }

    public DynamicLossScaler(float initialScale, int growthInterval, float minScale, float maxScale) {
        this(initialScale, growthInterval, minScale, maxScale, 0);
    }

    public DynamicLossScaler(
            float initialScale,
            int growthInterval,
            float minScale,
            float maxScale,
            int recoveryAfterMinStreak) {
        if (initialScale <= 0f) {
            throw new IllegalArgumentException("initialScale must be > 0");
        }
        if (growthInterval <= 0) {
            throw new IllegalArgumentException("growthInterval must be > 0");
        }
        if (minScale <= 0f || maxScale < minScale) {
            throw new IllegalArgumentException("invalid min/max scale");
        }
        if (recoveryAfterMinStreak < 0) {
            throw new IllegalArgumentException("recoveryAfterMinStreak must be >= 0");
        }
        this.baselineScale = Math.min(maxScale, Math.max(minScale, initialScale));
        this.scale = this.baselineScale;
        this.growthInterval = growthInterval;
        this.minScale = minScale;
        this.maxScale = maxScale;
        this.recoveryAfterMinStreak = recoveryAfterMinStreak;
        this.consecutiveNonOverflowSteps = 0;
        this.consecutiveOverflowsAtMinScale = 0;
    }

    /** Вернуть loss scale к значению после конструктора (например в начале эпохи). */
    public void resetToInitial() {
        this.scale = baselineScale;
        this.consecutiveNonOverflowSteps = 0;
        this.consecutiveOverflowsAtMinScale = 0;
    }

    /**
     * После eval / промежуточной генерации на том же CUDA-stream: первый train-step иногда даёт нечисловые
     * дельты в {@link com.veles.llm.jgpt.cuda.GpuPendingGradients}. Дополняет сброс ∂/pending в
     * {@link LLMTrainer} после инференса; при {@code JGPT_FP16_AUX_SOFTEN=0} деление scale не вызывается.
     * По умолчанию eval ÷8 (8192→1024): при ÷4 и полном cleanup рантайм всё ещё даёт overflow на шаге после eval.
     * Sample по умолчанию ÷64. Переопределение: {@code JGPT_FP16_AUX_SOFTEN_EVAL}, {@code JGPT_FP16_AUX_SOFTEN_SAMPLE}.
     *
     * @param reason для лога: {@code "eval"}, {@code "sample"} и т.п.
     */
    public void softenScaleAfterAuxiliaryGpuWork(String reason) {
        boolean sample = "sample".equals(reason);
        float divisor =
                readAuxiliarySoftenDivisor(
                        sample ? "JGPT_FP16_AUX_SOFTEN_SAMPLE" : "JGPT_FP16_AUX_SOFTEN_EVAL",
                        sample ? 64f : 8f);
        float old = scale;
        scale = Math.max(scale / divisor, minScale);
        if (scale < old) {
            consecutiveNonOverflowSteps = 0;
            String tag = sample ? "после генерации" : "после eval";
            log.info(
                    "{} scale {}× → {}× после infer ({}, ÷{})",
                    LogFmt.badge("FP16"),
                    String.format(Locale.ROOT, "%.4g", old),
                    String.format(Locale.ROOT, "%.4g", scale),
                    tag,
                    String.format(Locale.ROOT, "%.4g", divisor));
        }
    }

    /**
     * После вспомогательного GPU-инференса (eval, сэмпл): сбросить счётчик шагов до удвоения scale — рост
     * loss scale не должен совпасть сразу с первым train-step после смены режима.
     */
    public void resetConsecutiveNonOverflowAfterAuxiliaryGpuWork() {
        consecutiveNonOverflowSteps = 0;
    }

    /**
     * Вызывается после backward, до {@link #unscaleGradients(List)} / clip / optimizer.step.
     *
     * @param hasOverflow {@code true}, если обнаружены NaN/Inf в градиентах или loss — шаг оптимизатора
     *     не применять
     * @return {@code false} — пропустить обновление весов; {@code true} — можно снимать масштаб и
     *     вызывать оптимизатор
     */
    public boolean step(boolean hasOverflow) {
        if (hasOverflow) {
            float oldScale = scale;
            scale = Math.max(scale / 2.0f, minScale);
            consecutiveNonOverflowSteps = 0;
            if (oldScale > scale) {
                consecutiveOverflowsAtMinScale = 0;
                log.info(
                        "{} scale {}× → {}× (overflow / non-finite grads)",
                        LogFmt.badge("FP16"),
                        String.format(Locale.ROOT, "%.4g", oldScale),
                        String.format(Locale.ROOT, "%.4g", scale));
            } else {
                consecutiveOverflowsAtMinScale++;
                if (recoveryAfterMinStreak > 0
                        && consecutiveOverflowsAtMinScale >= recoveryAfterMinStreak) {
                    log.warn(
                            "{} recovery: reset scale к {}× после {} overflow подряд на min "
                                    + "(JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK={}); при нестабильности снизьте LR или {}",
                            LogFmt.badge("FP16"),
                            String.format(Locale.ROOT, "%.4g", baselineScale),
                            consecutiveOverflowsAtMinScale,
                            recoveryAfterMinStreak,
                            "JGPT_FP16_DYNAMIC_INITIAL");
                    scale = baselineScale;
                    consecutiveOverflowsAtMinScale = 0;
                    consecutiveNonOverflowSteps = 0;
                    return false;
                }
                String scaleStr = String.format(Locale.ROOT, "%.4g", scale);
                if (consecutiveOverflowsAtMinScale == 1) {
                    log.warn(
                            "{} overflow / non-finite grads при минимальном scale ({}×); "
                                    + "дальнейшие такие шаги — DEBUG, периодический напоминание каждые {} шагов. "
                                    + "Долгое застревание: {} или {}",
                            LogFmt.badge("FP16"),
                            scaleStr,
                            MIN_SCALE_OVERFLOW_WARN_INTERVAL,
                            "JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK>0",
                            "JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH=1");
                } else if (consecutiveOverflowsAtMinScale % MIN_SCALE_OVERFLOW_WARN_INTERVAL == 0) {
                    log.warn(
                            "{} всё ещё overflow при минимальном scale ({}×), подряд {} пропущенных шагов",
                            LogFmt.badge("FP16"),
                            scaleStr,
                            consecutiveOverflowsAtMinScale);
                } else {
                    log.debug(
                            "{} overflow / non-finite grads при минимальном scale ({}×)",
                            LogFmt.badge("FP16"),
                            scaleStr);
                }
            }
            return false;
        }
        consecutiveOverflowsAtMinScale = 0;
        consecutiveNonOverflowSteps++;
        if (consecutiveNonOverflowSteps >= growthInterval) {
            float oldScale = scale;
            scale = Math.min(scale * 2.0f, maxScale);
            consecutiveNonOverflowSteps = 0;
            if (oldScale != scale) {
                log.info(
                        "{} scale {}× → {}× ({} stable steps)",
                        LogFmt.badge("FP16"),
                        String.format(Locale.ROOT, "%.4g", oldScale),
                        String.format(Locale.ROOT, "%.4g", scale),
                        growthInterval);
            }
        }
        return true;
    }

    /** Снять loss scale с градиентов: {@code g *= 1/scale} для текущего {@link #getScale()}. */
    public void unscaleGradients(List<Tensor> tensors) {
        unscaleGradients(tensors, scale);
    }

    /** Как {@link #unscaleGradients(List)}, но ∂ на VRAM ({@link GpuTensor#gradBuffer()}), до D2H в host Adam. */
    public void unscaleGpuDeviceGrads(Map<Tensor, GpuTensor> paramMap) {
        unscaleGpuDeviceGrads(paramMap, scale);
    }

    /**
     * Снять фиксированный loss scale с градиентов параметров на device. При {@code lossScale <= 1} — no-op.
     */
    public static void unscaleGpuDeviceGrads(Map<Tensor, GpuTensor> paramMap, float lossScale) {
        if (paramMap == null || lossScale <= 1f) {
            return;
        }
        float inv = 1.0f / lossScale;
        for (Map.Entry<Tensor, GpuTensor> e : paramMap.entrySet()) {
            GpuTensor gt = e.getValue();
            if (gt != null && gt.hasGradBuffer()) {
                TensorOpsGPU.scaleInPlaceGpuDevice(gt.gradBuffer(), e.getKey().size(), inv);
            }
        }
    }

    /**
     * Снять loss scale с градиентов: {@code g *= 1/lossScale}. При {@code lossScale <= 1} — no-op.
     */
    public static void unscaleGradients(List<Tensor> tensors, float lossScale) {
        if (tensors == null || lossScale <= 1f) {
            return;
        }
        float inv = 1.0f / lossScale;
        for (Tensor t : tensors) {
            if (t != null && t.hasGrad()) {
                scaleGradBufferInPlace(t, inv);
            }
        }
    }

    private static void scaleGradBufferInPlace(Tensor t, float factor) {
        float[] g = t.gradBuffer();
        if (g.length <= 0) {
            return;
        }
        TensorOpsGPU.scaleInPlaceGPU(g, g.length, factor);
    }

    public float getScale() {
        return scale;
    }

    public float getInvScale() {
        return 1.0f / scale;
    }

    public int getGrowthInterval() {
        return growthInterval;
    }

    public float getMaxScale() {
        return maxScale;
    }

    public int getRecoveryAfterMinStreak() {
        return recoveryAfterMinStreak;
    }

    public float getBaselineScale() {
        return baselineScale;
    }

    /**
     * При {@link TensorOpsGPU#useFp16Matmul()} — по умолчанию включается динамический scaler (см.
     * {@code JGPT_FP16_DYNAMIC_INITIAL}, {@code JGPT_FP16_DYNAMIC_GROWTH_INTERVAL} или приоритетный
     * {@code JGPT_AMP_GROWTH_INTERVAL}, {@code JGPT_FP16_DYNAMIC_MAX} и зеркальные {@code jgpt.fp16.dynamic*});
     * иначе {@code null}.
     */
    public static DynamicLossScaler fromEnvironmentIfFp16() {
        if (!TensorOpsGPU.useFp16Matmul()) {
            return null;
        }
        float initial = readPositiveFloat("JGPT_FP16_DYNAMIC_INITIAL", 65536f);
        int growth =
                readPositiveIntPreferKeys(
                        new String[] {"JGPT_AMP_GROWTH_INTERVAL", "JGPT_FP16_DYNAMIC_GROWTH_INTERVAL"},
                        2000);
        float maxScale = readPositiveFloat("JGPT_FP16_DYNAMIC_MAX", 65536f);
        int recovery = readNonNegativeInt("JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK", 0);
        return new DynamicLossScaler(initial, growth, 1f, maxScale, recovery);
    }

    private static String envKeyToProperty(String envKey) {
        if (!envKey.startsWith("JGPT_")) {
            return envKey;
        }
        return "jgpt."
                + envKey.substring("JGPT_".length())
                        .toLowerCase(Locale.ROOT)
                        .replace('_', '.');
    }

    private static float readPositiveFloat(String envKey, float defaultValue) {
        try {
            String e = System.getenv(envKey);
            if (e != null && !e.isBlank()) {
                float v = Float.parseFloat(e.trim());
                if (v > 0f) {
                    return v;
                }
            }
        } catch (Exception ignored) {
        }
        try {
            String p = System.getProperty(envKeyToProperty(envKey));
            if (p != null && !p.isBlank()) {
                float v = Float.parseFloat(p.trim());
                if (v > 0f) {
                    return v;
                }
            }
        } catch (Exception ignored) {
        }
        return defaultValue;
    }

    private static int readNonNegativeInt(String envKey, int defaultValue) {
        try {
            String e = System.getenv(envKey);
            if (e != null && !e.isBlank()) {
                int parsed = Integer.parseInt(e.trim());
                return Math.max(0, parsed);
            }
        } catch (Exception ignored) {
        }
        try {
            String p = System.getProperty(envKeyToProperty(envKey));
            if (p != null && !p.isBlank()) {
                int parsed = Integer.parseInt(p.trim());
                return Math.max(0, parsed);
            }
        } catch (Exception ignored) {
        }
        return defaultValue;
    }

    private static int readPositiveInt(String envKey, int defaultValue) {
        try {
            String e = System.getenv(envKey);
            if (e != null && !e.isBlank()) {
                int parsed = Integer.parseInt(e.trim());
                if (parsed > 0) {
                    return parsed;
                }
            }
        } catch (Exception ignored) {
        }
        try {
            String p = System.getProperty(envKeyToProperty(envKey));
            if (p != null && !p.isBlank()) {
                int parsed = Integer.parseInt(p.trim());
                if (parsed > 0) {
                    return parsed;
                }
            }
        } catch (Exception ignored) {
        }
        return defaultValue;
    }

    /** Первый ключ с валидным env или system property; иначе {@code defaultValue}. */
    private static int readPositiveIntPreferKeys(String[] envKeys, int defaultValue) {
        for (String k : envKeys) {
            int v = readPositiveInt(k, -1);
            if (v > 0) {
                return v;
            }
        }
        return defaultValue;
    }

    /** Делитель для {@link #softenScaleAfterAuxiliaryGpuWork}; минимум 1. */
    private static float readAuxiliarySoftenDivisor(String envKey, float defaultDivisor) {
        if (defaultDivisor < 1f) {
            defaultDivisor = 1f;
        }
        try {
            String e = System.getenv(envKey);
            if (e != null && !e.isBlank()) {
                float v = Float.parseFloat(e.trim());
                if (v >= 1f) {
                    return v;
                }
            }
        } catch (Exception ignored) {
        }
        try {
            String p = System.getProperty(envKeyToProperty(envKey));
            if (p != null && !p.isBlank()) {
                float v = Float.parseFloat(p.trim());
                if (v >= 1f) {
                    return v;
                }
            }
        } catch (Exception ignored) {
        }
        return defaultDivisor;
    }
}
