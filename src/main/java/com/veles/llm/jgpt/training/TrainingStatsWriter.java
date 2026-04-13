package com.veles.llm.jgpt.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;

/**
 * Пишет state/stats.json при каждом значимом событии обучения.
 * Dashboard читает этот файл напрямую — без внешних скриптов.
 * Запись атомарна: сначала tmp-файл, потом rename.
 */
public final class TrainingStatsWriter {

    private static final Logger log = LoggerFactory.getLogger(TrainingStatsWriter.class);
    private static final DateTimeFormatter DT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final int MAX_SAMPLES = 10;
    /** Максимум точек в eval/train рядах в stats.json; переопределение: {@code JGPT_STATS_MAX_SERIES}. */
    private static final int MAX_SERIES = resolveMaxSeries();

    private static int resolveMaxSeries() {
        String v = System.getenv("JGPT_STATS_MAX_SERIES");
        if (v != null && !v.isBlank()) {
            try {
                int n = Integer.parseInt(v.trim());
                return Math.max(10, Math.min(n, 500_000));
            } catch (NumberFormatException ignored) {
                // fall through
            }
        }
        return 20_000;
    }

    private final Path stateDir;
    private final Path outFile;
    private final Path tmpFile;

    // Конфиг (заполняется один раз)
    private int totalSteps;
    private String preset = "?";
    private String presetIdx = "?";
    private String cfgSeqLen = "-";
    private String cfgBatch = "1";
    private String cfgEpochs = "-";
    private String cfgNumLayers = "-";
    private String cfgFp16Max = "-";
    private String cfgFp16GrowInterval = "-";
    private String cfgEarlyStopPatience = "-";
    private String cfgCeCandidates = "-";
    /** {@code validation} — hold-out из {@code JGPT_VAL_FRACTION}; иначе {@code train}. */
    private String cfgEvalData = "train";
    private String cfgDescription = "";

    // Скаляры (обновляются по событиям)
    private int    currentStep;
    /** До первого {@link #onStep} — «1/totalEpochs» из конфига (см. {@link #setConfig}). */
    private String currentEpoch = "1/?";
    private float  bestLoss     = Float.MAX_VALUE;
    private float  lastEvalLoss;
    private float  lastPerplexity;
    private float  lastTrainLoss;
    private int    tokensPerSec;
    private String lr = "-";
    private int    skippedSteps;
    private int    nonFiniteCount;
    private int    oomErrors;
    private int    fp16StuckCount;
    private String lastSample = "";
    private String lastSampleStep = "";

    // Временны́е ряды (ArrayDeque: O(1) pollFirst вместо O(n) ArrayList.remove(0))
    private final ArrayDeque<Integer> evalSteps    = new ArrayDeque<>();
    private final ArrayDeque<Float>   evalLoss     = new ArrayDeque<>();
    private final ArrayDeque<Float>   perplexity   = new ArrayDeque<>();
    /** Epoch millis на момент точки eval (для дашборда: фильтр по времени). */
    private final ArrayDeque<Long>    evalTimeMs   = new ArrayDeque<>();
    private final ArrayDeque<Integer> trainSteps   = new ArrayDeque<>();
    private final ArrayDeque<Float>   trainLoss    = new ArrayDeque<>();
    /** Epoch millis на момент залогированного train-шага. */
    private final ArrayDeque<Long>    trainTimeMs  = new ArrayDeque<>();
    private final ArrayDeque<Integer> overflowSteps = new ArrayDeque<>();
    private final List<String>        samples      = new ArrayList<>();

    public TrainingStatsWriter(Path stateDir) {
        this.stateDir = stateDir;
        this.outFile  = stateDir.resolve("stats.json");
        this.tmpFile  = stateDir.resolve("stats.json.tmp");
    }

    // ── Инициализация конфига ─────────────────────────────────

    /** Задать описание пресета (отображается в конфиг-блоке дашборда). Вызывать после {@link #setConfig}. */
    public void setDescription(String description) {
        this.cfgDescription = description != null ? description : "";
    }

    public void setConfig(TrainingConfig cfg, int totalSteps, String preset, String presetIdx) {
        this.totalSteps = totalSteps;
        this.preset     = preset;
        this.presetIdx  = presetIdx;
        this.currentEpoch = "1/" + cfg.epochs;
        this.cfgSeqLen            = String.valueOf(cfg.maxSeqLen);
        this.cfgBatch             = String.valueOf(cfg.batchSize);
        this.cfgEpochs            = String.valueOf(cfg.epochs);
        this.cfgNumLayers         = String.valueOf(cfg.numLayers);
        this.cfgEarlyStopPatience = String.valueOf(cfg.earlyStopEvalPatience);
        // Остальные параметры FP16/CE берём из env-переменных
        this.cfgFp16Max           = envOrDash("JGPT_FP16_DYNAMIC_MAX");
        this.cfgFp16GrowInterval  = envOrDash("JGPT_FP16_DYNAMIC_GROWTH_INTERVAL");
        this.cfgCeCandidates      = envOrDash("JGPT_SAMPLED_CE_CANDIDATES");
        write();
    }

    /**
     * Источник метрик eval в логах/stats: {@code validation} (hold-out) или {@code train}.
     * Вызывать после {@link #setConfig} при необходимости перезаписать авто-вывод из env.
     */
    public void setEvalDataSource(String source) {
        if (source != null && !source.isBlank()) {
            this.cfgEvalData = source.trim();
        }
        write();
    }

    /**
     * Обновить отображаемый прогресс после {@link LLMTrainer#loadCheckpoint} — до первого {@link #onStep}
     * иначе в stats.json остаётся current_step=0.
     */
    public void syncProgressFromResume(int step, int epochOneBased, int totalEpochs) {
        this.currentStep = Math.max(0, step);
        int ep = Math.max(1, Math.min(epochOneBased, Math.max(1, totalEpochs)));
        this.currentEpoch = ep + "/" + Math.max(1, totalEpochs);
        write();
    }

    /**
     * Синхронизировать {@code bestLoss} после загрузки чекпоинта.
     * Без этого до первого {@link #onEval} в stats.json остаётся «0.0000» (артефакт {@code Float.MAX_VALUE}).
     */
    public void syncBestLossFromResume(float bestLoss) {
        if (bestLoss > 0f && Float.isFinite(bestLoss)) {
            this.bestLoss = bestLoss;
            write();
        }
    }

    // ── События ──────────────────────────────────────────────

    /** Вызывать после каждого успешного шага оптимизатора. */
    public void onStep(int step, int epoch, int totalEpochs, float loss, float lrValue, int tokPerSec) {
        this.currentStep  = step;
        this.currentEpoch = epoch + "/" + totalEpochs;
        this.lastTrainLoss = loss;
        this.tokensPerSec  = tokPerSec;
        this.lr = String.format(Locale.ROOT, "%.2e", lrValue);
        addTrainSeries(step, loss);
        write();
    }

    /** Вызывать после вычисления eval loss и перплексии. */
    public void onEval(int step, float loss, float perp, float best) {
        this.lastEvalLoss  = loss;
        this.lastPerplexity = perp;
        this.bestLoss      = best;
        addEvalSeries(step, loss, perp);
        write();
    }

    /** Вызывать при успешной генерации. */
    public void onSample(int step, String text) {
        this.lastSample     = text;
        this.lastSampleStep = String.valueOf(step);
        if (samples.size() >= MAX_SAMPLES) samples.remove(0);
        samples.add(text);
        write();
    }

    /** Вызывать при пропуске шага из-за переполнения градиентов. */
    public void onOverflow(int step) {
        skippedSteps++;
        if (overflowSteps.size() >= MAX_SERIES) overflowSteps.pollFirst();
        overflowSteps.addLast(step);
        write();
    }

    /** Вызывать при non-finite градиентах. */
    public void onNonFiniteGradient() {
        nonFiniteCount++;
        write();
    }

    /** Вызывать при OOM. */
    public void onOom() {
        oomErrors++;
        write();
    }

    /** Вызывать при FP16 scale=1.0× (залип). */
    public void onFp16Stuck() {
        fp16StuckCount++;
        write();
    }

    // ── Запись JSON ───────────────────────────────────────────

    private void write() {
        try {
            Files.createDirectories(stateDir);
            String json = buildJson();
            Files.writeString(tmpFile, json, StandardCharsets.UTF_8);
            Files.move(tmpFile, outFile, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            log.debug("TrainingStatsWriter: не удалось записать stats.json: {}", e.getMessage());
        }
    }

    private String buildJson() {
        StringBuilder sb = new StringBuilder(4096);
        sb.append("{\n");
        sb.append("  \"updated\": \"").append(LocalDateTime.now().format(DT)).append("\",\n");
        sb.append("  \"updated_ms\": ").append(System.currentTimeMillis()).append(",\n");
        sb.append("  \"current_step\": ").append(currentStep).append(",\n");
        sb.append("  \"total_steps\": ").append(totalSteps).append(",\n");
        sb.append("  \"current_epoch\": \"").append(currentEpoch).append("\",\n");
        sb.append("  \"best_loss\": ").append(fmt(bestLoss == Float.MAX_VALUE ? 0f : bestLoss)).append(",\n");
        sb.append("  \"last_eval_loss\": ").append(fmt(lastEvalLoss)).append(",\n");
        sb.append("  \"last_perplexity\": ").append(fmt(lastPerplexity)).append(",\n");
        sb.append("  \"last_train_loss\": ").append(fmt(lastTrainLoss)).append(",\n");
        sb.append("  \"tokens_per_sec\": ").append(tokensPerSec).append(",\n");
        sb.append("  \"lr\": \"").append(lr).append("\",\n");
        sb.append("  \"skipped_steps\": ").append(skippedSteps).append(",\n");
        sb.append("  \"non_finite\": ").append(nonFiniteCount).append(",\n");
        sb.append("  \"oom_errors\": ").append(oomErrors).append(",\n");
        sb.append("  \"fp16_stuck\": ").append(fp16StuckCount).append(",\n");
        sb.append("  \"preset\": \"").append(escape(preset)).append("\",\n");
        sb.append("  \"preset_idx\": \"").append(escape(presetIdx)).append("\",\n");
        sb.append("  \"stats_max_series\": ").append(MAX_SERIES).append(",\n");
        sb.append("  \"config\": {\n");
        sb.append("    \"epochs\": \"").append(cfgEpochs).append("\",\n");
        sb.append("    \"num_layers\": \"").append(cfgNumLayers).append("\",\n");
        sb.append("    \"seq_len\": \"").append(cfgSeqLen).append("\",\n");
        sb.append("    \"batch\": \"").append(cfgBatch).append("\",\n");
        sb.append("    \"fp16_max\": \"").append(cfgFp16Max).append("\",\n");
        sb.append("    \"fp16_grow_interval\": \"").append(cfgFp16GrowInterval).append("\",\n");
        sb.append("    \"early_stop_patience\": \"").append(cfgEarlyStopPatience).append("\",\n");
        sb.append("    \"ce_candidates\": \"").append(cfgCeCandidates).append("\",\n");
        sb.append("    \"eval_data\": \"").append(escape(cfgEvalData)).append("\",\n");
        sb.append("    \"val_fraction\": \"").append(envOrDash("JGPT_VAL_FRACTION")).append("\",\n");
        sb.append("    \"description\": \"").append(escape(cfgDescription)).append("\"\n");
        sb.append("  },\n");
        sb.append("  \"last_sample\": \"").append(escape(lastSample)).append("\",\n");
        sb.append("  \"last_sample_step\": \"").append(lastSampleStep).append("\",\n");
        sb.append("  \"samples\": ").append(samplesToJson()).append(",\n");
        sb.append("  \"eval_steps\": ").append(intListToJson(evalSteps)).append(",\n");
        sb.append("  \"eval_loss\": ").append(floatListToJson(evalLoss)).append(",\n");
        sb.append("  \"perplexity\": ").append(floatListToJson(perplexity)).append(",\n");
        sb.append("  \"eval_time_ms\": ").append(longListToJson(evalTimeMs)).append(",\n");
        sb.append("  \"train_steps\": ").append(intListToJson(trainSteps)).append(",\n");
        sb.append("  \"train_loss\": ").append(floatListToJson(trainLoss)).append(",\n");
        sb.append("  \"train_time_ms\": ").append(longListToJson(trainTimeMs)).append(",\n");
        sb.append("  \"overflow_steps\": ").append(intListToJson(overflowSteps)).append("\n");
        sb.append("}\n");
        return sb.toString();
    }

    // ── Вспомогалки ───────────────────────────────────────────

    private static String fmt(float v) {
        if (!Float.isFinite(v)) return "0";
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private static String escape(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if      (c == '\\') sb.append("\\\\");
            else if (c == '"')  sb.append("\\\"");
            else if (c == '\n') sb.append(' ');
            else if (c == '\t') sb.append(' ');
            else if (c >= 0x20) sb.append(c);
            // \r и остальные управляющие символы (< 0x20) пропускаются — невалидны в JSON-строке
        }
        return sb.toString();
    }

    private static String envOrDash(String key) {
        String v = System.getenv(key);
        return (v != null && !v.isBlank()) ? v.trim() : "-";
    }

    /** Eval loss и перплексия по одним и тем же шагам (иначе графики в dashboard расходятся). */
    private void addEvalSeries(int step, float loss, float perp) {
        long now = System.currentTimeMillis();
        if (evalSteps.size() >= MAX_SERIES) {
            evalSteps.pollFirst();
            evalLoss.pollFirst();
            perplexity.pollFirst();
            evalTimeMs.pollFirst();
        }
        evalSteps.addLast(step);
        evalLoss.addLast(loss);
        perplexity.addLast(perp);
        evalTimeMs.addLast(now);
    }

    private void addTrainSeries(int step, float loss) {
        long now = System.currentTimeMillis();
        if (trainSteps.size() >= MAX_SERIES) {
            trainSteps.pollFirst();
            trainLoss.pollFirst();
            trainTimeMs.pollFirst();
        }
        trainSteps.addLast(step);
        trainLoss.addLast(loss);
        trainTimeMs.addLast(now);
    }

    private String intListToJson(Collection<Integer> coll) {
        if (coll.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for (int v : coll) {
            if (!first) sb.append(',');
            sb.append(v);
            first = false;
        }
        return sb.append(']').toString();
    }

    private String longListToJson(Collection<Long> coll) {
        if (coll.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for (long v : coll) {
            if (!first) sb.append(',');
            sb.append(v);
            first = false;
        }
        return sb.append(']').toString();
    }

    private String floatListToJson(Collection<Float> coll) {
        if (coll.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        for (float v : coll) {
            if (!first) sb.append(',');
            sb.append(fmt(v));
            first = false;
        }
        return sb.append(']').toString();
    }

    private String samplesToJson() {
        if (samples.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < samples.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append("{\"text\":\"").append(escape(samples.get(i))).append("\"}");
        }
        return sb.append(']').toString();
    }
}
