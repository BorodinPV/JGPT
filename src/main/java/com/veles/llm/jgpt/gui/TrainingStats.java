package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Снимок {@code state/stats.json} ({@code TrainingStatsWriter}). */
record TrainingStats(
        String updated,
        long updatedMs,
        int currentStep,
        int totalSteps,
        String currentEpoch,
        double bestLoss,
        double lastEvalLoss,
        double lastPerplexity,
        double lastTrainLoss,
        int tokensPerSec,
        String lr,
        int skippedSteps,
        int nonFinite,
        int oomErrors,
        int fp16Stuck,
        String preset,
        Map<String, Object> config,
        double[] evalSteps,
        double[] evalLoss,
        double[] perplexity,
        double[] trainSteps,
        double[] trainLoss,
        double[] trainTimeMs,
        double[] overflowSteps) {

    static TrainingStats read(Path file) throws IOException {
        String raw = Files.readString(file, StandardCharsets.UTF_8);
        if (raw.isBlank() || raw.charAt(0) == '\0') {
            throw new IOException("stats.json пуст или повреждён");
        }
        Map<String, Object> m = JsonLite.obj(JsonLite.parse(raw));
        return new TrainingStats(
                JsonLite.str(m, "updated", "-"),
                (long) JsonLite.num(m, "updated_ms", 0),
                (int) JsonLite.num(m, "current_step", 0),
                (int) JsonLite.num(m, "total_steps", 0),
                JsonLite.str(m, "current_epoch", "-"),
                JsonLite.num(m, "best_loss", 0),
                JsonLite.num(m, "last_eval_loss", 0),
                JsonLite.num(m, "last_perplexity", 0),
                JsonLite.num(m, "last_train_loss", 0),
                (int) JsonLite.num(m, "tokens_per_sec", 0),
                JsonLite.str(m, "lr", "-"),
                (int) JsonLite.num(m, "skipped_steps", 0),
                (int) JsonLite.num(m, "non_finite", 0),
                (int) JsonLite.num(m, "oom_errors", 0),
                (int) JsonLite.num(m, "fp16_stuck", 0),
                JsonLite.str(m, "preset", "-"),
                JsonLite.obj(m.get("config")),
                JsonLite.nums(m, "eval_steps"),
                JsonLite.nums(m, "eval_loss"),
                JsonLite.nums(m, "perplexity"),
                JsonLite.nums(m, "train_steps"),
                JsonLite.nums(m, "train_loss"),
                JsonLite.nums(m, "train_time_ms"),
                JsonLite.nums(m, "overflow_steps"));
    }

    /** Свежесть: тренер пишет stats.json на каждом шаге (секунды), 90 с без записи — процесс стоит. */
    boolean isLive(long nowMs) {
        return updatedMs > 0 && nowMs - updatedMs < 90_000;
    }

    double progress() {
        return totalSteps > 0 ? Math.min(1.0, (double) currentStep / totalSteps) : 0;
    }

    /** Оценка оставшегося времени по последним ~200 train-точкам (мс), {@code -1} если данных мало. */
    long etaMs() {
        int n = Math.min(trainSteps.length, trainTimeMs.length);
        if (n < 5 || totalSteps <= currentStep) {
            return -1;
        }
        int from = Math.max(0, n - 200);
        double dSteps = trainSteps[n - 1] - trainSteps[from];
        double dMs = trainTimeMs[n - 1] - trainTimeMs[from];
        if (dSteps <= 0 || dMs <= 0) {
            return -1;
        }
        double msPerStep = dMs / dSteps;
        return (long) ((totalSteps - currentStep) * msPerStep);
    }

    static String fmtDuration(long ms) {
        if (ms < 0) {
            return "—";
        }
        long s = ms / 1000;
        long h = s / 3600;
        long m = (s % 3600) / 60;
        if (h > 0) {
            return h + " ч " + m + " мин";
        }
        return m + " мин " + (s % 60) + " с";
    }
}
