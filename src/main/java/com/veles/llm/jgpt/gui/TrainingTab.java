package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.TitledPane;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;

/** Вкладка «Обучение»: запуск/остановка, живые метрики из stats.json, графики loss / perplexity. */
final class TrainingTab extends BorderPane {

    private static final int MAX_CHART_POINTS = 1500;
    private static final int MAX_CONSOLE_LINES = 3000;

    private final ProjectPaths paths;
    private final TrainingProcess proc;
    private final ScheduledExecutorService poller =
            Executors.newSingleThreadScheduledExecutor(
                    r -> {
                        Thread t = new Thread(r, "jgpt-gui-stats");
                        t.setDaemon(true);
                        return t;
                    });

    private final ComboBox<TrainRun> runBox = new ComboBox<>();
    private final CheckBox freshBox = new CheckBox("--fresh");
    private final CheckBox restartBox = new CheckBox("--restart-plan");
    private final CheckBox rebuildBox = new CheckBox("пересобрать CUDA");
    private final Button startBtn = new Button("Старт");
    private final Button stopBtn = new Button("Стоп (мягко)");
    private final Button killBtn = new Button("Убить");
    private final Label procLabel = new Label("процесс: —");

    private final Label stepValue = value("—", "cyan");
    private final Label epochValue = value("—", "blue");
    private final Label trainValue = value("—", "blue");
    private final Label valValue = value("—", "green");
    private final Label pplValue = value("—", "yellow");
    private final Label speedValue = value("—", "purple");
    private final Label stepSub = sub("");
    private final Label epochSub = sub("");
    private final Label trainSub = sub("");
    private final Label valSub = sub("");
    private final Label pplSub = sub("");
    private final Label speedSub = sub("");
    private final ProgressBar progress = new ProgressBar(0);
    private final Label etaLabel = sub("");
    private final Label healthLabel = sub("");
    private final Label updatedLabel = sub("");

    private final XYChart.Series<Number, Number> trainSeries = new XYChart.Series<>();
    private final XYChart.Series<Number, Number> valSeries = new XYChart.Series<>();
    private final XYChart.Series<Number, Number> pplSeries = new XYChart.Series<>();
    private final ObservableList<String> console = FXCollections.observableArrayList();

    private int lastStep = -1;
    private int lastEvalCount = -1;
    private long lastUpdatedMs = -1;

    TrainingTab(ProjectPaths paths, TrainingProcess proc, List<TrainRun> runs) {
        this.paths = paths;
        this.proc = proc;
        setPadding(new Insets(10));
        setTop(buildControls(runs));
        setCenter(buildCenter());
        poller.scheduleAtFixedRate(this::pollSafe, 0, 2, TimeUnit.SECONDS);
    }

    void shutdown() {
        poller.shutdownNow();
    }

    // ── UI ────────────────────────────────────────────────────────────────

    private VBox buildControls(List<TrainRun> runs) {
        runBox.getItems().setAll(runs);
        runBox.setPrefWidth(220);
        runs.stream().filter(r -> r.name().equals("28L-wide-sft")).findFirst().ifPresentOrElse(
                runBox.getSelectionModel()::select,
                () -> {
                    if (!runs.isEmpty()) {
                        runBox.getSelectionModel().select(0);
                    }
                });
        runBox.setTooltip(new Tooltip("scripts\\windows\\jgpt-train-<имя>.cmd"));
        freshBox.setTooltip(new Tooltip("Архивировать каталог чекпоинтов этого пресета и начать с нуля"));
        restartBox.setTooltip(
                new Tooltip("Веса + Adam из последнего чекпоинта, но шаг/LR/эпоха/best с нуля (JGPT_FINETUNE=1)"));
        rebuildBox.setTooltip(new Tooltip("Без флага --no-build: перед запуском собрать jgpt_cuda.dll"));

        startBtn.getStyleClass().add("primary");
        stopBtn.getStyleClass().add("warn");
        killBtn.getStyleClass().add("danger");
        stopBtn.setTooltip(new Tooltip("Создать state\\STOP: тренер допишет checkpoint_final и выйдет"));
        killBtn.setTooltip(new Tooltip("Убить java тренера без checkpoint_final (потеря до JGPT_SAVE_EVERY_STEPS шагов)"));
        startBtn.setOnAction(_ -> startTraining());
        stopBtn.setOnAction(_ -> softStop());
        killBtn.setOnAction(_ -> kill());

        Label presetInfo = sub("");
        runBox.valueProperty().addListener((_, _, r) -> presetInfo.setText(describe(r)));
        presetInfo.setText(describe(runBox.getValue()));

        HBox row1 = new HBox(10, new Label("Запуск:"), runBox, freshBox, restartBox, rebuildBox, spacer(), startBtn, stopBtn, killBtn);
        row1.setAlignment(Pos.CENTER_LEFT);
        presetInfo.setWrapText(true);
        procLabel.setWrapText(true);
        procLabel.getStyleClass().add("muted");
        VBox box = new VBox(4, row1, procLabel, presetInfo);
        box.getStyleClass().add("card");
        BorderPane.setMargin(box, new Insets(0, 0, 10, 0));
        return box;
    }

    private static String describe(TrainRun r) {
        if (r == null || r.preset() == null) {
            return "";
        }
        EnvPreset p = r.preset();
        return "пресет " + p.name() + ": " + p.geometry().describe()
                + "  loss=" + p.get("JGPT_TRAIN_LOSS_MODE", "?")
                + "  lr=" + p.get("JGPT_LEARNING_RATE", "?")
                + "  epochs=" + p.get("JGPT_EPOCHS", "?")
                + "  batch=" + p.get("JGPT_BATCH_SIZE", "?") + "×" + p.get("JGPT_ACCUMULATION_STEPS", "?")
                + "  dropout=" + p.get("JGPT_DROPOUT", "0")
                + "  → " + paths(r);
    }

    private static String paths(TrainRun r) {
        return r.ckptDir().getFileName() + "  /  " + r.logFile().getFileName();
    }

    private VBox buildCenter() {
        HBox stats =
                new HBox(
                        10,
                        card("ШАГ", stepValue, stepSub),
                        card("ЭПОХА", epochValue, epochSub),
                        card("TRAIN LOSS", trainValue, trainSub),
                        card("VAL LOSS", valValue, valSub),
                        card("PERPLEXITY", pplValue, pplSub),
                        card("ТОКЕНОВ/С", speedValue, speedSub));
        for (var n : stats.getChildren()) {
            HBox.setHgrow(n, Priority.ALWAYS);
        }

        progress.setMaxWidth(Double.MAX_VALUE);
        progress.setPrefHeight(12);
        HBox progRow = new HBox(12, progress, etaLabel, healthLabel, spacer(), updatedLabel);
        progRow.setAlignment(Pos.CENTER_LEFT);
        HBox.setHgrow(progress, Priority.ALWAYS);
        VBox progCard = new VBox(progRow);
        progCard.getStyleClass().add("card");

        LineChart<Number, Number> lossChart = chart("шаг", "loss");
        trainSeries.setName("train (среднее по окну)");
        valSeries.setName("val (hold-out)");
        lossChart.getData().add(trainSeries);
        lossChart.getData().add(valSeries);

        LineChart<Number, Number> pplChart = chart("шаг", "perplexity (val)");
        pplSeries.setName("perplexity");
        pplChart.getData().add(pplSeries);
        pplChart.setLegendVisible(false);

        HBox charts = new HBox(10, lossChart, pplChart);
        HBox.setHgrow(lossChart, Priority.ALWAYS);
        HBox.setHgrow(pplChart, Priority.ALWAYS);
        lossChart.setPrefWidth(0);
        pplChart.setPrefWidth(0);
        lossChart.setMaxWidth(Double.MAX_VALUE);
        pplChart.setMaxWidth(Double.MAX_VALUE);
        VBox.setVgrow(charts, Priority.ALWAYS);

        ListView<String> consoleView = new ListView<>(console);
        consoleView.setPrefHeight(160);
        console.addListener(
                (javafx.collections.ListChangeListener<String>) c -> {
                    if (!console.isEmpty()) {
                        consoleView.scrollTo(console.size() - 1);
                    }
                });
        TitledPane consolePane = new TitledPane("Вывод запускателя (jgpt-train-*.cmd)", consoleView);
        consolePane.setExpanded(false);
        consolePane.setAnimated(false);

        VBox center = new VBox(10, stats, progCard, charts, consolePane);
        return center;
    }

    private static LineChart<Number, Number> chart(String x, String y) {
        NumberAxis xa = new NumberAxis();
        xa.setLabel(x);
        xa.setForceZeroInRange(false);
        NumberAxis ya = new NumberAxis();
        ya.setLabel(y);
        ya.setForceZeroInRange(false);
        LineChart<Number, Number> ch = new LineChart<>(xa, ya);
        ch.setAnimated(false);
        ch.setCreateSymbols(true);
        ch.setHorizontalGridLinesVisible(true);
        ch.setVerticalGridLinesVisible(false);
        return ch;
    }

    private static VBox card(String title, Label value, Label sub) {
        Label t = new Label(title);
        t.getStyleClass().add("card-title");
        VBox box = new VBox(2, t, value, sub);
        box.getStyleClass().add("card");
        return box;
    }

    private static Label value(String text, String color) {
        Label l = new Label(text);
        l.getStyleClass().addAll("stat-value", color);
        return l;
    }

    private static Label sub(String text) {
        Label l = new Label(text);
        l.getStyleClass().add("muted");
        return l;
    }

    private static Region spacer() {
        Region r = new Region();
        HBox.setHgrow(r, Priority.ALWAYS);
        return r;
    }

    // ── Действия ──────────────────────────────────────────────────────────

    private void startTraining() {
        TrainRun run = runBox.getValue();
        if (run == null) {
            return;
        }
        Optional<TrainingProcess.Trainer> ext = TrainingProcess.findTrainerJava();
        if (ext.isPresent()) {
            alert(Alert.AlertType.WARNING, "Обучение уже идёт",
                    "Найден java-процесс AllBooksTrain (" + ext.get().describe()
                            + ").\nОстановите его (Стоп) прежде чем запускать новый прогон: два тренера не поместятся в VRAM.");
            return;
        }
        if (freshBox.isSelected()) {
            Alert a = new Alert(Alert.AlertType.CONFIRMATION,
                    "--fresh переместит содержимое " + run.ckptDir().getFileName()
                            + " в *_prev_backup и начнёт обучение с нуля. Продолжить?",
                    ButtonType.OK, ButtonType.CANCEL);
            a.setHeaderText("Начать с нуля?");
            Theme.apply(a.getDialogPane().getScene());
            if (a.showAndWait().orElse(ButtonType.CANCEL) != ButtonType.OK) {
                return;
            }
        }
        try {
            console.clear();
            proc.start(
                    run,
                    new TrainingProcess.Flags(freshBox.isSelected(), restartBox.isSelected(), rebuildBox.isSelected()),
                    line -> Platform.runLater(() -> appendConsole(line)),
                    () -> Platform.runLater(this::refreshProcessState));
            freshBox.setSelected(false);
            restartBox.setSelected(false);
        } catch (IOException e) {
            alert(Alert.AlertType.ERROR, "Не удалось запустить", e.getMessage());
        }
        refreshProcessState();
    }

    private void softStop() {
        try {
            proc.requestSoftStop();
            appendConsole("[GUI] создан state\\STOP — ждём [SHUTDOWN] checkpoint сохранён");
        } catch (IOException e) {
            alert(Alert.AlertType.ERROR, "Не удалось создать STOP", e.getMessage());
        }
    }

    private void kill() {
        Alert a = new Alert(Alert.AlertType.CONFIRMATION,
                "Убить java тренера без сохранения checkpoint_final?\nПотеряются шаги после последнего checkpoint_step_N.",
                ButtonType.OK, ButtonType.CANCEL);
        a.setHeaderText("Жёсткая остановка");
        Theme.apply(a.getDialogPane().getScene());
        if (a.showAndWait().orElse(ButtonType.CANCEL) != ButtonType.OK) {
            return;
        }
        boolean found = proc.killTrainerJava();
        appendConsole(found ? "[GUI] процесс тренера убит" : "[GUI] процесс тренера не найден");
        refreshProcessState();
    }

    private void appendConsole(String line) {
        console.add(line);
        if (console.size() > MAX_CONSOLE_LINES) {
            console.remove(0, console.size() - MAX_CONSOLE_LINES);
        }
    }

    private void alert(Alert.AlertType type, String header, String text) {
        Alert a = new Alert(type, text, ButtonType.OK);
        a.setHeaderText(header);
        Theme.apply(a.getDialogPane().getScene());
        a.showAndWait();
    }

    // ── Опрос ─────────────────────────────────────────────────────────────

    private void pollSafe() {
        try {
            poll();
        } catch (Throwable t) {
            Platform.runLater(() -> updatedLabel.setText("stats: " + t.getMessage()));
        }
    }

    private void poll() {
        TrainingStats stats = null;
        String err = null;
        if (Files.isRegularFile(paths.statsJson())) {
            try {
                stats = TrainingStats.read(paths.statsJson());
            } catch (IOException | IllegalArgumentException e) {
                err = e.getMessage();
            }
        }
        Optional<TrainingProcess.Trainer> trainer = TrainingProcess.findTrainerJava();
        boolean stopRequested = Files.exists(paths.stopFile());
        final TrainingStats st = stats;
        final String error = err;
        Platform.runLater(() -> {
            applyProcessState(trainer, stopRequested);
            if (st != null) {
                applyStats(st);
            } else if (error != null) {
                updatedLabel.setText("stats.json: " + error);
            }
        });
    }

    private void refreshProcessState() {
        applyProcessState(TrainingProcess.findTrainerJava(), Files.exists(paths.stopFile()));
    }

    private void applyProcessState(Optional<TrainingProcess.Trainer> trainer, boolean stopRequested) {
        boolean running = trainer.isPresent();
        startBtn.setDisable(running || proc.isChildAlive());
        stopBtn.setDisable(!running);
        killBtn.setDisable(!running);
        if (running) {
            procLabel.setText("процесс: идёт, " + trainer.get().describe()
                    + (proc.isChildAlive() ? " (запущен из GUI)" : " (запущен снаружи)")
                    + (stopRequested ? "  — STOP запрошен, ждём checkpoint_final" : ""));
            procLabel.getStyleClass().removeAll("muted", "red");
            procLabel.getStyleClass().add(stopRequested ? "yellow" : "green");
        } else {
            procLabel.setText("процесс: не запущен" + (stopRequested ? "  (висит state\\STOP — будет подхвачен следующим запуском)" : ""));
            procLabel.getStyleClass().removeAll("green", "yellow");
            procLabel.getStyleClass().add("muted");
        }
    }

    private void applyStats(TrainingStats s) {
        boolean live = s.isLive(System.currentTimeMillis());
        stepValue.setText(fmtInt(s.currentStep()));
        stepSub.setText("из " + fmtInt(s.totalSteps()) + String.format(Locale.ROOT, "  (%.1f%%)", s.progress() * 100));
        epochValue.setText(s.currentEpoch());
        epochSub.setText("пресет " + s.preset());
        trainValue.setText(fmt4(s.lastTrainLoss()));
        trainSub.setText("lr " + s.lr());
        valValue.setText(fmt4(s.lastEvalLoss()));
        valSub.setText("best " + fmt4(s.bestLoss()) + "  ·  " + JsonLite.str(s.config(), "eval_data", "?"));
        pplValue.setText(fmt2(s.lastPerplexity()));
        pplSub.setText("eval-точек: " + s.evalSteps().length);
        speedValue.setText(live ? fmtInt(s.tokensPerSec()) : "—");
        speedSub.setText(live ? "шаг ≈ " + stepSeconds(s) : "процесс не пишет stats.json");
        progress.setProgress(s.progress());
        etaLabel.setText("осталось ≈ " + (live ? TrainingStats.fmtDuration(s.etaMs()) : "—"));
        String health = "overflow " + s.skippedSteps() + "  ·  non-finite " + s.nonFinite()
                + "  ·  OOM " + s.oomErrors() + "  ·  fp16-stuck " + s.fp16Stuck();
        healthLabel.setText(health);
        healthLabel.getStyleClass().removeAll("red", "muted");
        healthLabel.getStyleClass().add(s.skippedSteps() + s.nonFinite() + s.oomErrors() + s.fp16Stuck() > 0 ? "red" : "muted");
        updatedLabel.setText("обновлено " + s.updated() + " UTC" + (live ? "" : "  (устарело)"));

        if (s.currentStep() != lastStep || s.updatedMs() != lastUpdatedMs) {
            lastStep = s.currentStep();
            lastUpdatedMs = s.updatedMs();
            trainSeries.setData(downsample(s.trainSteps(), s.trainLoss()));
        }
        if (s.evalSteps().length != lastEvalCount) {
            lastEvalCount = s.evalSteps().length;
            valSeries.setData(points(s.evalSteps(), s.evalLoss()));
            pplSeries.setData(points(s.evalSteps(), s.perplexity()));
        }
    }

    private static String stepSeconds(TrainingStats s) {
        int n = Math.min(s.trainSteps().length, s.trainTimeMs().length);
        if (n < 3) {
            return "—";
        }
        int from = Math.max(0, n - 50);
        double d = (s.trainTimeMs()[n - 1] - s.trainTimeMs()[from]) / Math.max(1, s.trainSteps()[n - 1] - s.trainSteps()[from]);
        return String.format(Locale.ROOT, "%.1f с", d / 1000.0);
    }

    private static ObservableList<XYChart.Data<Number, Number>> points(double[] xs, double[] ys) {
        ObservableList<XYChart.Data<Number, Number>> out = FXCollections.observableArrayList();
        int n = Math.min(xs.length, ys.length);
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(ys[i])) {
                out.add(new XYChart.Data<>(xs[i], ys[i]));
            }
        }
        return out;
    }

    /** Усреднение по корзинам, чтобы на графике было ≤ {@link #MAX_CHART_POINTS} точек. */
    private static ObservableList<XYChart.Data<Number, Number>> downsample(double[] xs, double[] ys) {
        int n = Math.min(xs.length, ys.length);
        if (n <= MAX_CHART_POINTS) {
            return points(xs, ys);
        }
        ObservableList<XYChart.Data<Number, Number>> out = FXCollections.observableArrayList();
        int bucket = (int) Math.ceil(n / (double) MAX_CHART_POINTS);
        for (int i = 0; i < n; i += bucket) {
            int end = Math.min(n, i + bucket);
            double sy = 0;
            int c = 0;
            for (int k = i; k < end; k++) {
                if (!Double.isNaN(ys[k])) {
                    sy += ys[k];
                    c++;
                }
            }
            if (c > 0) {
                out.add(new XYChart.Data<>(xs[end - 1], sy / c));
            }
        }
        return out;
    }

    private static String fmtInt(int v) {
        return String.format(Locale.ROOT, "%,d", v).replace(',', ' ');
    }

    private static String fmt4(double v) {
        return v <= 0 ? "—" : String.format(Locale.ROOT, "%.4f", v);
    }

    private static String fmt2(double v) {
        return v <= 0 ? "—" : String.format(Locale.ROOT, "%.2f", v);
    }
}
