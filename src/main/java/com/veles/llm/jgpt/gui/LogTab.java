package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;

/** Вкладка «Лог»: хвост training_*.log с фильтрами по типу строки. */
final class LogTab extends BorderPane {

    private static final int MAX_LINES = 6000;
    private static final long INITIAL_TAIL_BYTES = 2L * 1024 * 1024;

    private final ComboBox<TrainRun> runBox = new ComboBox<>();
    private final CheckBox stepBox = new CheckBox("STEP");
    private final CheckBox evalBox = new CheckBox("EVAL");
    private final CheckBox ckptBox = new CheckBox("CKPT");
    private final CheckBox fp16Box = new CheckBox("FP16");
    private final CheckBox warnBox = new CheckBox("WARN/ERROR");
    private final CheckBox otherBox = new CheckBox("прочее");
    private final CheckBox perfBox = new CheckBox("PERF/VRAM");
    private final CheckBox followBox = new CheckBox("следить");
    private final TextField searchField = new TextField();
    private final Label statusLabel = new Label();
    private final ObservableList<String> visible = FXCollections.observableArrayList();
    private final ListView<String> view = new ListView<>(visible);
    private final ArrayDeque<String> all = new ArrayDeque<>();

    private final ScheduledExecutorService poller =
            Executors.newSingleThreadScheduledExecutor(
                    r -> {
                        Thread t = new Thread(r, "jgpt-gui-log");
                        t.setDaemon(true);
                        return t;
                    });

    private Path currentFile;
    private long position;
    private byte[] partial = new byte[0];
    /** Вкладка на экране: scrollTo на невидимом ListView заставляет VirtualFlow жаловаться на каждом опросе. */
    private volatile boolean active;

    LogTab(List<TrainRun> runs, TrainRun initial) {
        setPadding(new Insets(10));
        runBox.getItems().setAll(runs);
        runBox.setPrefWidth(200);
        if (initial != null) {
            runBox.getSelectionModel().select(initial);
        } else if (!runs.isEmpty()) {
            runBox.getSelectionModel().select(0);
        }
        runBox.valueProperty().addListener((_, _, r) -> switchFile(r == null ? null : r.logFile()));

        for (CheckBox b : List.of(stepBox, evalBox, ckptBox, fp16Box, warnBox, otherBox, followBox)) {
            b.setSelected(true);
        }
        perfBox.setSelected(false);
        for (CheckBox b : List.of(stepBox, evalBox, ckptBox, fp16Box, warnBox, otherBox, perfBox)) {
            b.selectedProperty().addListener((_, _, _) -> refilter());
        }
        searchField.setPromptText("поиск (подстрока)");
        searchField.setPrefWidth(220);
        searchField.textProperty().addListener((_, _, _) -> refilter());
        Button clearBtn = new Button("Очистить");
        clearBtn.setOnAction(_ -> {
            all.clear();
            visible.clear();
        });
        Button reloadBtn = new Button("Перечитать");
        reloadBtn.setOnAction(_ -> switchFile(currentFile));

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox controls = new HBox(8, new Label("Лог:"), runBox, stepBox, evalBox, ckptBox, fp16Box, warnBox, otherBox,
                perfBox, searchField, spacer, followBox, reloadBtn, clearBtn);
        controls.setAlignment(Pos.CENTER_LEFT);
        controls.getStyleClass().add("card");
        statusLabel.getStyleClass().add("muted");
        BorderPane.setMargin(controls, new Insets(0, 0, 8, 0));
        setTop(controls);

        view.setCellFactory(_ -> new LogCell());
        setCenter(view);
        setBottom(statusLabel);
        BorderPane.setMargin(statusLabel, new Insets(6, 0, 0, 0));

        switchFile(runBox.getValue() == null ? null : runBox.getValue().logFile());
        poller.scheduleAtFixedRate(this::pollSafe, 1, 2, TimeUnit.SECONDS);
    }

    void shutdown() {
        poller.shutdownNow();
    }

    void showRun(TrainRun run) {
        if (run != null) {
            runBox.getSelectionModel().select(run);
        }
    }

    void setActive(boolean on) {
        active = on;
        if (on) {
            Platform.runLater(this::scrollToEnd);
        }
    }

    private void scrollToEnd() {
        if (active && followBox.isSelected() && !visible.isEmpty()) {
            view.scrollTo(visible.size() - 1);
        }
    }

    // ── Чтение файла ─────────────────────────────────────────────────────

    private synchronized void switchFile(Path file) {
        currentFile = file;
        position = 0;
        partial = new byte[0];
        all.clear();
        visible.clear();
        if (file == null || !Files.isRegularFile(file)) {
            statusLabel.setText(file == null ? "лог не выбран" : "нет файла: " + file);
            return;
        }
        try {
            long size = Files.size(file);
            position = Math.max(0, size - INITIAL_TAIL_BYTES);
            readNew(true);
        } catch (IOException e) {
            statusLabel.setText("ошибка чтения: " + e.getMessage());
        }
    }

    private void pollSafe() {
        try {
            readNew(false);
        } catch (Throwable t) {
            Platform.runLater(() -> statusLabel.setText("ошибка чтения: " + t.getMessage()));
        }
    }

    private synchronized void readNew(boolean initial) throws IOException {
        Path file = currentFile;
        if (file == null || !Files.isRegularFile(file)) {
            return;
        }
        long size = Files.size(file);
        if (size < position) {
            // файл пересоздан (--fresh / новый прогон) — начинаем с начала
            position = 0;
            partial = new byte[0];
        }
        if (size == position) {
            return;
        }
        List<String> lines = new ArrayList<>();
        try (RandomAccessFile raf = new RandomAccessFile(file.toFile(), "r")) {
            raf.seek(position);
            long toRead = size - position;
            byte[] buf = new byte[(int) Math.min(toRead, 8L * 1024 * 1024)];
            int n = raf.read(buf);
            if (n <= 0) {
                return;
            }
            position += n;
            byte[] data = new byte[partial.length + n];
            System.arraycopy(partial, 0, data, 0, partial.length);
            System.arraycopy(buf, 0, data, partial.length, n);
            int start = 0;
            for (int i = 0; i < data.length; i++) {
                if (data[i] == '\n') {
                    int end = i > start && data[i - 1] == '\r' ? i - 1 : i;
                    lines.add(new String(data, start, end - start, StandardCharsets.UTF_8));
                    start = i + 1;
                }
            }
            partial = new byte[data.length - start];
            System.arraycopy(data, start, partial, 0, partial.length);
        }
        if (initial && !lines.isEmpty()) {
            lines.remove(0); // первая строка хвоста обычно обрезана
        }
        if (lines.isEmpty()) {
            return;
        }
        final long sz = size;
        Platform.runLater(() -> append(lines, sz));
    }

    private void append(List<String> lines, long size) {
        boolean atBottom = followBox.isSelected();
        for (String l : lines) {
            all.addLast(l);
            if (all.size() > MAX_LINES) {
                all.pollFirst();
            }
            if (passes(l)) {
                visible.add(l);
            }
        }
        if (visible.size() > MAX_LINES) {
            visible.remove(0, visible.size() - MAX_LINES);
        }
        statusLabel.setText(currentFile + "  ·  " + (size / 1024 / 1024) + " МБ  ·  строк в буфере " + all.size()
                + ", показано " + visible.size());
        if (atBottom) {
            scrollToEnd();
        }
    }

    private void refilter() {
        List<String> out = new ArrayList<>();
        for (String l : all) {
            if (passes(l)) {
                out.add(l);
            }
        }
        visible.setAll(out);
        scrollToEnd();
    }

    private boolean passes(String l) {
        String q = searchField.getText();
        if (q != null && !q.isBlank() && !l.toLowerCase().contains(q.toLowerCase())) {
            return false;
        }
        Kind k = Kind.of(l);
        return switch (k) {
            case STEP -> stepBox.isSelected();
            case EVAL -> evalBox.isSelected();
            case CKPT -> ckptBox.isSelected();
            case FP16 -> fp16Box.isSelected();
            case WARN, ERROR -> warnBox.isSelected();
            case PERF -> perfBox.isSelected();
            case OTHER -> otherBox.isSelected();
        };
    }

    enum Kind {
        STEP, EVAL, CKPT, FP16, WARN, ERROR, PERF, OTHER;

        static Kind of(String l) {
            if (l.contains("ERROR") || l.contains("Exception") || l.contains("FATAL")) {
                return ERROR;
            }
            if (l.contains(" WARN ") || l.contains("[STOP]") || l.contains("[SHUTDOWN]")) {
                return WARN;
            }
            if (l.contains("[STEP]")) {
                return STEP; // строка [STEP] несёт и хвост [PERF] — это шаг, а не perf-шум
            }
            if (l.contains("[PERF]") || l.contains("[VRAM]") || l.startsWith("   ")) {
                return PERF;
            }
            if (l.contains("[EVAL]") || l.contains("[EPOCH]") || l.contains("[SAMPLE]")) {
                return EVAL;
            }
            if (l.contains("[CKPT]")) {
                return CKPT;
            }
            if (l.contains("[FP16]")) {
                return FP16;
            }
            return OTHER;
        }

        String style() {
            return switch (this) {
                case STEP -> "log-step";
                case EVAL -> "log-eval";
                case CKPT -> "log-ckpt";
                case FP16 -> "log-fp16";
                case WARN -> "log-warn";
                case ERROR -> "log-error";
                case PERF, OTHER -> "log-other";
            };
        }
    }

    private static final class LogCell extends ListCell<String> {
        @Override
        protected void updateItem(String item, boolean empty) {
            super.updateItem(item, empty);
            getStyleClass().removeAll("log-step", "log-eval", "log-ckpt", "log-fp16", "log-warn", "log-error", "log-other");
            if (empty || item == null) {
                setText(null);
                return;
            }
            setText(item);
            getStyleClass().add(Kind.of(item).style());
        }
    }
}
