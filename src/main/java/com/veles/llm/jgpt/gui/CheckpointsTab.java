package com.veles.llm.jgpt.gui;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import java.util.stream.Stream;

import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;

import com.veles.llm.jgpt.training.LLMTrainer;

/** Вкладка «Чекпоинты»: все файлы в checkpoints/*, шаг из заголовка чекпоинта, чат/удаление. */
final class CheckpointsTab extends BorderPane {

    private static final DateTimeFormatter DT = DateTimeFormatter.ofPattern("dd.MM HH:mm");

    record Row(Path file, String dir, String name, String kind, String step, String sizeMb, String modified, long modifiedMs) {}

    private final ProjectPaths paths;
    private final ModelCatalog catalog;
    private final Consumer<ModelCatalog.ModelEntry> openInChat;
    private final ObservableList<Row> rows = FXCollections.observableArrayList();
    private final TableView<Row> table = new TableView<>(rows);
    private final Label status = new Label();
    private final Button chatBtn = new Button("Открыть в чате");
    private final Button deleteBtn = new Button("Удалить");

    CheckpointsTab(ProjectPaths paths, ModelCatalog catalog, Consumer<ModelCatalog.ModelEntry> openInChat) {
        this.paths = paths;
        this.catalog = catalog;
        this.openInChat = openInChat;
        setPadding(new Insets(10));

        Button refresh = new Button("Обновить");
        refresh.setOnAction(_ -> reload());
        chatBtn.getStyleClass().add("primary");
        deleteBtn.getStyleClass().add("danger");
        chatBtn.setDisable(true);
        deleteBtn.setDisable(true);
        chatBtn.setOnAction(_ -> openSelectedInChat());
        deleteBtn.setOnAction(_ -> deleteSelected());
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        status.getStyleClass().add("muted");
        HBox top = new HBox(8, new Label("checkpoints\\"), refresh, status, spacer, chatBtn, deleteBtn);
        top.setAlignment(Pos.CENTER_LEFT);
        top.getStyleClass().add("card");
        BorderPane.setMargin(top, new Insets(0, 0, 8, 0));
        setTop(top);

        table.getColumns().addAll(List.of(
                col("Каталог", r -> r.dir(), 200),
                col("Файл", r -> r.name(), 220),
                col("Тип", r -> r.kind(), 110),
                col("Шаг", r -> r.step(), 80),
                col("МБ", r -> r.sizeMb(), 80),
                col("Изменён", r -> r.modified(), 110)));
        table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_ALL_COLUMNS);
        table.getSelectionModel().selectedItemProperty().addListener((_, _, r) -> {
            boolean model = r != null && r.kind().startsWith("веса");
            chatBtn.setDisable(!model || catalog.presetFor(r.file().getParent()).isEmpty());
            deleteBtn.setDisable(r == null);
        });
        table.setPlaceholder(new Label("нет чекпоинтов"));
        setCenter(table);
        reload();
    }

    private static TableColumn<Row, String> col(String title, java.util.function.Function<Row, String> f, int width) {
        TableColumn<Row, String> c = new TableColumn<>(title);
        c.setCellValueFactory(d -> new SimpleStringProperty(f.apply(d.getValue())));
        c.setPrefWidth(width);
        return c;
    }

    void reload() {
        status.setText("сканирую…");
        Thread t = new Thread(() -> {
            List<Row> out = scan();
            Platform.runLater(() -> {
                rows.setAll(out);
                long total = out.stream().mapToLong(r -> {
                    try {
                        return Files.size(r.file());
                    } catch (IOException _) {
                        return 0;
                    }
                }).sum();
                status.setText(out.size() + " файлов, " + String.format(Locale.ROOT, "%.1f ГБ", total / 1024.0 / 1024 / 1024));
            });
        }, "jgpt-gui-ckpt-scan");
        t.setDaemon(true);
        t.start();
    }

    private List<Row> scan() {
        List<Row> out = new ArrayList<>();
        Path root = paths.checkpointsDir();
        if (!Files.isDirectory(root)) {
            return out;
        }
        try (Stream<Path> dirs = Files.list(root)) {
            for (Path dir : dirs.filter(Files::isDirectory).sorted().toList()) {
                try (Stream<Path> files = Files.list(dir)) {
                    for (Path f : files.filter(x -> x.getFileName().toString().endsWith(".bin")).sorted().toList()) {
                        out.add(row(dir, f));
                    }
                }
            }
            try (Stream<Path> files = Files.list(root)) {
                for (Path f : files.filter(x -> Files.isRegularFile(x) && x.getFileName().toString().endsWith(".bin")).sorted().toList()) {
                    out.add(row(root, f));
                }
            }
        } catch (IOException _) {
            // частичный результат
        }
        out.sort(Comparator.comparing(Row::dir).thenComparing(Comparator.comparingLong(Row::modifiedMs).reversed()));
        return out;
    }

    private Row row(Path dir, Path f) {
        String name = f.getFileName().toString();
        String kind;
        String step = "";
        if (name.startsWith("checkpoint_")) {
            kind = "чекпоинт+Adam";
            try {
                int s = LLMTrainer.peekCheckpointGlobalStep(f);
                step = s > 0 ? Integer.toString(s) : "?";
            } catch (Throwable _) {
                step = "?";
            }
        } else if (name.startsWith("model_")) {
            kind = "веса";
            Path paired = dir.resolve(name.replaceFirst("^model_", "checkpoint_"));
            if (Files.isRegularFile(paired)) {
                try {
                    int s = LLMTrainer.peekCheckpointGlobalStep(paired);
                    step = s > 0 ? Integer.toString(s) : "";
                } catch (Throwable _) {
                    step = "";
                }
            }
        } else if (name.startsWith("tokenizer")) {
            kind = "токенизатор";
        } else {
            kind = "файл";
        }
        long size = 0;
        long mod = 0;
        try {
            size = Files.size(f);
            mod = Files.getLastModifiedTime(f).toMillis();
        } catch (IOException _) {
            // оставляем нули
        }
        String dirName = dir.equals(paths.checkpointsDir()) ? "." : dir.getFileName().toString();
        return new Row(f, dirName, name, kind, step, String.format(Locale.ROOT, "%,d", size / 1024 / 1024).replace(',', ' '),
                mod == 0 ? "" : DT.format(Instant.ofEpochMilli(mod).atZone(ZoneId.systemDefault())), mod);
    }

    private void openSelectedInChat() {
        Row r = table.getSelectionModel().getSelectedItem();
        if (r == null) {
            return;
        }
        catalog.entryFor(r.file()).ifPresent(openInChat);
    }

    private void deleteSelected() {
        Row r = table.getSelectionModel().getSelectedItem();
        if (r == null) {
            return;
        }
        Alert a = new Alert(Alert.AlertType.CONFIRMATION,
                "Удалить " + r.dir() + "\\" + r.name() + " (" + r.sizeMb() + " МБ)?\nЭто необратимо.",
                ButtonType.OK, ButtonType.CANCEL);
        a.setHeaderText("Удаление файла");
        Theme.apply(a.getDialogPane().getScene());
        if (a.showAndWait().orElse(ButtonType.CANCEL) != ButtonType.OK) {
            return;
        }
        try {
            Files.deleteIfExists(r.file());
            rows.remove(r);
        } catch (IOException e) {
            Alert err = new Alert(Alert.AlertType.ERROR, e.getMessage(), ButtonType.OK);
            err.setHeaderText("Не удалось удалить");
            Theme.apply(err.getDialogPane().getScene());
            err.showAndWait();
        }
    }
}
