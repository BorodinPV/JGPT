package com.veles.llm.jgpt.gui;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Slider;
import javafx.scene.control.TextArea;
import javafx.scene.control.Tooltip;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.scene.text.TextFlow;

/** Вкладка «Чат»: модель в процессе GUI, потоковый ответ, настройки сэмплинга. */
final class ChatTab extends BorderPane {

    private final ProjectPaths paths;
    private final ModelCatalog catalog;
    private final ChatEngine engine;

    private final ComboBox<ModelCatalog.ModelEntry> modelBox = new ComboBox<>();
    private final Button loadBtn = new Button("Загрузить");
    private final Button unloadBtn = new Button("Выгрузить");
    private final Label modelStatus = new Label("модель не загружена");
    private final Label vramWarn = new Label();
    private final CheckBox templateBox = new CheckBox("шаблон <user>/<assistant>");
    private final Slider temperature = slider(0, 1.5, 0.7, 0.05);
    private final Slider topK = slider(0, 100, 40, 1);
    private final Slider topP = slider(0.5, 1.0, 0.95, 0.01);
    private final Slider repPenalty = slider(1.0, 1.6, 1.15, 0.01);
    private final Slider maxNew = slider(16, 512, 200, 8);
    private final TextFlow transcript = new TextFlow();
    private final ScrollPane scroll = new ScrollPane(transcript);
    private final TextArea input = new TextArea();
    private final Button sendBtn = new Button("Отправить");
    private final Button stopBtn = new Button("Стоп");
    private final Button clearBtn = new Button("Новый диалог");

    private final List<ChatEngine.Turn> history = new ArrayList<>();
    private Text currentAnswer;
    private boolean generating;

    ChatTab(ProjectPaths paths, ModelCatalog catalog, ChatEngine engine) {
        this.paths = paths;
        this.catalog = catalog;
        this.engine = engine;
        setPadding(new Insets(10));
        setTop(buildModelBar());
        setCenter(buildCenter());
        setRight(buildSettings());
        refreshModels();
    }

    // ── UI ────────────────────────────────────────────────────────────────

    private VBox buildModelBar() {
        modelBox.setPrefWidth(340);
        modelBox.setTooltip(new Tooltip("checkpoints\\<каталог>\\model_*.bin с известным пресетом (env/*.env)"));
        Button refresh = new Button("↻");
        refresh.setOnAction(_ -> refreshModels());
        loadBtn.getStyleClass().add("primary");
        loadBtn.setOnAction(_ -> load(modelBox.getValue()));
        unloadBtn.setOnAction(_ -> unload());
        unloadBtn.setDisable(true);
        modelStatus.getStyleClass().add("muted");
        vramWarn.getStyleClass().add("yellow");
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox row1 = new HBox(8, new Label("Модель:"), modelBox, refresh, loadBtn, unloadBtn, spacer, templateBox);
        row1.setAlignment(Pos.CENTER_LEFT);
        HBox row2 = new HBox(16, modelStatus, vramWarn);
        VBox box = new VBox(6, row1, row2);
        box.getStyleClass().add("card");
        BorderPane.setMargin(box, new Insets(0, 0, 10, 0));
        modelBox.valueProperty().addListener((_, _, e) -> {
            if (e != null && !engine.isLoaded()) {
                templateBox.setSelected(e.preset().flag("JGPT_SFT"));
            }
        });
        return box;
    }

    private VBox buildCenter() {
        transcript.setPadding(new Insets(10));
        transcript.setLineSpacing(3);
        scroll.setFitToWidth(true);
        scroll.setStyle("-fx-background: #0b0f18; -fx-background-color: #0b0f18; -fx-border-color: " + Theme.BORDER + ";");
        transcript.setStyle("-fx-background-color: #0b0f18;");
        transcript.heightProperty().addListener((_, _, _) -> scroll.setVvalue(1.0));
        VBox.setVgrow(scroll, Priority.ALWAYS);

        input.setPromptText("Вопрос (Enter — отправить, Shift+Enter — новая строка)");
        input.setPrefRowCount(3);
        input.setWrapText(true);
        input.setOnKeyPressed(ev -> {
            if (ev.getCode() == KeyCode.ENTER && !ev.isShiftDown()) {
                ev.consume();
                send();
            }
        });
        sendBtn.getStyleClass().add("primary");
        sendBtn.setOnAction(_ -> send());
        stopBtn.getStyleClass().add("warn");
        stopBtn.setOnAction(_ -> engine.cancelGeneration());
        stopBtn.setDisable(true);
        clearBtn.setOnAction(_ -> {
            history.clear();
            transcript.getChildren().clear();
        });
        VBox buttons = new VBox(6, sendBtn, stopBtn, clearBtn);
        for (Button b : List.of(sendBtn, stopBtn, clearBtn)) {
            b.setMaxWidth(Double.MAX_VALUE);
        }
        HBox inputRow = new HBox(8, input, buttons);
        HBox.setHgrow(input, Priority.ALWAYS);
        VBox center = new VBox(8, scroll, inputRow);
        BorderPane.setMargin(center, new Insets(0, 10, 0, 0));
        return center;
    }

    private VBox buildSettings() {
        VBox box = new VBox(10,
                title("СЭМПЛИНГ"),
                labeled("temperature", temperature, "%.2f", "0 = жадный argmax"),
                labeled("top-k", topK, "%.0f", "0 = выкл."),
                labeled("top-p", topP, "%.2f", "1.0 = выкл."),
                labeled("repetition penalty", repPenalty, "%.2f", "1.0 = выкл.; штраф последним 256 токенам"),
                labeled("max new tokens", maxNew, "%.0f", "лимит длины ответа"));
        box.getStyleClass().add("card");
        box.setPrefWidth(250);
        return box;
    }

    private static Label title(String s) {
        Label l = new Label(s);
        l.getStyleClass().add("card-title");
        return l;
    }

    private static Slider slider(double min, double max, double val, double step) {
        Slider s = new Slider(min, max, val);
        s.setBlockIncrement(step);
        s.setMajorTickUnit(Math.max(step, (max - min) / 4));
        s.setSnapToTicks(false);
        return s;
    }

    private static VBox labeled(String name, Slider s, String fmt, String hint) {
        Label value = new Label(String.format(Locale.ROOT, fmt, s.getValue()));
        value.getStyleClass().add("cyan");
        s.valueProperty().addListener((_, _, v) -> value.setText(String.format(Locale.ROOT, fmt, v.doubleValue())));
        Region sp = new Region();
        HBox.setHgrow(sp, Priority.ALWAYS);
        HBox head = new HBox(6, new Label(name), sp, value);
        Label h = new Label(hint);
        h.getStyleClass().add("muted");
        return new VBox(2, head, s, h);
    }

    // ── модель ────────────────────────────────────────────────────────────

    void refreshModels() {
        ModelCatalog.ModelEntry sel = modelBox.getValue();
        List<ModelCatalog.ModelEntry> models = catalog.models();
        modelBox.getItems().setAll(models);
        if (sel != null) {
            models.stream().filter(m -> m.file().equals(sel.file())).findFirst().ifPresent(modelBox::setValue);
        }
        if (modelBox.getValue() == null && !models.isEmpty()) {
            models.stream()
                    .filter(m -> m.file().getFileName().toString().equals("model_best.bin"))
                    .findFirst()
                    .ifPresentOrElse(modelBox::setValue, () -> modelBox.setValue(models.get(0)));
        }
    }

    /** Из вкладки «Чекпоинты»: выбрать и загрузить. */
    void openModel(ModelCatalog.ModelEntry entry) {
        refreshModels();
        modelBox.getItems().stream().filter(m -> m.file().equals(entry.file())).findFirst()
                .ifPresentOrElse(modelBox::setValue, () -> {
                    modelBox.getItems().add(0, entry);
                    modelBox.setValue(entry);
                });
        load(entry);
    }

    private void load(ModelCatalog.ModelEntry entry) {
        if (entry == null || generating) {
            return;
        }
        loadBtn.setDisable(true);
        modelStatus.setText("загрузка " + entry + " …");
        vramWarn.setText(TrainingProcess.findTrainerJava().isPresent()
                ? "идёт обучение: модель чата займёт ещё ~0.7 ГБ VRAM" : "");
        engine.load(entry).whenComplete((desc, err) -> Platform.runLater(() -> {
            loadBtn.setDisable(false);
            if (err != null) {
                Throwable c = err.getCause() != null ? err.getCause() : err;
                modelStatus.setText("ошибка: " + c.getMessage());
                modelStatus.getStyleClass().add("red");
                unloadBtn.setDisable(true);
                return;
            }
            modelStatus.getStyleClass().remove("red");
            modelStatus.setText(entry + "  ·  " + desc);
            unloadBtn.setDisable(false);
            templateBox.setSelected(entry.preset().flag("JGPT_SFT") && engine.hasRoleTokens());
            templateBox.setDisable(!engine.hasRoleTokens());
        }));
    }

    private void unload() {
        engine.unload().whenComplete((_, _) -> Platform.runLater(() -> {
            modelStatus.setText("модель не загружена");
            unloadBtn.setDisable(true);
        }));
    }

    // ── диалог ────────────────────────────────────────────────────────────

    private void send() {
        String q = input.getText() == null ? "" : input.getText().trim();
        if (q.isEmpty() || generating) {
            return;
        }
        if (!engine.isLoaded()) {
            modelStatus.setText("сначала загрузите модель");
            return;
        }
        input.clear();
        history.add(new ChatEngine.Turn(true, q));
        appendText("Вы: ", "chat-user");
        appendText(q + "\n", "chat-assistant");
        appendText("Модель: ", "chat-user");
        currentAnswer = appendText("", "chat-assistant");
        generating = true;
        sendBtn.setDisable(true);
        stopBtn.setDisable(false);
        loadBtn.setDisable(true);
        long t0 = System.nanoTime();
        ChatEngine.Sampling s = new ChatEngine.Sampling(
                (float) temperature.getValue(),
                (int) Math.round(topK.getValue()),
                (float) topP.getValue(),
                (float) repPenalty.getValue(),
                (int) Math.round(maxNew.getValue()));
        engine.generate(history, templateBox.isSelected(), s, piece -> Platform.runLater(() -> {
            if (currentAnswer == null) {
                return;
            }
            if (piece.startsWith("\u0000")) {
                currentAnswer.setText(piece.substring(1));
            } else {
                currentAnswer.setText(currentAnswer.getText() + piece);
            }
        })).whenComplete((answer, err) -> Platform.runLater(() -> {
            generating = false;
            sendBtn.setDisable(false);
            stopBtn.setDisable(true);
            loadBtn.setDisable(false);
            if (err != null) {
                Throwable c = err.getCause() != null ? err.getCause() : err;
                appendText("\n[ошибка генерации: " + c.getMessage() + "]\n", "log-error");
                history.remove(history.size() - 1);
                currentAnswer = null;
                return;
            }
            String a = answer == null ? "" : answer;
            if (currentAnswer != null) {
                currentAnswer.setText(a);
            }
            history.add(new ChatEngine.Turn(false, a));
            double sec = (System.nanoTime() - t0) / 1e9;
            appendText(String.format(Locale.ROOT, "\n   %.1f с\n\n", sec), "chat-meta");
            currentAnswer = null;
        }));
    }

    private Text appendText(String s, String style) {
        Text t = new Text(s);
        t.getStyleClass().add(style);
        transcript.getChildren().add(t);
        return t;
    }
}
