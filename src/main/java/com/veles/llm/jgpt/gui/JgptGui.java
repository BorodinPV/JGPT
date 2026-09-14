package com.veles.llm.jgpt.gui;

import java.util.List;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.stage.Stage;

/**
 * Десктопный GUI JGPT (JavaFX из Liberica Full): обучение (старт/стоп, живые графики), лог, чекпоинты, чат с моделью.
 *
 * <p>Запуск: {@code scripts\windows\jgpt-gui.cmd}. Обучение — отдельный процесс ({@code jgpt-train-*.cmd}), метрики —
 * {@code state/stats.json}; чат грузит веса в этот JVM.
 */
public final class JgptGui extends Application {

    private TrainingTab trainingTab;
    private LogTab logTab;
    private ChatEngine chatEngine;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) {
        ProjectPaths paths = ProjectPaths.detect();
        List<TrainRun> runs = TrainRun.discover(paths);
        ModelCatalog catalog = new ModelCatalog(paths, runs);
        TrainingProcess proc = new TrainingProcess(paths);
        chatEngine = new ChatEngine(paths);

        trainingTab = new TrainingTab(paths, proc, runs);
        TrainRun initialLog = runs.stream().filter(r -> r.name().equals("28L-wide-sft")).findFirst()
                .orElse(runs.isEmpty() ? null : runs.get(0));
        logTab = new LogTab(runs, initialLog);
        ChatTab chatTab = new ChatTab(paths, catalog, chatEngine);
        TabPane tabs = new TabPane();
        CheckpointsTab ckptTab = new CheckpointsTab(paths, catalog, entry -> {
            tabs.getSelectionModel().select(3);
            chatTab.openModel(entry);
        });

        tabs.getTabs().addAll(
                tab("Обучение", trainingTab),
                tab("Лог", logTab),
                tab("Чекпоинты", ckptTab),
                tab("Чат", chatTab));
        tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);
        tabs.getSelectionModel().selectedItemProperty().addListener((_, _, t) -> {
            logTab.setActive(t != null && "Лог".equals(t.getText()));
            if (t != null && "Чекпоинты".equals(t.getText())) {
                ckptTab.reload();
            }
            if (t != null && "Чат".equals(t.getText())) {
                chatTab.refreshModels();
            }
        });

        // -Djgpt.gui.tab=N — открыть вкладку N при старте (для скриптов/скриншотов).
        try {
            int idx = Integer.parseInt(System.getProperty("jgpt.gui.tab", "0"));
            if (idx >= 0 && idx < tabs.getTabs().size()) {
                tabs.getSelectionModel().select(idx);
            }
        } catch (NumberFormatException _) {
            // по умолчанию первая вкладка
        }
        logTab.setActive("Лог".equals(tabs.getSelectionModel().getSelectedItem().getText()));

        Scene scene = new Scene(tabs, 1380, 860);
        Theme.apply(scene);
        stage.setTitle("JGPT — " + paths.root());
        stage.setScene(scene);
        stage.setMinWidth(1000);
        stage.setMinHeight(640);
        stage.show();
    }

    private static Tab tab(String title, javafx.scene.Node content) {
        Tab t = new Tab(title, content);
        t.setClosable(false);
        return t;
    }

    @Override
    public void stop() {
        if (trainingTab != null) {
            trainingTab.shutdown();
        }
        if (logTab != null) {
            logTab.shutdown();
        }
        if (chatEngine != null) {
            chatEngine.shutdown();
        }
        Platform.exit();
    }
}
