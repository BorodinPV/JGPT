package com.veles.llm.jgpt.gui;

import java.nio.charset.StandardCharsets;
import java.util.Base64;

import javafx.scene.Scene;

/** Тёмная тема. CSS встроен строкой и подключается как data:-URL, чтобы GUI собирался одним javac без ресурсов. */
final class Theme {

    private Theme() {}

    static final String BG = "#0f1420";
    static final String CARD = "#161c2a";
    static final String BORDER = "#252d3d";
    static final String TEXT = "#d6deeb";
    static final String MUTED = "#7d8aa5";
    static final String CYAN = "#22d3ee";
    static final String BLUE = "#4f8ef7";
    static final String GREEN = "#22c55e";
    static final String YELLOW = "#f0b429";
    static final String RED = "#f04747";
    static final String PURPLE = "#a78bfa";

    private static final String CSS =
            """
            .root { -fx-base: %CARD%; -fx-background: %BG%; -fx-background-color: %BG%; -fx-text-fill: %TEXT%;
                    -fx-font-family: "Segoe UI", system-ui, sans-serif; -fx-font-size: 13px; -fx-accent: %CYAN%;
                    -fx-focus-color: %CYAN%; -fx-faint-focus-color: #22d3ee22; -fx-control-inner-background: %CARD%; }
            .label { -fx-text-fill: %TEXT%; }
            .muted { -fx-text-fill: %MUTED%; -fx-font-size: 11px; }
            .card { -fx-background-color: %CARD%; -fx-background-radius: 8; -fx-border-color: %BORDER%;
                    -fx-border-radius: 8; -fx-padding: 12; }
            .card-title { -fx-text-fill: %MUTED%; -fx-font-size: 11px; -fx-font-weight: bold; }
            .stat-value { -fx-font-size: 26px; -fx-font-weight: bold; }
            .cyan { -fx-text-fill: %CYAN%; } .blue { -fx-text-fill: %BLUE%; } .green { -fx-text-fill: %GREEN%; }
            .yellow { -fx-text-fill: %YELLOW%; } .red { -fx-text-fill: %RED%; } .purple { -fx-text-fill: %PURPLE%; }
            .tab-pane .tab-header-area .tab-header-background { -fx-background-color: %BG%; }
            .tab-pane .tab { -fx-background-color: %CARD%; -fx-background-radius: 6 6 0 0; -fx-padding: 6 16 6 16; }
            .tab-pane .tab:selected { -fx-background-color: #1e2637; }
            .tab-pane .tab .tab-label { -fx-text-fill: %MUTED%; -fx-font-size: 13px; }
            .tab-pane .tab:selected .tab-label { -fx-text-fill: %TEXT%; }
            .tab-pane .tab:selected .focus-indicator { -fx-border-color: transparent; }
            .button { -fx-background-color: #1e2637; -fx-text-fill: %TEXT%; -fx-background-radius: 6;
                      -fx-border-color: %BORDER%; -fx-border-radius: 6; -fx-padding: 6 14 6 14; -fx-cursor: hand; }
            .button:hover { -fx-background-color: #263048; }
            .button:disabled { -fx-opacity: 0.45; }
            .button.primary { -fx-background-color: %CYAN%; -fx-text-fill: #04121a; -fx-font-weight: bold; -fx-border-color: transparent; }
            .button.primary:hover { -fx-background-color: #5ee2f5; }
            .button.danger { -fx-background-color: #3a1a1f; -fx-text-fill: #ff8a8a; -fx-border-color: #6b2a32; }
            .button.danger:hover { -fx-background-color: #4d2028; }
            .button.warn { -fx-background-color: #3a2f12; -fx-text-fill: %YELLOW%; -fx-border-color: #6b5620; }
            .combo-box, .choice-box, .text-field, .text-area, .spinner { -fx-background-color: #1e2637;
                      -fx-text-fill: %TEXT%; -fx-background-radius: 6; -fx-border-color: %BORDER%; -fx-border-radius: 6; }
            .combo-box .list-cell { -fx-text-fill: %TEXT%; -fx-background-color: transparent; }
            .combo-box-popup .list-view { -fx-background-color: %CARD%; -fx-border-color: %BORDER%; }
            .combo-box-popup .list-view .list-cell { -fx-text-fill: %TEXT%; -fx-background-color: %CARD%; }
            .combo-box-popup .list-view .list-cell:hover, .combo-box-popup .list-view .list-cell:filled:selected
                      { -fx-background-color: #263048; }
            .text-area .content { -fx-background-color: #0b0f18; }
            .text-area, .text-field { -fx-prompt-text-fill: %MUTED%; -fx-highlight-fill: #22d3ee55; }
            .check-box { -fx-text-fill: %TEXT%; }
            .check-box .box { -fx-background-color: #1e2637; -fx-border-color: %BORDER%; -fx-border-radius: 3; -fx-background-radius: 3; }
            .check-box:selected .mark { -fx-background-color: %CYAN%; }
            .progress-bar .track { -fx-background-color: #1e2637; -fx-background-radius: 4; }
            .progress-bar .bar { -fx-background-color: linear-gradient(to right, %BLUE%, %CYAN%); -fx-background-radius: 4; -fx-padding: 5; }
            .slider .track { -fx-background-color: #1e2637; }
            .slider .thumb { -fx-background-color: %CYAN%; }
            .list-view, .table-view { -fx-background-color: %CARD%; -fx-border-color: %BORDER%; -fx-border-radius: 6; -fx-background-radius: 6; }
            .list-view .list-cell { -fx-background-color: transparent; -fx-text-fill: %TEXT%; -fx-padding: 2 6 2 6;
                                    -fx-font-family: "Cascadia Mono", Consolas, monospace; -fx-font-size: 12px; }
            .list-view .list-cell:filled:selected { -fx-background-color: #263048; }
            .list-view .list-cell.log-step { -fx-text-fill: %TEXT%; }
            .list-view .list-cell.log-eval { -fx-text-fill: %GREEN%; }
            .list-view .list-cell.log-ckpt { -fx-text-fill: %PURPLE%; }
            .list-view .list-cell.log-fp16 { -fx-text-fill: %BLUE%; }
            .list-view .list-cell.log-warn { -fx-text-fill: %YELLOW%; }
            .list-view .list-cell.log-error { -fx-text-fill: %RED%; }
            .list-view .list-cell.log-other { -fx-text-fill: %MUTED%; }
            .log-error { -fx-fill: %RED%; }
            .table-view .column-header-background { -fx-background-color: #1e2637; -fx-background-radius: 6 6 0 0; }
            .table-view .column-header, .table-view .filler { -fx-background-color: transparent; -fx-border-color: %BORDER%; }
            .table-view .column-header .label { -fx-text-fill: %MUTED%; -fx-font-weight: bold; }
            .table-row-cell { -fx-background-color: %CARD%; -fx-text-fill: %TEXT%; -fx-border-color: transparent transparent %BORDER% transparent; }
            .table-row-cell:odd { -fx-background-color: #131927; }
            .table-row-cell:selected { -fx-background-color: #263048; }
            .table-row-cell .table-cell { -fx-text-fill: %TEXT%; -fx-border-color: transparent; }
            .table-view .placeholder .label { -fx-text-fill: %MUTED%; }
            .chart { -fx-padding: 4; }
            .chart-plot-background { -fx-background-color: #0b0f18; }
            .chart-vertical-grid-lines, .chart-horizontal-grid-lines { -fx-stroke: #1e2637; }
            .chart-alternative-row-fill { -fx-fill: transparent; -fx-stroke: transparent; }
            .axis { -fx-tick-label-fill: %MUTED%; -fx-font-size: 10px; }
            .axis-label { -fx-text-fill: %MUTED%; }
            .chart-legend { -fx-background-color: transparent; }
            .chart-legend-item { -fx-text-fill: %MUTED%; }
            .chart-series-line { -fx-stroke-width: 1.6px; }
            .default-color0.chart-series-line { -fx-stroke: %BLUE%; }
            .default-color1.chart-series-line { -fx-stroke: %GREEN%; }
            .default-color2.chart-series-line { -fx-stroke: %YELLOW%; }
            .default-color0.chart-line-symbol, .default-color1.chart-line-symbol, .default-color2.chart-line-symbol
                      { -fx-background-color: transparent, transparent; -fx-padding: 0; }
            .default-color1.chart-line-symbol { -fx-background-color: %GREEN%, #0b0f18; -fx-padding: 2.5px; -fx-background-radius: 3px; }
            .scroll-bar { -fx-background-color: %BG%; }
            .scroll-bar .thumb { -fx-background-color: #2b3549; -fx-background-radius: 4; }
            .scroll-bar .increment-button, .scroll-bar .decrement-button { -fx-background-color: transparent; -fx-padding: 0 4 0 4; }
            .scroll-bar .increment-arrow, .scroll-bar .decrement-arrow { -fx-background-color: %MUTED%; }
            .split-pane { -fx-background-color: %BG%; }
            .split-pane .split-pane-divider { -fx-background-color: %BORDER%; -fx-padding: 0 1 0 1; }
            .chat-user { -fx-fill: %CYAN%; -fx-font-weight: bold; }
            .chat-assistant { -fx-fill: %TEXT%; }
            .chat-meta { -fx-fill: %MUTED%; -fx-font-size: 11px; }
            .tooltip { -fx-background-color: #1e2637; -fx-text-fill: %TEXT%; -fx-border-color: %BORDER%; }
            .dialog-pane { -fx-background-color: %CARD%; }
            .dialog-pane .label, .dialog-pane .content { -fx-text-fill: %TEXT%; }
            .dialog-pane .header-panel { -fx-background-color: #1e2637; }
            .dialog-pane .header-panel .label { -fx-text-fill: %TEXT%; -fx-font-size: 15px; }
            .titled-pane > .title { -fx-background-color: #1e2637; -fx-text-fill: %TEXT%; }
            .titled-pane > .content { -fx-background-color: %CARD%; -fx-border-color: %BORDER%; }
            """
                    .replace("%BG%", BG)
                    .replace("%CARD%", CARD)
                    .replace("%BORDER%", BORDER)
                    .replace("%TEXT%", TEXT)
                    .replace("%MUTED%", MUTED)
                    .replace("%CYAN%", CYAN)
                    .replace("%BLUE%", BLUE)
                    .replace("%GREEN%", GREEN)
                    .replace("%YELLOW%", YELLOW)
                    .replace("%RED%", RED)
                    .replace("%PURPLE%", PURPLE);

    static String stylesheetUrl() {
        return "data:text/css;base64," + Base64.getEncoder().encodeToString(CSS.getBytes(StandardCharsets.UTF_8));
    }

    static void apply(Scene scene) {
        scene.getStylesheets().add(stylesheetUrl());
    }
}
