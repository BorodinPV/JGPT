package com.veles.llm.jgpt.gui;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Запуск/остановка обучения. Обучение — всегда отдельный процесс (тот же {@code jgpt-train-*.cmd}, что и из консоли),
 * чтобы GUI можно было закрыть, а VRAM тренера не делить с чатом. Мягкая остановка — файл {@code state/STOP}.
 */
final class TrainingProcess {

    private final ProjectPaths paths;
    private volatile Process child;
    private volatile Thread pump;

    TrainingProcess(ProjectPaths paths) {
        this.paths = paths;
    }

    /** Флаги запуска, как у .cmd. */
    record Flags(boolean fresh, boolean restartPlan, boolean rebuildCuda) {}

    synchronized void start(TrainRun run, Flags flags, Consumer<String> lineSink, Runnable onExit) throws IOException {
        if (isChildAlive()) {
            throw new IOException("обучение уже запущено из этого GUI");
        }
        List<String> cmd = new ArrayList<>();
        cmd.add("cmd.exe");
        cmd.add("/c");
        cmd.add(run.cmd().toString());
        if (!flags.rebuildCuda()) {
            cmd.add("--no-build");
        }
        if (flags.fresh()) {
            cmd.add("--fresh");
        }
        if (flags.restartPlan()) {
            cmd.add("--restart-plan");
        }
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.directory(paths.root().toFile());
        pb.redirectErrorStream(true);
        // GUI сам может быть запущен с JGPT_FINETUNE в окружении — дочернему передаём только явный флаг.
        pb.environment().remove("JGPT_FINETUNE");
        Process p = pb.start();
        child = p;
        lineSink.accept("> " + String.join(" ", cmd));
        Thread t =
                new Thread(
                        () -> {
                            try (BufferedReader r =
                                    new BufferedReader(
                                            new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                                String line;
                                while ((line = r.readLine()) != null) {
                                    lineSink.accept(line);
                                }
                            } catch (IOException _) {
                                // поток закрыт вместе с процессом
                            }
                            int code;
                            try {
                                code = p.waitFor();
                            } catch (InterruptedException _) {
                                Thread.currentThread().interrupt();
                                code = -1;
                            }
                            lineSink.accept("[GUI] процесс завершён, код " + code);
                            onExit.run();
                        },
                        "jgpt-train-pump");
        t.setDaemon(true);
        t.start();
        pump = t;
    }

    boolean isChildAlive() {
        Process p = child;
        return p != null && p.isAlive();
    }

    /** Мягкая остановка: {@code state/STOP}; тренер допишет checkpoint_final и выйдет сам. */
    void requestSoftStop() throws IOException {
        Files.createDirectories(paths.stateDir());
        Files.writeString(paths.stopFile(), "stop\n", StandardCharsets.UTF_8);
    }

    /** Найденный java-процесс тренера. */
    record Trainer(long pid, String dataDir) {
        String describe() {
            return "pid " + pid + (dataDir.isEmpty() ? "" : "  data=" + dataDir);
        }
    }

    private static final String MAIN_CLASS = "com.veles.llm.jgpt.app.AllBooksTrain";
    private static final long CIM_CACHE_MS = 8_000;
    private static volatile long cimCheckedAt;
    private static volatile Optional<Trainer> cimResult = Optional.empty();

    /**
     * Найти java-процесс {@code AllBooksTrain} — наш дочерний или запущенный из консоли. {@code ProcessHandle} на
     * Windows не отдаёт командную строку чужих процессов, поэтому для внешних используется {@code Win32_Process}
     * через PowerShell (кэш {@value #CIM_CACHE_MS} мс).
     */
    static Optional<Trainer> findTrainerJava() {
        Optional<Trainer> viaHandle =
                ProcessHandle.allProcesses()
                        .filter(ph -> ph.info().commandLine().map(l -> l.contains(MAIN_CLASS)).orElse(false))
                        .findFirst()
                        .map(ph -> new Trainer(ph.pid(), dataDirOf(ph.info().commandLine().orElse(""))));
        if (viaHandle.isPresent()) {
            return viaHandle;
        }
        long now = System.currentTimeMillis();
        if (now - cimCheckedAt < CIM_CACHE_MS) {
            return cimResult.filter(t -> ProcessHandle.of(t.pid()).map(ProcessHandle::isAlive).orElse(false));
        }
        cimCheckedAt = now;
        cimResult = queryCim();
        return cimResult;
    }

    private static Optional<Trainer> queryCim() {
        if (!System.getProperty("os.name", "").toLowerCase().contains("win")) {
            return Optional.empty();
        }
        // -EncodedCommand: кавычки внутри аргумента ProcessBuilder на Windows не экранируются.
        String script =
                "Get-CimInstance -Query \"SELECT ProcessId, CommandLine FROM Win32_Process WHERE Name='java.exe'\""
                        + " | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }";
        String encoded = java.util.Base64.getEncoder().encodeToString(script.getBytes(java.nio.charset.StandardCharsets.UTF_16LE));
        List<String> cmd = List.of("powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded);
        try {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            Process p = pb.start();
            StringBuilder sb = new StringBuilder();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = r.readLine()) != null) {
                    sb.append(line).append('\n');
                }
            }
            if (!p.waitFor(10, java.util.concurrent.TimeUnit.SECONDS)) {
                p.destroyForcibly();
                return Optional.empty();
            }
            for (String line : sb.toString().split("\n")) {
                int tab = line.indexOf('\t');
                if (tab <= 0 || !line.contains(MAIN_CLASS)) {
                    continue;
                }
                try {
                    long pid = Long.parseLong(line.substring(0, tab).trim());
                    return Optional.of(new Trainer(pid, dataDirOf(line.substring(tab + 1))));
                } catch (NumberFormatException _) {
                    // строка без pid
                }
            }
        } catch (IOException _) {
            // PowerShell недоступен — считаем, что внешнего тренера нет
        } catch (InterruptedException _) {
            Thread.currentThread().interrupt();
        }
        return Optional.empty();
    }

    /** Аргумент {@code --data-dir} из командной строки тренера — чтобы понять, какой запуск идёт. */
    private static String dataDirOf(String line) {
        int k = line.indexOf("--data-dir");
        if (k < 0) {
            return "";
        }
        String data = line.substring(k + "--data-dir".length()).trim();
        if (data.startsWith("\"")) {
            int e = data.indexOf('"', 1);
            return e > 0 ? data.substring(1, e) : data;
        }
        int e = data.indexOf(' ');
        return e > 0 ? data.substring(0, e) : data;
    }

    /** Жёсткое убийство java тренера (без checkpoint_final). Возвращает {@code true}, если процесс был найден. */
    boolean killTrainerJava() {
        Optional<Trainer> t = findTrainerJava();
        t.flatMap(x -> ProcessHandle.of(x.pid())).ifPresent(ProcessHandle::destroyForcibly);
        Process p = child;
        if (p != null && p.isAlive()) {
            p.descendants().forEach(ProcessHandle::destroyForcibly);
            p.destroyForcibly();
        }
        cimCheckedAt = 0;
        return t.isPresent();
    }
}
