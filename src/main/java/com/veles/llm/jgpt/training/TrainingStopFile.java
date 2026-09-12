package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.util.LogFmt;

import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Мягкая остановка обучения: файл {@code state/STOP} (или {@code JGPT_STOP_FILE}) и SIGINT/SIGTERM.
 *
 * <p>На Windows Ctrl+C в {@code .cmd} даёт «Terminate batch job / завершить пакет?» и часто убивает JVM
 * без shutdown hook — чекпоинт не пишется. Файл STOP читается из цикла {@link LLMTrainer#train()},
 * после чего вызывающий сохраняет {@code checkpoint_final.bin} на основном потоке.
 */
public final class TrainingStopFile {

    private static final Logger log = LoggerFactory.getLogger(TrainingStopFile.class);
    private static final AtomicBoolean interruptInstalled = new AtomicBoolean();
    private static final AtomicBoolean interruptLogged = new AtomicBoolean();

    private TrainingStopFile() {}

    public static Path resolveFromEnv() {
        String e = System.getenv("JGPT_STOP_FILE");
        if (e != null && !e.isBlank()) {
            return Path.of(e.trim());
        }
        return Path.of("state", "STOP");
    }

    public static boolean isPresent(Path path) {
        return path != null && Files.isRegularFile(path);
    }

    public static void consume(Path path) {
        if (path == null) {
            return;
        }
        try {
            Files.deleteIfExists(path);
        } catch (IOException e) {
            log.warn("{} не удалось удалить STOP-файл {}: {}", LogFmt.badge("STOP"), path, e.toString());
        }
    }

    /** Удаляет STOP от прошлого запуска, чтобы новый прогон сразу не вышел. */
    public static void clearStaleAtStartup() {
        Path p = resolveFromEnv();
        if (!isPresent(p)) {
            return;
        }
        consume(p);
        log.info("{} удалён старый {} (иначе обучение сразу остановилось бы)", LogFmt.badge("STOP"), p);
    }

    /**
     * Перехватывает SIGINT/SIGTERM, чтобы JVM не выходила сразу: цикл train дойдёт до проверки флага
     * и сохранит чекпоинт на основном потоке. На Windows с piped java сигнал может не дойти — тогда STOP-файл.
     */
    public static void installOsInterrupt(Runnable onStop) {
        if (onStop == null || !interruptInstalled.compareAndSet(false, true)) {
            return;
        }
        Runnable once =
                () -> {
                    if (interruptLogged.compareAndSet(false, true)) {
                        log.warn(
                                "{} SIGINT/SIGTERM — мягкая остановка (дождитесь checkpoint_final, не закрывайте окно)",
                                LogFmt.badge("STOP"));
                    }
                    onStop.run();
                };
        boolean any = installOneSignal("INT", once) | installOneSignal("TERM", once);
        if (!any) {
            log.info(
                    "{} OS-сигнал не установлен — останавливайте через файл {}",
                    LogFmt.badge("STOP"),
                    resolveFromEnv());
        }
    }

    private static boolean installOneSignal(String name, Runnable onStop) {
        try {
            Class<?> sigCl = Class.forName("sun.misc.Signal");
            Class<?> handlerCl = Class.forName("sun.misc.SignalHandler");
            Object sig = sigCl.getConstructor(String.class).newInstance(name);
            Object handler =
                    Proxy.newProxyInstance(
                            handlerCl.getClassLoader(),
                            new Class<?>[] {handlerCl},
                            (proxy, method, args) -> {
                                if ("handle".equals(method.getName())) {
                                    onStop.run();
                                }
                                return null;
                            });
            sigCl.getMethod("handle", sigCl, handlerCl).invoke(null, sig, handler);
            return true;
        } catch (Throwable e) {
            log.debug("{} не удалось поставить handler {}: {}", LogFmt.badge("STOP"), name, e.toString());
            return false;
        }
    }
}
