package com.veles.llm.jgpt.cuda;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Единая загрузка {@code libjgpt_cuda.so}: {@code -Djgpt.cuda.lib}, переменная {@code JGPT_CUDA_LIB},
 * затем {@code build/libjgpt_cuda.so} относительно {@code user.dir}, иначе {@link System#loadLibrary}.
 *
 * <p><b>Потокобезопасность:</b> повторные вызовы {@link #load()} после успешной загрузки — no-op (быстрый путь).
 * Первый успех фиксируется под монитором класса; параллельные первые вызовы не приводят к повторному
 * {@link System#load} для того же class loader.
 *
 * <p><b>Жизненный цикл:</b> выгрузить нативную библиотеку из стандартной JVM нельзя; новая загрузка возможна
 * только в другом {@link ClassLoader}. {@link #getLastLoadedPath()} осмысленен только после успешного
 * {@link #load()} (обновляется сразу после успешного {@code System.load} / {@code loadLibrary}).
 */
public final class TensorCudaLibrary {

    private static volatile boolean loaded;
    private static volatile String lastLoadedPath;

    private TensorCudaLibrary() {}

    /** {@code true} после первой успешной загрузки в этом загрузчике классов. */
    public static boolean isLoaded() {
        return loaded;
    }

    /** Путь к загруженному файлу или метка {@code jgpt_cuda (java.library.path)}; до {@link #load()} — {@code null}. */
    public static String getLastLoadedPath() {
        return lastLoadedPath;
    }

    /**
     * @throws UnsatisfiedLinkError если библиотека не найдена
     */
    public static void load() {
        if (loaded) {
            return;
        }
        synchronized (TensorCudaLibrary.class) {
            if (loaded) {
                return;
            }

            String override = System.getProperty("jgpt.cuda.lib");
            if (override != null && !override.isBlank()) {
                Path p = Path.of(override.trim());
                if (Files.isRegularFile(p)) {
                    String abs = p.toAbsolutePath().toString();
                    System.load(abs);
                    lastLoadedPath = abs;
                    loaded = true;
                    return;
                }
                throw new UnsatisfiedLinkError("jgpt.cuda.lib not found: " + p);
            }

            String env = System.getenv("JGPT_CUDA_LIB");
            if (env != null && !env.isBlank()) {
                Path p = Path.of(env.trim());
                if (Files.isRegularFile(p)) {
                    String abs = p.toAbsolutePath().toString();
                    System.load(abs);
                    lastLoadedPath = abs;
                    loaded = true;
                    return;
                }
                throw new UnsatisfiedLinkError("JGPT_CUDA_LIB not found: " + p);
            }

            for (String rel : relativeCandidatePaths()) {
                Path p = Path.of(rel).normalize();
                if (Files.isRegularFile(p)) {
                    String abs = p.toAbsolutePath().toString();
                    System.load(abs);
                    lastLoadedPath = abs;
                    loaded = true;
                    return;
                }
            }

            try {
                System.loadLibrary("jgpt_cuda");
                lastLoadedPath = "jgpt_cuda (java.library.path)";
                loaded = true;
                return;
            } catch (UnsatisfiedLinkError e) {
                System.err.println(
                        "[TensorCudaLibrary] System.loadLibrary(jgpt_cuda) не удался: " + e.getMessage());
                System.err.println(
                        "[TensorCudaLibrary] java.library.path="
                                + System.getProperty("java.library.path", "<пусто>"));
            }

            throw new UnsatisfiedLinkError(buildErrorMessage());
        }
    }

    private static String[] relativeCandidatePaths() {
        String userDir = System.getProperty("user.dir", ".");
        return new String[] {userDir + "/build/libjgpt_cuda.so", userDir + "/../build/libjgpt_cuda.so"};
    }

    private static String buildErrorMessage() {
        String[] candidates = relativeCandidatePaths();
        String env = System.getenv("JGPT_CUDA_LIB");
        return "libjgpt_cuda.so не найден. Порядок поиска:\n"
                + "  1. -Djgpt.cuda.lib="
                + System.getProperty("jgpt.cuda.lib", "<не задано>")
                + "\n"
                + "  2. JGPT_CUDA_LIB="
                + (env != null && !env.isBlank() ? env : "<не задано>")
                + "\n"
                + "  3. Относительно user.dir: "
                + String.join(", ", candidates)
                + "\n"
                + "  4. java.library.path="
                + System.getProperty("java.library.path", "<пусто>")
                + "\n"
                + "Сборка: cd src/main/cpp && cmake -B ../../build -S . && cmake --build ../../build";
    }
}
