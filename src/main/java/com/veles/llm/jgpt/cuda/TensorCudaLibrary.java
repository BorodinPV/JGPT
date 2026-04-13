package com.veles.llm.jgpt.cuda;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Загрузка нативных библиотек CUDA: сначала {@code libjgpt_cuda_extra.so} (JNI/ядра из {@code jgpt_cuda_extra.cu}),
 * затем {@code libjgpt_cuda.so} (основной модуль). Обе собираются в {@code build/} и должны лежать рядом (или в
 * {@code java.library.path} при {@link System#loadLibrary}).
 *
 * <p>Порядок: {@code -Djgpt.cuda.lib} / {@code JGPT_CUDA_LIB} (путь к <b>основной</b> {@code libjgpt_cuda.so}),
 * относительные пути к {@code build/libjgpt_cuda.so}, иначе {@link System#loadLibrary}.
 *
 * <p><b>Потокобезопасность:</b> повторные вызовы {@link #load()} после успешной загрузки — no-op (быстрый путь).
 * Первый успех фиксируется под монитором класса; параллельные первые вызовы не приводят к повторному
 * {@link System#load} для того же class loader.
 *
 * <p><b>Жизненный цикл:</b> выгрузить нативную библиотеку из стандартной JVM нельзя; новая загрузка возможна
 * только в другом {@link ClassLoader}. {@link #getLastLoadedPath()} осмысленен только после успешного
 * {@link #load()} (обновляется сразу после успешного {@code System.load} основной библиотеки).
 */
public final class TensorCudaLibrary {

    private static volatile boolean loaded;
    private static volatile String lastLoadedPath;

    private TensorCudaLibrary() {}

    /** {@code true} после первой успешной загрузки в этом загрузчике классов. */
    public static boolean isLoaded() {
        return loaded;
    }

    /** Путь к загруженному основному {@code .so} или метка {@code jgpt_cuda (java.library.path)}; до {@link #load()} — {@code null}. */
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
                    loadMainWithCompanionExtra(p);
                    lastLoadedPath = p.toAbsolutePath().toString();
                    loaded = true;
                    return;
                }
                throw new UnsatisfiedLinkError("jgpt.cuda.lib not found: " + p);
            }

            String env = System.getenv("JGPT_CUDA_LIB");
            if (env != null && !env.isBlank()) {
                Path p = Path.of(env.trim());
                if (Files.isRegularFile(p)) {
                    loadMainWithCompanionExtra(p);
                    lastLoadedPath = p.toAbsolutePath().toString();
                    loaded = true;
                    return;
                }
                throw new UnsatisfiedLinkError("JGPT_CUDA_LIB not found: " + p);
            }

            for (String rel : relativeCandidatePaths()) {
                Path p = Path.of(rel).normalize();
                if (Files.isRegularFile(p)) {
                    loadMainWithCompanionExtra(p);
                    lastLoadedPath = p.toAbsolutePath().toString();
                    loaded = true;
                    return;
                }
            }

            try {
                System.loadLibrary("jgpt_cuda_extra");
                System.loadLibrary("jgpt_cuda");
                lastLoadedPath = "jgpt_cuda_extra + jgpt_cuda (java.library.path)";
                loaded = true;
                return;
            } catch (UnsatisfiedLinkError e) {
                System.err.println(
                        "[TensorCudaLibrary] loadLibrary(jgpt_cuda_extra/jgpt_cuda) не удался: " + e.getMessage());
                System.err.println(
                        "[TensorCudaLibrary] java.library.path="
                                + System.getProperty("java.library.path", "<пусто>"));
            }

            throw new UnsatisfiedLinkError(buildErrorMessage());
        }
    }

    /** Сначала companion {@code jgpt_cuda_extra}, затем основной модуль (тот же каталог, что и {@code mainSo}). */
    private static void loadMainWithCompanionExtra(Path mainSo) {
        Path dir = mainSo.toAbsolutePath().getParent();
        if (dir != null) {
            Path extra = dir.resolve(companionExtraFileName());
            if (!Files.isRegularFile(extra)) {
                throw new UnsatisfiedLinkError(
                        "Рядом с "
                                + mainSo
                                + " ожидается "
                                + extra
                                + " (соберите cmake-таргеты jgpt_cuda_extra и jgpt_cuda).");
            }
            System.load(extra.toAbsolutePath().toString());
        }
        System.load(mainSo.toAbsolutePath().toString());
    }

    private static String companionExtraFileName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) {
            return "jgpt_cuda_extra.dll";
        }
        if (os.contains("mac")) {
            return "libjgpt_cuda_extra.dylib";
        }
        return "libjgpt_cuda_extra.so";
    }

    private static String[] relativeCandidatePaths() {
        String userDir = System.getProperty("user.dir", ".");
        return new String[] {userDir + "/build/libjgpt_cuda.so", userDir + "/../build/libjgpt_cuda.so"};
    }

    private static String buildErrorMessage() {
        String[] candidates = relativeCandidatePaths();
        String env = System.getenv("JGPT_CUDA_LIB");
        return "libjgpt_cuda.so не найден (и/или рядом нет "
                + companionExtraFileName()
                + "). Порядок поиска:\n"
                + "  1. -Djgpt.cuda.lib="
                + System.getProperty("jgpt.cuda.lib", "<не задано>")
                + "\n"
                + "  2. JGPT_CUDA_LIB="
                + (env != null && !env.isBlank() ? env : "<не задано>")
                + "\n"
                + "  3. Относительно user.dir: "
                + String.join(", ", candidates)
                + "\n"
                + "  4. java.library.path (нужны и jgpt_cuda_extra, и jgpt_cuda): "
                + System.getProperty("java.library.path", "<пусто>")
                + "\n"
                + "Сборка: cd src/main/cpp && cmake -B ../../build -S . && cmake --build ../../build";
    }
}
