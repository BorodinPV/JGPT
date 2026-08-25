package com.veles.llm.jgpt.cuda;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Загрузка нативных библиотек CUDA.
 *
 * <p>Linux: extra ({@code libjgpt_cuda_extra.so}), затем основной {@code libjgpt_cuda.so} — оба в {@code build/}.
 * Windows: один {@code jgpt_cuda.dll} (оба .cu в одной DLL); extra не требуется.
 *
 * <p>Порядок: {@code -Djgpt.cuda.lib} / {@code JGPT_CUDA_LIB} (путь к основной библиотеке),
 * относительные пути {@code build/jgpt_cuda.dll} или {@code build/libjgpt_cuda.so}, иначе {@link System#loadLibrary}.
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
                if (!windows()) {
                    System.loadLibrary("jgpt_cuda_extra");
                }
                System.loadLibrary("jgpt_cuda");
                lastLoadedPath = "jgpt_cuda (java.library.path)";
                loaded = true;
                return;
            } catch (UnsatisfiedLinkError e) {
                if (!quietMissingNative()) {
                    System.err.println(
                            "[TensorCudaLibrary] loadLibrary(jgpt_cuda_extra/jgpt_cuda) не удался: "
                                    + e.getMessage());
                    System.err.println(
                            "[TensorCudaLibrary] java.library.path="
                                    + System.getProperty("java.library.path", "<пусто>"));
                }
            }

            throw new UnsatisfiedLinkError(buildErrorMessage());
        }
    }

    /**
     * Linux: companion extra, затем основной модуль. Windows: CUDA runtime DLL из того же каталога,
     * затем основной {@code jgpt_cuda.dll} (extra опционален).
     */
    private static void loadMainWithCompanionExtra(Path mainSo) {
        Path dir = mainSo.toAbsolutePath().getParent();
        if (dir != null) {
            preloadWindowsCudaRuntimeDlls(dir);
            Path extra = dir.resolve(companionExtraFileName());
            if (Files.isRegularFile(extra)) {
                System.load(extra.toAbsolutePath().toString());
            } else if (!windows()) {
                throw new UnsatisfiedLinkError(
                        "Рядом с "
                                + mainSo
                                + " ожидается "
                                + extra
                                + " (соберите cmake-таргеты jgpt_cuda_extra и jgpt_cuda).");
            }
        }
        System.load(mainSo.toAbsolutePath().toString());
    }

    /** Уже загруженные DLL удовлетворяют импорт jgpt_cuda.dll (PATH Java.exe extra не видит). */
    private static void preloadWindowsCudaRuntimeDlls(Path dir) {
        if (!windows()) {
            return;
        }
        String[] prefixes = {"cudart64_", "nvJitLink", "cublasLt64_", "cublas64_"};
        for (Path search : windowsCudaRuntimeSearchDirs(dir)) {
            for (String prefix : prefixes) {
                try (var stream = Files.list(search)) {
                    stream.filter(p -> {
                                String n = p.getFileName().toString();
                                return n.startsWith(prefix) && n.toLowerCase().endsWith(".dll");
                            })
                            .sorted()
                            .forEach(p -> {
                                try {
                                    System.load(p.toAbsolutePath().toString());
                                } catch (UnsatisfiedLinkError ignored) {
                                    // already loaded or missing transitive dep — next prefix/dir
                                }
                            });
                } catch (Exception ignored) {
                    // missing dir
                }
            }
        }
    }

    private static Path[] windowsCudaRuntimeSearchDirs(Path dllDir) {
        java.util.LinkedHashSet<Path> dirs = new java.util.LinkedHashSet<>();
        if (dllDir != null) {
            dirs.add(dllDir);
        }
        String cudaPath = System.getenv("CUDA_PATH");
        if (cudaPath != null && !cudaPath.isBlank()) {
            Path root = Path.of(cudaPath.trim());
            dirs.add(root.resolve("bin").resolve("x64"));
            dirs.add(root.resolve("bin"));
        }
        return dirs.toArray(Path[]::new);
    }

    private static boolean windows() {
        return System.getProperty("os.name", "").toLowerCase().contains("win");
    }

    private static String companionExtraFileName() {
        if (windows()) {
            return "jgpt_cuda_extra.dll";
        }
        if (System.getProperty("os.name", "").toLowerCase().contains("mac")) {
            return "libjgpt_cuda_extra.dylib";
        }
        return "libjgpt_cuda_extra.so";
    }

    private static String[] relativeCandidatePaths() {
        String userDir = System.getProperty("user.dir", ".");
        String mainName = mainLibraryFileName();
        return new String[] {
            userDir + "/build/" + mainName,
            userDir + "/../build/" + mainName,
            userDir + "/build/Release/" + mainName,
            userDir + "/build/Debug/" + mainName
        };
    }

    private static String mainLibraryFileName() {
        if (windows()) {
            return "jgpt_cuda.dll";
        }
        if (System.getProperty("os.name", "").toLowerCase().contains("mac")) {
            return "libjgpt_cuda.dylib";
        }
        return "libjgpt_cuda.so";
    }

    private static boolean quietMissingNative() {
        try {
            if (Boolean.getBoolean("jgpt.allow.no.gpu")) {
                return true;
            }
        } catch (Exception ignored) {
            // ignore
        }
        try {
            String e = System.getenv("JGPT_ALLOW_NO_GPU");
            if (e != null) {
                String t = e.trim();
                if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                    return true;
                }
            }
        } catch (Exception ignored) {
            // ignore
        }
        return false;
    }

    private static String buildErrorMessage() {
        String[] candidates = relativeCandidatePaths();
        String env = System.getenv("JGPT_CUDA_LIB");
        return mainLibraryFileName()
                + " не найден"
                + (windows() ? "" : " (и/или рядом нет " + companionExtraFileName() + ")")
                + ". Порядок поиска:\n"
                + "  1. -Djgpt.cuda.lib="
                + System.getProperty("jgpt.cuda.lib", "<не задано>")
                + "\n"
                + "  2. JGPT_CUDA_LIB="
                + (env != null && !env.isBlank() ? env : "<не задано>")
                + "\n"
                + "  3. Относительно user.dir: "
                + String.join(", ", candidates)
                + "\n"
                + "  4. java.library.path: "
                + System.getProperty("java.library.path", "<пусто>")
                + "\n"
                + "Сборка: из корня репозитория Linux ./scripts/build-cuda.sh ; Windows .\\scripts\\build-cuda.ps1";
    }
}
