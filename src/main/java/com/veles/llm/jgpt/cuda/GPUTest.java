package com.veles.llm.jgpt.cuda;

import com.veles.llm.jgpt.TensorOpsGPU;

import java.util.Arrays;

/**
 * Загрузка {@code libjgpt_cuda.so} и вызов {@link TensorOpsGPU}.
 *
 * <p>Запуск (рабочая директория — корень проекта {@code JGPT/}, где лежит {@code target/classes}
 * после {@code mvn compile}):
 * <pre>{@code
 * cd /path/to/JGPT
 * mvn -q compile
 * java --enable-native-access=ALL-UNNAMED --add-modules=jdk.incubator.vector --enable-preview \
 *     -cp target/classes com.veles.llm.jgpt.GPUTest [warmup] [runs] [itersPerRun]
 * }</pre>
 * Необязательные аргументы: число итераций прогрева, число замеров, итераций matmul в каждом замере (по умолчанию 32, 7, 50).
 * Из {@code src/main/cpp} используйте {@code -cp ../../../target/classes} или сначала {@code cd} в {@code JGPT/}.
 *
 * <p>Путь к .so: свойство {@code -Djgpt.cuda.lib=...}, переменная {@code JGPT_CUDA_LIB},
 * или автопоиск {@code build/libjgpt_cuda.so} относительно текущей директории.
 */
public class GPUTest {

    static {
        try {
            TensorCudaLibrary.load();
            System.out.println("✅ Нативная библиотека загружена: " + TensorCudaLibrary.getLastLoadedPath());
        } catch (UnsatisfiedLinkError e) {
            System.err.println("❌ " + e.getMessage());
            e.printStackTrace();
            printBuildHint();
        }
    }

    private static void printBuildHint() {
        System.err.println();
        System.err.println("Соберите нативную библиотеку (нужны CUDA, nvcc, JNI):");
        System.err.println("  cd src/main/cpp && cmake -B ../../build -S . && cmake --build ../../build");
        System.err.println("Ожидается файл: JGPT/build/libjgpt_cuda.so");
        System.err.println("Или укажите путь: -Djgpt.cuda.lib=/полный/путь/libjgpt_cuda.so");
        System.err.println();
    }

    public static void main(String[] args) {
        System.out.println("🧪 Проверка GPU...");

        boolean available = TensorOpsGPU.isGpuAvailable();
        System.out.println("GPU доступен: " + available);

        if (available) {
            System.out.println("🎉 GPU готов к matmul (проверьте производительность на своём устройстве).");

            int warmup = 32;
            int runs = 7;
            int itersPerRun = 50;
            if (args.length >= 1) {
                warmup = Integer.parseInt(args[0]);
            }
            if (args.length >= 2) {
                runs = Integer.parseInt(args[1]);
            }
            if (args.length >= 3) {
                itersPerRun = Integer.parseInt(args[2]);
            }
            if (warmup < 0 || runs < 1 || itersPerRun < 1) {
                System.err.println("Неверные аргументы: прогрев≥0, замеров≥1, итерацийВЗамере≥1");
                return;
            }
            runGpuMatmulSanityTiming(warmup, runs, itersPerRun);
            TensorOpsGPU.synchronizeStream();
            TensorOpsGPU.drainDeferredGpuBuffers();
            TensorOpsGPU.cudaTrimDeviceMemoryPoolsBestEffort();
        }
    }

    /** Несколько замеров matmul на GPU для sanity-check после загрузки JNI. */
    private static void runGpuMatmulSanityTiming(int warmup, int numRuns, int itersPerRun) {
        int M = 512, K = 512, N = 512;
        float[] A = new float[M * K];
        float[] B = new float[K * N];
        float[] C = new float[M * N];

        for (int i = 0; i < A.length; i++) {
            A[i] = (float) Math.random();
            B[i] = (float) Math.random();
        }

        for (int i = 0; i < warmup; i++) {
            TensorOpsGPU.matmulGPU(A, B, C, M, K, N);
        }

        double flopsPerBatch = 2.0 * M * K * N * itersPerRun;
        double[] gflopsPerRun = new double[numRuns];
        long[] nsPerRun = new long[numRuns];

        for (int r = 0; r < numRuns; r++) {
            long t0 = System.nanoTime();
            for (int i = 0; i < itersPerRun; i++) {
                TensorOpsGPU.matmulGPU(A, B, C, M, K, N);
            }
            long t1 = System.nanoTime();
            long dt = t1 - t0;
            nsPerRun[r] = dt;
            double sec = dt / 1e9;
            gflopsPerRun[r] = flopsPerBatch / sec / 1e9;
        }

        double[] gSorted = Arrays.copyOf(gflopsPerRun, numRuns);
        Arrays.sort(gSorted);
        long[] tSorted = Arrays.copyOf(nsPerRun, numRuns);
        Arrays.sort(tSorted);

        double gMin = gSorted[0];
        double gMax = gSorted[numRuns - 1];
        double gMed = medianSorted(gSorted);
        double msMed = medianSortedAsMsPerIter(tSorted, itersPerRun);

        System.out.printf("🚀 GPU matmul %dx%dx%d:%n", M, K, N);
        System.out.printf("   Прогрев: %d ит.; замеры: %d прогонов × %d ит. каждый%n",
                warmup, numRuns, itersPerRun);
        System.out.printf("   GFLOPS (медиана / мин / макс): %.1f / %.1f / %.1f%n", gMed, gMin, gMax);
        System.out.printf("   мс/ит. (медиана средних по прогону): %.3f%n", msMed);
        System.out.printf("   (ориентир CPU ~175 GFLOPS; GPU с PCIe+JNI часто ~400–900)%n");
    }

    private static double medianSorted(double[] sorted) {
        int n = sorted.length;
        if ((n & 1) == 1) {
            return sorted[n / 2];
        }
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static double medianSorted(long[] sorted) {
        int n = sorted.length;
        if ((n & 1) == 1) {
            return sorted[n / 2];
        }
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    /** Медиана длительности замера / itersPerRun → среднее время одной matmul в мс. */
    private static double medianSortedAsMsPerIter(long[] sortedNsPerRun, int itersPerRun) {
        return medianSorted(sortedNsPerRun) / itersPerRun / 1_000_000.0;
    }
}
