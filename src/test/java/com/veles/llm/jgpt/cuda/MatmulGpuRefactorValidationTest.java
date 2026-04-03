package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * Проверки после рефакторинга JNI matmul (общий CUDA stream, меньше sync, mutex на init).
 *
 * <p>Вручную (профилирование / JNI / valgrind) см. {@code scripts/profile-matmul-refactor.sh}.
 */
class MatmulGpuRefactorValidationTest {

    private static final float TOL_STRESS = 2e-3f;

    /** Эталон: накопление в double от тех же float-входов, затем усечение к float. */
    private static void matmulHostFp64(float[] a, float[] b, float[] c, int m, int k, int n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int t = 0; t < k; t++) {
                    s += (double) a[i * k + t] * (double) b[t * n + j];
                }
                c[i * n + j] = (float) s;
            }
        }
    }

    @Test
    void matmulGpu_edge_1x1x1() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        float[] a = {2.0f};
        float[] b = {3.0f};
        float[] c = {0.0f};
        TensorOpsGPU.matmulGPU(a, b, c, 1, 1, 1);
        assertEquals(6.0f, c[0], 1e-6f);
    }

    @Test
    void matmulGpu_edge_skinny_rectangles_matchHost() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        // M=1, K=5, N=7 и M=11, K=1, N=3
        runCompareToHost(1, 5, 7, 0x111);
        runCompareToHost(11, 1, 3, 0x222);
    }

    private static void runCompareToHost(int m, int k, int n, int seed) {
        float[] a = new float[m * k];
        float[] b = new float[k * n];
        float[] c = new float[m * n];
        float[] expected = new float[m * n];
        Random r = new Random(seed);
        for (int i = 0; i < a.length; i++) {
            a[i] = r.nextFloat() * 2.0f - 1.0f;
        }
        for (int i = 0; i < b.length; i++) {
            b[i] = r.nextFloat() * 2.0f - 1.0f;
        }
        matmulHostFp64(a, b, expected, m, k, n);
        Arrays.fill(c, Float.NaN);
        TensorOpsGPU.matmulGPU(a, b, c, m, k, n);
        for (int i = 0; i < c.length; i++) {
            assertEquals(expected[i], c[i], TOL_STRESS, "m,k,n=" + m + "," + k + "," + n + " i=" + i);
        }
    }

    /**
     * Большой квадрат без полного эталона O(n³) на CPU: константные матрицы и проверка одного элемента.
     * ~12 MiB на матрицу при n=1024 — укладывается в типичную дискретку.
     */
    @Test
    void matmulGpu_large_square_spotCheck() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        int n = 1024;
        float[] a = new float[n * n];
        float[] b = new float[n * n];
        float[] c = new float[n * n];
        Arrays.fill(a, 0.001f);
        Arrays.fill(b, 0.002f);
        TensorOpsGPU.matmulGPU(a, b, c, n, n, n);
        float expected = n * 0.001f * 0.002f;
        assertEquals(expected, c[0], Math.max(1e-3f, Math.abs(expected) * 1e-2f), "C[0]");
        assertEquals(expected, c[n * n - 1], Math.max(1e-3f, Math.abs(expected) * 1e-2f), "C[last]");
    }

    /**
     * Много потоков вызывают {@link TensorOpsGPU#matmulGPU} с независимыми буферами — ловим гонки при общем stream /
     * кэшах.
     */
    @Test
    void matmulGpu_parallelThreads_matchesHostReference() throws InterruptedException {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        final int threadCount = 16;
        final int innerIters = 64;
        final int m = 48;
        final int k = 48;
        final int n = 48;

        ExecutorService pool = Executors.newFixedThreadPool(threadCount);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threadCount);
        AtomicReference<AssertionError> firstFailure = new AtomicReference<>();

        for (int t = 0; t < threadCount; t++) {
            final int tid = t;
            pool.submit(() -> {
                try {
                    start.await();
                    Random r = new Random(0xDECAFBAD + tid * 1_000_003);
                    float[] a = new float[m * k];
                    float[] b = new float[k * n];
                    float[] c = new float[m * n];
                    float[] expected = new float[m * n];
                    for (int it = 0; it < innerIters; it++) {
                        for (int i = 0; i < a.length; i++) {
                            a[i] = r.nextFloat() * 2.0f - 1.0f;
                        }
                        for (int i = 0; i < b.length; i++) {
                            b[i] = r.nextFloat() * 2.0f - 1.0f;
                        }
                        matmulHostFp64(a, b, expected, m, k, n);
                        Arrays.fill(c, Float.NaN);
                        /* Один глобальный CUDA-stream: параллельный enqueue с нескольких потоков даёт UB и ломает последующие тесты. */
                        synchronized (TensorOpsGPU.class) {
                            TensorOpsGPU.matmulGPU(a, b, c, m, k, n);
                        }
                        for (int i = 0; i < c.length; i++) {
                            if (Math.abs(expected[i] - c[i]) > TOL_STRESS) {
                                firstFailure.compareAndSet(
                                        null,
                                        new AssertionError(
                                                "tid=" + tid + " iter=" + it + " i=" + i + " exp="
                                                        + expected[i] + " got=" + c[i]));
                                return;
                            }
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    firstFailure.compareAndSet(null, new AssertionError(e));
                } finally {
                    done.countDown();
                }
            });
        }

        start.countDown();
        assertTrue(done.await(5, TimeUnit.MINUTES), "parallel matmul threads should finish");
        pool.shutdown();
        assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS));

        AssertionError err = firstFailure.get();
        if (err != null) {
            throw err;
        }
    }
}
