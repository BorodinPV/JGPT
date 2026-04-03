package com.veles.llm.jgpt.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class TensorCudaLibraryTest {

    @Test
    void loadIsIdempotentFromManyThreads() throws Exception {
        int threads = 24;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch gate = new CountDownLatch(1);
        AtomicInteger errors = new AtomicInteger();
        for (int i = 0; i < threads; i++) {
            pool.submit(
                    () -> {
                        try {
                            gate.await();
                            TensorCudaLibrary.load();
                        } catch (Throwable t) {
                            errors.incrementAndGet();
                        }
                    });
        }
        gate.countDown();
        pool.shutdown();
        assertTrue(pool.awaitTermination(60, TimeUnit.SECONDS), "thread pool hung");

        if (!TensorCudaLibrary.isLoaded()) {
            return;
        }
        assertEquals(0, errors.get(), "load() failed in some threads");
        String path = TensorCudaLibrary.getLastLoadedPath();
        assertTrue(path != null && !path.isBlank(), "lastLoadedPath: " + path);
    }

    @Test
    void secondLoadIsCheapNoThrow() {
        if (!TensorCudaLibrary.isLoaded()) {
            try {
                TensorCudaLibrary.load();
            } catch (UnsatisfiedLinkError e) {
                return;
            }
        }
        assertTrue(TensorCudaLibrary.isLoaded());
        String p1 = TensorCudaLibrary.getLastLoadedPath();
        TensorCudaLibrary.load();
        assertEquals(p1, TensorCudaLibrary.getLastLoadedPath());
    }
}
