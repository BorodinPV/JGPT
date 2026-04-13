package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.data.DataLoader;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;

/** Подготовка следующего батча в фоне (один поток на эпоху в {@link LLMTrainer#train()}). */
final class LlmTrainerBatchPrefetch {

    private LlmTrainerBatchPrefetch() {}

    static CompletableFuture<DataLoader.Batch> scheduleBatchPrefetch(
            DataLoader loader, ExecutorService prefetchExecutor) {
        if (prefetchExecutor == null || !loader.hasMore()) {
            return null;
        }
        return CompletableFuture.supplyAsync(loader::buildBatchNoAdvance, prefetchExecutor);
    }

    static DataLoader.Batch takeNextBatchPrefetched(
            DataLoader loader, CompletableFuture<DataLoader.Batch> prefetchFut) {
        try {
            DataLoader.Batch b = prefetchFut.get();
            loader.advanceAfterPreparedBatch();
            return b;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } catch (ExecutionException e) {
            Throwable c = e.getCause();
            if (c instanceof Error) {
                throw (Error) c;
            }
            if (c instanceof RuntimeException) {
                throw (RuntimeException) c;
            }
            throw new RuntimeException(c);
        }
    }
}
