package com.veles.llm.jgpt.junit;

import org.junit.platform.engine.TestExecutionResult;
import org.junit.platform.launcher.TestExecutionListener;
import org.junit.platform.launcher.TestIdentifier;
import org.junit.platform.launcher.TestPlan;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Выводит сводку по всем тестам в конце прогона (JUnit Platform auto-SPI).
 */
public final class TensorTestExecutionSummaryListener implements TestExecutionListener {

    private final AtomicInteger successful = new AtomicInteger();
    private final AtomicInteger failed = new AtomicInteger();
    private final AtomicInteger aborted = new AtomicInteger();
    private final AtomicInteger skipped = new AtomicInteger();

    @Override
    public void testPlanExecutionStarted(TestPlan testPlan) {
        successful.set(0);
        failed.set(0);
        aborted.set(0);
        skipped.set(0);
        TensorTestProbeResults.reset();
    }

    @Override
    public void executionFinished(TestIdentifier testIdentifier, TestExecutionResult testExecutionResult) {
        if (!testIdentifier.isTest()) {
            return;
        }
        switch (testExecutionResult.getStatus()) {
            case SUCCESSFUL -> successful.incrementAndGet();
            case FAILED -> failed.incrementAndGet();
            case ABORTED -> aborted.incrementAndGet();
            default -> {
                // defensive
            }
        }
    }

    @Override
    public void executionSkipped(TestIdentifier testIdentifier, String reason) {
        if (testIdentifier.isTest()) {
            skipped.incrementAndGet();
        }
    }

    @Override
    public void testPlanExecutionFinished(TestPlan testPlan) {
        int ok = successful.get();
        int bad = failed.get();
        int ab = aborted.get();
        int sk = skipped.get();
        int executed = ok + bad + ab;

        StringBuilder line = new StringBuilder(128);
        TensorTestProbeResults.appendTo(line);
        line.append("========== Итоги тестов (JUnit) ==========\n");
        line.append("Выполнено тестов: ").append(executed).append("  (успех: ").append(ok);
        line.append(", неудач: ").append(bad).append(", прервано: ").append(ab);
        line.append(", пропущено: ").append(sk).append(")\n");
        line.append("==========================================");
        System.out.println(line);
    }
}
