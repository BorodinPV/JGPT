package com.veles.llm.jgpt;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import org.junit.jupiter.api.extension.ConditionEvaluationResult;
import org.junit.jupiter.api.extension.ExecutionCondition;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * Skip when {@code jgpt_cuda} is not loaded. Surefire sets {@code -Djgpt.allow.no.gpu=true}
 * so the JVM starts; TensorOps still requires CUDA.
 */
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@ExtendWith(EnabledIfGpu.Condition.class)
public @interface EnabledIfGpu {

    final class Condition implements ExecutionCondition {
        @Override
        public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
            if (TensorOpsGPU.isGpuAvailable()) {
                return ConditionEvaluationResult.enabled("CUDA");
            }
            return ConditionEvaluationResult.disabled("skip: no jgpt_cuda / GPU");
        }
    }
}
