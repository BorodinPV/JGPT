package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class StepBeyondPlanPolicyTest {

    @Test
    void fromEnv_parsesAliases() {
        assertEquals(StepBeyondPlanPolicy.SKIP, StepBeyondPlanPolicy.fromEnv(null));
        assertEquals(StepBeyondPlanPolicy.SKIP, StepBeyondPlanPolicy.fromEnv(""));
        assertEquals(StepBeyondPlanPolicy.SKIP, StepBeyondPlanPolicy.fromEnv("skip"));
        assertEquals(StepBeyondPlanPolicy.SKIP, StepBeyondPlanPolicy.fromEnv("SKIP"));
        assertEquals(StepBeyondPlanPolicy.SKIP, StepBeyondPlanPolicy.fromEnv("unknown"));

        assertEquals(StepBeyondPlanPolicy.RESTART_SCHEDULE, StepBeyondPlanPolicy.fromEnv("restart_schedule"));
        assertEquals(StepBeyondPlanPolicy.RESTART_SCHEDULE, StepBeyondPlanPolicy.fromEnv("restart-schedule"));
        assertEquals(StepBeyondPlanPolicy.RESTART_SCHEDULE, StepBeyondPlanPolicy.fromEnv("restart"));

        assertEquals(StepBeyondPlanPolicy.FAIL, StepBeyondPlanPolicy.fromEnv("fail"));
        assertEquals(StepBeyondPlanPolicy.FAIL, StepBeyondPlanPolicy.fromEnv("ERROR"));
    }
}
