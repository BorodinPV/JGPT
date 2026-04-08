package com.veles.llm.jgpt.training;

/** Выбрасывается при {@link StepBeyondPlanPolicy#FAIL} и исчерпанном плане ({@code globalStep >= totalTrainingSteps}). */
public final class TrainingPlanExhaustedException extends RuntimeException {

    public TrainingPlanExhaustedException(String message) {
        super(message);
    }
}
