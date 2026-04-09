package com.veles.llm.jgpt.training;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class CheckpointPrunerTest {

    @Test
    void pruneStepTriplesKeepsNewest(@TempDir Path dir) throws IOException {
        touch(dir, "checkpoint_step_1000.bin");
        touch(dir, "model_step_1000.bin");
        touch(dir, "tokenizer_step_1000.bin");
        touch(dir, "checkpoint_step_2000.bin");
        touch(dir, "model_step_2000.bin");
        touch(dir, "tokenizer_step_2000.bin");
        touch(dir, "checkpoint_step_3000.bin");
        touch(dir, "model_step_3000.bin");
        touch(dir, "tokenizer_step_3000.bin");

        CheckpointPruner.pruneStepTriples(dir, 2);

        assertFalse(Files.exists(dir.resolve("checkpoint_step_1000.bin")));
        assertFalse(Files.exists(dir.resolve("model_step_1000.bin")));
        assertTrue(Files.exists(dir.resolve("checkpoint_step_2000.bin")));
        assertTrue(Files.exists(dir.resolve("checkpoint_step_3000.bin")));
    }

    @Test
    void pruneEpochTriplesDoesNotTouchFinal(@TempDir Path dir) throws IOException {
        touch(dir, "checkpoint_epoch_1.bin");
        touch(dir, "model_epoch_1.bin");
        touch(dir, "checkpoint_epoch_2.bin");
        touch(dir, "checkpoint_final.bin");

        CheckpointPruner.pruneEpochTriples(dir, 1);

        assertFalse(Files.exists(dir.resolve("checkpoint_epoch_1.bin")));
        assertTrue(Files.exists(dir.resolve("checkpoint_epoch_2.bin")));
        assertTrue(Files.exists(dir.resolve("checkpoint_final.bin")));
    }

    @Test
    void pruneAfterSaveRunsEpochPruneOnFinal(@TempDir Path dir) throws IOException {
        touch(dir, "checkpoint_epoch_1.bin");
        touch(dir, "model_epoch_1.bin");
        touch(dir, "checkpoint_epoch_2.bin");
        touch(dir, "model_epoch_2.bin");
        touch(dir, "checkpoint_epoch_3.bin");
        touch(dir, "model_epoch_3.bin");

        CheckpointPruner.pruneAfterSave(dir, "final");

        assertTrue(Files.exists(dir.resolve("checkpoint_epoch_3.bin")));
        assertTrue(Files.exists(dir.resolve("checkpoint_epoch_2.bin")));
        assertFalse(Files.exists(dir.resolve("checkpoint_epoch_1.bin")));
    }

    private static void touch(Path dir, String name) throws IOException {
        Files.writeString(dir.resolve(name), "x");
    }
}
