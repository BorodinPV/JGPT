package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainingStopFileTest {

    @Test
    void missingFileIsNotAStop(@TempDir Path dir) {
        Path stop = dir.resolve("STOP");
        assertFalse(TrainingStopFile.isPresent(stop));
    }

    @Test
    void consumeDeletesStopFile(@TempDir Path dir) throws Exception {
        Path stop = dir.resolve("STOP");
        Files.writeString(stop, "stop\n");
        assertTrue(TrainingStopFile.isPresent(stop));
        TrainingStopFile.consume(stop);
        assertFalse(TrainingStopFile.isPresent(stop));
        TrainingStopFile.consume(stop);
    }
}
