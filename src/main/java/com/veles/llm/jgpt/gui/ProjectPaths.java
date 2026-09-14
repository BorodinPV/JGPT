package com.veles.llm.jgpt.gui;

import java.nio.file.Files;
import java.nio.file.Path;

/** Корень репозитория JGPT и производные каталоги. */
record ProjectPaths(Path root) {

    Path scriptsWindows() {
        return root.resolve("scripts").resolve("windows");
    }

    Path envDir() {
        return root.resolve("env");
    }

    Path checkpointsDir() {
        return root.resolve("checkpoints");
    }

    Path stateDir() {
        return root.resolve("state");
    }

    Path statsJson() {
        return stateDir().resolve("stats.json");
    }

    Path stopFile() {
        return stateDir().resolve("STOP");
    }

    /** {@code -Djgpt.root=...}, иначе — от cwd вверх до каталога с {@code pom.xml} и {@code env/}. */
    static ProjectPaths detect() {
        String prop = System.getProperty("jgpt.root");
        if (prop != null && !prop.isBlank()) {
            return new ProjectPaths(Path.of(prop).toAbsolutePath().normalize());
        }
        Path p = Path.of("").toAbsolutePath();
        for (Path cur = p; cur != null; cur = cur.getParent()) {
            if (Files.isRegularFile(cur.resolve("pom.xml")) && Files.isDirectory(cur.resolve("env"))) {
                return new ProjectPaths(cur);
            }
        }
        return new ProjectPaths(p);
    }
}
