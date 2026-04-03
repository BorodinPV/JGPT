package com.veles.llm.jgpt.training;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Properties;
import java.util.Set;

/**
 * Состояние очереди книг: какие уже обучены, какая сейчас в работе (для возобновления после рестарта).
 * Файл: обычно {@code boo/checkpoints/training_state.properties}.
 */
public final class BookTrainingState {

    private final LinkedHashSet<String> completed;
    private String current;

    public BookTrainingState(LinkedHashSet<String> completed, String current) {
        this.completed = completed;
        this.current = current;
    }

    public static BookTrainingState load(Path path) throws IOException {
        LinkedHashSet<String> done = new LinkedHashSet<>();
        String cur = null;
        if (!Files.isRegularFile(path)) {
            return new BookTrainingState(done, null);
        }
        Properties p = new Properties();
        try (InputStream in = Files.newInputStream(path)) {
            p.load(in);
        }
        String c = p.getProperty("completed", "");
        if (!c.isEmpty()) {
            for (String part : c.split("\\|")) {
                String s = part.trim();
                if (!s.isEmpty()) {
                    done.add(s);
                }
            }
        }
        String raw = p.getProperty("current");
        if (raw != null && !raw.isBlank()) {
            cur = raw.trim();
        }
        return new BookTrainingState(done, cur);
    }

    public void save(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Properties p = new Properties();
        p.setProperty("completed", String.join("|", completed));
        if (current != null) {
            p.setProperty("current", current);
        }
        try (OutputStream out = Files.newOutputStream(path)) {
            p.store(out, "Multi-book training state");
        }
    }

    public boolean isCompleted(String fileName) {
        return completed.contains(fileName);
    }

    public void markCompleted(String fileName) {
        completed.add(fileName);
    }

    public String getCurrent() {
        return current;
    }

    public void setCurrent(String fileNameOrNull) {
        this.current = fileNameOrNull;
    }

    public Set<String> completed() {
        return Set.copyOf(completed);
    }
}
