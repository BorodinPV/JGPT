package com.veles.llm.jgpt.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Загрузка и подготовка текстовых данных для обучения.
 */
public final class TextDataset {

    private static final Logger log = LoggerFactory.getLogger(TextDataset.class);

    private final List<String> texts;

    public TextDataset() {
        this.texts = new ArrayList<>();
    }

    /**
     * Загрузка текста из файла.
     */
    public void loadFile(String path) throws IOException {
        String content = Files.readString(Path.of(path));
        loadText(content);
        log.info("Файл прочитан: {} ({} символов)", path, String.format("%,d", content.length()));
    }

    /**
     * Загрузка текста из строки.
     */
    public void loadText(String text) {
        texts.add(text);
    }

    /** Общий размер данных (символы). */
    public int totalCharacters() {
        return texts.stream().mapToInt(String::length).sum();
    }

    /** Количество текстов. */
    public int size() {
        return texts.size();
    }

    public List<String> getTexts() {
        return texts;
    }

    /** Освободить память после обучения BPE (тексты больше не нужны). */
    public void clearTexts() {
        texts.clear();
    }
}
