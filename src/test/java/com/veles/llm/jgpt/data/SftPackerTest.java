package com.veles.llm.jgpt.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SftPackerTest {

    @Test
    void parseAlpacaAndMessages() {
        List<SftTurn> alpaca =
                SftJsonlParser.parseLine(
                        "{\"instruction\":\"Привет\",\"input\":\"\",\"output\":\"Здравствуй\"}");
        assertEquals(2, alpaca.size());
        assertEquals(SftTurn.Role.USER, alpaca.get(0).role);
        assertEquals(SftTurn.Role.ASSISTANT, alpaca.get(1).role);

        List<SftTurn> chat =
                SftJsonlParser.parseLine(
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"A\"},{\"role\":\"bot\",\"content\":\"B\"}]}");
        assertEquals(2, chat.size());
        assertEquals("A", chat.get(0).content);
        assertEquals("B", chat.get(1).content);

        assertTrue(
                SftJsonlParser.parseLine(
                                "{\"instruction\":\"x\",\"output\":\"y\",\"label\":\"bad_output\"}")
                        .isEmpty());

        List<SftTurn> repaired =
                SftJsonlParser.parseLine(
                        "{\"instruction\":\"x\",\"output\":\"плохо\",\"label\":\"bad_output\",\"alternative_output\":\"хорошо\"}");
        assertEquals(2, repaired.size());
        assertEquals("хорошо", repaired.get(1).content);

        List<SftTurn> withSystem =
                SftJsonlParser.parseLine(
                        "{\"messages\":[{\"role\":\"system\",\"content\":\"будь краток\"},{\"role\":\"user\",\"content\":\"A\"},{\"role\":\"bot\",\"content\":\"B\"}]}");
        assertEquals(2, withSystem.size());
        assertTrue(withSystem.get(0).content.contains("будь краток"));
        assertTrue(withSystem.get(0).content.contains("A"));
    }

    @Test
    void lossOnlyOnAssistant() {
        BPETokenizer tok = BPETokenizer.train(List.of("привет здравствуй пользователь ассистент"), 80);
        List<SftTurn> turns =
                List.of(
                        new SftTurn(SftTurn.Role.USER, "привет"),
                        new SftTurn(SftTurn.Role.ASSISTANT, "здравствуй"));
        SftExampleEncoder.Encoded ex = SftExampleEncoder.encode(tok, turns);
        assertTrue(ex.tokens.length > 4);
        assertEquals(tok.bosId(), ex.tokens[0]);
        assertEquals(tok.eosId(), ex.tokens[ex.tokens.length - 1]);
        boolean sawUser = false;
        boolean sawAsst = false;
        for (boolean s : ex.supervised) {
            if (s) {
                sawAsst = true;
            } else {
                sawUser = true;
            }
        }
        assertTrue(sawUser);
        assertTrue(sawAsst);
        assertTrue(!ex.supervised[0]);
        assertTrue(ex.supervised[ex.supervised.length - 1]);
    }

    @Test
    void packSetsIgnoreOnPrompt() {
        BPETokenizer tok = BPETokenizer.train(List.of("привет здравствуй пользователь ассистент"), 80);
        SftExampleEncoder.Encoded ex =
                SftExampleEncoder.encode(
                        tok,
                        List.of(
                                new SftTurn(SftTurn.Role.USER, "привет"),
                                new SftTurn(SftTurn.Role.ASSISTANT, "здравствуй")));
        int maxSeq = Math.max(8, ex.tokens.length + 2);
        List<SftWindowPacker.Window> windows = SftWindowPacker.pack(List.of(ex), maxSeq, tok.padId());
        assertEquals(1, windows.size());
        SftWindowPacker.Window w = windows.get(0);
        assertEquals(maxSeq + 1, w.tokens.length);
        assertEquals(maxSeq, w.targets.length);
        int supervised = 0;
        int ignored = 0;
        for (int t : w.targets) {
            if (t < 0) {
                ignored++;
            } else {
                supervised++;
            }
        }
        assertTrue(supervised > 0);
        assertTrue(ignored > 0);
    }

    @Test
    void packTwoShortDialogsIntoOneWindow() {
        BPETokenizer tok = BPETokenizer.train(List.of("привет здравствуй пользователь ассистент да нет"), 80);
        SftExampleEncoder.Encoded a =
                SftExampleEncoder.encode(
                        tok,
                        List.of(
                                new SftTurn(SftTurn.Role.USER, "привет"),
                                new SftTurn(SftTurn.Role.ASSISTANT, "да")));
        SftExampleEncoder.Encoded b =
                SftExampleEncoder.encode(
                        tok,
                        List.of(
                                new SftTurn(SftTurn.Role.USER, "нет"),
                                new SftTurn(SftTurn.Role.ASSISTANT, "здравствуй")));
        int maxSeq = a.tokens.length + b.tokens.length + 8;
        List<SftWindowPacker.Window> windows = SftWindowPacker.pack(List.of(a, b), maxSeq, tok.padId());
        assertEquals(1, windows.size());
        DataLoader loader = new DataLoader(tok, maxSeq, 1, false, false);
        loader.loadSftWindow(windows.get(0).tokens, windows.get(0).targets);
        DataLoader.Batch batch = loader.nextBatch();
        float[] tgt = batch.target.internalBuffer();
        int supervised = 0;
        int ignored = 0;
        for (float v : tgt) {
            if (v < 0f) {
                ignored++;
            } else {
                supervised++;
            }
        }
        assertTrue(supervised > 0);
        assertTrue(ignored > 0);
        assertEquals(-1f, tgt[0], 0f);
    }
}
