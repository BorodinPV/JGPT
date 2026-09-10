package com.veles.llm.jgpt.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
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

    @Test
    void wrapUserChatPromptAddsPrefixes() {
        String wrapped = SftExampleEncoder.wrapUserChatPrompt("привет");
        assertEquals(
                SftExampleEncoder.USER_PREFIX + "привет\n" + SftExampleEncoder.ASSISTANT_PREFIX,
                wrapped);
        String already = SftExampleEncoder.wrapUserChatPrompt("Пользователь: a");
        assertTrue(already.contains(SftExampleEncoder.ASSISTANT_PREFIX));
        String both = SftExampleEncoder.wrapUserChatPrompt("Пользователь: a\nАссистент: ");
        assertEquals("Пользователь: a\nАссистент:", both.trim());
    }

    @Test
    void dialogSplitIsDisjointAndDeterministic() {
        List<Integer> items = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            items.add(i);
        }
        List<Integer> trainA = new ArrayList<>();
        List<Integer> valA = new ArrayList<>();
        SftCorpus.splitShuffled(items, 0.05, 42L, trainA, valA);
        assertEquals(5, valA.size());
        assertEquals(95, trainA.size());
        for (Integer v : valA) {
            assertTrue(!trainA.contains(v));
        }
        List<Integer> trainB = new ArrayList<>();
        List<Integer> valB = new ArrayList<>();
        SftCorpus.splitShuffled(items, 0.05, 42L, trainB, valB);
        assertEquals(valA, valB);
        assertEquals(trainA, trainB);
        List<Integer> trainC = new ArrayList<>();
        List<Integer> valC = new ArrayList<>();
        SftCorpus.splitShuffled(items, 0.05, 7L, trainC, valC);
        assertTrue(!valA.equals(valC));
    }

    @Test
    void uniqueUserSplitKeepsRepeatCopiesTogether() {
        SftExampleEncoder.Encoded dummy =
                new SftExampleEncoder.Encoded(new int[] {2, 4, 5, 3}, new boolean[] {false, false, true, true});
        List<SftCorpus.LabeledExample> all = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            all.add(new SftCorpus.LabeledExample("столица франции", dummy));
        }
        all.add(new SftCorpus.LabeledExample("2+2", dummy));
        all.add(new SftCorpus.LabeledExample("2+2", dummy));
        all.add(new SftCorpus.LabeledExample("кто ты", dummy));
        all.add(new SftCorpus.LabeledExample("кто ты", dummy));
        all.add(new SftCorpus.LabeledExample("небо голубое", dummy));
        all.add(new SftCorpus.LabeledExample("небо голубое", dummy));
        List<SftExampleEncoder.Encoded> train = new ArrayList<>();
        List<SftExampleEncoder.Encoded> val = new ArrayList<>();
        int uniqueVal = SftCorpus.splitByUniqueUserKey(all, 0.25, 42L, train, val);
        assertEquals(1, uniqueVal);
        assertEquals(all.size(), train.size() + val.size());
        assertTrue(val.size() == 2 || val.size() == 4);
        List<SftExampleEncoder.Encoded> train2 = new ArrayList<>();
        List<SftExampleEncoder.Encoded> val2 = new ArrayList<>();
        SftCorpus.splitByUniqueUserKey(all, 0.25, 42L, train2, val2);
        assertEquals(val.size(), val2.size());
        assertEquals(train.size(), train2.size());
    }

    @Test
    void packOnePerWindowKeepsDialogsApart() {
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
        List<SftWindowPacker.Window> packed = SftWindowPacker.pack(List.of(a, b), maxSeq, tok.padId());
        assertEquals(1, packed.size());
        List<SftWindowPacker.Window> ones = SftWindowPacker.packOnePerWindow(List.of(a, b), maxSeq, tok.padId());
        assertEquals(2, ones.size());
    }

    @Test
    void roleTokensAndCasePreserved() {
        BPETokenizer tok = BPETokenizer.train(List.of("Париж столица Франции 2+2=4"), 200, false);
        assertTrue(tok.hasChatRoleTokens());
        int[] ids = tok.encode(BPETokenizer.USER_TOKEN + "Париж" + BPETokenizer.ASSISTANT_TOKEN, false, false);
        assertEquals(tok.userId(), ids[0]);
        assertEquals(tok.assistantId(), ids[ids.length - 1]);
        SftExampleEncoder.Encoded ex =
                SftExampleEncoder.encode(
                        tok,
                        List.of(
                                new SftTurn(SftTurn.Role.USER, "столица Франции"),
                                new SftTurn(SftTurn.Role.ASSISTANT, "Париж")));
        assertEquals(tok.bosId(), ex.tokens[0]);
        assertEquals(tok.userId(), ex.tokens[1]);
        boolean sawAsstTok = false;
        for (int id : ex.tokens) {
            if (id == tok.assistantId()) {
                sawAsstTok = true;
                break;
            }
        }
        assertTrue(sawAsstTok);
    }
}
