package com.veles.llm.jgpt.data;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class DataLoaderPackedDocumentsTest {

    @Test
    void packedDocumentsMatchConcatenatedLoadTokens() {
        BPETokenizer tok = BPETokenizer.train(List.of("a b c d e f g h"), 48);
        int maxSeq = 8;
        DataLoader concat = new DataLoader(tok, maxSeq, 1, false, false);
        DataLoader packed = new DataLoader(tok, maxSeq, 1, false, false);

        List<int[]> docs = new ArrayList<>();
        docs.add(range(1, 5));
        docs.add(range(6, 12));
        docs.add(range(13, 14));
        docs.add(range(15, 40));
        int[] stream = concatAll(docs);
        List<int[]> copy = copyDocs(docs);

        concat.loadTokens(stream);
        boolean[] isVal = new boolean[docs.size()];
        long nTok = packed.loadPackedDocuments(copy, isVal, false);

        assertEquals(stream.length, nTok);
        assertEquals(concat.numSequences(), packed.numSequences());
        List<int[]> a = concat.copySequences();
        List<int[]> b = packed.copySequences();
        for (int i = 0; i < a.size(); i++) {
            assertArrayEquals(a.get(i), b.get(i), "window " + i);
        }
        for (int[] d : copy) {
            assertNull(d);
        }
    }

    @Test
    void shortDocsStillFormWindowsAcrossBoundaries() {
        BPETokenizer tok = BPETokenizer.train(List.of("a b c"), 32);
        int maxSeq = 4;
        DataLoader packed = new DataLoader(tok, maxSeq, 1, false, false);
        List<int[]> docs = new ArrayList<>();
        docs.add(new int[] {1, 2});
        docs.add(new int[] {3});
        docs.add(new int[] {4, 5, 6, 7, 8, 9, 10});
        packed.loadPackedDocuments(docs, new boolean[3], false);
        assertEquals(2, packed.numSequences());
        assertArrayEquals(new int[] {1, 2, 3, 4, 5}, packed.copySequences().get(0));
        assertArrayEquals(new int[] {5, 6, 7, 8, 9}, packed.copySequences().get(1));
    }

    @Test
    void trainValSplitDoesNotMixDocuments() {
        BPETokenizer tok = BPETokenizer.train(List.of("a b c d e f g h"), 48);
        int maxSeq = 3;
        DataLoader train = new DataLoader(tok, maxSeq, 1, false, false);
        DataLoader val = new DataLoader(tok, maxSeq, 1, false, false);
        List<int[]> docs = new ArrayList<>();
        docs.add(new int[] {1, 2, 3, 4, 5, 6, 7, 8});
        docs.add(new int[] {9, 10, 11, 12, 13, 14, 15, 16});
        boolean[] isVal = {false, true};
        train.loadPackedDocuments(docs, isVal, false);
        val.loadPackedDocuments(docs, isVal, true);
        for (int[] w : train.copySequences()) {
            for (int t : w) {
                if (t >= 9) {
                    throw new AssertionError("val token in train window: " + Arrays.toString(w));
                }
            }
        }
        for (int[] w : val.copySequences()) {
            for (int t : w) {
                if (t > 0 && t < 9) {
                    throw new AssertionError("train token in val window: " + Arrays.toString(w));
                }
            }
        }
    }

    private static int[] range(int lo, int hiInclusive) {
        int[] a = new int[hiInclusive - lo + 1];
        for (int i = 0; i < a.length; i++) {
            a[i] = lo + i;
        }
        return a;
    }

    private static int[] concatAll(List<int[]> docs) {
        int n = 0;
        for (int[] d : docs) {
            n += d.length;
        }
        int[] out = new int[n];
        int off = 0;
        for (int[] d : docs) {
            System.arraycopy(d, 0, out, off, d.length);
            off += d.length;
        }
        return out;
    }

    private static List<int[]> copyDocs(List<int[]> docs) {
        List<int[]> copy = new ArrayList<>(docs.size());
        for (int[] d : docs) {
            copy.add(Arrays.copyOf(d, d.length));
        }
        return copy;
    }
}
