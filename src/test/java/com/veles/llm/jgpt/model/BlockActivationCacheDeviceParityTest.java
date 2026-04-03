package com.veles.llm.jgpt.model;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;

import java.lang.reflect.Field;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * Слоты {@link BlockActivationCacheDevice} после forward с device decoder совпадают с эталоном из host
 * {@link BlockActivationCache} (отдельная модель с {@code deviceDecoderBackward=false}).
 */
class BlockActivationCacheDeviceParityTest {

    private static final float SLOT_EPS = 1e-5f;

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    private static void assertSlotClose(
            String name, Tensor host, BlockActivationCacheDevice dc, BlockActivationCacheDevice.SlotId slot, int n) {
        float[] h = new float[n];
        float[] d = new float[n];
        System.arraycopy(host.internalBuffer(), 0, h, 0, n);
        dc.copySlotToHostFloat(slot, d, 0, n);
        float diff = maxAbsDiff(h, d);
        float eps = dc.isFp16ActivationStorage() ? 1e-3f : SLOT_EPS;
        assertTrue(diff < eps, name + ": max abs diff " + diff);
    }

    @Test
    void hostReferenceAndDeviceCacheMatch_afterForward() throws Exception {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel modelHost = new GPTModel(48, 8, 32, 4, 2, 64, true);
            modelHost.setDeviceLogitsEnabled(true);
            modelHost.setDeviceDecoderBackward(false);

            GPTModel modelDev = new GPTModel(48, 8, 32, 4, 2, 64, true);
            modelDev.setDeviceLogitsEnabled(true);
            modelDev.setDeviceDecoderBackward(true);

            List<Tensor> phList = modelHost.getParameters();
            List<Tensor> pdList = modelDev.getParameters();
            for (int i = 0; i < phList.size(); i++) {
                Tensor ph = phList.get(i);
                Tensor pd = pdList.get(i);
                System.arraycopy(ph.internalBuffer(), 0, pd.internalBuffer(), 0, ph.size());
            }
            modelDev.syncGpuResidentWeightsFromHost();

            Tensor input = new Tensor(new int[] {2, 8});
            float[] id = input.internalBuffer();
            for (int i = 0; i < id.length; i++) {
                id[i] = (i * 11 + 3) % 48;
            }

            modelHost.forward(input, true, true);
            modelDev.forward(input, true, true);

            Field fHost = GPTModel.class.getDeclaredField("blockCaches");
            Field fDev = GPTModel.class.getDeclaredField("blockCachesDevice");
            fHost.setAccessible(true);
            fDev.setAccessible(true);
            BlockActivationCache[] hostCaches = (BlockActivationCache[]) fHost.get(modelHost);
            BlockActivationCacheDevice[] devCaches = (BlockActivationCacheDevice[]) fDev.get(modelDev);
            assertTrue(hostCaches != null && devCaches != null);
            for (int li = 0; li < hostCaches.length; li++) {
                BlockActivationCache hc = hostCaches[li];
                BlockActivationCacheDevice dc = devCaches[li];
                int flat = 2 * 8 * 32;
                assertSlotClose(
                        "xIn L" + li,
                        hc.xIn.getTensor(),
                        dc,
                        BlockActivationCacheDevice.SlotId.X_IN,
                        flat);
                assertSlotClose("xNorm1", hc.xNorm1.getTensor(), dc, BlockActivationCacheDevice.SlotId.X_NORM1, flat);
                assertSlotClose("attnOut", hc.attnOut.getTensor(), dc, BlockActivationCacheDevice.SlotId.ATTN_OUT, flat);
                assertSlotClose("xRes1", hc.xRes1.getTensor(), dc, BlockActivationCacheDevice.SlotId.X_RES1, flat);
                assertSlotClose("xNorm2", hc.xNorm2.getTensor(), dc, BlockActivationCacheDevice.SlotId.X_NORM2, flat);
                assertSlotClose("ffnOut", hc.ffnOut.getTensor(), dc, BlockActivationCacheDevice.SlotId.FFN_OUT, flat);
                assertSlotClose("xOut", hc.xOut.getTensor(), dc, BlockActivationCacheDevice.SlotId.X_OUT, flat);
                int dInt = 64;
                int mid = 2 * 8 * dInt;
                assertSlotClose("ffnH1", hc.ffnH1.getTensor(), dc, BlockActivationCacheDevice.SlotId.FFN_H1, mid);
                assertSlotClose("ffnGate", hc.ffnGate.getTensor(), dc, BlockActivationCacheDevice.SlotId.FFN_GATE, mid);
                assertSlotClose("attnQ", hc.attnQHeads.getTensor(), dc, BlockActivationCacheDevice.SlotId.ATTN_Q_HEADS, flat);
                assertSlotClose("attnK", hc.attnKHeads.getTensor(), dc, BlockActivationCacheDevice.SlotId.ATTN_K_HEADS, flat);
                assertSlotClose("attnV", hc.attnVHeads.getTensor(), dc, BlockActivationCacheDevice.SlotId.ATTN_V_HEADS, flat);
                int bh = 2 * 4;
                int probs = bh * 8 * 8;
                assertSlotClose(
                        "attnProbs", hc.attnProbs.getTensor(), dc, BlockActivationCacheDevice.SlotId.ATTN_PROBS, probs);
                assertSlotClose(
                        "attnConcat", hc.attnConcat.getTensor(), dc, BlockActivationCacheDevice.SlotId.ATTN_CONCAT, flat);
            }
            modelHost.closeGpuResidentWeights();
            modelDev.closeGpuResidentWeights();
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
        }
    }
}
