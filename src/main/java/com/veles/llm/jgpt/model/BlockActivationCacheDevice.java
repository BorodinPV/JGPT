package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.GpuHalfBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Активации одного decoder-блока на VRAM для fused backward без полного кэша на хосте ({@link
 * BlockActivationCache}).
 *
 * <p>При {@code JGPT_ACTIVATION_CACHE_FP16=1} (или {@code -Djgpt.activationCache.fp16=true}) слоты хранятся как
 * <b>FP16 на device</b> (~2 байта на скаляр); иначе — {@link GpuFloatBuffer} (FP32). Чтение/запись — через {@link
 * #copySlotFromDeviceFloat(SlotId, GpuFloatBuffer, int)} и сёстры; прямой доступ к сырым буферам не поддерживается.
 *
 * <p>{@code JGPT_BLOCK_CACHE_MAX_BYTES}: сравнение с оценкой {@code totalFloats × bytesPerElem}, где
 * {@code bytesPerElem} равно 2 при FP16-хранении и 4 при FP32.
 *
 * <p><b>Потокобезопасность:</b> не потокобезопасен; вызывать из одного потока обучения.
 */
public final class BlockActivationCacheDevice implements AutoCloseable {

    private static final Logger CACHE_LOG = Logger.getLogger(BlockActivationCacheDevice.class.getName());

    /** Слоты кэша (одинаковое число логических float в FP16 и FP32 режимах). */
    public enum SlotId {
        X_IN,
        X_NORM1,
        ATTN_OUT,
        X_RES1,
        X_NORM2,
        FFN_OUT,
        X_OUT,
        FFN_H1,
        FFN_GATE,
        ATTN_Q_HEADS,
        ATTN_K_HEADS,
        ATTN_V_HEADS,
        ATTN_PROBS,
        ATTN_CONCAT,
        /** FlashAttention: log-sum-exp per query row [BH, S]. */
        ATTN_LSE,
        /** FlashAttention: head-wise attention output O [BH, S, Dh], needed for backward D computation. */
        ATTN_OUT_HEADS
    }

    private GpuFloatBuffer xIn;
    private GpuFloatBuffer xNorm1;
    private GpuFloatBuffer attnOut;
    private GpuFloatBuffer xRes1;
    private GpuFloatBuffer xNorm2;
    private GpuFloatBuffer ffnOut;
    private GpuFloatBuffer xOut;
    private GpuFloatBuffer ffnH1;
    private GpuFloatBuffer ffnGate;
    private GpuFloatBuffer attnQHeads;
    private GpuFloatBuffer attnKHeads;
    private GpuFloatBuffer attnVHeads;
    private GpuFloatBuffer attnProbs;
    private GpuFloatBuffer attnConcat;
    /** FlashAttention: LSE [BH*S] — float only (no fp16 variant needed, tiny buffer). */
    private GpuFloatBuffer attnLse;
    /** FlashAttention: O_heads [BH*S*Dh] — head-wise attention output before concatHeads. */
    private GpuFloatBuffer attnOutHeads;

    private GpuHalfBuffer hxIn;
    private GpuHalfBuffer hxNorm1;
    private GpuHalfBuffer hAttnOut;
    private GpuHalfBuffer hxRes1;
    private GpuHalfBuffer hxNorm2;
    private GpuHalfBuffer hFfnOut;
    private GpuHalfBuffer hxOut;
    private GpuHalfBuffer hFfnH1;
    private GpuHalfBuffer hFfnGate;
    private GpuHalfBuffer hAttnQHeads;
    private GpuHalfBuffer hAttnKHeads;
    private GpuHalfBuffer hAttnVHeads;
    private GpuHalfBuffer hAttnProbs;
    private GpuHalfBuffer hAttnConcat;

    /** {@code true} после первой аллокации, если слоты — half. */
    private boolean storageFp16;

    /**
     * Только SDPA attention probs (FP16-слоты): указатель участвует в CUDA Graph декодера — не перевыделять из путей
     * {@link #copySlotFromHostFloat}/{@link #copySlotToHostFloat}.
     */
    private GpuFloatBuffer attnProbsFloatStage;
    private long attnProbsFloatStageCap;

    /** Staging half↔float для произвольных слотов; отделён от {@link #attnProbsFloatStage}. */
    private GpuFloatBuffer halfSlotFloatStage;
    private long halfSlotFloatStageCap;

    private int cachedBatch = -1;
    private int cachedSeqLen = -1;
    private int cachedDModel = -1;
    private int cachedNumHeads = -1;
    private int cachedDInt = -1;

    /**
     * Инкремент при любом перевыделении device-буферов слотов/Flash/stage, на которые может ссылаться CUDA graph
     * декодера; участвует в {@link com.veles.llm.jgpt.model.GPTModel} {@code decoderLayerGraphKey}.
     */
    private long graphCaptureGeneration;

    private static final ThreadLocal<Map<ArchKey, ArrayDeque<PooledBuffers>>> BLOCK_CACHE_POOL =
            ThreadLocal.withInitial(HashMap::new);

    public BlockActivationCacheDevice() {}

    /** Как фактически выделен кэш после {@link #ensure}; до ensure — {@code false}. */
    public boolean isFp16ActivationStorage() {
        return storageFp16;
    }

    /** Поколение буферов для инвалидации decoder CUDA graph при смене указателей на VRAM. */
    public long graphCaptureGeneration() {
        return graphCaptureGeneration;
    }

    /** Только диагностика decoder CUDA graph: строка с cacheGen и ключевыми device pointer'ами слота. */
    void appendDecoderGraphDevicePointers(StringBuilder sb) {
        sb.append("cacheGen=").append(graphCaptureGeneration);
        if (!isAllocated()) {
            sb.append(" cache=!alloc");
            return;
        }
        if (TensorOpsGPU.FLASH_ATTENTION) {
            if (attnLse != null && !attnLse.isClosed()) {
                sb.append(" lse=0x").append(Long.toHexString(attnLse.devicePointer()));
            }
            if (attnOutHeads != null && !attnOutHeads.isClosed()) {
                sb.append(" attnOutHeads=0x").append(Long.toHexString(attnOutHeads.devicePointer()));
            }
        }
        if (attnProbsFloatStage != null && !attnProbsFloatStage.isClosed()) {
            sb.append(" probsFloatStage=0x").append(Long.toHexString(attnProbsFloatStage.devicePointer()));
        }
    }

    /**
     * Заполняет {@code s[8..10]} для сравнения снимков (LSE, O_heads, FP32 staging probs); остальные индексы —
     * {@link com.veles.llm.jgpt.model.GPTModel}.
     */
    void fillDecoderGraphDebugSnapshotSlots(long[] s) {
        if (s == null || s.length < 11) {
            return;
        }
        if (!isAllocated()) {
            return;
        }
        if (TensorOpsGPU.FLASH_ATTENTION) {
            if (attnLse != null && !attnLse.isClosed()) {
                s[8] = attnLse.devicePointer();
            }
            if (attnOutHeads != null && !attnOutHeads.isClosed()) {
                s[9] = attnOutHeads.devicePointer();
            }
        }
        if (attnProbsFloatStage != null && !attnProbsFloatStage.isClosed()) {
            s[10] = attnProbsFloatStage.devicePointer();
        }
    }

    private void bumpGraphCaptureGeneration() {
        graphCaptureGeneration++;
    }

    /**
     * Закрывает только transient FP32 staging ({@link #attnProbsFloatStage}, {@link #halfSlotFloatStage}). Имеет смысл
     * после {@link com.veles.llm.jgpt.model.GPTModel#destroyDecoderLayerCudaGraphs()} на OOM: основные слоты кэша не
     * трогаем (backward), а staging снимает фрагментацию перед eager-остатком forward.
     *
     * @return {@code true}, если что-то было освобождено
     */
    public boolean releaseTransientFloatStagingBuffers() {
        boolean changed = false;
        if (attnProbsFloatStage != null) {
            if (!attnProbsFloatStage.isClosed()) {
                attnProbsFloatStage.close();
                changed = true;
            }
            attnProbsFloatStage = null;
            attnProbsFloatStageCap = 0L;
        }
        if (halfSlotFloatStage != null) {
            if (!halfSlotFloatStage.isClosed()) {
                halfSlotFloatStage.close();
                changed = true;
            }
            halfSlotFloatStage = null;
            halfSlotFloatStageCap = 0L;
        }
        if (changed) {
            bumpGraphCaptureGeneration();
        }
        return changed;
    }

    /** Returns true for slots that are always stored as float32 regardless of storageFp16. */
    private static boolean isAlwaysFloat32(SlotId id) {
        return id == SlotId.ATTN_LSE || id == SlotId.ATTN_OUT_HEADS;
    }

    public void copySlotFromDeviceFloat(SlotId id, GpuFloatBuffer src, int n) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(src, "src");
        if (n <= 0) {
            return;
        }
        if (storageFp16 && !isAlwaysFloat32(id)) {
            TensorOpsGPU.convertFloatDeviceToHalfDevice(src.devicePointer(), requireHalf(id).devicePointer(), n);
        } else {
            dstFloat(id).copyFromDevice(src, n);
        }
    }

    public void copySlotToDeviceFloat(SlotId id, GpuFloatBuffer dst, int n) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(dst, "dst");
        if (n <= 0) {
            return;
        }
        if (storageFp16 && !isAlwaysFloat32(id)) {
            TensorOpsGPU.convertHalfDeviceToFloatDevice(requireHalf(id).devicePointer(), dst.devicePointer(), n);
        } else {
            dst.copyFromDevice(dstFloat(id), n);
        }
    }

    public void copySlotFromHostFloat(SlotId id, float[] src, int srcOff, int n) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(src, "src");
        if (n <= 0) {
            return;
        }
        if (storageFp16 && !isAlwaysFloat32(id)) {
            ensureHalfSlotFloatStage(n);
            halfSlotFloatStage.copyFrom(src, srcOff, n);
            TensorOpsGPU.convertFloatDeviceToHalfDevice(
                    halfSlotFloatStage.devicePointer(), requireHalf(id).devicePointer(), n);
        } else {
            dstFloat(id).copyFrom(src, srcOff, n);
        }
    }

    /**
     * Буфер для прямой записи attention probs с device (FP32-слот или {@link #attnProbsFloatStage} при
     * FP16-хранилище).
     */
    public GpuFloatBuffer attnProbsWriteBufferAsFloat(int probFloats) {
        if (probFloats <= 0) {
            throw new IllegalArgumentException("probFloats must be positive");
        }
        if (storageFp16) {
            ensureAttnProbsFloatStage(probFloats);
            return attnProbsFloatStage;
        }
        GpuFloatBuffer b = dstFloat(SlotId.ATTN_PROBS);
        if (b.numFloats() < probFloats) {
            throw new IllegalStateException("ATTN_PROBS buffer too small: " + b.numFloats() + " < " + probFloats);
        }
        return b;
    }

    /**
     * Перед {@code cudaStreamBeginCapture} на декодере: для FP16-слотов выделить {@link #attnProbsFloatStage} под SDPA
     * probs (изолирован от {@link #halfSlotFloatStage}), иначе ленивое выделение внутри захвата вызвало бы {@code
     * cudaMalloc} (недопустимо при построении графа).
     */
    public void ensureAttentionProbsFloatStageForGraphCapture(int probFloats) {
        if (probFloats <= 0 || !isAllocated()) {
            return;
        }
        if (storageFp16) {
            ensureAttnProbsFloatStage(probFloats);
        }
    }

    /** После SDPA в float staging: конвертация в half-слот (no-op если кэш уже FP32). */
    public void storeAttnProbsFromFloatStagingIfFp16(GpuFloatBuffer floatSrc, int n) {
        Objects.requireNonNull(floatSrc, "floatSrc");
        if (!storageFp16 || n <= 0) {
            return;
        }
        TensorOpsGPU.convertFloatDeviceToHalfDevice(
                floatSrc.devicePointer(), requireHalf(SlotId.ATTN_PROBS).devicePointer(), n);
    }

    public void copySlotToHostFloat(SlotId id, float[] dst, int dstOff, int n) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(dst, "dst");
        if (n <= 0) {
            return;
        }
        if (storageFp16 && !isAlwaysFloat32(id)) {
            ensureHalfSlotFloatStage(n);
            TensorOpsGPU.convertHalfDeviceToFloatDevice(
                    requireHalf(id).devicePointer(), halfSlotFloatStage.devicePointer(), n);
            halfSlotFloatStage.copyTo(dst, dstOff, n);
        } else {
            dstFloat(id).copyTo(dst, dstOff, n);
        }
    }

    private void ensureAttnProbsFloatStage(long needFloats) {
        if (attnProbsFloatStage != null
                && !attnProbsFloatStage.isClosed()
                && attnProbsFloatStageCap >= needFloats) {
            return;
        }
        if (attnProbsFloatStage != null) {
            attnProbsFloatStage.close();
        }
        attnProbsFloatStage = GpuFloatBuffer.allocate(needFloats);
        attnProbsFloatStageCap = needFloats;
        bumpGraphCaptureGeneration();
    }

    private void ensureHalfSlotFloatStage(long needFloats) {
        if (halfSlotFloatStage != null
                && !halfSlotFloatStage.isClosed()
                && halfSlotFloatStageCap >= needFloats) {
            return;
        }
        if (halfSlotFloatStage != null) {
            halfSlotFloatStage.close();
        }
        halfSlotFloatStage = GpuFloatBuffer.allocate(needFloats);
        halfSlotFloatStageCap = needFloats;
        bumpGraphCaptureGeneration();
    }

    private GpuFloatBuffer dstFloat(SlotId id) {
        return switch (id) {
            case X_IN -> requireBuf(xIn, "xIn");
            case X_NORM1 -> requireBuf(xNorm1, "xNorm1");
            case ATTN_OUT -> requireBuf(attnOut, "attnOut");
            case X_RES1 -> requireBuf(xRes1, "xRes1");
            case X_NORM2 -> requireBuf(xNorm2, "xNorm2");
            case FFN_OUT -> requireBuf(ffnOut, "ffnOut");
            case X_OUT -> requireBuf(xOut, "xOut");
            case FFN_H1 -> requireBuf(ffnH1, "ffnH1");
            case FFN_GATE -> requireBuf(ffnGate, "ffnGate");
            case ATTN_Q_HEADS -> requireBuf(attnQHeads, "attnQHeads");
            case ATTN_K_HEADS -> requireBuf(attnKHeads, "attnKHeads");
            case ATTN_V_HEADS -> requireBuf(attnVHeads, "attnVHeads");
            case ATTN_PROBS -> requireBuf(attnProbs, "attnProbs");
            case ATTN_CONCAT -> requireBuf(attnConcat, "attnConcat");
            case ATTN_LSE -> requireBuf(attnLse, "attnLse");
            case ATTN_OUT_HEADS -> requireBuf(attnOutHeads, "attnOutHeads");
        };
    }

    private GpuHalfBuffer requireHalf(SlotId id) {
        GpuHalfBuffer b =
                switch (id) {
                    case X_IN -> hxIn;
                    case X_NORM1 -> hxNorm1;
                    case ATTN_OUT -> hAttnOut;
                    case X_RES1 -> hxRes1;
                    case X_NORM2 -> hxNorm2;
                    case FFN_OUT -> hFfnOut;
                    case X_OUT -> hxOut;
                    case FFN_H1 -> hFfnH1;
                    case FFN_GATE -> hFfnGate;
                    case ATTN_Q_HEADS -> hAttnQHeads;
                    case ATTN_K_HEADS -> hAttnKHeads;
                    case ATTN_V_HEADS -> hAttnVHeads;
                    case ATTN_PROBS -> hAttnProbs;
                    case ATTN_CONCAT -> hAttnConcat;
                    // LSE and OUT_HEADS are always float — no fp16 variant
                    case ATTN_LSE, ATTN_OUT_HEADS -> throw new IllegalStateException(
                            id + " is always stored as float32, use dstFloat path");
                };
        if (b == null || b.isClosed()) {
            throw new IllegalStateException("Call ensure() before accessing " + id);
        }
        return b;
    }

    private static GpuFloatBuffer requireBuf(GpuFloatBuffer b, String name) {
        if (b == null || b.isClosed()) {
            throw new IllegalStateException("Call ensure() before accessing " + name);
        }
        return b;
    }

    /**
     * Выделяет или переиспользует буферы под текущий forward (FP32 или FP16 на device по env).
     */
    public void ensure(int batch, int seqLen, int dModel, int numHeads, int dIntermediate) {
        if (batch == cachedBatch
                && seqLen == cachedSeqLen
                && dModel == cachedDModel
                && numHeads == cachedNumHeads
                && dIntermediate == cachedDInt) {
            return;
        }
        long rows = mulExact("batch*seqLen", (long) batch, (long) seqLen);
        long plane = mulExact("batch*seqLen*dModel", rows, (long) dModel);
        int dHead = dModel / numHeads;
        long batchHeads = mulExact("batch*numHeads", (long) batch, (long) numHeads);
        long headFlat = mulExact("batchHeads*seqLen*dHead", mulExact("batchHeads*seqLen", batchHeads, (long) seqLen), (long) dHead);
        // In flash attention mode attnProbs (S²) is replaced by LSE + O_heads (both tiny/moderate).
        boolean flash = TensorOpsGPU.FLASH_ATTENTION;
        long probs = flash ? 0L :
                mulExact("batchHeads*seqLen*seqLen", mulExact("batchHeads*seqLen", batchHeads, (long) seqLen), (long) seqLen);
        long lseFlat = flash ? mulExact("batchHeads*seqLen", batchHeads, (long) seqLen) : 0L;
        long ffnMid = mulExact("batch*seqLen*dIntermediate", rows, (long) dIntermediate);

        // lseFlat and headFlat (for attnOutHeads) are float32-only; add them to size estimate
        long totalFloats =
                Math.addExact(
                        Math.addExact(
                                Math.addExact(
                                        Math.addExact(Math.multiplyExact(8L, plane), Math.multiplyExact(2L, ffnMid)),
                                        Math.multiplyExact(3L, headFlat)),
                                probs),
                        flash ? Math.addExact(lseFlat, headFlat) : 0L);
        long maxBytes = blockActivationCacheMaxBytesFromEnv();
        if (maxBytes > 0L) {
            int bpe = blockActivationCacheBytesPerElement();
            long needBytes = Math.multiplyExact(totalFloats, (long) bpe);
            if (needBytes > maxBytes) {
                purgeBlockActivationCachePool();
            }
            if (needBytes > maxBytes) {
                long actualApprox;
                try {
                    actualApprox = Math.multiplyExact(totalFloats, storageFp16 ? 2L : 4L);
                } catch (ArithmeticException ex) {
                    actualApprox = -1L;
                }
                CACHE_LOG.log(
                        Level.WARNING,
                        "JGPT_BLOCK_CACHE_MAX_BYTES={0} below estimate {1} B (totalFloats={2}, bytes/elem={3}); "
                                + "allocating anyway. Purged thread-local pool first. Disable cap with JGPT_BLOCK_CACHE_MAX_BYTES=0. "
                                + "Actual device cache VRAM ~{4} B ({5}).",
                        new Object[] {
                            maxBytes,
                            needBytes,
                            totalFloats,
                            bpe,
                            actualApprox,
                            activationCacheFp16StorageFromEnv() ? "FP16 slots" : "FP32 slots"
                        });
            }
        }

        boolean wantFp16 = activationCacheFp16StorageFromEnv();
        if (blockActivationCacheGrowOnlyFromEnv()
                && isAllocatedForMode(wantFp16)
                && dModel == cachedDModel
                && numHeads == cachedNumHeads
                && dIntermediate == cachedDInt
                && plane <= slotCapacityPlane(wantFp16)
                && headFlat <= slotCapacityHeadFlat(wantFp16)
                && (flash || probs <= slotCapacityProbs(wantFp16))
                && ffnMid <= slotCapacityFfnMid(wantFp16)) {
            cachedBatch = batch;
            cachedSeqLen = seqLen;
            // Flash attention buffers may need (re)allocation if grow-only reused a pool object
            ensureFlashBuffers(flash, lseFlat, headFlat);
            return;
        }

        returnOrFreeBuffers();

        PooledBuffers borrowed = tryBorrowFromPool(plane, headFlat, probs, ffnMid, dModel, numHeads, dIntermediate, wantFp16);
        if (borrowed != null) {
            borrowed.moveInto(this);
            cachedBatch = batch;
            cachedSeqLen = seqLen;
            cachedDModel = dModel;
            cachedNumHeads = numHeads;
            cachedDInt = dIntermediate;
            storageFp16 = wantFp16;
            ensureFlashBuffers(flash, lseFlat, headFlat);
            return;
        }

        storageFp16 = wantFp16;
        if (wantFp16) {
            clearFp32Slots();
            hxIn = GpuHalfBuffer.allocate(plane);
            hxNorm1 = GpuHalfBuffer.allocate(plane);
            hAttnOut = GpuHalfBuffer.allocate(plane);
            hxRes1 = GpuHalfBuffer.allocate(plane);
            hxNorm2 = GpuHalfBuffer.allocate(plane);
            hFfnOut = GpuHalfBuffer.allocate(plane);
            hxOut = GpuHalfBuffer.allocate(plane);
            hFfnH1 = GpuHalfBuffer.allocate(ffnMid);
            hFfnGate = GpuHalfBuffer.allocate(ffnMid);
            hAttnQHeads = GpuHalfBuffer.allocate(headFlat);
            hAttnKHeads = GpuHalfBuffer.allocate(headFlat);
            hAttnVHeads = GpuHalfBuffer.allocate(headFlat);
            if (!flash) hAttnProbs = GpuHalfBuffer.allocate(probs);
            hAttnConcat = GpuHalfBuffer.allocate(plane);
        } else {
            clearFp16Slots();
            xIn = GpuFloatBuffer.allocate(plane);
            xNorm1 = GpuFloatBuffer.allocate(plane);
            attnOut = GpuFloatBuffer.allocate(plane);
            xRes1 = GpuFloatBuffer.allocate(plane);
            xNorm2 = GpuFloatBuffer.allocate(plane);
            ffnOut = GpuFloatBuffer.allocate(plane);
            xOut = GpuFloatBuffer.allocate(plane);
            ffnH1 = GpuFloatBuffer.allocate(ffnMid);
            ffnGate = GpuFloatBuffer.allocate(ffnMid);
            attnQHeads = GpuFloatBuffer.allocate(headFlat);
            attnKHeads = GpuFloatBuffer.allocate(headFlat);
            attnVHeads = GpuFloatBuffer.allocate(headFlat);
            if (!flash) attnProbs = GpuFloatBuffer.allocate(probs);
            attnConcat = GpuFloatBuffer.allocate(plane);
        }
        ensureFlashBuffers(flash, lseFlat, headFlat);

        cachedBatch = batch;
        cachedSeqLen = seqLen;
        cachedDModel = dModel;
        cachedNumHeads = numHeads;
        cachedDInt = dIntermediate;
        bumpGraphCaptureGeneration();
    }

    /**
     * Allocates/resizes flash-attention-only buffers (LSE and O_heads).
     * Called from ensure() after the main pool/alloc path. No-op when flash=false.
     *
     * <p>При росте {@code lseFlat}/{@code headFlat} старый {@link GpuFloatBuffer} закрывается перед новым allocate —
     * это не утечка, а смена ёмкости (см. {@link #graphCaptureGeneration}).
     */
    private void ensureFlashBuffers(boolean flash, long lseFlat, long headFlat) {
        if (!flash) return;
        if (attnLse == null || attnLse.isClosed() || attnLse.numFloats() < lseFlat) {
            if (attnLse != null && !attnLse.isClosed()) attnLse.close();
            attnLse = GpuFloatBuffer.allocate(lseFlat);
            bumpGraphCaptureGeneration();
        }
        if (attnOutHeads == null || attnOutHeads.isClosed() || attnOutHeads.numFloats() < headFlat) {
            if (attnOutHeads != null && !attnOutHeads.isClosed()) attnOutHeads.close();
            attnOutHeads = GpuFloatBuffer.allocate(headFlat);
            bumpGraphCaptureGeneration();
        }
    }

    /** FlashAttention: device buffer for LSE [BH*S]. Valid only when FLASH_ATTENTION=true. */
    public GpuFloatBuffer attnLseBuffer() {
        return requireBuf(attnLse, "attnLse");
    }

    /** FlashAttention: device buffer for O_heads [BH*S*Dh]. Valid only when FLASH_ATTENTION=true. */
    public GpuFloatBuffer attnOutHeadsBuffer() {
        return requireBuf(attnOutHeads, "attnOutHeads");
    }

    private boolean isAllocatedForMode(boolean fp16) {
        if (fp16) {
            return hxIn != null && !hxIn.isClosed();
        }
        return xIn != null && !xIn.isClosed();
    }

    private long slotCapacityPlane(boolean fp16) {
        if (fp16) {
            return hxIn != null && !hxIn.isClosed() ? hxIn.numHalfs() : 0L;
        }
        return xIn != null && !xIn.isClosed() ? xIn.numFloats() : 0L;
    }

    private long slotCapacityHeadFlat(boolean fp16) {
        if (fp16) {
            return hAttnQHeads != null && !hAttnQHeads.isClosed() ? hAttnQHeads.numHalfs() : 0L;
        }
        return attnQHeads != null && !attnQHeads.isClosed() ? attnQHeads.numFloats() : 0L;
    }

    private long slotCapacityProbs(boolean fp16) {
        if (fp16) {
            return hAttnProbs != null && !hAttnProbs.isClosed() ? hAttnProbs.numHalfs() : 0L;
        }
        return attnProbs != null && !attnProbs.isClosed() ? attnProbs.numFloats() : 0L;
    }

    private long slotCapacityFfnMid(boolean fp16) {
        if (fp16) {
            return hFfnH1 != null && !hFfnH1.isClosed() ? hFfnH1.numHalfs() : 0L;
        }
        return ffnH1 != null && !ffnH1.isClosed() ? ffnH1.numFloats() : 0L;
    }

    private static long mulExact(String what, long a, long b) {
        try {
            return Math.multiplyExact(a, b);
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("Size overflow (" + what + "): " + a + " * " + b, e);
        }
    }

    private static int blockActivationCacheBytesPerElement() {
        return activationCacheFp16StorageFromEnv() ? 2 : 4;
    }

    /**
     * Та же политика, что и {@link GPTModel} для {@code useFp16ActivationCache}: env, затем
     * {@code -Djgpt.activationCache.fp16}. Явные {@code 0} / {@code false} / {@code no} в env отключают FP16 даже при
     * системном property.
     */
    static boolean activationCacheFp16StorageFromEnv() {
        try {
            String e = System.getenv("JGPT_ACTIVATION_CACHE_FP16");
            if (e != null && !e.isBlank()) {
                String t = e.trim();
                if ("1".equals(t) || "true".equalsIgnoreCase(t)) {
                    return true;
                }
                if ("0".equals(t) || "false".equalsIgnoreCase(t) || "no".equalsIgnoreCase(t)) {
                    return false;
                }
            }
        } catch (Exception ignored) {
            /* SecurityManager и т.п. */
        }
        return Boolean.getBoolean("jgpt.activationCache.fp16");
    }

    private static void purgeBlockActivationCachePool() {
        Map<ArchKey, ArrayDeque<PooledBuffers>> m = BLOCK_CACHE_POOL.get();
        for (ArrayDeque<PooledBuffers> dq : m.values()) {
            while (!dq.isEmpty()) {
                PooledBuffers p = dq.poll();
                if (p != null) {
                    p.closeAll();
                }
            }
        }
        m.clear();
    }

    private static boolean blockActivationCacheGrowOnlyFromEnv() {
        String e = System.getenv("JGPT_BLOCK_CACHE_GROW_ONLY");
        if (e == null) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    private static boolean blockActivationCachePoolFromEnv() {
        /*
         * BUG-FIX: GROW_ONLY + POOL несовместимы (pool steal'ит буферы из cache,
         * после чего growOnly не может их переиспользовать). При GROW_ONLY
         * отключаем пул — буферы переиспользуются напрямую в cache.
         */
        if (blockActivationCacheGrowOnlyFromEnv()) {
            return false;
        }
        String e = System.getenv("JGPT_BLOCK_CACHE_POOL");
        if (e == null) {
            return false;
        }
        String t = e.trim();
        return "1".equals(t) || "true".equalsIgnoreCase(t);
    }

    private static int blockActivationCachePoolMaxFromEnv() {
        String e = System.getenv("JGPT_BLOCK_CACHE_POOL_MAX");
        if (e == null) {
            return 2;
        }
        String t = e.trim();
        if (t.isEmpty()) {
            return 2;
        }
        try {
            int v = Integer.parseInt(t);
            return v < 1 ? 2 : Math.min(v, 64);
        } catch (NumberFormatException ex) {
            return 2;
        }
    }

    private static long blockActivationCacheMaxBytesFromEnv() {
        String e = System.getenv("JGPT_BLOCK_CACHE_MAX_BYTES");
        if (e == null) {
            return 0L;
        }
        String t = e.trim();
        if (t.isEmpty()) {
            return 0L;
        }
        try {
            long v = Long.parseLong(t);
            return v < 0L ? 0L : v;
        } catch (NumberFormatException ex) {
            return 0L;
        }
    }

    private void returnOrFreeBuffers() {
        if (!isAllocated()) {
            return;
        }
        /*
         * BUG-FIX: GROW_ONLY + POOL несовместимы. При GROW_ONLY буферы
         * переиспользуются в cache напрямую (ensure() early return),
         * а returnOrFreeBuffers() через stealFrom() опустошает cache,
         * после чего isAllocatedForMode=false и growOnly больше не срабатывает.
         * Результат — каждый forward аллоцирует новые буферы → OOM.
         * Решение: при GROW_ONLY закрываем буферы напрямую, минуя пул.
         */
        if (blockActivationCacheGrowOnlyFromEnv()) {
            closeBuffersHard();
            return;
        }
        if (blockActivationCachePoolFromEnv()) {
            ArchKey key = new ArchKey(cachedDModel, cachedNumHeads, cachedDInt, storageFp16);
            PooledBuffers pooled = PooledBuffers.stealFrom(this);
            Map<ArchKey, ArrayDeque<PooledBuffers>> m = BLOCK_CACHE_POOL.get();
            int cap = blockActivationCachePoolMaxFromEnv();
            ArrayDeque<PooledBuffers> q = m.computeIfAbsent(key, k -> new ArrayDeque<>(2));
            while (q.size() >= cap) {
                PooledBuffers ev = q.pollLast();
                if (ev != null) {
                    ev.closeAll();
                }
            }
            q.offerFirst(pooled);
            return;
        }
        closeBuffersHard();
    }

    private static PooledBuffers tryBorrowFromPool(
            long plane,
            long headFlat,
            long probs,
            long ffnMid,
            int dModel,
            int numHeads,
            int dIntermediate,
            boolean fp16) {
        if (!blockActivationCachePoolFromEnv()) {
            return null;
        }
        Map<ArchKey, ArrayDeque<PooledBuffers>> m = BLOCK_CACHE_POOL.get();
        ArchKey key = new ArchKey(dModel, numHeads, dIntermediate, fp16);
        ArrayDeque<PooledBuffers> q = m.get(key);
        if (q == null || q.isEmpty()) {
            return null;
        }
        ArrayDeque<PooledBuffers> keep = new ArrayDeque<>();
        PooledBuffers found = null;
        while (!q.isEmpty()) {
            PooledBuffers p = q.pollFirst();
            if (found == null && p.covers(plane, headFlat, probs, ffnMid)) {
                found = p;
            } else {
                keep.addLast(p);
            }
        }
        q.addAll(keep);
        return found;
    }

    private static final class ArchKey {
        final int dModel;
        final int numHeads;
        final int dInt;
        final boolean fp16;

        ArchKey(int dModel, int numHeads, int dInt, boolean fp16) {
            this.dModel = dModel;
            this.numHeads = numHeads;
            this.dInt = dInt;
            this.fp16 = fp16;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof ArchKey)) {
                return false;
            }
            ArchKey archKey = (ArchKey) o;
            return dModel == archKey.dModel
                    && numHeads == archKey.numHeads
                    && dInt == archKey.dInt
                    && fp16 == archKey.fp16;
        }

        @Override
        public int hashCode() {
            return Objects.hash(dModel, numHeads, dInt, fp16);
        }
    }

    private static final class PooledBuffers {
        private GpuFloatBuffer xIn;
        private GpuFloatBuffer xNorm1;
        private GpuFloatBuffer attnOut;
        private GpuFloatBuffer xRes1;
        private GpuFloatBuffer xNorm2;
        private GpuFloatBuffer ffnOut;
        private GpuFloatBuffer xOut;
        private GpuFloatBuffer ffnH1;
        private GpuFloatBuffer ffnGate;
        private GpuFloatBuffer attnQHeads;
        private GpuFloatBuffer attnKHeads;
        private GpuFloatBuffer attnVHeads;
        private GpuFloatBuffer attnProbs;
        private GpuFloatBuffer attnConcat;

        private GpuHalfBuffer hxIn;
        private GpuHalfBuffer hxNorm1;
        private GpuHalfBuffer hAttnOut;
        private GpuHalfBuffer hxRes1;
        private GpuHalfBuffer hxNorm2;
        private GpuHalfBuffer hFfnOut;
        private GpuHalfBuffer hxOut;
        private GpuHalfBuffer hFfnH1;
        private GpuHalfBuffer hFfnGate;
        private GpuHalfBuffer hAttnQHeads;
        private GpuHalfBuffer hAttnKHeads;
        private GpuHalfBuffer hAttnVHeads;
        private GpuHalfBuffer hAttnProbs;
        private GpuHalfBuffer hAttnConcat;

        private GpuFloatBuffer attnLse;
        private GpuFloatBuffer attnOutHeads;

        private boolean fp16;

        boolean covers(long plane, long headFlat, long probs, long ffnMid) {
            if (fp16) {
                return hxIn != null
                        && !hxIn.isClosed()
                        && hxIn.numHalfs() >= plane
                        && hxNorm1.numHalfs() >= plane
                        && hAttnOut.numHalfs() >= plane
                        && hxRes1.numHalfs() >= plane
                        && hxNorm2.numHalfs() >= plane
                        && hFfnOut.numHalfs() >= plane
                        && hxOut.numHalfs() >= plane
                        && hAttnConcat.numHalfs() >= plane
                        && hFfnH1.numHalfs() >= ffnMid
                        && hFfnGate.numHalfs() >= ffnMid
                        && hAttnQHeads.numHalfs() >= headFlat
                        && hAttnKHeads.numHalfs() >= headFlat
                        && hAttnVHeads.numHalfs() >= headFlat
                        && (probs == 0 || (hAttnProbs != null && hAttnProbs.numHalfs() >= probs));
            }
            return xIn != null
                    && !xIn.isClosed()
                    && xIn.numFloats() >= plane
                    && xNorm1.numFloats() >= plane
                    && attnOut.numFloats() >= plane
                    && xRes1.numFloats() >= plane
                    && xNorm2.numFloats() >= plane
                    && ffnOut.numFloats() >= plane
                    && xOut.numFloats() >= plane
                    && attnConcat.numFloats() >= plane
                    && ffnH1.numFloats() >= ffnMid
                    && ffnGate.numFloats() >= ffnMid
                    && attnQHeads.numFloats() >= headFlat
                    && attnKHeads.numFloats() >= headFlat
                    && attnVHeads.numFloats() >= headFlat
                    && (probs == 0 || (attnProbs != null && attnProbs.numFloats() >= probs));
        }

        void moveInto(BlockActivationCacheDevice c) {
            c.xIn = xIn;
            c.xNorm1 = xNorm1;
            c.attnOut = attnOut;
            c.xRes1 = xRes1;
            c.xNorm2 = xNorm2;
            c.ffnOut = ffnOut;
            c.xOut = xOut;
            c.ffnH1 = ffnH1;
            c.ffnGate = ffnGate;
            c.attnQHeads = attnQHeads;
            c.attnKHeads = attnKHeads;
            c.attnVHeads = attnVHeads;
            c.attnProbs = attnProbs;
            c.attnConcat = attnConcat;
            c.hxIn = hxIn;
            c.hxNorm1 = hxNorm1;
            c.hAttnOut = hAttnOut;
            c.hxRes1 = hxRes1;
            c.hxNorm2 = hxNorm2;
            c.hFfnOut = hFfnOut;
            c.hxOut = hxOut;
            c.hFfnH1 = hFfnH1;
            c.hFfnGate = hFfnGate;
            c.hAttnQHeads = hAttnQHeads;
            c.hAttnKHeads = hAttnKHeads;
            c.hAttnVHeads = hAttnVHeads;
            c.hAttnProbs = hAttnProbs;
            c.hAttnConcat = hAttnConcat;
            c.attnLse = attnLse;
            c.attnOutHeads = attnOutHeads;
            c.storageFp16 = fp16;
            c.bumpGraphCaptureGeneration();
            xIn = null;
            xNorm1 = null;
            attnOut = null;
            xRes1 = null;
            xNorm2 = null;
            ffnOut = null;
            xOut = null;
            ffnH1 = null;
            ffnGate = null;
            attnQHeads = null;
            attnKHeads = null;
            attnVHeads = null;
            attnProbs = null;
            attnConcat = null;
            hxIn = null;
            hxNorm1 = null;
            hAttnOut = null;
            hxRes1 = null;
            hxNorm2 = null;
            hFfnOut = null;
            hxOut = null;
            hFfnH1 = null;
            hFfnGate = null;
            hAttnQHeads = null;
            hAttnKHeads = null;
            hAttnVHeads = null;
            hAttnProbs = null;
            hAttnConcat = null;
            attnLse = null;
            attnOutHeads = null;
        }

        static PooledBuffers stealFrom(BlockActivationCacheDevice c) {
            PooledBuffers p = new PooledBuffers();
            p.fp16 = c.storageFp16;
            p.xIn = c.xIn;
            p.xNorm1 = c.xNorm1;
            p.attnOut = c.attnOut;
            p.xRes1 = c.xRes1;
            p.xNorm2 = c.xNorm2;
            p.ffnOut = c.ffnOut;
            p.xOut = c.xOut;
            p.ffnH1 = c.ffnH1;
            p.ffnGate = c.ffnGate;
            p.attnQHeads = c.attnQHeads;
            p.attnKHeads = c.attnKHeads;
            p.attnVHeads = c.attnVHeads;
            p.attnProbs = c.attnProbs;
            p.attnConcat = c.attnConcat;
            p.hxIn = c.hxIn;
            p.hxNorm1 = c.hxNorm1;
            p.hAttnOut = c.hAttnOut;
            p.hxRes1 = c.hxRes1;
            p.hxNorm2 = c.hxNorm2;
            p.hFfnOut = c.hFfnOut;
            p.hxOut = c.hxOut;
            p.hFfnH1 = c.hFfnH1;
            p.hFfnGate = c.hFfnGate;
            p.hAttnQHeads = c.hAttnQHeads;
            p.hAttnKHeads = c.hAttnKHeads;
            p.hAttnVHeads = c.hAttnVHeads;
            p.hAttnProbs = c.hAttnProbs;
            p.hAttnConcat = c.hAttnConcat;
            p.attnLse = c.attnLse;
            p.attnOutHeads = c.attnOutHeads;
            c.clearAllSlotRefs();
            c.cachedBatch = -1;
            c.cachedSeqLen = -1;
            c.cachedDModel = -1;
            c.cachedNumHeads = -1;
            c.cachedDInt = -1;
            c.storageFp16 = false;
            c.bumpGraphCaptureGeneration();
            return p;
        }

        void closeAll() {
            xIn = closeBufF(xIn);
            xNorm1 = closeBufF(xNorm1);
            attnOut = closeBufF(attnOut);
            xRes1 = closeBufF(xRes1);
            xNorm2 = closeBufF(xNorm2);
            ffnOut = closeBufF(ffnOut);
            xOut = closeBufF(xOut);
            ffnH1 = closeBufF(ffnH1);
            ffnGate = closeBufF(ffnGate);
            attnQHeads = closeBufF(attnQHeads);
            attnKHeads = closeBufF(attnKHeads);
            attnVHeads = closeBufF(attnVHeads);
            attnProbs = closeBufF(attnProbs);
            attnConcat = closeBufF(attnConcat);
            hxIn = closeBufH(hxIn);
            hxNorm1 = closeBufH(hxNorm1);
            hAttnOut = closeBufH(hAttnOut);
            hxRes1 = closeBufH(hxRes1);
            hxNorm2 = closeBufH(hxNorm2);
            hFfnOut = closeBufH(hFfnOut);
            hxOut = closeBufH(hxOut);
            hFfnH1 = closeBufH(hFfnH1);
            hFfnGate = closeBufH(hFfnGate);
            hAttnQHeads = closeBufH(hAttnQHeads);
            hAttnKHeads = closeBufH(hAttnKHeads);
            hAttnVHeads = closeBufH(hAttnVHeads);
            hAttnProbs = closeBufH(hAttnProbs);
            hAttnConcat = closeBufH(hAttnConcat);
            attnLse = closeBufF(attnLse);
            attnOutHeads = closeBufF(attnOutHeads);
        }

        private static GpuFloatBuffer closeBufF(GpuFloatBuffer b) {
            if (b != null && !b.isClosed()) {
                b.close();
            }
            return null;
        }

        private static GpuHalfBuffer closeBufH(GpuHalfBuffer b) {
            if (b != null && !b.isClosed()) {
                b.close();
            }
            return null;
        }
    }

    public boolean isAllocated() {
        return (storageFp16 && hxIn != null && !hxIn.isClosed()) || (!storageFp16 && xIn != null && !xIn.isClosed());
    }

    /** После {@link #ensure}; до вызова — {@code -1}. */
    public int batch() {
        return cachedBatch;
    }

    public int seqLen() {
        return cachedSeqLen;
    }

    public int dModel() {
        return cachedDModel;
    }

    public int numHeads() {
        return cachedNumHeads;
    }

    public int dIntermediate() {
        return cachedDInt;
    }

    private void clearAllSlotRefs() {
        xIn = null;
        xNorm1 = null;
        attnOut = null;
        xRes1 = null;
        xNorm2 = null;
        ffnOut = null;
        xOut = null;
        ffnH1 = null;
        ffnGate = null;
        attnQHeads = null;
        attnKHeads = null;
        attnVHeads = null;
        attnProbs = null;
        attnConcat = null;
        attnLse = null;
        attnOutHeads = null;
        hxIn = null;
        hxNorm1 = null;
        hAttnOut = null;
        hxRes1 = null;
        hxNorm2 = null;
        hFfnOut = null;
        hxOut = null;
        hFfnH1 = null;
        hFfnGate = null;
        hAttnQHeads = null;
        hAttnKHeads = null;
        hAttnVHeads = null;
        hAttnProbs = null;
        hAttnConcat = null;
    }

    private void clearFp32Slots() {
        xIn = closeBuf(xIn);
        xNorm1 = closeBuf(xNorm1);
        attnOut = closeBuf(attnOut);
        xRes1 = closeBuf(xRes1);
        xNorm2 = closeBuf(xNorm2);
        ffnOut = closeBuf(ffnOut);
        xOut = closeBuf(xOut);
        ffnH1 = closeBuf(ffnH1);
        ffnGate = closeBuf(ffnGate);
        attnQHeads = closeBuf(attnQHeads);
        attnKHeads = closeBuf(attnKHeads);
        attnVHeads = closeBuf(attnVHeads);
        attnProbs = closeBuf(attnProbs);
        attnConcat = closeBuf(attnConcat);
        attnLse = closeBuf(attnLse);
        attnOutHeads = closeBuf(attnOutHeads);
    }

    private void clearFp16Slots() {
        hxIn = closeHalf(hxIn);
        hxNorm1 = closeHalf(hxNorm1);
        hAttnOut = closeHalf(hAttnOut);
        hxRes1 = closeHalf(hxRes1);
        hxNorm2 = closeHalf(hxNorm2);
        hFfnOut = closeHalf(hFfnOut);
        hxOut = closeHalf(hxOut);
        hFfnH1 = closeHalf(hFfnH1);
        hFfnGate = closeHalf(hFfnGate);
        hAttnQHeads = closeHalf(hAttnQHeads);
        hAttnKHeads = closeHalf(hAttnKHeads);
        hAttnVHeads = closeHalf(hAttnVHeads);
        hAttnProbs = closeHalf(hAttnProbs);
        hAttnConcat = closeHalf(hAttnConcat);
    }

    private void closeBuffersHard() {
        clearFp32Slots();
        clearFp16Slots();
        if (attnProbsFloatStage != null) {
            attnProbsFloatStage.close();
            attnProbsFloatStage = null;
            attnProbsFloatStageCap = 0L;
        }
        if (halfSlotFloatStage != null) {
            halfSlotFloatStage.close();
            halfSlotFloatStage = null;
            halfSlotFloatStageCap = 0L;
        }
        cachedBatch = -1;
        cachedSeqLen = -1;
        cachedDModel = -1;
        cachedNumHeads = -1;
        cachedDInt = -1;
        storageFp16 = false;
        bumpGraphCaptureGeneration();
    }

    private static GpuFloatBuffer closeBuf(GpuFloatBuffer b) {
        if (b != null && !b.isClosed()) {
            b.close();
        }
        return null;
    }

    private static GpuHalfBuffer closeHalf(GpuHalfBuffer b) {
        if (b != null && !b.isClosed()) {
            b.close();
        }
        return null;
    }

    @Override
    public String toString() {
        if (!isAllocated()) {
            return "BlockActivationCacheDevice{unallocated}";
        }
        return String.format(
                "BlockActivationCacheDevice{batch=%d, seqLen=%d, dModel=%d, numHeads=%d, dIntermediate=%d, fp16=%s}",
                cachedBatch, cachedSeqLen, cachedDModel, cachedNumHeads, cachedDInt, storageFp16);
    }

    @Override
    public void close() {
        closeBuffersHard();
    }
}
