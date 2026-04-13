package com.veles.llm.jgpt;

import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.cuda.GpuTensor;
import com.veles.llm.jgpt.cuda.TensorCudaLibrary;
import com.veles.llm.jgpt.training.LLMTrainer;
import java.nio.ByteBuffer;
import java.util.Map;

/**
 * GPU-ускоренные операции через JNI + CUDA/cuBLAS.
 *
 * <p>Политика: обучение и полный forward требуют CUDA; см. {@link #requireCuda(String)} в точках входа. При
 * первой загрузке класса, если натив не поднялся или {@code initGPU()} вернул {@code false}, JVM падает с
 * {@link ExceptionInInitializerError}, если не задан явный override {@code -Djgpt.allow.no.gpu=true} или
 * {@code JGPT_ALLOW_NO_GPU=1} (для CI/JVM без GPU). Методы {@code shouldUseGpu*} отражают доступность GPU и
 * положительные размеры.
 *
 * <p><b>Потокобезопасность:</b> только read-only проверки ({@link #isGpuAvailable()}, пороги) безопасны с любых потоков.
 * JNI-вызовы с записью в буферы вызывающего кода и мутации глобального CUDA-контекста требуют внешней координации:
 * один поток обучения/инференса на процесс или явная сериализация. Буфер mean-loss для CE хранится в {@code ThreadLocal}
 * и сбрасывается в {@code 0f} перед JNI (см. {@link TensorOpsGpuCrossEntropyHost}).
 */
public final class TensorOpsGPU {
    /** Совпадает с {@code cudaErrorMemoryAllocation} в CUDA Runtime (OOM при graph launch и т.п.). */
    public static final int CUDA_ERROR_MEMORY_ALLOCATION = 2;

    /**
     * Закрывает thread-local scratch для путей host→GPU ({@code splitHeads}/{@code concatHeads}, SDPA с {@code float[]}
     * на хосте). Без этого при {@link ThreadLocal#remove()} или завершении потока буферы освобождает только отложенный
     * путь ({@link com.veles.llm.jgpt.GpuFloatBuffer#drainLeaked()} после GC) — в stderr уходит предупреждение
     * «незакрытого буфера».
     *
     * <p>Вызывается из {@link com.veles.llm.jgpt.ops.GpuWorkspaceCleanup#releaseAllGpuWorkspacesThreadLocal()}.
     */
    public static void releaseHostPathScratchThreadLocal() {
        TensorOpsGpuHostPathScratch.releaseThreadLocal();
    }

    /**
     * Историческое значение порога GEMM (m×k×n); маршрутизация больше не от него зависит — при доступной GPU GEMM
     * идёт на GPU для любых положительных размеров. Оставлено для совместимости/логов.
     */
    public static final long GPU_MATMUL_MIN_OPS = 16_777_216L;

    /**
     * Исторический порог по числу элементов для поэлементных op; маршрутизация не использует (см. {@link
     * #shouldUseGpuElementwise}).
     */
    public static final int GPU_ELEMENTWISE_MIN = 65536;

    /**
     * Исторический порог для оптимизатора; fused Adam по всем параметрам не от него зависит (см. {@link
     * #shouldUseGpuOptimizer}).
     */
    public static final int GPU_OPTIMIZER_MIN = 1_000_000;

    /** Кэш: проверка GPU один раз при загрузке класса. */
    private static final boolean GPU_AVAILABLE;

    /**
     * GEMM через FP16 Tensor Cores (входы конвертируются float→half на GPU, выход FP32).
     * Включается: {@code JGPT_FP16_MATMUL=1} или {@code -Djgpt.fp16.matmul=true}.
     */
    private static final boolean FP16_MATMUL;

    /**
     * Epsilon для RMSNorm во всех горячих путях. Переопределение: {@code JGPT_RMSNORM_EPS} или
     * {@code -Djgpt.rmsnorm.eps}. При включённом FP16-matmul дефолт {@code 1e-5f} (устойчивее backward при малой
     * дисперсии); иначе {@code 1e-6f}.
     */
    private static final float RMSNORM_EPS;

    private static final String GPU_NAME;
    private static final long GPU_MEMORY_MB;

    /** Устар.: раньше ограничивал CE на GPU; {@link #shouldUseGpuCrossEntropy} больше не использует порог. */
    private static final int CE_GPU_MIN_ELEMENTS;

    /**
     * FlashAttention-2 forward/backward вместо классического O(S²) softmax.
     * Включается: {@code JGPT_FLASH_ATTENTION=1}.
     * Требует d_head == 16 (kFaDh в CUDA).  При несоответствии автоматически отключается.
     */
    public static final boolean FLASH_ATTENTION;

    static {
        boolean available = false;
        String name = "Unknown";
        long memory = 0L;

        try {
            TensorCudaLibrary.load();

            available = initGPU();
            if (available) {
                name = getGPUName();
                memory = getGPUMemory();
                System.out.println("[TensorOpsGPU] GPU инициализирован: " + name + ", " + memory + " МБ");
            } else {
                System.err.println("[TensorOpsGPU] CUDA или initGPU недоступны; без GPU работа не поддерживается.");
            }
        } catch (UnsatisfiedLinkError e) {
            System.err.println("[TensorOpsGPU] Не удалось загрузить нативную библиотеку: " + e.getMessage());
        }

        GPU_AVAILABLE = available;
        GPU_NAME = name;
        GPU_MEMORY_MB = memory;

        FP16_MATMUL = TensorOpsGpuInit.resolveFp16Matmul(available);
        if (FP16_MATMUL) {
            System.out.println("[TensorOpsGPU] FP16 matmul (GemmEx): включён");
        }

        RMSNORM_EPS = TensorOpsGpuInit.resolveRmsNormEps(FP16_MATMUL);

        CE_GPU_MIN_ELEMENTS = TensorOpsGpuInit.resolveCeGpuMinElements();

        FLASH_ATTENTION = TensorOpsGpuInit.resolveFlashAttention(available);
        if (FLASH_ATTENTION) {
            System.out.println("[TensorOpsGPU] FlashAttention-2: включён (d_head=16 обязателен)");
        }

        if (!available && !TensorOpsGpuInit.allowNoGpuOverride()) {
            throw new ExceptionInInitializerError(
                    new IllegalStateException(
                            "CUDA / libjgpt_cuda недоступны: обучение и TensorOps требуют GPU. "
                                    + "Для JVM без GPU (CI, юнит-тесты) задайте -Djgpt.allow.no.gpu=true или "
                                    + "JGPT_ALLOW_NO_GPU=1 (Maven Surefire в проекте передаёт -Djgpt.allow.no.gpu=true)."));
        }
    }

    // ========== NATIVE МЕТОДЫ ==========

    /** Инициализация GPU (один раз из static-блока). */
    private static native boolean initGPU();

    private static native String getGPUName();

    private static native long getGPUMemory();

    private static native void synchronizeStream0();

    /**
     * Освобождает thread-local cuBLAS/кэши matmul/CE текущего потока. Вызывать при остановке воркера пула,
     * выполнявшего JNI на GPU, иначе возможна утечка дескрипторов до завершения процесса.
     */
    private static native void cleanupCudaThreadResources0();

    /** Ждёт завершения работы на <b>всех</b> потоках устройства (для тестов / диагностики гонок). */
    private static native void synchronizeDevice0();

    /**
     * Полная синхронизация устройства, {@code cudaDeviceGraphMemTrim} (CUDA 12+) и trim default/device memory pool —
     * после graph-OOM перед eager, чтобы снизить ложные OOM на {@code cudaMallocAsync}.
     */
    private static native void cudaTrimDeviceMemoryPoolsBestEffort0();

    /** Текущее использование VRAM (bytes): {@code total − free} из {@code cudaMemGetInfo}. */
    private static native long getGpuMemoryAllocated0();

    /** Объём VRAM устройства (bytes): {@code total} из {@code cudaMemGetInfo} (не «reserved» PyTorch). */
    private static native long getGpuMemoryReserved0();

    /** D2D на stream: {@code dstHalf[i] = half_rn(srcFloat[i])}, длина {@code n}. */
    static native void nativeConvertFloatDeviceToHalfDevice(long srcFloatDevice, long dstHalfDevice, int n);

    /** D2D на stream: {@code dstFloat[i] = half2float(srcHalf[i])}, длина {@code n}. */
    static native void nativeConvertHalfDeviceToFloatDevice(long srcHalfDevice, long dstFloatDevice, int n);

    /** Matmul: C = A × B */
    public static native void matmulGPU(float[] A, float[] B, float[] C, int M, int K, int N);

    /** Matmul FP16 Tensor Cores: те же буферы float, внутри конвертация в half. */
    static native void matmulGPUFp16(float[] A, float[] B, float[] C, int M, int K, int N);

    /**
     * Matmul по указателям на уже загруженные float-буферы на GPU (см. {@link GpuFloatBuffer}).
     * Формы: A [M×K], B [K×N], C [M×N] в row-major.
     */
    public static native void matmulGPUDevice(long dA, long dB, long dC, int M, int K, int N);

    /**
     * Device GEMM с опциональным транспонированием входов. {@code beta} — множитель существующего {@code C}
     * (cuBLAS: {@code C = alpha * op(A)*op(B) + beta * C}).
     */
    public static native void matmulGPUDeviceEx(
            long dA,
            long dB,
            long dC,
            int M,
            int K,
            int N,
            boolean transposeA,
            boolean transposeB,
            float beta);

    /**
     * Три независимых GEMM (как три раза {@link #matmulGpuDeviceEx} без транспонирования) с общим {@code xNorm [M×K]};
     * веса {@code Wq, Wk, Wv [K×N]}; выходы {@code Q, K, V [M×N]}. Реализация: D2D упаковка весов и один
     * {@code cublasSgemmStridedBatched} (batch=3, нулевой stride второго операнда для повторного X) вместо трёх отдельных
     * {@code cublasSgemm}.
     */
    static native void matmulGpuDeviceQkvProjections0(
            long dXnorm, long dWq, long dWk, long dWv, long dQ, long dK, long dV, int M, int K, int N);

    /**
     * Два GEMM SwiGLU после второй RMSNorm: общий {@code xNorm [M×K]}, веса {@code W1, W3 [K×N]} ({@code N}=dIntermediate),
     * выходы {@code h1, gate [M×N]}. Один {@code cublasSgemmStridedBatched} (batch=2) вместо двух {@code cublasSgemm}.
     */
    static native void matmulGpuDeviceFfnW1W3Projections0(
            long dXnorm, long dW1, long dW3, long dH1, long dGate, int M, int K, int N);

    /**
     * RMSNorm на device; {@code useFp16} — ядро с округлением x/γ до FP16 (см. {@link #useFp16Matmul()}).
     */
    public static native void rmsNormGPUDevice(
            long dX, long dGamma, float eps, long dOut, int outer, int lastDim, boolean useFp16);

    public static native void sigmoidGPUDevice(long dSrc, long dDst, int n);

    public static native void multiplyGPUDevice(long dA, long dB, long dC, int n);

    /**
     * {@code C = ReLU(A × B + bias)}; bias длины {@code N}, без промежуточного тензора на CPU.
     */
    public static native void matmulAddReluGPU(float[] A, float[] B, float[] bias, float[] C, int M, int K, int N);

    static native void matmulAddReluGPUFp16(float[] A, float[] B, float[] bias, float[] C, int M, int K, int N);

    /**
     * Батчевый matmul: для каждого {@code i ∈ [0, batch)} независимо {@code C[i] = A[i] × B[i]}.
     * {@code A} — плотный {@code [batch][M][K]}, {@code B} — {@code [batch][K][N]}, {@code C} — {@code [batch][M][N]}.
     */
    public static native void matmulBatchedGPU(float[] A, float[] B, float[] C, int M, int K, int N, int batchCount);

    static native void matmulBatchedGPUFp16(
            float[] A, float[] B, float[] C, int M, int K, int N, int batchCount);

    /** Add: C = A + B */
    public static native void addGPU(float[] a, float[] b, float[] c, int n);

    /** Subtract: C = A − B (без копии операндов на стороне Java). */
    public static native void subtractGPU(float[] a, float[] b, float[] c, int n);

    /** ReLU: b = max(0, a) */
    public static native void reluGPU(float[] a, float[] b, int n);

    /**
     * Softmax по последней оси 3D. При {@code useFp16Softmax} — дифф к max округляется до FP16 перед exp,
     * как в {@link #crossEntropySoftmaxGradLossGpu} (согласовано с fused CE backward).
     */
    public static native void softmaxLastDimGPU(
            float[] src, float[] dst, int batch, int mid, int inner, boolean useFp16Softmax);

    /**
     * Fused CE + градиент по logits: (softmax − one_hot) × {@code gradScaleOverTotalTokens}.
     * При {@link #useFp16Matmul()} — softmax/exp с округлением диффа до FP16 (см. CUDA); иначе полностью FP32.
     * Масштаб из {@link LLMTrainer} (в т.ч. loss scale) передаётся в {@code gradScaleOverTotalTokens} — без
     * дополнительного множителя в ядре.
     *
     * @return средний CE по валидным токенам (как {@link LLMTrainer} на CPU)
     */
    public static float crossEntropySoftmaxGradLossGpu(
            float[] logits,
            float[] targets,
            float[] gradOut,
            int batch,
            int seqLen,
            int vocab,
            float gradScaleOverTotalTokens) {
        return TensorOpsGpuCrossEntropyHost.crossEntropySoftmaxGradLossGpu(
                logits, targets, gradOut, batch, seqLen, vocab, gradScaleOverTotalTokens);
    }

    /**
     * То же, что {@link #crossEntropySoftmaxGradLossGpu}, с явным выбором FP16-softmax (для тестов и отладки).
     */
    public static float crossEntropySoftmaxGradLossGpuEx(
            float[] logits,
            float[] targets,
            float[] gradOut,
            int batch,
            int seqLen,
            int vocab,
            float gradScaleOverTotalTokens,
            boolean useFp16Softmax) {
        return TensorOpsGpuCrossEntropyHost.crossEntropySoftmaxGradLossGpuEx(
                logits,
                targets,
                gradOut,
                batch,
                seqLen,
                vocab,
                gradScaleOverTotalTokens,
                useFp16Softmax);
    }

    /**
     * Как {@link #crossEntropySoftmaxGradLossGpuEx}, но логиты и таргеты читаются из direct {@link ByteBuffer}
     * (pinned/direct host) без {@code float[]}-зеркала. Градиент по логитам пишется в heap {@code gradOut}.
     */
    public static float crossEntropySoftmaxGradLossGpuDirectEx(
            ByteBuffer logits,
            long logitsByteOffset,
            ByteBuffer targets,
            long targetsByteOffset,
            float[] gradOut,
            int batch,
            int seqLen,
            int vocab,
            float gradScaleOverTotalTokens,
            boolean useFp16Softmax) {
        return TensorOpsGpuCrossEntropyHost.crossEntropySoftmaxGradLossGpuDirectEx(
                logits,
                logitsByteOffset,
                targets,
                targetsByteOffset,
                gradOut,
                batch,
                seqLen,
                vocab,
                gradScaleOverTotalTokens,
                useFp16Softmax);
    }

    /**
     * H2D: {@code nTokens} значений float (id токена) из direct буфера → int32 на GPU ({@link GpuIntBuffer}),
     * без промежуточного {@code int[]} на хосте.
     */
    public static void uploadTokenIdsFromFloatDirectToGpuInt(
            ByteBuffer hostFloatsAsTokenIds, long byteOffset, int nTokens, GpuIntBuffer dst) {
        TensorOpsGpuHostUpload.uploadTokenIdsFromFloatDirectToGpuInt(
                hostFloatsAsTokenIds, byteOffset, nTokens, dst);
    }

    static native void crossEntropySoftmaxGradLossGPU(
            float[] logits,
            float[] targets,
            float[] gradOut,
            int batch,
            int seqLen,
            int vocab,
            float gradScaleOverTotalTokens,
            float[] lossOut,
            boolean useFp16Softmax);

    static native void crossEntropySoftmaxGradLossGPUDirect(
            ByteBuffer logits,
            long logitsByteOffset,
            ByteBuffer targets,
            long targetsByteOffset,
            float[] gradOut,
            int batch,
            int seqLen,
            int vocab,
            float gradScaleOverTotalTokens,
            float[] lossOut,
            boolean useFp16Softmax);

    static native void copyHostFloatBufferToGpuIntTokenIds(
            ByteBuffer hostFloats, long byteOffset, int nTokens, long deviceIntPtr);

    public static native void layerNormGPU(
            float[] x, float[] gamma, float[] beta, float[] out, int outer, int lastDim, float eps);

    public static native void rmsNormGPU(
            float[] x, float[] gamma, float[] out, int outer, int lastDim, float eps, boolean useFp16);

    public static native void geluGPU(float[] src, float[] dst, int n);

    public static native void sigmoidGPU(float[] src, float[] dst, int n);

    public static native void multiplyGPU(float[] a, float[] b, float[] c, int n);

    public static native void multiplyScalarGPU(float[] a, float[] b, int n, float scalar);

    public static native void applyCausalMask3DGPU(float[] scores, float[] mask, float[] out, int batch, int seqLen);

    public static native void transpose2DLastGPU(float[] src, float[] dst, int d0, int d1, int d2);

    /**
     * [batch, seq, d_model] → [batch, heads, seq, d_head] с временными буферами на GPU; см. {@link
     * com.veles.llm.jgpt.ops.TensorOps#splitHeads}.
     */
    public static void splitHeadsFromHost(float[] src, float[] dst, int batch, int seqLen, int dModel, int numHeads) {
        TensorOpsGpuHostPathScratch.splitHeadsFromHost(src, dst, batch, seqLen, dModel, numHeads);
    }

    /**
     * Обратно к {@link #splitHeadsFromHost}; вход [batch, heads, seq, d_head], выход [batch, seq, d_model].
     */
    public static void concatHeadsFromHost(float[] src, float[] dst, int batch, int numHeads, int seqLen, int dHead) {
        TensorOpsGpuHostPathScratch.concatHeadsFromHost(src, dst, batch, numHeads, seqLen, dHead);
    }

    /**
     * positions может быть {@code null}: тогда используется {@code i + posBaseOffset} по токену {@code i}
     * (при {@code posBaseOffset == 0} — 0…seqLen−1).
     */
    public static native void applyRoPE4DGPU(
            float[] src,
            float[] dst,
            int batch,
            int numHeads,
            int seqLen,
            int dHead,
            int[] positions,
            int posBaseOffset);

    /**
     * RoPE на уже загруженных float-буферах; in-place допустимо при {@code dSrc == dDst}.
     * При {@code positions == null} — {@code i + posBaseOffset}.
     */
    public static native void applyRoPE4DGPUDevice(
            long dSrc,
            long dDst,
            int batch,
            int numHeads,
            int seqLen,
            int dHead,
            int[] positions,
            int posBaseOffset);

    /** Обратный RoPE: {@code gradX += R^T gradY}; positions может быть {@code null}. */
    public static native void applyRoPEBackward4DGPU(
            float[] gradY,
            float[] gradX,
            int batch,
            int numHeads,
            int seqLen,
            int dHead,
            int[] positions,
            int posBaseOffset);

    public static native void softmaxLastDimBackward3DGPU(
            float[] gOut, float[] probs, float[] gIn, int batch, int mid, int inner);

    public static native void layerNormBackwardGPU(
            float[] gOut,
            float[] x,
            float[] gamma,
            float eps,
            float[] gX,
            float[] gGamma,
            float[] gBeta,
            int outer,
            int lastDim);

    public static native void rmsNormBackwardGPU(
            float[] gOut, float[] x, float[] gamma, float eps, float[] gX, float[] gGamma, int outer, int lastDim);

    public static native void multiplyBackwardGPU(float[] gOut, float[] a, float[] b, float[] gA, float[] gB, int n);

    public static native void accumulateAddGPU(float[] acc, float[] delta, int n);

    /** {@code acc[i] += delta[i] * scale} (для {@code subtractBackward}: {@code scale = -1}). */
    public static native void accumulateScaledAddGPU(float[] acc, float[] delta, float scale, int n);

    public static native void multiplyBackwardGPUDevice(long dGOut, long dA, long dB, long dGA, long dGB, int n);

    public static native void sigmoidBackwardGPUDevice(long dGOut, long dInp, long dGIn, int n);

    public static native void accumulateAddGPUDevice(long dAcc, long dDelta, int n);

    /** Device-only: {@code dSrc[i] *= scalar}. */
    public static native void scaleInPlaceGPUDevice(long dSrc, int n, float scalar);

    /** Device-only: sum of squares via cublasSdot (tree reduction, high precision). */
    public static native double sumSquaresGPUDevice(long dSrc, int n);

    /**
     * Сумма {@code sum_i ||x_i||²} по нескольким device-векторам; один JNI и одна синхронизация потока в конце.
     */
    public static native double sumSquaresGPUDeviceFused(long[] dPtrs, int[] lens, int nBufs);

    /** Device-only: returns true if any element is NaN or Inf. */
    public static native boolean anyNonFiniteGPUDevice(long dSrc, int n);

    /**
     * Фьюзированная проверка NaN/Inf по нескольким device-буферам; один JNI-вызов и одна
     * синхронизация стрима в конце. Быстрее чем N вызовов {@link #anyNonFiniteGPUDevice}.
     *
     * @param dPtrs  device-указатели (как {@code long})
     * @param lens   число float в каждом буфере
     * @param nBufs  число буферов
     * @return {@code true}, если хотя бы один элемент — NaN или Inf
     */
    public static native boolean anyNonFiniteGPUDeviceMulti(long[] dPtrs, int[] lens, int nBufs);

    /**
     * CE + softmax + grad on device pointers; targets H2D (small O(B×S)), loss D2H scalar.
     * Logits are modified in-place (softmax applied), grad written to dGrad.
     *
     * @return mean CE loss over valid tokens
     */
    public static native float crossEntropySoftmaxGradLossGPUDevice(
            long dLogits, float[] hTargets, long dGrad,
            int batch, int seqLen, int vocab, float gradScale,
            boolean useFp16);

    /**
     * Как {@link #crossEntropySoftmaxGradLossGPUDevice}, но targets уже на device (int32 на строку,
     * индекс токена как в float-CE).
     */
    public static native float crossEntropySoftmaxGradLossGPUDeviceTargetsDevice(
            long dLogits,
            long dTargetsInt,
            long dGrad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean useFp16);

    /**
     * Как {@link #crossEntropySoftmaxGradLossGPUDeviceTargetsDevice}, но без синхронизации стрима: D2H loss/valid
     * в pinned host через {@code cudaMemcpyAsync}. Перед чтением скаляра — {@link #synchronizeStream()}, затем
     * {@link #crossEntropySoftmaxGradLossGPUDeviceReadPendingFromHost()}.
     *
     * <p>Совместимо с FP16 matmul при <b>одном</b> потоке обучения на общем stream: буферы суммы loss / valid и
     * (для host-float targets) pinned targets — thread-local в native, не разделяются с cuBLAS дескриптором.
     * Несколько потоков на одном CUDA stream не поддержаны.
     */
    public static native void crossEntropySoftmaxGradLossGPUDeviceTargetsDeviceAsync(
            long dLogits,
            long dTargetsInt,
            long dGrad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean useFp16);

    /**
     * Async CE с float targets на хосте (как {@link #crossEntropySoftmaxGradLossGPUDevice}); pinned-копия + H2D
     * async, затем тот же {@link #crossEntropySoftmaxGradLossGPUDeviceReadPendingFromHost()} после sync.
     */
    public static native void crossEntropySoftmaxGradLossGPUDeviceHostFloatTargetsAsync(
            long dLogits,
            float[] hTargets,
            long dGrad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean useFp16);

    /** Чтение среднего CE после {@link #synchronizeStream()} (async-путь). Без sync — гонка. */
    public static native float crossEntropySoftmaxGradLossGPUDeviceReadPendingFromHost();

    /** Один JNI: RMSNorm по строкам + LM-head matmul {@code logits = norm(x) × W} (как {@link #matmulGPUDevice}). */
    public static native void rmsNormMatmulLmHeadGPUDevice(
            long dX,
            long dGamma,
            float eps,
            long dNormOut,
            long dW,
            long dLogits,
            int rows,
            int dModel,
            int vocab,
            boolean useFp16Rms);

    /**
     * Один JNI: второй RMSNorm по строкам + strided batched {@code norm·W1}, {@code norm·W3} (как {@link
     * #matmulGpuDeviceFfnW1W3Projections} после {@link #rmsNormGpuDevice}).
     */
    public static native void rmsNormMatmulFfnW1W3GPUDevice(
            long dX,
            long dGamma,
            float eps,
            long dNormOut,
            long dW1,
            long dW3,
            long dH1,
            long dGate,
            int rows,
            int dModel,
            int dIntermediate,
            boolean useFp16Rms);

    /** H2D delta[off..off+len] into temp device buf, then accumulate_add_kernel into dAcc. */
    public static native void accumulateAddFromHostGPUDevice(long dAcc, float[] hDelta, int off, int len);

    /** In-place: {@code C[i,j] += bias[j]}; {@code C} — M×N row-major, {@code bias} — N элементов. */
    public static native void addBiasBroadcastGPUDevice(long dC, long dBias, int M, int N);

    /**
     * {@code dst[j] = beta * dst[j] + sum_i src[i*N + j]}; {@code src} — M×N row-major, {@code dst} — длина N. При
     * {@code beta == 0} эквивалентно перезаписи суммой столбца.
     */
    public static native void sumColumnsGPUDevice(long dSrc, long dDst, int M, int N, float beta);

    public static native void splitHeadsGPUDevice(long dSrc, long dDst, int batch, int seqLen, int dModel, int numHeads);

    public static native void concatHeadsGPUDevice(long dSrc, long dDst, int batch, int numHeads, int seqLen, int dHead);

    /**
     * Копия среза батча из 4D голов (row-major [batch, H, seq, dHead]) в плоский кэш
     * {@code head * maxSeqLen * dHead + pos * dHead}.
     */
    public static native void copyKvHeads4dToCacheGPUDevice(
            long dSrcHeads4d,
            long dDstCache,
            int numHeads,
            int seqLen,
            int maxSeqLen,
            int dHead,
            int batchIdx,
            int batch);

    /** При обучении на device: {@code positions == null} → {@code i + posBaseOffset}. */
    public static native void applyRoPEBackward4DGPUDevice(
            long dGradY, long dGradX, int batch, int numHeads, int seqLen, int dHead, int posBaseOffset);

    public static native void scaledDotProductAttentionBackwardGPUDevice(
            long dGradOut,
            long dProbs,
            long dQ,
            long dK,
            long dV,
            long dGradQ,
            long dGradK,
            long dGradV,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            long dMask,
            boolean useFp16Softmax);

    public static native void rmsNormBackwardGPUDevice(
            long dGOut, long dX, long dGamma, float eps, long dGX, long dGGamma, int outer, int lastDim);

    public static native void geluBackwardGPU(float[] gOut, float[] inp, float[] gIn, int n);

    public static native void sigmoidBackwardGPU(float[] gOut, float[] inp, float[] gIn, int n);

    public static native void reluBackwardGPU(float[] gOut, float[] inp, float[] gIn, int n);

    /**
     * Token embedding forward: {@code out[b,s,:] = weights[token[b,s], :]}; буферы row-major,
     * {@code weights} — [vocabSize, dModel].
     */
    public static native void embeddingTokenForwardGPU(
            float[] tokens,
            float[] weights,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            boolean useFp16Gather);

    /**
     * То же, что {@link #embeddingTokenForwardGPU}, но токены — direct {@link java.nio.ByteBuffer} (float32
     * row-major), без {@code GetFloatArrayElements} для индексов.
     */
    public static native void embeddingTokenForwardGPUDirect(
            java.nio.ByteBuffer directTokens,
            float[] weights,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            boolean useFp16Gather);

    /**
     * Gather эмбеддингов: веса уже на device ({@code weightsDevicePtr}); токены — хостовый {@code float[]}.
     */
    public static native void embeddingTokenForwardGPUDeviceWeights(
            float[] tokens,
            long weightsDevicePtr,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            boolean useFp16Gather);

    /**
     * То же, что {@link #embeddingTokenForwardGPUDeviceWeights}, токены — direct {@link java.nio.ByteBuffer}.
     */
    public static native void embeddingTokenForwardGPUDirectDeviceWeights(
            java.nio.ByteBuffer directTokens,
            long weightsDevicePtr,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            boolean useFp16Gather);

    /**
     * Как {@link #embeddingTokenForwardGPUDeviceWeights}, но результат пишется в уже выделенный device-буфер
     * ({@code outDevicePtr}); без D2H в {@code float[]}.
     */
    public static native void embeddingTokenForwardGPUDeviceWeightsToDevice(
            float[] tokens,
            long weightsDevicePtr,
            long outDevicePtr,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            boolean useFp16Gather);

    /** То же с direct-токенами (см. {@link #embeddingTokenForwardGPUDirectDeviceWeights}). */
    public static native void embeddingTokenForwardGPUDirectDeviceWeightsToDevice(
            java.nio.ByteBuffer directTokens,
            long weightsDevicePtr,
            long outDevicePtr,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            boolean useFp16Gather);

    /** Scatter-add: gradWeights[token[b,s], :] += gradOut[b,s,:]. */
    public static native void embeddingTokenBackwardGPU(
            float[] tokens, float[] gradOut, float[] gradWeights, int batch, int seqLen, int dModel, int vocabSize);

    /**
     * То же, что {@link #embeddingTokenBackwardGPU}, но накопление в уже выделенный device-буфер градиента
     * ({@code gradWeightsDevicePtr}); без {@code cudaMalloc} под всю таблицу весов.
     */
    public static native void embeddingTokenBackwardGPUDeviceGradWeights(
            float[] tokens,
            float[] gradOut,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            long gradWeightsDevicePtr);

    /** gradPos[s, :] += sum_b gradCombined[b,s,:] для s in [0, seqLen). */
    public static native void embeddingPositionBackwardGPU(
            float[] gradCombined, float[] gradWeights, int batch, int seqLen, int dModel);

    /**
     * То же, что {@link #embeddingPositionBackwardGPU}, но накопление в device-буфер градиента таблицы позиций
     * ({@code gradWeightsDevicePtr}).
     */
    public static native void embeddingPositionBackwardGPUDeviceGradWeights(
            float[] gradCombined, int batch, int seqLen, int dModel, long gradWeightsDevicePtr);

    /** Как {@link #embeddingTokenBackwardGPUDeviceGradWeights}, но gradOut уже на device ({@code dGradOut}). */
    public static native void embeddingTokenBackwardGPUDeviceGradWeightsDeviceGrad(
            float[] tokens,
            long dGradOut,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            long gradWeightsDevicePtr);

    /** Как {@link #embeddingPositionBackwardGPUDeviceGradWeights}, но gradCombined уже на device. */
    public static native void embeddingPositionBackwardGPUDeviceGradWeightsDeviceGrad(
            long dGradCombined, int batch, int seqLen, int dModel, long gradWeightsDevicePtr);

    /**
     * На месте: {@code x[b,s,:] += posWeights[posRowStart + s,:]}; таблица на device (полный {@code maxSeq × dModel}).
     */
    static native void addPositionEmbeddingGPUDeviceWeightsWithOffset(
            float[] xBatchSeqD, long posWeightsDevicePtr, int batch, int seqLen, int dModel, int posRowStart);

    /** Как {@link #addPositionEmbeddingGPUDeviceWeightsWithOffset} с {@code posRowStart = 0}. */
    public static void addPositionEmbeddingGPUDeviceWeights(
            float[] xBatchSeqD, long posWeightsDevicePtr, int batch, int seqLen, int dModel) {
        TensorOpsGpuEmbeddingHost.addPositionEmbeddingGPUDeviceWeights(
                xBatchSeqD, posWeightsDevicePtr, batch, seqLen, dModel);
    }

    public static void addPositionEmbeddingGPUDeviceWeights(
            float[] xBatchSeqD, long posWeightsDevicePtr, int batch, int seqLen, int dModel, int posRowStart) {
        TensorOpsGpuEmbeddingHost.addPositionEmbeddingGPUDeviceWeights(
                xBatchSeqD, posWeightsDevicePtr, batch, seqLen, dModel, posRowStart);
    }

    static native void addPositionEmbeddingGPUDeviceBuffersWithOffset(
            long xDevicePtr, long posWeightsDevicePtr, int batch, int seqLen, int dModel, int posRowStart);

    public static void addPositionEmbeddingGPUDeviceBuffers(
            long xDevicePtr, long posWeightsDevicePtr, int batch, int seqLen, int dModel) {
        TensorOpsGpuEmbeddingHost.addPositionEmbeddingGPUDeviceBuffers(
                xDevicePtr, posWeightsDevicePtr, batch, seqLen, dModel);
    }

    public static void addPositionEmbeddingGPUDeviceBuffers(
            long xDevicePtr, long posWeightsDevicePtr, int batch, int seqLen, int dModel, int posRowStart) {
        TensorOpsGpuEmbeddingHost.addPositionEmbeddingGPUDeviceBuffers(
                xDevicePtr, posWeightsDevicePtr, batch, seqLen, dModel, posRowStart);
    }

    /**
     * Host {@code x} и contiguous строки {@code [seqLen * dModel]} (уже смещённые при необходимости); in-place на CPU
     * буфере через GPU.
     */
    public static native void addPositionEmbeddingGPUHostPosSlice(
            float[] xBatchSeqD, float[] posRowsContiguous, int batch, int seqLen, int dModel);

    /**
     * Gather токен-эмбеддингов на GPU; при {@link #useFp16Matmul()} — чтение весов с округлением до FP16.
     */
    public static void embeddingTokenForwardGpu(
            float[] tokens, float[] weights, float[] out, int batch, int seqLen, int dModel, int vocabSize) {
        TensorOpsGpuEmbeddingHost.embeddingTokenForwardGpu(tokens, weights, out, batch, seqLen, dModel, vocabSize);
    }

    /** Direct-буфер токенов (см. {@link #embeddingTokenForwardGPUDirect}). */
    public static void embeddingTokenForwardGpuDirect(
            java.nio.ByteBuffer directTokens,
            float[] weights,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGpuEmbeddingHost.embeddingTokenForwardGpuDirect(
                directTokens, weights, out, batch, seqLen, dModel, vocabSize);
    }

    public static void embeddingTokenForwardGpuDeviceWeights(
            float[] tokens,
            long weightsDevicePtr,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGpuEmbeddingHost.embeddingTokenForwardGpuDeviceWeights(
                tokens, weightsDevicePtr, out, batch, seqLen, dModel, vocabSize);
    }

    public static void embeddingTokenForwardGpuDirectDeviceWeights(
            java.nio.ByteBuffer directTokens,
            long weightsDevicePtr,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGpuEmbeddingHost.embeddingTokenForwardGpuDirectDeviceWeights(
                directTokens, weightsDevicePtr, out, batch, seqLen, dModel, vocabSize);
    }

    public static void embeddingTokenForwardGpuDeviceWeightsToDevice(
            float[] tokens,
            long weightsDevicePtr,
            GpuFloatBuffer outDevice,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGpuEmbeddingHost.embeddingTokenForwardGpuDeviceWeightsToDevice(
                tokens, weightsDevicePtr, outDevice, batch, seqLen, dModel, vocabSize);
    }

    public static void embeddingTokenForwardGpuDirectDeviceWeightsToDevice(
            java.nio.ByteBuffer directTokens,
            long weightsDevicePtr,
            GpuFloatBuffer outDevice,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize) {
        TensorOpsGpuEmbeddingHost.embeddingTokenForwardGpuDirectDeviceWeightsToDevice(
                directTokens, weightsDevicePtr, outDevice, batch, seqLen, dModel, vocabSize);
    }

    public static void addPositionEmbeddingGpuDevice(
            GpuFloatBuffer xData, GpuFloatBuffer posWeightsData, int batch, int seqLen, int dModel) {
        TensorOpsGpuEmbeddingHost.addPositionEmbeddingGpuDevice(xData, posWeightsData, batch, seqLen, dModel);
    }

    public static void addPositionEmbeddingGpuDevice(
            GpuFloatBuffer xData,
            GpuFloatBuffer posWeightsData,
            int batch,
            int seqLen,
            int dModel,
            int posRowStart) {
        TensorOpsGpuEmbeddingHost.addPositionEmbeddingGpuDevice(
                xData, posWeightsData, batch, seqLen, dModel, posRowStart);
    }

    public static void addPositionEmbeddingInPlaceHostSlice(
            float[] xBatchSeqD, float[] posRowsContiguous, int batch, int seqLen, int dModel) {
        TensorOpsGpuEmbeddingHost.addPositionEmbeddingInPlaceHostSlice(
                xBatchSeqD, posRowsContiguous, batch, seqLen, dModel);
    }

    /** Явный выбор FP16-gather (тесты). */
    public static void embeddingTokenForwardGpuEx(
            float[] tokens,
            float[] weights,
            float[] out,
            int batch,
            int seqLen,
            int dModel,
            int vocabSize,
            boolean useFp16Gather) {
        TensorOpsGpuEmbeddingHost.embeddingTokenForwardGpuEx(
                tokens, weights, out, batch, seqLen, dModel, vocabSize, useFp16Gather);
    }

    /**
     * Fused scaled-dot-product attention: H2D {@code q,k,v} → нативное ядро {@link
     * #scaledDotProductAttentionForwardGPUDevice} → D2H {@code output} и {@code probs}. Отдельного JNI только под
     * {@code float[]} нет (один CUDA-путь по device-указателям).
     *
     * <p>Размеры как у прежнего нативного forward; softmax — FP16-exp при {@code useFp16Softmax}.
     */
    public static void scaledDotProductAttentionForwardFromHost(
            float[] q,
            float[] k,
            float[] v,
            float[] mask,
            float[] output,
            float[] probs,
            int batch,
            int seqLen,
            int dK,
            int dV,
            float scale,
            boolean useFp16Softmax) {
        TensorOpsGpuHostPathScratch.scaledDotProductAttentionForwardFromHost(
                q, k, v, mask, output, probs, batch, seqLen, dK, dV, scale, useFp16Softmax);
    }

    /**
     * Q/K/V и выход attention на device; маска с CPU при необходимости. {@code h_probs} может быть {@code null} —
     * без D2H вероятностей.
     */
    public static native void scaledDotProductAttentionForwardGPUDevice(
            long dQPtr,
            long dKPtr,
            long dVPtr,
            float[] mask,
            long dOutPtr,
            float[] h_probs,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            boolean useFp16Softmax);

    /**
     * SDPA целиком на device: опционально {@code dMask} (seqLen×seqLen float), опционально {@code dProbsOut} — копия
     * вероятностей D2D без синхронизации потока (для CUDA graph и training cache на VRAM).
     */
    public static native void scaledDotProductAttentionForwardGPUDeviceResident(
            long dQPtr,
            long dKPtr,
            long dVPtr,
            long dOutPtr,
            long dMaskPtr,
            long dProbsOutPtr,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            boolean useFp16Softmax);

    /**
     * Размер головы для скомпилированных CUDA ядер FlashAttention-2 (causal); иной d_head на этом пути не
     * поддерживается.
     */
    public static final int FLASH_ATTENTION_D_HEAD = 16;

    /** FlashAttention-2 forward (causal). Q/K/V/O=[BH,S,dHead], LSE=[BH,S]. BH=batch*numHeads. */
    static native void flashAttentionForwardGPUDeviceResident(
            long dQPtr, long dKPtr, long dVPtr, long dOutPtr, long dLSEPtr,
            int BH, int S, int dHead, float scale);

    /** FlashAttention-2 backward (causal). Q/K/V/O/dO/LSE → dQ/dK/dV. */
    static native void flashAttentionBackwardGPUDeviceResident(
            long dQPtr, long dKPtr, long dVPtr,
            long dOPtr, long dOGradPtr, long dLSEPtr,
            long dGradQPtr, long dGradKPtr, long dGradVPtr,
            int BH, int S, int dHead, float scale);

    /** Вызов FlashAttention-2 forward. BH = batch*numHeads; {@code dHead} должен быть {@link #FLASH_ATTENTION_D_HEAD}. */
    public static void flashAttentionForwardGpuDeviceResident(
            GpuFloatBuffer dQ, GpuFloatBuffer dK, GpuFloatBuffer dV,
            GpuFloatBuffer dOut, GpuFloatBuffer dLSE,
            int BH, int S, int dHead, float scale) {
        TensorOpsGpuFlashAttention.flashAttentionForwardGpuDeviceResident(
                dQ, dK, dV, dOut, dLSE, BH, S, dHead, scale);
    }

    /** Вызов FlashAttention-2 backward; {@code dHead} — как во forward, должен быть {@link #FLASH_ATTENTION_D_HEAD}. */
    public static void flashAttentionBackwardGpuDeviceResident(
            GpuFloatBuffer dQ, GpuFloatBuffer dK, GpuFloatBuffer dV,
            GpuFloatBuffer dO, GpuFloatBuffer dOGrad, GpuFloatBuffer dLSE,
            GpuFloatBuffer dGradQ, GpuFloatBuffer dGradK, GpuFloatBuffer dGradV,
            int BH, int S, int dHead, float scale) {
        TensorOpsGpuFlashAttention.flashAttentionBackwardGpuDeviceResident(
                dQ, dK, dV, dO, dOGrad, dLSE, dGradQ, dGradK, dGradV, BH, S, dHead, scale);
    }

    static native boolean ensureStridedBatchedPackScratch0(long rows, int dModel, int dIntermediate);

    /**
     * Привязать QKV/FFN strided-batched pack к внешним device-буферам (время жизни ≥ decoder CUDA graph).
     * Снимать {@link #clearStridedBatchedPackOverride()} в {@code finally}.
     */
    static native void setStridedBatchedPackOverride0(
            long wDevicePtr, long cDevicePtr, long capWElems, long capCElems);

    static native void clearStridedBatchedPackOverride0();

    static native void decoderGraphPrewarmDeviceOps0(
            int batch, int seqLen, int dModel, int numHeads, int dIntermediate);

    static native boolean cudaStreamBeginCapture0();

    static native long cudaStreamEndCaptureAndInstantiate0();

    static native void abortCudaStreamCaptureIfActive0();

    /** @return {@code false} если {@code cudaGraphLaunch} вернул ошибку (граф не исполнялся). */
    static native boolean cudaGraphExecLaunch0(long execPtr);

    /** Код последней ошибки {@code cudaGraphLaunch} на потоке (0 если успех); см. {@link #cudaErrorMemoryAllocation}. */
    static native int decoderGraphExecLaunchLastCudaError0();

    /**
     * Зонд перед/после decoder graph launch (device, capture, streamQuery, версии, флаги exec при поддержке CUDA).
     * Длина 9: {@code [dev, capStatus, streamQueryCode, noPrelaunchSyncEnv1, driverVer, runtimeVer, execFlagsOrNeg,
     * streamPtr, execPtr]}.
     */
    static native long[] decoderGraphLaunchProbe0(long execPtr);

    static native void cudaGraphExecDestroy0(long execPtr);

    /** Снимок thread-local SDPA aux на GPU: {@code [fwdPtr, graphPtr, fwdBytes, graphBytes]}. */
    static native long[] decoderGraphDebugNativeAuxSnapshot0();

    /**
     * FNV-1a-микс thread-local указателей/размеров, влияющих на decoder CUDA graph (graph SDPA aux, fwd aux, prewarm
     * warmup, Flash {@code D}, QKV pack override/TL).
     */
    static native long decoderGraphNativeStabilityToken0();

    /**
     * Fused scaled-dot-product attention backward: вычисляет {@code dQ,dK,dV}.
     * При {@code useFp16Softmax} градиент по softmax совпадает с forward ({@link
     * #scaledDotProductAttentionForwardFromHost} / {@link #scaledDotProductAttentionForwardGPUDevice}) при FP16-exp.
     */
    public static native void scaledDotProductAttentionBackwardGPU(
            float[] gradOut,
            float[] probs,
            float[] q,
            float[] k,
            float[] v,
            float[] gradQ,
            float[] gradK,
            float[] gradV,
            int batch,
            int seqLen,
            int dK,
            int dV,
            float scale,
            float[] mask,
            boolean useFp16Softmax);

    /** Сумма квадратов элементов: {@code sum_i x_i^2}. */
    public static native double sumSquaresGPU(float[] src, int n);

    /** Масштабирование на месте: {@code src[i] *= scalar}. */
    public static native void scaleInPlaceGPU(float[] src, int n, float scalar);

    /**
     * Один AdamW step на векторе параметров. Ядро на GPU; нативный код держит thread-local пул из четырёх
     * device-буферов (рост по мере max(n)), без cudaMalloc/free на каждый вызов. Веса и моменты по-прежнему
     * в {@code float[]} на хосте — полный H2D/D2H на шаг сохраняется.
     */
    public static native void adamWStepGPU(
            float[] param,
            float[] grad,
            float[] m,
            float[] v,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            float weightDecay,
            float invBias1,
            float invBias2,
            int n);

    /**
     * AdamW на уже выделенных device-буферах (без H2D/D2H); то же ядро, что {@link #adamWStepGPU}.
     */
    public static native void adamWStepGPUDevice(
            long dParam,
            long dGrad,
            long dM,
            long dV,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            float weightDecay,
            float invBias1,
            float invBias2,
            int n);

    /**
     * Несколько независимых device-буферов: один launch (по CUDA-блоку на сегмент), без склейки данных в один
     * contiguous буфер.
     */
    public static native void adamWStepGPUDeviceFused(
            long[] paramDevicePtrs,
            long[] gradDevicePtrs,
            long[] mDevicePtrs,
            long[] vDevicePtrs,
            int[] segmentLengths,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            float weightDecay,
            float invBias1,
            float invBias2);

    /**
     * Один contiguous host-блок длины {@code numParams} после склейки в Java: один JNI и один kernel
     * launch (тот же {@code adamw_kernel}, что у {@link #adamWStepGPU}) — не отдельное fused-CUDA-ядро.
     * Настоящая device-мультибуферная «сборка» — {@link #adamWStepGPUDeviceFused}.
     */
    public static native void adamWStepFusedGPU(
            float[] param,
            float[] grad,
            float[] m,
            float[] v,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            float weightDecay,
            float invBias1,
            float invBias2,
            int numParams);

    /** Обёртка над {@link #adamWStepGPUDevice} для {@link GpuFloatBuffer}. */
    public static void adamWStepGpuDevice(
            GpuFloatBuffer param,
            GpuFloatBuffer grad,
            GpuFloatBuffer m,
            GpuFloatBuffer v,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            float weightDecay,
            float invBias1,
            float invBias2,
            int n) {
        TensorOpsGpuDeviceAdamW.adamWStepGpuDevice(
                param,
                grad,
                m,
                v,
                learningRate,
                beta1,
                beta2,
                epsilon,
                weightDecay,
                invBias1,
                invBias2,
                n);
    }

    // ========== JAVA-ОБЁРТКИ ==========

    /** Доступен ли GPU (без повторных нативных вызовов). */
    public static boolean isGpuAvailable() {
        return GPU_AVAILABLE;
    }

    /**
     * Жёсткое требование CUDA. Без инициализированного GPU и {@code jgpt_cuda} обучение/inference с
     * {@link TensorOps} не поддерживаются.
     */
    public static void requireCuda(String context) {
        if (!GPU_AVAILABLE) {
            throw new IllegalStateException(
                    context
                            + ": требуется CUDA и нативная библиотека jgpt_cuda; без GPU работа не поддерживается.");
        }
    }

    /**
     * Ждёт завершения очереди TensorOpsGPU на GPU (нативно: {@code cudaStreamSynchronize} для
     * {@code cudaStreamNonBlocking} библиотеки). Не вызывает {@code cudaDeviceSynchronize} — работа в
     * других {@code cudaStream_t} на том же устройстве не блокируется. Без GPU — no-op.
     *
     * <p><b>Граница шага обучения:</b> {@link com.veles.llm.jgpt.training.LLMTrainer} вызывает синхронизацию
     * потока на границах шага/чекпоинта, чтобы хостовые чтения (логи, сравнение весов) видели завершённые
     * ядра. Внутри одного JNI-вызова с {@code jfloatArray} нативный код сам синхронизирует поток до
     * {@code ReleaseFloatArrayElements}; между подряд идущими {@code *GPUDevice} на том же
     * {@code kTensorCudaStream} полагаемся на порядок запуска без лишнего sync до явной границы (этот метод
     * или следующий JNI с хостовым буфером).
     *
     * @throws IllegalStateException если {@code cudaStreamSynchronize} вернул ошибку (в т.ч. после сбоя ядра на
     *     stream); sticky CUDA error сбрасывается в native.
     */
    public static void synchronizeStream() {
        if (!GPU_AVAILABLE) {
            return;
        }
        synchronizeStream0();
    }

    /** То же, что {@link #synchronizeStream()} — короткое имя для замеров. */
    public static void synchronize() {
        synchronizeStream();
    }

    /**
     * Освобождает нативные thread-local ресурсы CUDA/cuBLAS текущего потока. Без активного GPU не вызывает JNI.
     */
    public static void cleanupCudaThreadResources() {
        if (!GPU_AVAILABLE) {
            return;
        }
        cleanupCudaThreadResources0();
    }

    /**
     * Глобальная синхронизация устройства. Не использовать в горячем цикле обучения: дороже
     * {@link #synchronizeStream()} и нужна, если возможны ядра вне {@code kTensorCudaStream}.
     */
    public static void synchronizeDevice() {
        if (!GPU_AVAILABLE) {
            return;
        }
        synchronizeDevice0();
    }

    /**
     * Обрабатывает очереди отложенных освобождений VRAM ({@link GpuFloatBuffer#drainLeaked()},
     * {@link GpuHalfBuffer#drainLeaked()}, {@link GpuIntBuffer#drainLeaked()}, {@link GpuTensor#drainLeaked()}) —
     * записи появляются после GC, если не вызывали {@code close()}. Порядок: сначала «листовые» буферы, затем
     * {@link GpuTensor} (закрывает вложенные {@link GpuFloatBuffer}). Имеет смысл вызывать перед
     * {@link #cudaTrimDeviceMemoryPoolsBestEffort()} в известных точках синхронизации обучения.
     */
    public static void drainDeferredGpuBuffers() {
        GpuFloatBuffer.drainLeaked();
        GpuHalfBuffer.drainLeaked();
        GpuIntBuffer.drainLeaked();
        GpuTensor.drainLeaked();
    }

    /**
     * После graph-OOM / проактивного отказа от graph: полная sync устройства, trim graph memory и async memory pools
     * (нативно), чтобы снизить ложные OOM на {@code cudaMallocAsync} в eager-пути.
     */
    public static void cudaTrimDeviceMemoryPoolsBestEffort() {
        if (!GPU_AVAILABLE) {
            return;
        }
        cudaTrimDeviceMemoryPoolsBestEffort0();
    }

    /** Байты занятые сейчас: {@code total − free} из {@code cudaMemGetInfo}. */
    public static long getGpuMemoryAllocated() {
        if (!GPU_AVAILABLE) {
            return 0L;
        }
        return getGpuMemoryAllocated0();
    }

    /** Общий объём VRAM устройства (байты), см. {@code cudaMemGetInfo}. */
    public static long getGpuMemoryReserved() {
        if (!GPU_AVAILABLE) {
            return 0L;
        }
        return getGpuMemoryReserved0();
    }

    /**
     * Лог VRAM по {@link #getGpuMemoryAllocated()} / {@link #getGpuMemoryReserved()} перед крупной аллокацией float
     * на устройстве. Срабатывает если {@code numFloats} ≥ ~60&nbsp;MiB буфера или задано
     * {@code -Djgpt.debug.vramBeforeAlloc=true} (тогда — для любого положительного размера).
     */
    public static void logVramBeforeDeviceFloatAlloc(long numFloats, String context) {
        TensorOpsGpuVramLog.logBeforeDeviceFloatAlloc(numFloats, context);
    }

    public static String getGpuName() {
        return GPU_NAME;
    }

    public static long getGpuMemory() {
        return GPU_MEMORY_MB;
    }

    /** GEMM на GPU, если CUDA инициализирована и размеры положительны. */
    public static boolean shouldUseGpuMatmul(long m, long k, long n) {
        return GPU_AVAILABLE && m > 0 && k > 0 && n > 0;
    }

    /** Батчевый GEMM на GPU при положительных размерах. */
    public static boolean shouldUseGpuMatmulBatched(long batch, long m, long k, long n) {
        return GPU_AVAILABLE && batch > 0 && m > 0 && k > 0 && n > 0;
    }

    /** Поэлементные/редукционные op на GPU при ненулевой длине буфера. */
    public static boolean shouldUseGpuElementwise(int numElements) {
        return GPU_AVAILABLE && numElements > 0;
    }

    /**
     * Fused CE + softmax + ∂logits на GPU при инициализированной CUDA и ненулевой длине логитов.
     */
    public static boolean shouldUseGpuCrossEntropy(int logitNumFloats) {
        return GPU_AVAILABLE && logitNumFloats > 0;
    }

    /** Устар.; раньше влиял на CE. Сейчас {@link #shouldUseGpuCrossEntropy} порог не использует. */
    public static int ceGpuMinElements() {
        return CE_GPU_MIN_ELEMENTS;
    }

    /**
     * AdamW/clip/unscale на GPU для любого ненулевого буфера при доступной CUDA ({@link #adamWStepGPU}).
     */
    public static boolean shouldUseGpuOptimizer(int numElements) {
        return GPU_AVAILABLE && numElements > 0;
    }

    /** Включён ли GEMM через FP16 Tensor Cores ({@link #FP16_MATMUL}). */
    public static boolean useFp16Matmul() {
        return FP16_MATMUL;
    }

    /** {@code eps} для RMSNorm: env/property или дефолт по {@link #useFp16Matmul()}. */
    public static float rmsNormEps() {
        return RMSNORM_EPS;
    }

    /**
     * Один GEMM на GPU: при {@link #useFp16Matmul()} — FP16 compute, иначе FP32+TF32.
     */
    public static void matmulGPUMaybeFp16(float[] A, float[] B, float[] C, int M, int K, int N) {
        TensorOpsGpuHostMatmul.matmulGPUMaybeFp16(A, B, C, M, K, N);
    }

    /** Батчевый GEMM: FP16 или FP32 по {@link #useFp16Matmul()}. */
    public static void matmulBatchedGPUMaybeFp16(
            float[] A, float[] B, float[] C, int M, int K, int N, int batchCount) {
        TensorOpsGpuHostMatmul.matmulBatchedGPUMaybeFp16(A, B, C, M, K, N, batchCount);
    }

    /** {@code matmul + bias + ReLU} на GPU: FP16 или FP32 по {@link #useFp16Matmul()}. */
    public static void matmulAddReluGPUMaybeFp16(
            float[] A, float[] B, float[] bias, float[] C, int M, int K, int N) {
        TensorOpsGpuHostMatmul.matmulAddReluGPUMaybeFp16(A, B, bias, C, M, K, N);
    }

    /** In-place: {@code C += bias} по столбцам (как в {@link #addBiasBroadcastGPUDevice}). */
    public static void addBiasBroadcastGpuDevice(GpuFloatBuffer c, GpuFloatBuffer bias, int M, int N) {
        TensorOpsGpuDeviceGemm.addBiasBroadcastGpuDevice(c, bias, M, N);
    }

    /** Сумма по строкам для каждого столбца (градиент bias после linear). {@code dst <- 0*dst + sum}. */
    public static void sumColumnsGpuDevice(GpuFloatBuffer src, GpuFloatBuffer dst, int M, int N) {
        TensorOpsGpuDeviceGemm.sumColumnsGpuDevice(src, dst, M, N);
    }

    /**
     * Как {@link #sumColumnsGpuDevice(GpuFloatBuffer, GpuFloatBuffer, int, int)} с {@code dst[j] = beta*dst[j] +
     * sum_i src[i*N+j]}.
     */
    public static void sumColumnsGpuDevice(GpuFloatBuffer src, GpuFloatBuffer dst, int M, int N, float beta) {
        TensorOpsGpuDeviceGemm.sumColumnsGpuDevice(src, dst, M, N, beta);
    }

    /** GEMM на GPU; буферы должны вмещать M×K, K×N и M×N float соответственно. */
    public static void matmulGpuDevice(GpuFloatBuffer a, GpuFloatBuffer b, GpuFloatBuffer c, int M, int K, int N) {
        TensorOpsGpuDeviceGemm.matmulGpuDevice(a, b, c, M, K, N);
    }

    /** GEMM с {@code beta = 0} (перезапись {@code C}). */
    public static void matmulGpuDeviceEx(
            GpuFloatBuffer a,
            GpuFloatBuffer b,
            GpuFloatBuffer c,
            int M,
            int K,
            int N,
            boolean transposeA,
            boolean transposeB) {
        TensorOpsGpuDeviceGemm.matmulGpuDeviceEx(a, b, c, M, K, N, transposeA, transposeB);
    }

    public static void matmulGpuDeviceEx(
            GpuFloatBuffer a,
            GpuFloatBuffer b,
            GpuFloatBuffer c,
            int M,
            int K,
            int N,
            boolean transposeA,
            boolean transposeB,
            float beta) {
        TensorOpsGpuDeviceGemm.matmulGpuDeviceEx(a, b, c, M, K, N, transposeA, transposeB, beta);
    }

    /**
     * Проекции Q, K, V после первой RMSNorm в attention (формы как у трёх {@link #matmulGpuDeviceEx} с общим
     * {@code xNorm}).
     */
    public static void matmulGpuDeviceQkvProjections(
            GpuFloatBuffer xNorm,
            GpuFloatBuffer wq,
            GpuFloatBuffer wk,
            GpuFloatBuffer wv,
            GpuFloatBuffer q,
            GpuFloatBuffer k,
            GpuFloatBuffer v,
            int rows,
            int dModel) {
        TensorOpsGpuDeviceGemm.matmulGpuDeviceQkvProjections(xNorm, wq, wk, wv, q, k, v, rows, dModel);
    }

    /**
     * Проекции {@code xNorm·W1} и {@code xNorm·W3} для SwiGLU FFN (как два {@link #matmulGpuDeviceEx} без транспонирования
     * с общим {@code xNorm}).
     */
    public static void matmulGpuDeviceFfnW1W3Projections(
            GpuFloatBuffer xNorm,
            GpuFloatBuffer w1,
            GpuFloatBuffer w3,
            GpuFloatBuffer h1,
            GpuFloatBuffer gate,
            int rows,
            int dModel,
            int dIntermediate) {
        TensorOpsGpuDeviceGemm.matmulGpuDeviceFfnW1W3Projections(
                xNorm, w1, w3, h1, gate, rows, dModel, dIntermediate);
    }

    public static void rmsNormGpuDevice(
            GpuFloatBuffer x, GpuFloatBuffer gamma, float eps, GpuFloatBuffer out, int outer, int lastDim) {
        TensorOpsGpuDeviceUnary.rmsNormGpuDevice(x, gamma, eps, out, outer, lastDim);
    }

    public static void sigmoidGpuDevice(GpuFloatBuffer src, GpuFloatBuffer dst, int n) {
        TensorOpsGpuDeviceUnary.sigmoidGpuDevice(src, dst, n);
    }

    public static void multiplyGpuDevice(GpuFloatBuffer a, GpuFloatBuffer b, GpuFloatBuffer c, int n) {
        TensorOpsGpuDeviceUnary.multiplyGpuDevice(a, b, c, n);
    }

    public static void multiplyBackwardGpuDevice(
            GpuFloatBuffer gOut, GpuFloatBuffer a, GpuFloatBuffer b, GpuFloatBuffer gA, GpuFloatBuffer gB, int n) {
        TensorOpsGpuDeviceUnary.multiplyBackwardGpuDevice(gOut, a, b, gA, gB, n);
    }

    public static void sigmoidBackwardGpuDevice(GpuFloatBuffer gOut, GpuFloatBuffer inp, GpuFloatBuffer gIn, int n) {
        TensorOpsGpuDeviceUnary.sigmoidBackwardGpuDevice(gOut, inp, gIn, n);
    }

    public static void accumulateAddGpuDevice(GpuFloatBuffer acc, GpuFloatBuffer delta, int n) {
        TensorOpsGpuDeviceUnary.accumulateAddGpuDevice(acc, delta, n);
    }

    public static void scaleInPlaceGpuDevice(GpuFloatBuffer buf, int n, float scalar) {
        TensorOpsGpuDeviceUnary.scaleInPlaceGpuDevice(buf, n, scalar);
    }

    public static double sumSquaresGpuDevice(GpuFloatBuffer buf, int n) {
        return TensorOpsGpuDeviceUnary.sumSquaresGpuDevice(buf, n);
    }

    /**
     * Сумма квадратов всех VRAM-градиентов в {@code paramMap} (только записи с {@link
     * GpuTensor#hasGradBuffer()}).
     */
    public static double sumSquaresGpuDeviceParamGrads(Map<Tensor, GpuTensor> paramMap) {
        return TensorOpsGpuDeviceUnary.sumSquaresGpuDeviceParamGrads(paramMap);
    }

    public static boolean anyNonFiniteGpuDevice(GpuFloatBuffer buf, int n) {
        return TensorOpsGpuDeviceUnary.anyNonFiniteGpuDevice(buf, n);
    }

    /**
     * Фьюзированная проверка NaN/Inf по списку буферов на GPU; один JNI-вызов и одна синхронизация.
     * Пропускает null/closed буферы.
     *
     * @param bufs   буферы для проверки
     * @param lens   длины (в float) каждого буфера; {@code lens[i] <= 0} → буфер пропускается
     * @param nBufs  реальное число элементов в массивах (≤ {@code bufs.length})
     * @return {@code true}, если хотя бы в одном буфере есть NaN или Inf
     */
    public static boolean anyNonFiniteGpuDeviceMulti(GpuFloatBuffer[] bufs, int[] lens, int nBufs) {
        return TensorOpsGpuDeviceUnary.anyNonFiniteGpuDeviceMulti(bufs, lens, nBufs);
    }

    /**
     * CE + softmax + grad on device; targets are host-side (small O(B×S)).
     *
     * @return mean CE loss
     */
    public static float crossEntropySoftmaxGradLossGpuDevice(
            GpuFloatBuffer logits,
            float[] targets,
            GpuFloatBuffer grad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean fp16) {
        return TensorOpsGpuDeviceCeLm.crossEntropySoftmaxGradLossGpuDevice(
                logits, targets, grad, batch, seqLen, vocab, gradScale, fp16);
    }

    /**
     * CE на device; {@link GpuIntBuffer} — int32 target id на каждую строку {@code batch×seqLen}.
     * Возвращаемый loss — среднее по токенам; native слой синхронизирует стрим перед D2H скаляра (безопасно для
     * scaler). Не убирать эту синхронизацию в паре с FP16 до переноса проверки overflow на GPU.
     */
    public static float crossEntropySoftmaxGradLossGpuDeviceTargetsDevice(
            GpuFloatBuffer logits,
            GpuIntBuffer targets,
            GpuFloatBuffer grad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean fp16) {
        return TensorOpsGpuDeviceCeLm.crossEntropySoftmaxGradLossGpuDeviceTargetsDevice(
                logits, targets, grad, batch, seqLen, vocab, gradScale, fp16);
    }

    /**
     * Async device-CE (см. native): очередь kernel + D2H скаляра. Перед чтением ∂logits на device (backward) вызвать
     * {@link #synchronizeStream()}; loss — {@link #crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost()} (pinned
     * host, валиден после того же sync).
     */
    public static void crossEntropySoftmaxGradLossGpuDeviceTargetsDeviceAsync(
            GpuFloatBuffer logits,
            GpuIntBuffer targets,
            GpuFloatBuffer grad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean fp16) {
        TensorOpsGpuDeviceCeLm.crossEntropySoftmaxGradLossGpuDeviceTargetsDeviceAsync(
                logits, targets, grad, batch, seqLen, vocab, gradScale, fp16);
    }

    /** Async CE с {@code float[]} targets (device logits); контракт как у {@link #crossEntropySoftmaxGradLossGpuDeviceTargetsDeviceAsync}. */
    public static void crossEntropySoftmaxGradLossGpuDeviceHostFloatTargetsAsync(
            GpuFloatBuffer logits,
            float[] targets,
            GpuFloatBuffer grad,
            int batch,
            int seqLen,
            int vocab,
            float gradScale,
            boolean fp16) {
        TensorOpsGpuDeviceCeLm.crossEntropySoftmaxGradLossGpuDeviceHostFloatTargetsAsync(
                logits, targets, grad, batch, seqLen, vocab, gradScale, fp16);
    }

    public static float crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost() {
        return TensorOpsGpuDeviceCeLm.crossEntropySoftmaxGradLossGpuDeviceReadPendingFromHost();
    }

    public static void gatherLogitsByIdsGpuDevice(
            GpuFloatBuffer logits,
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateLogits,
            int rows,
            int vocab,
            int candidates) {
        TensorOpsGpuDeviceCeLm.gatherLogitsByIdsGpuDevice(
                logits, candidateIds, candidateLogits, rows, vocab, candidates);
    }

    /**
     * LM-head только по кандидатам: эквивалентно {@code matmul(normedHidden, W)} с последующим gather по id, без
     * материализации полных логитов {@code rows × vocab}.
     */
    public static void lmHeadCandidateLogitsGpuDevice(
            GpuFloatBuffer normedHidden,
            GpuFloatBuffer lmHeadWeights,
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateLogits,
            int rows,
            int dModel,
            int vocab,
            int candidates) {
        TensorOpsGpuDeviceCeLm.lmHeadCandidateLogitsGpuDevice(
                normedHidden, lmHeadWeights, candidateIds, candidateLogits, rows, dModel, vocab, candidates);
    }

    public static float sampledCrossEntropyGradLossGpuDeviceFirstSlot(
            GpuFloatBuffer candidateLogits,
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateGrad,
            int rows,
            int candidates,
            float gradScale) {
        return TensorOpsGpuDeviceCeLm.sampledCrossEntropyGradLossGpuDeviceFirstSlot(
                candidateLogits, candidateIds, candidateGrad, rows, candidates, gradScale);
    }

    public static void sampledLmHeadBackwardGpuDevice(
            GpuIntBuffer candidateIds,
            GpuFloatBuffer candidateGrad,
            GpuFloatBuffer normedHidden,
            GpuFloatBuffer lmHeadWeights,
            GpuFloatBuffer dHidden,
            GpuFloatBuffer dLmHead,
            int rows,
            int dModel,
            int vocab,
            int candidates) {
        TensorOpsGpuDeviceCeLm.sampledLmHeadBackwardGpuDevice(
                candidateIds,
                candidateGrad,
                normedHidden,
                lmHeadWeights,
                dHidden,
                dLmHead,
                rows,
                dModel,
                vocab,
                candidates);
    }

    public static void rmsNormMatmulLmHeadGpuDevice(
            GpuFloatBuffer x,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer normOut,
            GpuFloatBuffer w,
            GpuFloatBuffer logits,
            int rows,
            int dModel,
            int vocab,
            boolean fp16Rms) {
        TensorOpsGpuDeviceFusedLayout.rmsNormMatmulLmHeadGpuDevice(
                x, gamma, eps, normOut, w, logits, rows, dModel, vocab, fp16Rms);
    }

    public static void rmsNormMatmulFfnW1W3GpuDevice(
            GpuFloatBuffer xRes1,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer normOut,
            GpuFloatBuffer w1,
            GpuFloatBuffer w3,
            GpuFloatBuffer h1,
            GpuFloatBuffer gate,
            int rows,
            int dModel,
            int dIntermediate,
            boolean fp16Rms) {
        TensorOpsGpuDeviceFusedLayout.rmsNormMatmulFfnW1W3GpuDevice(
                xRes1, gamma, eps, normOut, w1, w3, h1, gate, rows, dModel, dIntermediate, fp16Rms);
    }

    public static void accumulateAddGpuFromHost(GpuFloatBuffer acc, float[] host, int off, int len) {
        TensorOpsGpuDeviceFusedLayout.accumulateAddGpuFromHost(acc, host, off, len);
    }

    public static void splitHeadsGpuDevice(
            GpuFloatBuffer src, GpuFloatBuffer dst, int batch, int seqLen, int dModel, int numHeads) {
        TensorOpsGpuDeviceFusedLayout.splitHeadsGpuDevice(src, dst, batch, seqLen, dModel, numHeads);
    }

    public static void concatHeadsGpuDevice(
            GpuFloatBuffer src, GpuFloatBuffer dst, int batch, int numHeads, int seqLen, int dHead) {
        TensorOpsGpuDeviceFusedLayout.concatHeadsGpuDevice(src, dst, batch, numHeads, seqLen, dHead);
    }

    public static void copyKvHeads4dToCacheGpuDevice(
            GpuFloatBuffer srcHeads4d,
            GpuFloatBuffer dstCache,
            int numHeads,
            int seqLen,
            int maxSeqLen,
            int dHead,
            int batchIdx,
            int batch) {
        TensorOpsGpuDeviceFusedLayout.copyKvHeads4dToCacheGpuDevice(
                srcHeads4d, dstCache, numHeads, seqLen, maxSeqLen, dHead, batchIdx, batch);
    }

    public static void applyRoPEBackwardGpuDevice(
            GpuFloatBuffer gradY, GpuFloatBuffer gradX, int batch, int numHeads, int seqLen, int dHead) {
        TensorOpsGpuDeviceFusedLayout.applyRoPEBackwardGpuDevice(gradY, gradX, batch, numHeads, seqLen, dHead);
    }

    public static void applyRoPE4dGpuDevice(
            GpuFloatBuffer src,
            GpuFloatBuffer dst,
            int batch,
            int numHeads,
            int seqLen,
            int dHead,
            int[] positions,
            int posBaseOffset) {
        TensorOpsGpuDeviceFusedLayout.applyRoPE4dGpuDevice(
                src, dst, batch, numHeads, seqLen, dHead, positions, posBaseOffset);
    }

    public static void scaledDotProductAttentionForwardGpuDevice(
            GpuFloatBuffer dQ,
            GpuFloatBuffer dK,
            GpuFloatBuffer dV,
            float[] maskOrNull,
            GpuFloatBuffer dOut,
            float[] h_probsOrNull,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            boolean useFp16Softmax) {
        TensorOpsGpuDeviceFusedLayout.scaledDotProductAttentionForwardGpuDevice(
                dQ,
                dK,
                dV,
                maskOrNull,
                dOut,
                h_probsOrNull,
                batch,
                seqLen,
                dKDim,
                dVDim,
                scale,
                useFp16Softmax);
    }

    /**
     * Как {@link #scaledDotProductAttentionForwardGpuDevice}, но маска уже на GPU ({@code dMaskOrNull}) и при
     * необходимости выход probs в device-буфер без D2H и без {@code cudaStreamSynchronize} внутри нативного вызова.
     */
    public static void scaledDotProductAttentionForwardGpuDeviceResident(
            GpuFloatBuffer dQ,
            GpuFloatBuffer dK,
            GpuFloatBuffer dV,
            GpuFloatBuffer dOut,
            GpuFloatBuffer dMaskOrNull,
            GpuFloatBuffer dProbsOutOrNull,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            boolean useFp16Softmax) {
        TensorOpsGpuDeviceFusedLayout.scaledDotProductAttentionForwardGpuDeviceResident(
                dQ,
                dK,
                dV,
                dOut,
                dMaskOrNull,
                dProbsOutOrNull,
                batch,
                seqLen,
                dKDim,
                dVDim,
                scale,
                useFp16Softmax);
    }

    /**
     * Подготовка к {@link #cudaStreamBeginCapture}: device scratch для QKV/FFN pack и инициализация thread-local
     * cuBLAS handle на {@code kTensorCudaStream}. Во время захвата графа недопустимы {@code cudaMalloc}/{@code
     * cudaFree} и ленивый {@code cublasCreate}.
     */
    public static void ensureStridedBatchedPackScratch(long rows, int dModel, int dIntermediate) {
        TensorOpsGpuDeviceStreamGraph.ensureStridedBatchedPackScratch(rows, dModel, dIntermediate);
    }

    /**
     * Число float в W- и C-pack для {@link #ensureStridedBatchedPackScratch(long, int, int)} (как в нативе).
     *
     * @return {@code [wNeed, cNeed]}
     */
    public static long[] stridedBatchedPackNeed(long rows, int dModel, int dIntermediate) {
        return TensorOpsGpuDeviceStreamGraph.stridedBatchedPackNeed(rows, dModel, dIntermediate);
    }

    public static void setStridedBatchedPackOverride(
            long wDevicePtr, long cDevicePtr, long capWElems, long capCElems) {
        TensorOpsGpuDeviceStreamGraph.setStridedBatchedPackOverride(
                wDevicePtr, cDevicePtr, capWElems, capCElems);
    }

    /** Снять {@link #setStridedBatchedPackOverride}; безопасно при неактивном override. */
    public static void clearStridedBatchedPackOverride() {
        TensorOpsGpuDeviceStreamGraph.clearStridedBatchedPackOverride();
    }

    /**
     * Перед {@link #cudaStreamBeginCapture}: предвыделить SDPA aux на device, прогреть strided-batched QKV/FFN и
     * batched GEMM attention на том же stream/handle, что forward — без {@code cudaMalloc} и ленивого workspace
     * cuBLAS внутри графа.
     */
    public static void decoderGraphPrewarmDeviceOps(
            int batch, int seqLen, int dModel, int numHeads, int dIntermediate) {
        TensorOpsGpuDeviceStreamGraph.decoderGraphPrewarmDeviceOps(
                batch, seqLen, dModel, numHeads, dIntermediate);
    }

    /**
     * Начать захват графа на {@code kTensorCudaStream} (режим global).
     *
     * <p>Диагностика VRAM: env {@code JGPT_DECODER_GRAPH_MEM_PROBE=1} — нативно пишет {@code cudaMemGetInfo} в stderr и
     * NDJSON при begin capture, после instantiate и до/после {@link #cudaGraphExecLaunch(long)}.
     */
    public static boolean cudaStreamBeginCapture() {
        return TensorOpsGpuDeviceStreamGraph.cudaStreamBeginCapture();
    }

    /** Завершить захват и вернуть указатель {@code cudaGraphExec_t} или {@code 0} при ошибке. */
    public static long cudaStreamEndCaptureAndInstantiate() {
        return TensorOpsGpuDeviceStreamGraph.cudaStreamEndCaptureAndInstantiate();
    }

    /**
     * Если основной stream TensorOpsGPU всё ещё в режиме graph capture (например после исключения до {@link
     * #cudaStreamEndCaptureAndInstantiate}), завершает захват и уничтожает неинстанциированный граф. Безопасно, если
     * захвата нет.
     */
    public static void abortCudaStreamCaptureIfActive() {
        TensorOpsGpuDeviceStreamGraph.abortCudaStreamCaptureIfActive();
    }

    /**
     * Запуск захваченного графа на {@code kTensorCudaStream}.
     *
     * <p>Диагностика (натив): при ошибке в stderr — коды {@code cudaGraphLaunch} и {@code cudaGetLastError}, статус
     * capture потока и снимок SDPA aux. По умолчанию перед запуском выполняется {@code cudaStreamSynchronize} на
     * основном потоке TensorOps (снижает ложные {@code cudaErrorInvalidValue} от отложенных ошибок). Отключить:
     * {@code JGPT_DECODER_CUDA_GRAPH_NO_PRELAUNCH_SYNC=1}.
     *
     * @return {@code false} при ошибке драйвера (в т.ч. illegal memory access) — вызывающий должен отключить graph
     *     и выполнить eager-путь.
     */
    public static boolean cudaGraphExecLaunch(long execPtr) {
        return TensorOpsGpuDeviceStreamGraph.cudaGraphExecLaunch(execPtr);
    }

    /**
     * Код CUDA после последнего {@link #cudaGraphExecLaunch(long)} с ненулевым {@code execPtr} на этом потоке:
     * {@code 0} — успех, иначе {@code cudaError*} (например {@link #CUDA_ERROR_MEMORY_ALLOCATION}).
     */
    public static int decoderGraphExecLaunchLastCudaError() {
        return TensorOpsGpuDeviceStreamGraph.decoderGraphExecLaunchLastCudaError();
    }

    /** См. {@link #decoderGraphLaunchProbe0(long)}. */
    public static long[] decoderGraphLaunchProbe(long execPtr) {
        return TensorOpsGpuDeviceStreamGraph.decoderGraphLaunchProbe(execPtr);
    }

    public static void cudaGraphExecDestroy(long execPtr) {
        TensorOpsGpuDeviceStreamGraph.cudaGraphExecDestroy(execPtr);
    }

    /**
     * Текущие указатели/размеры thread-local буферов SDPA для отладки decoder CUDA graph (см. нативный stderr при
     * неудачном {@link #cudaGraphExecLaunch}).
     */
    public static long[] decoderGraphDebugNativeAuxSnapshot() {
        return TensorOpsGpuDeviceStreamGraph.decoderGraphDebugNativeAuxSnapshot();
    }

    /** См. {@link #decoderGraphNativeStabilityToken0()}. */
    public static long decoderGraphNativeStabilityToken() {
        return TensorOpsGpuDeviceStreamGraph.decoderGraphNativeStabilityToken();
    }

    public static void scaledDotProductAttentionBackwardGpuDevice(
            GpuFloatBuffer gradOut,
            GpuFloatBuffer probs,
            GpuFloatBuffer q,
            GpuFloatBuffer k,
            GpuFloatBuffer v,
            GpuFloatBuffer gradQ,
            GpuFloatBuffer gradK,
            GpuFloatBuffer gradV,
            int batch,
            int seqLen,
            int dKDim,
            int dVDim,
            float scale,
            long dMask,
            boolean useFp16Softmax) {
        TensorOpsGpuDeviceFusedLayout.scaledDotProductAttentionBackwardGpuDevice(
                gradOut,
                probs,
                q,
                k,
                v,
                gradQ,
                gradK,
                gradV,
                batch,
                seqLen,
                dKDim,
                dVDim,
                scale,
                dMask,
                useFp16Softmax);
    }

    public static void rmsNormBackwardGpuDevice(
            GpuFloatBuffer gOut,
            GpuFloatBuffer x,
            GpuFloatBuffer gamma,
            float eps,
            GpuFloatBuffer gX,
            GpuFloatBuffer gGamma,
            int outer,
            int lastDim) {
        TensorOpsGpuDeviceFusedLayout.rmsNormBackwardGpuDevice(
                gOut, x, gamma, eps, gX, gGamma, outer, lastDim);
    }

    /**
     * На текущем CUDA-stream: {@code dstHalf[i] = float_to_half_rn(srcFloat[i])} (оба указателя на device).
     */
    public static void convertFloatDeviceToHalfDevice(long srcFloatDevicePtr, long dstHalfDevicePtr, int numFloats) {
        TensorOpsGpuDeviceStreamGraph.convertFloatDeviceToHalfDevice(
                srcFloatDevicePtr, dstHalfDevicePtr, numFloats);
    }

    /**
     * На текущем CUDA-stream: {@code dstFloat[i] = half_to_float(srcHalf[i])} (оба указателя на device).
     */
    public static void convertHalfDeviceToFloatDevice(long srcHalfDevicePtr, long dstFloatDevicePtr, int numFloats) {
        TensorOpsGpuDeviceStreamGraph.convertHalfDeviceToFloatDevice(
                srcHalfDevicePtr, dstFloatDevicePtr, numFloats);
    }

    static native void gatherLogitsByIdsGPUDevice(
            long dLogits, long dCandidateIds, long dCandidateLogits, int rows, int vocab, int candidates);

    static native void lmHeadCandidateLogitsGPUDevice(
            long dNormedHidden,
            long dLmHead,
            long dCandidateIds,
            long dCandidateLogits,
            int rows,
            int dModel,
            int vocab,
            int candidates);

    static native float sampledCrossEntropyGradLossGPUDeviceFirstSlot(
            long dCandidateLogits, long dCandidateIds, long dCandidateGrad, int rows, int candidates, float gradScale);

    static native void sampledLmHeadBackwardGPUDevice(
            long dCandidateIds,
            long dCandidateGrad,
            long dNormedHidden,
            long dLmHeadWeights,
            long dHidden,
            long dLmHeadGrad,
            int rows,
            int dModel,
            int vocab,
            int candidates);
}
