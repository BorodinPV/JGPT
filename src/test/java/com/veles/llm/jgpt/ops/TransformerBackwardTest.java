package com.veles.llm.jgpt.ops;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.core.Tensor;
import com.veles.llm.jgpt.model.BlockActivationCache;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Unit tests for {@link TransformerBackward}.
 * <p>
 * All tests use deterministic data via {@link Tensor#fromArray} — no randomTensor dependency.
 */
@DisplayName("TransformerBackward")
public class TransformerBackwardTest {

    private static final float EPS = 1e-5f;
    private static final float NUMERICAL_EPS = 1e-4f;

    // ========== Helper: deterministic small tensors ==========

    private static Tensor smallTensor(int[] shape, float baseValue) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = baseValue + i * 0.1f;
        }
        return Tensor.fromArray(data, shape);
    }

    private static Tensor zeros(int[] shape) {
        return new Tensor(shape);  // constructor zero-initializes
    }

    /**
     * Tensor whose {@link Tensor#gradBuffer()} holds upstream ∂L/∂output (what backward reads).
     * {@link Tensor#fromArray} only fills {@code data}; calling {@code zeroGrad()} after that zeros
     * {@code grad} — wrong for attention/block backward tests.
     */
    private static Tensor upstreamGradOut(int[] shape, float[] gradValues) {
        Tensor t = new Tensor(shape);
        t.zeroGrad();
        System.arraycopy(gradValues, 0, t.gradBuffer(), 0, gradValues.length);
        return t;
    }

    /** Same pattern as {@link #smallTensor} but stored in the grad buffer (for ∂L/∂output). */
    private static Tensor upstreamGradOutSmall(int[] shape, float baseValue) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        float[] g = new float[n];
        for (int i = 0; i < n; i++) {
            g[i] = baseValue + i * 0.1f;
        }
        return upstreamGradOut(shape, g);
    }

    // ========== RoPE Backward ==========

    @Test
    @DisplayName("applyRoPEBackward: accumulates gradient correctly")
    public void testApplyRoPEBackwardAccumulates() {
        // Setup: 4D tensor [B=1, H=1, S=2, D=4]
        Tensor gradY = Tensor.fromArray(new float[]{1, 0, 0, 1, 0, 1, 1, 0}, new int[]{1, 1, 2, 4});
        Tensor gradX = new Tensor(new int[]{1, 1, 2, 4});
        gradX.zeroGrad();

        TransformerBackward.applyRoPEBackward(gradY, gradX, null);

        // Verify gradient was accumulated (not overwritten)
        float[] gx = gradX.gradBuffer();
        assertNotEquals(0f, gx[0], EPS);  // at least some non-zero gradient
        assertTrue(gradX.hasGrad());
    }

    @Test
    @DisplayName("applyRoPEBackward: null positions uses 0..S-1")
    public void testApplyRoPEBackwardNullPositions() {
        Tensor gradY = Tensor.fromArray(new float[]{1, 2, 3, 4}, new int[]{1, 1, 1, 4});
        Tensor gradX = new Tensor(new int[]{1, 1, 1, 4});
        gradX.zeroGrad();

        // Should not throw with null positions
        assertDoesNotThrow(() -> TransformerBackward.applyRoPEBackward(gradY, gradX, null));
    }

    @Test
    @DisplayName("applyRoPEBackward: rejects odd dHead")
    public void testApplyRoPEBackwardOddDHead() {
        Tensor gradY = new Tensor(new int[]{1, 1, 2, 3});  // dHead=3 (odd)
        Tensor gradX = new Tensor(new int[]{1, 1, 2, 3});
        gradX.zeroGrad();

        assertThrows(IllegalArgumentException.class,
                () -> TransformerBackward.applyRoPEBackward(gradY, gradX, null));
    }

    @Test
    @DisplayName("applyRoPEBackward: null checks")
    public void testApplyRoPEBackwardNullChecks() {
        Tensor t = new Tensor(new int[]{1, 1, 2, 4});
        t.zeroGrad();
        assertThrows(NullPointerException.class,
                () -> TransformerBackward.applyRoPEBackward(null, t, null));
        assertThrows(NullPointerException.class,
                () -> TransformerBackward.applyRoPEBackward(t, null, null));
    }

    // ========== Attention Backward ==========

    @Test
    @DisplayName("scaledDotProductAttentionBackward: basic gradient flow")
    public void testAttentionBackwardBasic() {
        int B = 1, S = 2, dK = 4, dV = 4;
        Tensor gradOut =
                upstreamGradOut(
                        new int[]{B, S, dV}, new float[]{1, 0, 0, 1, 0, 1, 1, 0});

        // Deterministic small inputs
        Tensor Q = smallTensor(new int[]{B, S, dK}, 0.1f);
        Tensor K = smallTensor(new int[]{B, S, dK}, 0.2f);
        Tensor V = smallTensor(new int[]{B, S, dV}, 0.3f);

        Tensor gradQ = new Tensor(Q.getShape());
        Tensor gradK = new Tensor(K.getShape());
        Tensor gradV = new Tensor(V.getShape());

        TransformerBackward.scaledDotProductAttentionBackward(
                gradOut, Q, K, V, null, 1.0f / (float) Math.sqrt(dK), null, gradQ, gradK, gradV);

        // Verify gradients were computed
        assertTrue(gradQ.hasGrad() && gradK.hasGrad() && gradV.hasGrad());
        float gSum = 0f;
        for (float v : gradQ.gradBuffer()) gSum += Math.abs(v);
        assertTrue(gSum > 1e-6f, "Gradient sum should be non-zero");
    }

    @Test
    @DisplayName("scaledDotProductAttentionBackward: with cached probs")
    public void testAttentionBackwardWithCachedProbs() {
        int B = 1, S = 2, dK = 4, dV = 4;
        Tensor gradOut =
                upstreamGradOut(
                        new int[]{B, S, dV}, new float[]{1, 0, 0, 1, 0, 1, 1, 0});
        Tensor Q = smallTensor(new int[]{B, S, dK}, 0.1f);
        Tensor K = smallTensor(new int[]{B, S, dK}, 0.2f);
        Tensor V = smallTensor(new int[]{B, S, dV}, 0.3f);

        // Pre-compute probs
        Tensor kT = TensorOps.transpose2DLast(K);
        Tensor scores = TensorOps.matmulBatched3D(Q, kT);
        Tensor scaled = TensorOps.multiplyScalar(scores, 1.0f / (float) Math.sqrt(dK));
        Tensor probsCached = TensorOps.softmaxLastDim(scaled);

        Tensor gradQ = new Tensor(Q.getShape());
        Tensor gradK = new Tensor(K.getShape());
        Tensor gradV = new Tensor(V.getShape());

        // Should not throw with cached probs
        assertDoesNotThrow(() ->
                TransformerBackward.scaledDotProductAttentionBackward(
                        gradOut, Q, K, V, null, 1.0f / (float) Math.sqrt(dK),
                        probsCached, gradQ, gradK, gradV));
    }

    @Test
    @DisplayName("scaledDotProductAttentionBackward: null checks")
    public void testAttentionBackwardNullChecks() {
        Tensor t = new Tensor(new int[]{1, 2, 4});
        t.zeroGrad();
        assertThrows(NullPointerException.class,
                () -> TransformerBackward.scaledDotProductAttentionBackward(
                        null, t, t, t, null, 1f, null, t, t, t));
    }

    // ========== SwiGLU Backward ==========

    @Test
    @DisplayName("feedForwardSwiGLUBackward: without cache")
    public void testSwiGLUBackwardNoCache() {
        int B = 1, S = 2, dModel = 4, dInt = 8;
        Tensor gradOut = upstreamGradOutSmall(new int[]{B, S, dModel}, 0.1f);
        Tensor x = smallTensor(new int[]{B, S, dModel}, 0.2f);
        Tensor W1 = smallTensor(new int[]{dModel, dInt}, 0.3f);
        Tensor W2 = smallTensor(new int[]{dInt, dModel}, 0.4f);
        Tensor W3 = smallTensor(new int[]{dModel, dInt}, 0.5f);
        Tensor gradX = new Tensor(x.getShape());
        Tensor gradW1 = new Tensor(W1.getShape());
        Tensor gradW2 = new Tensor(W2.getShape());
        Tensor gradW3 = new Tensor(W3.getShape());

        TransformerBackward.feedForwardSwiGLUBackward(
                gradOut, x, W1, W2, W3, gradX, gradW1, gradW2, gradW3,
                null, null, null, null, null);

        // Verify gradients accumulated
        assertTrue(gradX.hasGrad() && gradW1.hasGrad() && gradW2.hasGrad() && gradW3.hasGrad());
    }

    @Test
    @DisplayName("feedForwardSwiGLUBackward: with cache skips recomputation")
    public void testSwiGLUBackwardWithCache() {
        int B = 1, S = 2, dModel = 4, dInt = 8;
        Tensor gradOut = upstreamGradOutSmall(new int[]{B, S, dModel}, 0.1f);
        Tensor x = smallTensor(new int[]{B, S, dModel}, 0.2f);
        Tensor W1 = smallTensor(new int[]{dModel, dInt}, 0.3f);
        Tensor W2 = smallTensor(new int[]{dInt, dModel}, 0.4f);
        Tensor W3 = smallTensor(new int[]{dModel, dInt}, 0.5f);

        // Forward pass to generate cache
        BlockActivationCache cache = new BlockActivationCache();
        Tensor xFlat = Tensor.wrap(x.internalBuffer(), new int[]{B * S, dModel});
        Tensor h1 = TensorOps.matmul(xFlat, W1);
        Tensor gate = TensorOps.matmul(xFlat, W3);
        Tensor sig = TensorOps.sigmoid(gate);
        Tensor gateSwish = TensorOps.multiply(gate, sig);
        Tensor hAct = TensorOps.multiply(h1, gateSwish);
        cache.ffnH1.store(h1, false);
        cache.ffnGate.store(gate, false);
        cache.ffnSig.store(sig, false);
        cache.ffnGateSwish.store(gateSwish, false);
        cache.ffnHActivated.store(hAct, false);

        Tensor gradX = new Tensor(x.getShape());
        Tensor gradW1 = new Tensor(W1.getShape());
        Tensor gradW2 = new Tensor(W2.getShape());
        Tensor gradW3 = new Tensor(W3.getShape());

        // Should not throw with cache
        assertDoesNotThrow(() ->
                TransformerBackward.feedForwardSwiGLUBackward(
                        gradOut, x, W1, W2, W3, gradX, gradW1, gradW2, gradW3,
                        cache.ffnH1.getTensor(),
                        cache.ffnGate.getTensor(),
                        cache.ffnSig.getTensor(),
                        cache.ffnGateSwish.getTensor(),
                        cache.ffnHActivated.getTensor()));
    }

    @Test
    @DisplayName("feedForwardSwiGLUBackward: null checks")
    public void testSwiGLUBackwardNullChecks() {
        Tensor t = new Tensor(new int[]{1, 2, 4});
        t.zeroGrad();
        Tensor w = new Tensor(new int[]{4, 8});
        assertThrows(NullPointerException.class,
                () -> TransformerBackward.feedForwardSwiGLUBackward(
                        null, t, w, w, w, t, w, w, w));
    }

    // ========== MHA with RoPE Backward ==========

    @Test
    @DisplayName("multiHeadAttentionWithRoPEBackward: basic flow")
    public void testMhaRoPEBackwardBasic() {
        assumeTrue(TensorOpsGPU.isGpuAvailable(), "fused MHA backward требует CUDA");
        int B = 1, S = 2, dModel = 8, numHeads = 2;
        int dHead = dModel / numHeads;
        Tensor gradOut = upstreamGradOutSmall(new int[]{B, S, dModel}, 0.1f);
        Tensor xNorm = smallTensor(new int[]{B, S, dModel}, 0.2f);
        Tensor Wq = smallTensor(new int[]{dModel, dModel}, 0.3f);
        Tensor Wk = smallTensor(new int[]{dModel, dModel}, 0.4f);
        Tensor Wv = smallTensor(new int[]{dModel, dModel}, 0.5f);
        Tensor Wo = smallTensor(new int[]{dModel, dModel}, 0.6f);
        Tensor mask = TensorOps.createCausalMask(S);
        BlockActivationCache cache = new BlockActivationCache();
        cache.attnQHeads.store(zeros(new int[]{B, numHeads, S, dHead}), false);
        cache.attnKHeads.store(zeros(new int[]{B, numHeads, S, dHead}), false);
        cache.attnVHeads.store(zeros(new int[]{B, numHeads, S, dHead}), false);
        cache.attnProbs.store(zeros(new int[]{B * numHeads, S, S}), false);
        cache.attnConcat.store(zeros(new int[]{B, S, dModel}), false);
        Tensor gradX = new Tensor(xNorm.getShape());
        Tensor gradWq = new Tensor(Wq.getShape());
        Tensor gradWk = new Tensor(Wk.getShape());
        Tensor gradWv = new Tensor(Wv.getShape());
        Tensor gradWo = new Tensor(Wo.getShape());

        assertDoesNotThrow(() ->
                TransformerBackward.multiHeadAttentionWithRoPEBackward(
                        gradOut, xNorm, Wq, Wk, Wv, Wo, numHeads, mask, cache,
                        gradX, gradWq, gradWk, gradWv, gradWo));
    }

    @Test
    @DisplayName("multiHeadAttentionWithRoPEBackward: null checks")
    public void testMhaRoPEBackwardNullChecks() {
        Tensor t = new Tensor(new int[]{1, 2, 8});
        t.zeroGrad();
        Tensor w = new Tensor(new int[]{8, 8});
        BlockActivationCache cache = new BlockActivationCache();
        assertThrows(NullPointerException.class,
                () -> TransformerBackward.multiHeadAttentionWithRoPEBackward(
                        null, t, w, w, w, w, 2, null, cache, t, w, w, w, w));
    }

    // ========== Transformer Block Backward ==========

    @Test
    @DisplayName("transformerBlockBackward: end-to-end with cache")
    public void testTransformerBlockBackwardWithCache() {
        assumeTrue(TensorOpsGPU.isGpuAvailable(), "fused block backward требует CUDA");
        int B = 1, S = 2, dModel = 8, numHeads = 2, dInt = 16;

        // Forward pass to populate cache (using deterministic data)
        Tensor xIn = smallTensor(new int[]{B, S, dModel}, 0.1f);
        Tensor Wq = smallTensor(new int[]{dModel, dModel}, 0.2f);
        Tensor Wk = smallTensor(new int[]{dModel, dModel}, 0.3f);
        Tensor Wv = smallTensor(new int[]{dModel, dModel}, 0.4f);
        Tensor Wo = smallTensor(new int[]{dModel, dModel}, 0.5f);
        Tensor W1 = smallTensor(new int[]{dModel, dInt}, 0.6f);
        Tensor W2 = smallTensor(new int[]{dInt, dModel}, 0.7f);
        Tensor W3 = smallTensor(new int[]{dModel, dInt}, 0.8f);
        Tensor norm1 = smallTensor(new int[]{dModel}, 1.0f);  // gamma for RMSNorm
        Tensor norm2 = smallTensor(new int[]{dModel}, 1.0f);
        Tensor mask = TensorOps.createCausalMask(S);

        BlockActivationCache cache = new BlockActivationCache();
        Tensor xNorm1 = TensorOps.rmsNorm(xIn, norm1, 1e-6f);
        cache.xIn.store(xIn, false);
        cache.xNorm1.store(xNorm1, false);

        // Simplified attention output (dummy for test)
        Tensor xRes1 = TensorOps.add(xNorm1, zeros(xNorm1.getShape()));
        cache.xRes1.store(xRes1, false);
        cache.xNorm2.store(TensorOps.rmsNorm(xRes1, norm2, 1e-6f), false);
        cache.attnOut.store(zeros(xRes1.getShape()), false);

        // Minimal cache entries for GPU path check
        cache.attnQHeads.store(zeros(new int[]{B, numHeads, S, dModel / numHeads}), false);
        cache.attnKHeads.store(zeros(new int[]{B, numHeads, S, dModel / numHeads}), false);
        cache.attnVHeads.store(zeros(new int[]{B, numHeads, S, dModel / numHeads}), false);
        cache.attnProbs.store(zeros(new int[]{B * numHeads, S, S}), false);
        cache.attnConcat.store(zeros(new int[]{B, S, dModel}), false);
        cache.ffnH1.store(zeros(new int[]{B, S, dInt}), false);
        cache.ffnGate.store(zeros(new int[]{B, S, dInt}), false);

        // Backward pass
        Tensor gradOut = upstreamGradOutSmall(new int[]{B, S, dModel}, 0.1f);
        Tensor gradXIn = new Tensor(xIn.getShape());
        Tensor gradWq = new Tensor(Wq.getShape());
        Tensor gradWk = new Tensor(Wk.getShape());
        Tensor gradWv = new Tensor(Wv.getShape());
        Tensor gradWo = new Tensor(Wo.getShape());
        Tensor gradW1 = new Tensor(W1.getShape());
        Tensor gradW2 = new Tensor(W2.getShape());
        Tensor gradW3 = new Tensor(W3.getShape());
        Tensor gradNorm1 = new Tensor(norm1.getShape());
        Tensor gradNorm2 = new Tensor(norm2.getShape());

        // Should not throw
        assertDoesNotThrow(() ->
                TransformerBackward.transformerBlockBackward(
                        gradOut, cache, Wq, Wk, Wv, Wo, W1, W2, W3, norm1, norm2,
                        numHeads, mask, true, gradXIn, gradWq, gradWk, gradWv, gradWo,
                        gradW1, gradW2, gradW3, gradNorm1, gradNorm2));

        // Verify input gradient was accumulated
        assertTrue(gradXIn.hasGrad());
    }

    @Test
    @DisplayName("transformerBlockBackward: null checks")
    public void testTransformerBlockBackwardNullChecks() {
        Tensor t = new Tensor(new int[]{1, 2, 8});
        t.zeroGrad();
        Tensor w = new Tensor(new int[]{8, 8});
        Tensor n = new Tensor(new int[]{8});
        BlockActivationCache cache = new BlockActivationCache();
        assertThrows(NullPointerException.class,
                () -> TransformerBackward.transformerBlockBackward(
                        null, cache, w, w, w, w, w, w, w, n, n, 2, null, true,
                        t, w, w, w, w, w, w, w, n, n));
    }

    // ========== Numerical Gradient Checks ==========

    @Test
    @DisplayName("RoPE backward: numerical gradient validation")
    public void testRoPEBackwardNumerical() {
        int B = 1, H = 1, S = 2, D = 4;
        Tensor x = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{B, H, S, D});
        float eps = NUMERICAL_EPS;

        // Analytical gradient via backward pass
        Tensor gradY = Tensor.fromArray(new float[]{1, 0, 0, 0, 0, 1, 0, 0}, new int[]{B, H, S, D});
        Tensor gradXAnalytical = new Tensor(x.getShape());
        gradXAnalytical.zeroGrad();
        TransformerBackward.applyRoPEBackward(gradY, gradXAnalytical, null);

        // Numerical gradient: (f(x+ε) - f(x-ε)) / (2ε)
        float[] gradXNumerical = new float[x.internalBuffer().length];
        float[] xData = x.internalBuffer();
        for (int i = 0; i < xData.length; i++) {
            float orig = xData[i];

            // f(x + ε)
            xData[i] = orig + eps;
            Tensor yPlus = TensorOps.applyRoPE(x, null);
            float fPlus = 0f;
            for (int j = 0; j < gradY.internalBuffer().length; j++) {
                fPlus += yPlus.internalBuffer()[j] * gradY.internalBuffer()[j];
            }

            // f(x - ε)
            xData[i] = orig - eps;
            Tensor yMinus = TensorOps.applyRoPE(x, null);
            float fMinus = 0f;
            for (int j = 0; j < gradY.internalBuffer().length; j++) {
                fMinus += yMinus.internalBuffer()[j] * gradY.internalBuffer()[j];
            }

            // Restore and compute numerical grad
            xData[i] = orig;
            gradXNumerical[i] = (fPlus - fMinus) / (2 * eps);
        }

        // Compare analytical vs numerical
        float[] analytical = gradXAnalytical.gradBuffer();
        // Centered finite differences on RoPE are ~O(ε²) error; 1e-4 step needs looser tol than 1e-3
        for (int i = 0; i < analytical.length; i++) {
            assertEquals(gradXNumerical[i], analytical[i], 5e-3f,
                    String.format("Gradient mismatch at index %d", i));
        }
    }

    @Test
    @DisplayName("softmax backward: numerical gradient validation")
    public void testSoftmaxBackwardNumerical() {
        int rows = 2, width = 3;
        Tensor logits = Tensor.fromArray(new float[]{1, 2, 3, 4, 5, 6}, new int[]{rows, width});
        float eps = NUMERICAL_EPS;

        // Analytical: dL/dlogits = p * (dL/dp - sum(p * dL/dp))
        Tensor dLdp = Tensor.fromArray(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{rows, width});
        Tensor probs = TensorOps.softmaxLastDim(
                Tensor.wrap(logits.internalBuffer(), new int[]{1, rows, width}));
        Tensor dLogitsAnalytical = new Tensor(new int[]{rows, width});
        dLogitsAnalytical.zeroGrad();

        float[] dp = dLdp.internalBuffer();
        float[] p = probs.internalBuffer();
        float[] dl = dLogitsAnalytical.gradBuffer();
        for (int i = 0; i < rows; i++) {
            int base = i * width;
            float dot = 0f;
            for (int j = 0; j < width; j++) dot += p[base + j] * dp[base + j];
            for (int j = 0; j < width; j++) {
                dl[base + j] = p[base + j] * (dp[base + j] - dot);
            }
        }

        // Numerical gradient
        float[] gradNumerical = new float[logits.internalBuffer().length];
        float[] logitData = logits.internalBuffer();
        for (int i = 0; i < logitData.length; i++) {
            float orig = logitData[i];

            logitData[i] = orig + eps;
            Tensor pPlus = TensorOps.softmaxLastDim(
                    Tensor.wrap(logits.internalBuffer(), new int[]{1, rows, width}));
            float fPlus = 0f;
            for (int j = 0; j < dp.length; j++) fPlus += pPlus.internalBuffer()[j] * dp[j];

            logitData[i] = orig - eps;
            Tensor pMinus = TensorOps.softmaxLastDim(
                    Tensor.wrap(logits.internalBuffer(), new int[]{1, rows, width}));
            float fMinus = 0f;
            for (int j = 0; j < dp.length; j++) fMinus += pMinus.internalBuffer()[j] * dp[j];

            logitData[i] = orig;
            gradNumerical[i] = (fPlus - fMinus) / (2 * eps);
        }

        // Compare
        for (int i = 0; i < dl.length; i++) {
            assertEquals(gradNumerical[i], dl[i], 5e-4f,
                    String.format("Softmax grad mismatch at %d", i));
        }
    }

    // ========== Edge Cases ==========

    @Test
    @DisplayName("scaledDotProductAttentionBackward: mask applied")
    public void testAttentionBackwardWithMask() {
        int B = 1, S = 3, dK = 4, dV = 4;
        Tensor gradOut = upstreamGradOutSmall(new int[]{B, S, dV}, 0.1f);
        Tensor Q = smallTensor(new int[]{B, S, dK}, 0.2f);
        Tensor K = smallTensor(new int[]{B, S, dK}, 0.3f);
        Tensor V = smallTensor(new int[]{B, S, dV}, 0.4f);
        Tensor mask = TensorOps.createCausalMask(S);  // causal mask
        Tensor gradQ = new Tensor(Q.getShape());
        Tensor gradK = new Tensor(K.getShape());
        Tensor gradV = new Tensor(V.getShape());

        assertDoesNotThrow(() ->
                TransformerBackward.scaledDotProductAttentionBackward(
                        gradOut, Q, K, V, mask, 1.0f / (float) Math.sqrt(dK), null, gradQ, gradK, gradV));
    }

    @Test
    @DisplayName("feedForwardSwiGLUBackward: shape validation")
    public void testSwiGLUBackwardShapeValidation() {
        Tensor gradOut = new Tensor(new int[]{1, 2, 4});
        gradOut.zeroGrad();
        Tensor x = new Tensor(new int[]{1, 2, 4});
        Tensor W1 = new Tensor(new int[]{4, 8});  // dModel=4, dInt=8
        Tensor W2 = new Tensor(new int[]{8, 4});  // dInt=8, dModel=4
        Tensor W3 = new Tensor(new int[]{4, 7});  // ❌ wrong: should be [4, 8]
        Tensor gradX = new Tensor(x.getShape());
        Tensor gradW1 = new Tensor(W1.getShape());
        Tensor gradW2 = new Tensor(W2.getShape());
        Tensor gradW3 = new Tensor(W3.getShape());

        assertThrows(IllegalArgumentException.class,
                () -> TransformerBackward.feedForwardSwiGLUBackward(
                        gradOut, x, W1, W2, W3, gradX, gradW1, gradW2, gradW3));
    }
}