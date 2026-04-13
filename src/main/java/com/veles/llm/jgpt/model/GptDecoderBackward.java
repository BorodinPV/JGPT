package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.GpuFloatBuffer;
import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.ops.TensorOps;
import com.veles.llm.jgpt.ops.TransformerBackward;
import com.veles.llm.jgpt.util.DebugGpuTrain;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Обратный проход по стеку декодера на VRAM (ping-pong градиентов). */
final class GptDecoderBackward {

    private static final Logger log = LoggerFactory.getLogger(GptDecoderBackward.class);

    private GptDecoderBackward() {}

    static void backwardDecoderLayersDevice(
            GPTModel m, GpuFloatBuffer dGradBeforeNorm, int batch, int seqLen, boolean zeroParamGrads) {
        int flat = batch * seqLen * m.dModel;
        if (!m.deviceDecoderBackward || m.blockCachesDevice == null || !TensorOpsGPU.isGpuAvailable()) {
            throw new IllegalStateException(
                    "decoder backward on VRAM requires deviceDecoderBackward, CUDA, and device activation cache "
                            + "from forward(training=true); misconfiguration or missing GPU pipeline.");
        }
        m.decoderBwdGradPing = GPTModel.ensureGpuBuffer(m.decoderBwdGradPing, flat);
        m.decoderBwdGradPong = GPTModel.ensureGpuBuffer(m.decoderBwdGradPong, flat);

        GpuFloatBuffer gradCur = dGradBeforeNorm;
        GpuFloatBuffer gradNext = m.decoderBwdGradPing;

        for (int layer = m.numLayers - 1; layer >= 0; layer--) {
            DecoderBlock blk = m.blocks[layer];
            if (zeroParamGrads) {
                blk.zeroGradTensors();
            }
            TensorOps.GpuAttnResidentBuffers attnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[layer].attnBuffers() : null;
            TensorOps.GpuFfnResidentBuffers ffnResident =
                    m.gpuDecoderLayer != null ? m.gpuDecoderLayer[layer].ffnBuffers() : null;
            TransformerBackward.transformerBlockBackwardGpuDevice(
                    gradCur,
                    m.blockCachesDevice[layer],
                    blk.getWq(),
                    blk.getWk(),
                    blk.getWv(),
                    blk.getWo(),
                    blk.getW1(),
                    blk.getW2(),
                    blk.getW3(),
                    blk.getNorm1(),
                    blk.getNorm2(),
                    m.numHeads,
                    m.lastMask,
                    true,
                    gradNext,
                    blk.getWq(),
                    blk.getWk(),
                    blk.getWv(),
                    blk.getWo(),
                    blk.getW1(),
                    blk.getW2(),
                    blk.getW3(),
                    blk.getNorm1(),
                    blk.getNorm2(),
                    attnResident,
                    ffnResident);

            if (DebugGpuTrain.perLayerFiniteCheck()) {
                if (TensorOpsGPU.anyNonFiniteGpuDevice(gradNext, flat)) {
                    log.warn(
                            "Non-finite gradient after decoder layer {} ({} elements). "
                                    + "Enable JGPT_DEBUG_GPU_TRAIN=1 for JSONL diagnostics; "
                                    + "JGPT_BWD_LAYER_FINITE_CHECK=0 and no debug = one check after stack only.",
                            layer,
                            flat);
                    GptModelDebugLog.layerGrad(layer, flat);
                }
            }

            GpuFloatBuffer tmp = gradCur;
            gradCur = gradNext;
            gradNext = tmp;
        }

        if (!DebugGpuTrain.perLayerFiniteCheck()) {
            if (TensorOpsGPU.anyNonFiniteGpuDevice(gradCur, flat)) {
                log.warn(
                        "Non-finite gradient after decoder stack ({} elements). "
                                + "Enable JGPT_DEBUG_GPU_TRAIN=1 for JSONL or JGPT_BWD_LAYER_FINITE_CHECK=1 "
                                + "to localize by layer.",
                        flat);
                GptModelDebugLog.layerGrad(-1, flat);
            }
        }

        m.tokenEmbedding.backwardScatterFromDeviceGrad(m.lastInputTokens, gradCur, batch, seqLen);
        m.positionEmbedding.backwardAccumulateFromDeviceGrad(gradCur, batch, seqLen);
    }
}
