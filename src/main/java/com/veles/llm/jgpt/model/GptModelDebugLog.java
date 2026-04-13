package com.veles.llm.jgpt.model;

import com.veles.llm.jgpt.util.DebugGpuTrain;

/**
 * Отладочные JSON-строки для сессии b39372; не часть публичного API.
 */
final class GptModelDebugLog {

    private GptModelDebugLog() {}

    static void layerGrad(int layer, int flat) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H_layer_grad\",\"location\":\"GptDecoderBackward.backwardDecoderLayersDevice\",\"message\":\"nonfinite_grad_after_block\",\"data\":{\"layer\":"
                        + layer
                        + ",\"flat\":"
                        + flat
                        + "},\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }

    static void rmsBuf(
            String hypothesisId, String location, String message, float a, float b, float c, float d, long n) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\""
                        + hypothesisId
                        + "\",\"location\":\""
                        + location
                        + "\",\"message\":\""
                        + message
                        + "\",\"data\":{\"n\":"
                        + n
                        + ",\"s0\":"
                        + a
                        + ",\"s1\":"
                        + b
                        + ",\"s2\":"
                        + c
                        + ",\"s3\":"
                        + d
                        + "},\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }

    /**
     * mask: b0..b4 = bwdPing,bwdPong,chainPing,chainPong,lmFwdScratch; b5..b8 = lastHidden,xBeforeNorm,lastLogits,
     * lastLogitsGrad (1 if cleared).
     */
    static void scratchHandoff(int mask) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\"H_scratch_handoff\",\"location\":\"GPTModel.clearGpuResidentDecoderScratchForTrainHandoff\",\"message\":\"cleared_resident_scratch\",\"data\":{\"mask\":"
                        + mask
                        + "},\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }
}
