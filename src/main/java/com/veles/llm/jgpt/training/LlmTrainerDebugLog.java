package com.veles.llm.jgpt.training;

import com.veles.llm.jgpt.util.DebugGpuTrain;

/** JSON-строки для сессии b39372; не часть публичного API. */
final class LlmTrainerDebugLog {

    private LlmTrainerDebugLog() {}

    static String jsonEsc(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    static void b39372(String hypothesisId, String location, String message, String dataJson) {
        DebugGpuTrain.appendJsonLine(
                "{\"sessionId\":\"b39372\",\"hypothesisId\":\""
                        + hypothesisId
                        + "\",\"location\":\""
                        + location
                        + "\",\"message\":\""
                        + message
                        + "\",\"data\":"
                        + dataJson
                        + ",\"timestamp\":"
                        + System.currentTimeMillis()
                        + "}");
    }
}
