package com.veles.llm.jgpt.training;

import static org.junit.jupiter.api.Assertions.assertThrows;

import com.veles.llm.jgpt.TensorOpsGPU;
import com.veles.llm.jgpt.model.GPTModel;

import org.junit.jupiter.api.Test;

/** Прямые проверки {@link GPTModel#setDeviceDecoderBackward(boolean)}. */
class DeviceDecoderBackwardModelTest {

    @Test
    void setDeviceDecoderBackward_withoutDeviceLogits_throws() {
        if (!TensorOpsGPU.isGpuAvailable()) {
            return;
        }
        String prevPipe = System.getProperty("jgpt.decoder.gpu.pipeline");
        try {
            System.setProperty("jgpt.decoder.gpu.pipeline", "true");
            GPTModel model = new GPTModel(48, 8, 32, 4, 2, 64, true);
            try {
                if (!model.canFullGpuTrain()) {
                    return;
                }
                assertThrows(IllegalStateException.class, () -> model.setDeviceDecoderBackward(true));
            } finally {
                model.closeGpuResidentWeights();
            }
        } finally {
            if (prevPipe == null) {
                System.clearProperty("jgpt.decoder.gpu.pipeline");
            } else {
                System.setProperty("jgpt.decoder.gpu.pipeline", prevPipe);
            }
        }
    }
}
