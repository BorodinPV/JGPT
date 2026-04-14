#pragma once

#include <jni.h>
#include "jgpt_cuda_jni_raii.cuh"

/** Один float в массиве loss (CE): записать 0; безопасно при null. */
static inline void jgpt_jni_loss_out_zero(JNIEnv* env, jfloatArray loss_out) {
    if (loss_out == nullptr) {
        return;
    }
    JniFloatArrayScope scope(env, loss_out, 0);
    if (scope) {
        scope.ptr[0] = 0.f;
    }
}
