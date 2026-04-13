#pragma once

#include <jni.h>

/** Один float в массиве loss (CE): записать 0; безопасно при null. */
static inline void jgpt_jni_loss_out_zero(JNIEnv* env, jfloatArray loss_out) {
    if (loss_out == nullptr) {
        return;
    }
    jfloat* p = env->GetFloatArrayElements(loss_out, nullptr);
    if (p != nullptr) {
        p[0] = 0.f;
        env->ReleaseFloatArrayElements(loss_out, p, 0);
    }
}
