#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Единый CUDA stream для memcpy и ядер TensorOpsGPU (cudaStreamNonBlocking).
 *
 * Инициализация: лениво в jgpt_cuda_ensure_stream(); потокобезопасна (mutex + double-check в jgpt_cuda.cu).
 * Не смешивать с cudaDeviceSynchronize для всего устройства — достаточно синхронизации этого stream при необходимости.
 *
 * Политика sync: (A) JNI с записью в jfloatArray после D2H — sync на stream до Release*ArrayElements; (B) цепочки
 * только с device pointers или Async H2D → ядро на этом stream — без промежуточного cudaStreamSynchronize (порядок на
 * stream достаточен); перед cudaFree временного device-буфера после такой цепочки отдельный sync не нужен (cudaFree ждёт
 * использование указателя). В jgpt_cuda.cu sync в основном это (A), чтение скаляра с GPU, cuBLAS→host, либо согласование
 * после blocking H2D с работой на kTensorCudaStream (GpuFloatBuffer / GpuIntBuffer). Граница для хоста —
 * TensorOpsGPU.synchronizeStream() (например граница шага в LLMTrainer), явный D2H (GpuFloatBuffer.copyTo / JNI (A)), или чекпоинт.
 */
extern cudaStream_t g_jgpt_cuda_stream;

/** Гарантирует создание g_jgpt_cuda_stream; безопасен при параллельных вызовах. */
void jgpt_cuda_ensure_stream(void);

/**
 * Уничтожает stream и обнуляет указатель.
 * Под mutex; перед destroy выполняется cudaStreamSynchronize для данного stream.
 */
void jgpt_cuda_destroy_stream(void);

/**
 * Кэшированный cudaDevAttrMaxThreadsPerBlock для устройства 0 (после первого успешного запроса).
 * Возвращает значение из прошивки GPU (часто 1024).
 */
int jgpt_cuda_max_threads_per_block(void);

/**
 * Рекомендуемый blockDim.x для типовых 1D-сеток: min(256, jgpt_cuda_max_threads_per_block()).
 * Один вызов на запуск ядра — дешевле, чем повторять логику в каждом месте.
 */
int jgpt_cuda_get_optimal_block_size(void);

/**
 * До {@code cudaStreamBeginCapture}: ленивый {@code cublasCreate} для дескриптора в {@code jgpt_cuda_extra.cu}
 * (FP16 GemmEx и др.). Иначе первый вызов внутри захвата графа даёт ошибки cuBLAS / CUDA.
 */
void jgpt_cuda_extra_warmup_cublas(void);

/**
 * Освобождает thread-local CUDA/cuBLAS ресурсы текущего потока (pinned staging, кэши matmul, extra-буферы).
 * Перед освобождением device-памяти синхронизирует g_jgpt_cuda_stream, если он не nullptr.
 *
 * Вызывать из того же потока, где выделялись ресурсы. Для JNI_OnUnload см. сочетание с jgpt_cuda_destroy_stream().
 */
void jgpt_cuda_cleanup_thread_resources(void);

#ifdef __cplusplus
}
#endif
