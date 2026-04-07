#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Единый CUDA stream для memcpy и ядер TensorOpsGPU (cudaStreamNonBlocking).
 *
 * Инициализация: лениво в jgpt_cuda_ensure_stream(); потокобезопасна (mutex + double-check + std::atomic в jgpt_cuda.cu).
 * Указатель читать только через jgpt_cuda_stream_handle() или макрос kTensorCudaStream (релиз store — только в jgpt_cuda.cu).
 * Не смешивать с cudaDeviceSynchronize для всего устройства — достаточно синхронизации этого stream при необходимости.
 *
 * Политика sync: (A) JNI с записью в jfloatArray после D2H — sync на stream до Release*ArrayElements (при ошибке CUDA до Release — JNI_ABORT); (B) цепочки
 * только с device pointers или Async H2D → ядро на этом stream — без промежуточного cudaStreamSynchronize (порядок на
 * stream достаточен); перед cudaFree временного device-буфера после такой цепочки отдельный sync не нужен (cudaFree ждёт
 * использование указателя). В jgpt_cuda.cu sync в основном это (A), чтение скаляра с GPU, cuBLAS→host, либо согласование
 * после blocking H2D с работой на kTensorCudaStream (GpuFloatBuffer / GpuIntBuffer). Граница для хоста —
 * TensorOpsGPU.synchronizeStream() (например граница шага в LLMTrainer), явный D2H (GpuFloatBuffer.copyTo / JNI (A)), или чекпоинт.
 */
/** Текущий handle единого stream (acquire); до jgpt_cuda_ensure_stream может быть nullptr. */
cudaStream_t jgpt_cuda_stream_handle(void);

/** Гарантирует создание stream TensorOpsGPU; безопасен при параллельных вызовах. */
void jgpt_cuda_ensure_stream(void);

/**
 * {@code cudaStreamSynchronize(jgpt_cuda_stream_handle())} для границы перед чтением результата на CPU.
 * Если на stream активен {@code cudaStreamBeginCapture}, синхронизация не выполняется (иначе
 * cudaErrorStreamCaptureUnsupported / «operation not permitted when stream is capturing»).
 *
 * @return 1 при успехе или при активном capture; 0 при ошибке CUDA.
 */
int jgpt_cuda_sync_stream_unless_capturing(const char* ctx);

/**
 * Если на stream TensorOpsGPU активен захват графа — {@code cudaStreamEndCapture} и {@code cudaGraphDestroy}
 * (частичный граф отбрасывается). Нужно перед D2H в JNI и перед {@code cudaStreamSynchronize}, когда Java могла
 * вызвать копирование на хост при ещё незавершённом {@code cudaStreamBeginCapture} (shutdown / checkpoint).
 */
void jgpt_cuda_abort_stream_capture_discard_graph(void);

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
 * Перед освобождением thread-local device-памяти синхронизирует общий stream (если не nullptr), чтобы не освободить буферы
 * при ещё незавершённой работе на нём (один stream на процесс — ожидание может включать работу других потоков на этом stream).
 *
 * Вызывать из того же потока, где выделялись ресурсы. Для JNI_OnUnload см. сочетание с jgpt_cuda_destroy_stream().
 */
void jgpt_cuda_cleanup_thread_resources(void);

#ifdef __cplusplus
}
/** Stream для launch/async API; каждый раз читает актуальный handle (см. jgpt_cuda_stream_handle). */
#define kTensorCudaStream (jgpt_cuda_stream_handle())
#endif
