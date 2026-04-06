#!/usr/bin/env bash
# Общая среда и запуск Maven для GPU-обучения (AllBooks / MultiBook / TrainLLM / Profile).
# Подключать: source "$(dirname "$0")/jgpt-gpu-train-lib.sh" из scripts/ или
#   source "$ROOT/scripts/jgpt-gpu-train-lib.sh" после cd "$ROOT".
# Не использовать exec внутри этого файла — только функции и export.

_jgpt_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="$(cd "${_jgpt_lib_dir}/.." && pwd)"
cd "$ROOT" || return 2
unset _jgpt_lib_dir

jgpt__maven_opts() {
    if [[ "${MAVEN_OPTS:-}" != *enable-native-access* ]]; then
        export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED ${MAVEN_OPTS:-}"
    fi
    if [[ -n "${JGPT_JAVA_MEM:-}" ]]; then
        export MAVEN_OPTS="${JGPT_JAVA_MEM} ${MAVEN_OPTS:-}"
    fi
}

# Экспорты JGPT_* как в прежнем run-training-gpu.sh (дефолты для обучения).
jgpt__export_train_env() {
    export JGPT_ACTIVATION_CACHE_FP16="${JGPT_ACTIVATION_CACHE_FP16:-1}"

    export JGPT_BATCH_DIRECT="${JGPT_BATCH_DIRECT:-1}"
    export JGPT_BATCH_PINNED="${JGPT_BATCH_PINNED:-1}"
    export JGPT_BATCH_PREFETCH="${JGPT_BATCH_PREFETCH:-1}"
    export JGPT_BATCH_SIZE="${JGPT_BATCH_SIZE:-}"

    export JGPT_BLOCK_CACHE_GROW_ONLY="${JGPT_BLOCK_CACHE_GROW_ONLY:-1}"
    export JGPT_BLOCK_CACHE_MAX_BYTES="${JGPT_BLOCK_CACHE_MAX_BYTES:-0}"
    export JGPT_BLOCK_CACHE_POOL="${JGPT_BLOCK_CACHE_POOL:-1}"
    export JGPT_BLOCK_CACHE_POOL_MAX="${JGPT_BLOCK_CACHE_POOL_MAX:-2}"

    export JGPT_CE_GPU_MIN_ELEMENTS="${JGPT_CE_GPU_MIN_ELEMENTS:-0}"
    export JGPT_CE_ASYNC="${JGPT_CE_ASYNC:-1}"

    export JGPT_TRAIN_LOSS_MODE="${JGPT_TRAIN_LOSS_MODE:-full}"
    export JGPT_SAMPLED_CE_CANDIDATES="${JGPT_SAMPLED_CE_CANDIDATES:-128}"
    export JGPT_SAMPLED_CE_NEGATIVE_MODE="${JGPT_SAMPLED_CE_NEGATIVE_MODE:-batch_shared_uniform}"

    export JGPT_CHECKPOINT_ASYNC="${JGPT_CHECKPOINT_ASYNC:-0}"

    export JGPT_CUDA_LIB="${JGPT_CUDA_LIB:-}"
    if [[ -z "$JGPT_CUDA_LIB" && -f "$ROOT/build/libjgpt_cuda.so" ]]; then
        export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
    fi

    export JGPT_DECODER_GPU_PIPELINE="${JGPT_DECODER_GPU_PIPELINE:-1}"
    export JGPT_DEVICE_DECODER_BWD="${JGPT_DEVICE_DECODER_BWD:-1}"
    export JGPT_DEVICE_LOGITS_TRAIN="${JGPT_DEVICE_LOGITS_TRAIN:-1}"
    export JGPT_DECODER_LAYER_CUDA_GRAPH="${JGPT_DECODER_LAYER_CUDA_GRAPH:-1}"

    export JGPT_EXIT_AFTER_STEP="${JGPT_EXIT_AFTER_STEP:-0}"

    export JGPT_FP16_MATMUL="${JGPT_FP16_MATMUL:-1}"
    export JGPT_FP16_DYNAMIC_GROWTH_INTERVAL="${JGPT_FP16_DYNAMIC_GROWTH_INTERVAL:-2000}"
    export JGPT_FP16_DYNAMIC_INITIAL="${JGPT_FP16_DYNAMIC_INITIAL:-65536}"
    export JGPT_FP16_DYNAMIC_MAX="${JGPT_FP16_DYNAMIC_MAX:-65536}"

    export JGPT_FLASH_ATTENTION="${JGPT_FLASH_ATTENTION:-0}"

    export JGPT_FUSED_LM_HEAD="${JGPT_FUSED_LM_HEAD:-1}"

    export JGPT_FULL_GPU_TRAIN="${JGPT_FULL_GPU_TRAIN:-0}"
    export JGPT_GENERATE_GPU_KV="${JGPT_GENERATE_GPU_KV:-1}"
    export JGPT_GPU_E2E_TRAIN="${JGPT_GPU_E2E_TRAIN:-1}"

    export JGPT_LOG_COLOR="${JGPT_LOG_COLOR:-}"
    export JGPT_MAX_SEQUENCES="${JGPT_MAX_SEQUENCES:-}"

    export JGPT_PROFILE="${JGPT_PROFILE:-0}"
    export JGPT_PROFILE_STEPS="${JGPT_PROFILE_STEPS:-20}"
    export JGPT_TIMINGS="${JGPT_TIMINGS:-0}"
    # JGPT_TRAIN_PERF здесь не задаём: дефолт 0 ломал jgpt_e2e_train_overrides — после export 0
    # подстановка ${JGPT_TRAIN_PERF:-1} не срабатывала (0 считается заданным), [PERF] в e2e/smart пропадал.
    export JGPT_TRAIN_GPU_RESIDENT="${JGPT_TRAIN_GPU_RESIDENT:-1}"
}

# Дефолты сценария «e2e» (как в прежнем train-e2e-gpu.sh); вызывать после jgpt__export_train_env.
jgpt_e2e_train_overrides() {
    # Полные строки [PERF] (как JGPT_PROFILE=1 + JGPT_TIMINGS=1), если пользователь не задал иное.
    export JGPT_TRAIN_PERF="${JGPT_TRAIN_PERF:-1}"
    export JGPT_TRAIN_LOSS_MODE="${JGPT_TRAIN_LOSS_MODE:-full}"
    export JGPT_SAMPLED_CE_CANDIDATES="${JGPT_SAMPLED_CE_CANDIDATES:-128}"
    export JGPT_SAMPLED_CE_NEGATIVE_MODE="${JGPT_SAMPLED_CE_NEGATIVE_MODE:-batch_shared_uniform}"
    export JGPT_DECODER_LAYER_CUDA_GRAPH="${JGPT_DECODER_LAYER_CUDA_GRAPH:-1}"
    export JGPT_FUSED_FFN_RMS_W1W3="${JGPT_FUSED_FFN_RMS_W1W3:-1}"
    export JGPT_FP16_DYNAMIC_INITIAL="${JGPT_FP16_DYNAMIC_INITIAL:-8192}"
    export JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK="${JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK:-256}"
    export JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH="${JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH:-0}"
}

jgpt_cmake_build_cuda() {
    cmake -B build -S src/main/cpp
    cmake --build build
    export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
}

# Снять префиксы e2e и разрешить режим (single|train|allbooks|profile|по умолчанию MultiBook).
# Результат: массив JGPT_EXEC_CMD — вызывающий: exec "${JGPT_EXEC_CMD[@]}".
jgpt_resolve_mvn_command() {
    while [[ "${1:-}" == e2e ]]; do
        shift
        export JGPT_GPU_E2E_TRAIN=1
        export JGPT_FULL_GPU_TRAIN=1
    done

    local -a cmd=()

    case "${1:-}" in
        single|train)
            shift
            if [[ $# -gt 0 ]]; then
                cmd=(
                    mvn -q compile exec:java
                    -Dexec.mainClass=com.veles.llm.jgpt.app.TrainLLM
                    "-Dexec.args=$*"
                )
            else
                cmd=(mvn -q compile exec:java -Dexec.mainClass=com.veles.llm.jgpt.app.TrainLLM)
            fi
            ;;
        allbooks)
            shift
            if [[ $# -gt 0 ]]; then
                cmd=(
                    mvn -q compile exec:java
                    -Dexec.mainClass=com.veles.llm.jgpt.app.AllBooksTrain
                    "-Dexec.args=$*"
                )
            else
                cmd=(
                    mvn -q compile exec:java
                    -Dexec.mainClass=com.veles.llm.jgpt.app.AllBooksTrain
                    '-Dexec.args=--boo .'
                )
            fi
            ;;
        profile)
            shift
            if [[ $# -gt 0 ]]; then
                cmd=(
                    mvn -q compile exec:java
                    -Dexec.mainClass=com.veles.llm.jgpt.app.ProfileQuickRun
                    "-Dexec.args=$*"
                )
            else
                cmd=(mvn -q compile exec:java -Dexec.mainClass=com.veles.llm.jgpt.app.ProfileQuickRun)
            fi
            ;;
    esac

    if [[ ${#cmd[@]} -eq 0 ]]; then
        cmd=(
            mvn -q compile exec:java
            -Dexec.mainClass=com.veles.llm.jgpt.app.MultiBookTrain
            '-Dexec.args=--boo .'
        )
    fi

    JGPT_EXEC_CMD=("${cmd[@]}")
}
