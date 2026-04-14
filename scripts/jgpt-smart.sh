#!/usr/bin/env bash
# =============================================================
# jgpt-smart.sh — Адаптивное обучение с авто-переключением пресетов
#
# Запускает обучение, следит за логом и автоматически:
#   - понижает пресет при OOM / зависании FP16
#   - повышает пресет обратно после стабильной работы
#   - после штатного завершения AllBooksTrain переходит к следующему пресету по кругу
#     (цикл не завершается — остановка: Ctrl+C в этом скрипте)
#   - делает resume после каждого переключения
#
# Иерархия пресетов (от быстрого к безопасному), по кругу:
#   00 → 01 → 02 → 03 → 04 → 00 → …
#
# Использование:
#   ./scripts/jgpt-smart.sh                    # с текущего пресета
#   ./scripts/jgpt-smart.sh 01-aggressive      # явный стартовый пресет
#   Ctrl+C — остановить (checkpoint сохраняется через shutdown hook)
#
# Сборка CUDA + env + Maven: функции jgpt__* / jgpt_* ниже в этом файле.
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ─── Maven / CUDA / env (раньше jgpt-gpu-train-lib.sh) ───────

jgpt__maven_opts() {
    if [[ "${MAVEN_OPTS:-}" != *enable-native-access* ]]; then
        export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED ${MAVEN_OPTS:-}"
    fi
    if [[ -n "${JGPT_JAVA_MEM:-}" ]]; then
        export MAVEN_OPTS="${JGPT_JAVA_MEM} ${MAVEN_OPTS:-}"
    fi
}

jgpt__export_train_env() {
    export JGPT_ACTIVATION_CACHE_FP16="${JGPT_ACTIVATION_CACHE_FP16:-1}"

    export JGPT_BATCH_DIRECT="${JGPT_BATCH_DIRECT:-1}"
    export JGPT_BATCH_PINNED="${JGPT_BATCH_PINNED:-1}"
    export JGPT_BATCH_PREFETCH="${JGPT_BATCH_PREFETCH:-1}"
    export JGPT_BATCH_SIZE="${JGPT_BATCH_SIZE:-}"

    export JGPT_BLOCK_CACHE_GROW_ONLY="${JGPT_BLOCK_CACHE_GROW_ONLY:-1}"
    export JGPT_BLOCK_CACHE_MAX_BYTES="${JGPT_BLOCK_CACHE_MAX_BYTES:-0}"
    export JGPT_BLOCK_CACHE_POOL="${JGPT_BLOCK_CACHE_POOL:-1}"
    export JGPT_BLOCK_CACHE_POOL_MAX="${JGPT_BLOCK_CACHE_POOL_MAX:-4}"

    export JGPT_CE_GPU_MIN_ELEMENTS="${JGPT_CE_GPU_MIN_ELEMENTS:-0}"
    export JGPT_CE_ASYNC="${JGPT_CE_ASYNC:-1}"

    export JGPT_TRAIN_LOSS_MODE="${JGPT_TRAIN_LOSS_MODE:-full}"
    export JGPT_SAMPLED_CE_CANDIDATES="${JGPT_SAMPLED_CE_CANDIDATES:-128}"
    export JGPT_SAMPLED_CE_NEGATIVE_MODE="${JGPT_SAMPLED_CE_NEGATIVE_MODE:-batch_shared_uniform}"

    export JGPT_CHECKPOINT_ASYNC="${JGPT_CHECKPOINT_ASYNC:-0}"

    export JGPT_CUDA_LIB="${JGPT_CUDA_LIB:-}"
    export JGPT_CUDA_TRIM_EVERY_STEPS="${JGPT_CUDA_TRIM_EVERY_STEPS:-500}"
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
    export JGPT_TRAIN_GPU_RESIDENT="${JGPT_TRAIN_GPU_RESIDENT:-1}"
}

jgpt_e2e_train_overrides() {
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

# Абсолютный путь к nvcc для CMake (кэш часто содержит /usr/bin/nvcc без пакета nvidia-cuda-toolkit).
jgpt__resolve_nvcc() {
    local p cand
    if [[ -n "${CUDACXX:-}" && -x "${CUDACXX}" ]]; then
        readlink -f "${CUDACXX}" 2>/dev/null || echo "${CUDACXX}"
        return 0
    fi
    p="$(command -v nvcc 2>/dev/null || true)"
    if [[ -n "$p" && -x "$p" ]]; then
        readlink -f "$p" 2>/dev/null || echo "$p"
        return 0
    fi
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        readlink -f /usr/local/cuda/bin/nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc
        return 0
    fi
    shopt -s nullglob
    local -a vers=(/usr/local/cuda-*/bin/nvcc)
    shopt -u nullglob
    if [[ ${#vers[@]} -gt 0 ]]; then
        while IFS= read -r cand; do
            [[ -x "$cand" ]] || continue
            readlink -f "$cand" 2>/dev/null || echo "$cand"
            return 0
        done < <(printf '%s\n' "${vers[@]}" | sort -Vr)
    fi
    if [[ -x /opt/cuda/bin/nvcc ]]; then
        readlink -f /opt/cuda/bin/nvcc 2>/dev/null || echo /opt/cuda/bin/nvcc
        return 0
    fi
    echo "[jgpt-smart] nvcc не найден. Установите CUDA Toolkit или задайте CUDACXX=/полный/путь/к/nvcc" >&2
    return 1
}

# g++ для хост-кода nvcc: CUDA 12.6 официально только до GCC 13; GCC 14+ без g++-13 ломает cudafe++.
jgpt__resolve_cudahostcxx() {
    local p maj
    if [[ -n "${CUDAHOSTCXX:-}" && -x "${CUDAHOSTCXX}" ]]; then
        readlink -f "${CUDAHOSTCXX}" 2>/dev/null || echo "${CUDAHOSTCXX}"
        return 0
    fi
    for p in \
        /usr/bin/g++-13 /usr/bin/x86_64-linux-gnu-g++-13 \
        /usr/bin/g++-12 /usr/bin/x86_64-linux-gnu-g++-12 \
        /usr/bin/g++-11 /usr/bin/x86_64-linux-gnu-g++-11
    do
        [[ -x "$p" ]] || continue
        readlink -f "$p" 2>/dev/null || echo "$p"
        return 0
    done
    maj="$(gcc -dumpversion 2>/dev/null | cut -d. -f1)"
    maj="${maj//[^0-9]/}"
    maj="${maj:-99}"
    if [[ "$maj" -gt 13 ]]; then
        echo "[jgpt-smart] Системный GCC ${maj} не поддерживается nvcc 12.x как хост-компилятор." >&2
        echo "[jgpt-smart] Установите: sudo apt install g++-13" >&2
        echo "[jgpt-smart] Или задайте CUDAHOSTCXX=/usr/bin/g++-13 (после установки пакета)." >&2
        return 1
    fi
    return 0
}

# Патч одного crt/math_functions.h (glibc 2.41+ / noexcept); файл должен быть доступен на запись.
jgpt__patch_cuda_math_functions_h_file() {
    perl -i -pe '
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+rsqrt\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+rsqrtf\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+sinpi\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+sinpif\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+cospi\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+cospif\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x) noexcept (true);/;
' "$1"
}

# glibc 2.41+: nvcc иначе тянет системный crt/math_functions.h (нужен sudo для правки в /usr/local).
# Копируем targets/.../include в build/cuda_include_mirror и подставляем -I… раньше пути toolkit (см. nvcc -v).
jgpt__resolve_cuda_include_mirror_if_needed() {
    local nvcc_real inc_root mirror meta new_ts sys_mf
    nvcc_real="$(readlink -f "$1")"
    inc_root="$(readlink -f "$(dirname "$nvcc_real")/../targets/x86_64-linux/include")"
    if [[ ! -d "$inc_root" ]]; then
        echo "[jgpt-smart] Нет каталога заголовков CUDA: $inc_root" >&2
        return 1
    fi
    sys_mf="$inc_root/crt/math_functions.h"
    if grep -q 'cospi(double x) noexcept' "$sys_mf" 2>/dev/null; then
        return 0
    fi
    mirror="$ROOT/build/cuda_include_mirror"
    meta="$mirror/.jgpt_cuda_mirror_meta"
    new_ts="$(stat -c '%Y' "$inc_root/cuda_runtime.h" 2>/dev/null || echo 0)"
    if [[ -f "$mirror/crt/math_functions.h" ]] \
        && grep -q 'cospi(double x) noexcept' "$mirror/crt/math_functions.h" 2>/dev/null \
        && [[ -f "$meta" ]] && [[ "$(<"$meta")" == "${inc_root}|${new_ts}" ]]; then
        echo "$mirror"
        return 0
    fi
    echo "[jgpt-smart] Копирую заголовки CUDA в build/cuda_include_mirror (обход glibc без sudo)…" >&2
    rm -rf "$mirror"
    mkdir -p "$ROOT/build"
    mkdir -p "$mirror"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$inc_root/" "$mirror/"
    else
        cp -a "$inc_root/." "$mirror/"
    fi
    jgpt__patch_cuda_math_functions_h_file "$mirror/crt/math_functions.h" || return 1
    if ! grep -q 'cospi(double x) noexcept' "$mirror/crt/math_functions.h"; then
        echo "[jgpt-smart] Патч math_functions.h не применился (другая версия CUDA?)." >&2
        return 1
    fi
    printf '%s|%s\n' "$inc_root" "$new_ts" > "$meta"
    echo "$mirror"
    return 0
}

jgpt_cmake_build_cuda() {
    local nvcc_path cuda_bin hostcxx cuda_mirror
    local -a cmake_args
    nvcc_path="$(jgpt__resolve_nvcc)" || return 1
    export CUDACXX="$nvcc_path"
    cuda_mirror="$(jgpt__resolve_cuda_include_mirror_if_needed "$nvcc_path")" || return 1
    cuda_bin="$(dirname "$nvcc_path")"
    case ":${PATH}:" in
        *:"${cuda_bin}":*) ;;
        *) export PATH="${cuda_bin}:${PATH}" ;;
    esac
    cmake_args=(-DCMAKE_CUDA_COMPILER="$nvcc_path")
    hostcxx="$(jgpt__resolve_cudahostcxx)" || return 1
    if [[ -n "$hostcxx" ]]; then
        export CUDAHOSTCXX="$hostcxx"
        cmake_args+=(-DCMAKE_CUDA_HOST_COMPILER="$hostcxx")
    fi
    # Убрать устаревший флаг из кэша (например -allow-unsupported-compiler при уже выбранном g++-13).
    # Путь к зеркалу с пробелами/кириллицей ломает CMAKE_CUDA_FLAGS: CMake режет по пробелу → nvcc fatal.
    # Симлинк в /tmp без пробелов — стабильный -I и для try_compile (определение компилятора CUDA).
    if [[ -n "$cuda_mirror" ]]; then
        local cuda_inc_link h
        h="$(printf '%s' "$cuda_mirror" | cksum | awk '{print $1}')"
        cuda_inc_link="/tmp/jgpt-cuda-include-${UID:-0}-${h}"
        ln -sfn "$cuda_mirror" "$cuda_inc_link"
        cmake_args+=("-DCMAKE_CUDA_FLAGS=-I${cuda_inc_link}")
    fi
    cmake -B build -U CMAKE_CUDA_FLAGS -S src/main/cpp "${cmake_args[@]}"
    cmake --build build
    export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
}

jgpt_resolve_mvn_command() {
    while [[ "${1:-}" == e2e ]]; do
        shift
        export JGPT_GPU_E2E_TRAIN=1
        export JGPT_FULL_GPU_TRAIN=1
    done
    # Совместимость: старый вызов «… e2e allbooks …» — слово allbooks игнорируем.
    while [[ "${1:-}" == allbooks ]]; do
        shift
    done

    # Каталог с .txt для AllBooksTrain (рекурсивно). Пример: JGPT_DATA_DIR=data/books/libru_txt
    local train_args="--boo ."
    if [[ -n "${JGPT_DATA_DIR:-}" ]]; then
        train_args+=" --data-dir ${JGPT_DATA_DIR}"
    fi

    local -a cmd
    if [[ $# -gt 0 ]]; then
        local user_args="$*"
        if [[ -n "${JGPT_DATA_DIR:-}" ]]; then
            user_args+=" --data-dir ${JGPT_DATA_DIR}"
        fi
        cmd=(
            mvn -q compile exec:java
            -Dexec.mainClass=com.veles.llm.jgpt.app.AllBooksTrain
            "-Dexec.args=${user_args}"
        )
    else
        cmd=(
            mvn -q compile exec:java
            -Dexec.mainClass=com.veles.llm.jgpt.app.AllBooksTrain
            "-Dexec.args=${train_args}"
        )
    fi
    JGPT_EXEC_CMD=("${cmd[@]}")
}

# ─── Параметры smart-монитора ─────────────────────────────────
PRESETS=("00-max-throughput" "01-aggressive" "02-stable" "03-recovery" "04-minimal")
# Порог: сколько раз за сегмент eval зафиксировал НОВЫЙ (строго лучший) лучший loss — только если JGPT_SMART_UPGRADE=1
STABLE_EVALS_FOR_UPGRADE=30
# Авто-upgrade к более быстрому пресету (idx−1). По умолчанию выкл.: старый счётчик считал почти все строки [EVAL]
# и давал качели только между 00 и 01 (с 00 upgrade не делается, с 01 через ~30 eval снова 00).
JGPT_SMART_UPGRADE="${JGPT_SMART_UPGRADE:-0}"
# Дублировать сообщения [SMART] в training_allbooks.log (раньше они были только в терминале).
JGPT_SMART_LOG_TO_FILE="${JGPT_SMART_LOG_TO_FILE:-1}"
# Если 1 — не переключать пресет автоматически (перезапускать тот же).
# Можно задавать в env/<preset>.env, напр. в 02-stable.env.
JGPT_SMART_STICKY_PRESET="${JGPT_SMART_STICKY_PRESET:-0}"
OOM_THRESHOLD=1
FP16_STUCK_THRESHOLD=8
HANG_SECONDS="${HANG_SECONDS:-900}"
# Подряд eval без улучшения best — не раньше PLATEAU_MIN_STEP_LINES строк [STEP] в сегменте (после длинной загрузки данных).
PLATEAU_THRESHOLD=35
PLATEAU_MIN_STEP_LINES="${PLATEAU_MIN_STEP_LINES:-5}"
MONITOR_INTERVAL=30

STATE_DIR="$ROOT/state"
LOG_FILE="$ROOT/training_allbooks.log"
PID_FILE="$STATE_DIR/training.pid"
PRESET_FILE="$STATE_DIR/current_preset_idx"

mkdir -p "$STATE_DIR"

# ─── Разбор аргументов ────────────────────────────────────────
START_PRESET=""
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            echo "Использование: $0 [ПРЕСЕТ]"
            echo "Пресеты: ${PRESETS[*]}"
            echo "После штатного завершения обучения пресет переключается по кругу (цикл бесконечный)."
            echo "Остановить: Ctrl+C в этом терминале."
            echo "JGPT_SMART_UPGRADE=1 — включить авто-upgrade к более быстрому пресету (порог: STABLE_EVALS_FOR_UPGRADE)."
            echo "JGPT_SMART_LOG_TO_FILE=0 — не писать [SMART] в training_allbooks.log."
            echo "JGPT_DATA_DIR=<каталог> — AllBooksTrain: откуда брать .txt (рекурсивно), эквивалент --data-dir."
            echo "JGPT_IF_STEP_BEYOND_PLAN — по умолчанию restart_schedule (см. README); для чистого skip: export перед запуском."
            echo "PLATEAU_MIN_STEP_LINES — мин. число строк [STEP] в сегменте до проверки плато (по умолчанию $PLATEAU_MIN_STEP_LINES)."
            exit 0
            ;;
        --*) ;;
        *) START_PRESET="$arg" ;;
    esac
done

preset_index() {
    local name="$1"
    local i
    for i in "${!PRESETS[@]}"; do
        if [[ "${PRESETS[$i]}" == "$name" ]]; then
            echo "$i"
            return 0
        fi
    done
    echo "-1"
    return 0
}

# Допустимый индекс пресета [0, n) или сообщение в stderr и fallback.
clamp_preset_idx() {
    local idx="$1"
    local max=$(( ${#PRESETS[@]} - 1 ))
    if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
        echo "  [SMART] [WARN] Некорректный индекс пресета «${idx}», используем 01-aggressive (1)." >&2
        echo 1
        return
    fi
    if (( idx < 0 || idx > max )); then
        echo "  [SMART] [WARN] Индекс пресета $idx вне [0,$max], используем 1." >&2
        echo 1
        return
    fi
    echo "$idx"
}

# Начальный пресет
if [[ -n "$START_PRESET" ]]; then
    CURRENT_IDX=$(preset_index "$START_PRESET")
    if [[ "$CURRENT_IDX" -lt 0 ]]; then
        echo "[ERROR] Неизвестный пресет: $START_PRESET"
        exit 1
    fi
elif [[ -f "$PRESET_FILE" ]]; then
    CURRENT_IDX=$(clamp_preset_idx "$(tr -d ' \t\r\n' < "$PRESET_FILE")")
elif [[ -L "$STATE_DIR/current.env" ]]; then
    LINKED=$(basename "$(readlink "$STATE_DIR/current.env")" .env)
    CURRENT_IDX=$(preset_index "$LINKED")
    if [[ "$CURRENT_IDX" -lt 0 ]]; then
        CURRENT_IDX=1
    fi
else
    CURRENT_IDX=1
fi
CURRENT_IDX=$(clamp_preset_idx "$CURRENT_IDX")

# ─── Вспомогательные функции smart ───────────────────────────

apply_preset() {
    local idx="$1"
    idx=$(clamp_preset_idx "$idx")
    local name="${PRESETS[$idx]}"
    local env_file="$ROOT/env/${name}.env"
    if [[ ! -f "$env_file" ]]; then
        echo "  [SMART] [ERROR] Нет файла пресета: $env_file"
        exit 1
    fi
    ln -sf "../env/${name}.env" "$STATE_DIR/current.env"
    echo "$idx" > "$PRESET_FILE"
    # source только в текущем shell, не в $(...).
    set -a
    # shellcheck source=/dev/null
    source "$env_file"
    set +a
    APPLIED_PRESET_NAME="$name"
}

count_pattern_from_line() {
    local file="$1" pattern="$2" start_line="$3"
    awk -v start="$start_line" -v pat="$pattern" \
        'NR >= start && $0 ~ pat { count++ } END { print count+0 }' "$file" 2>/dev/null || echo 0
}

# Максимум подряд eval без улучшения best_loss (в сегменте с start_line).
count_plateau_evals() {
    local file="$1" start_line="$2"
    awk -v start="$start_line" '
        NR < start { next }
        /\[EVAL\].*: loss=.*/ {
            n1 = split($0, a, /: loss=/)
            if (n1 < 2) next
            cur = a[2]; sub(/ .*/, "", cur)
            n2 = split($0, b, /лучший сохранённый=/)
            if (n2 < 2) next
            best = b[2]; sub(/[^0-9.].*/, "", best)
            if (cur == best) {
                consec = 0
            } else {
                consec++
                if (consec > max_c) max_c = consec
            }
        }
        END { print max_c+0 }
    ' "$file" 2>/dev/null || echo 0
}

# Сколько раз в сегменте лога значение «лучший сохранённый» стало строго лучше (меньше), чем на предыдущем [EVAL].
count_eval_best_improvements() {
    local file="$1" start_line="$2"
    awk -v start="$start_line" '
        NR < start { next }
        /\[EVAL\].*лучший сохранённый=/ {
            n = split($0, b, /лучший сохранённый=/)
            if (n < 2) next
            best = b[2]
            sub(/[^0-9.,].*/, "", best)
            gsub(/,/, ".", best)
            if (best == "" || best !~ /^[0-9]/) next
            v = best + 0
            if (has_prev && v < prev) imp++
            prev = v
            has_prev = 1
        }
        END { print imp+0 }
    ' "$file" 2>/dev/null || echo 0
}

smart_sink() {
    if [[ "${JGPT_SMART_LOG_TO_FILE:-1}" == "1" ]]; then
        tee -a "$LOG_FILE"
    else
        cat
    fi
}

# Секунды от полуночи до «сейчас» минус секунды от полуночи последней метки HH:MM:SS
# в строке [STEP] или [PERF]…шаг N в сегменте лога (с start_line). Учёт перехода через полночь — грубый.
last_step_time() {
    local start_line="${1:-1}"
    local last_line
    last_line=$(awk -v start="$start_line" '
        NR >= start && /\[STEP\]/ { line = $0 }
        NR >= start && /\[PERF\]/ && /шаг[[:space:]]+[0-9]/ { line = $0 }
        END { if (line != "") print line }
    ' "$LOG_FILE" 2>/dev/null || true)
    local stamp
    stamp=$(grep -oE '^[0-9]{2}:[0-9]{2}:[0-9]{2}' <<< "$last_line" || true)
    if [[ -z "$stamp" ]]; then
        echo 0
        return
    fi
    local h m s
    IFS=: read -r h m s <<< "$stamp"
    local step_secs=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
    local nh nm ns
    IFS=: read -r nh nm ns <<< "$(date +%H:%M:%S)"
    local now_day_secs=$((10#$nh * 3600 + 10#$nm * 60 + 10#$ns))
    local diff=$((now_day_secs - step_secs))
    if (( diff < 0 )); then
        diff=$((diff + 86400))
    fi
    echo "$diff"
}

# Текущее число строк в логе (0 если файла нет).
log_file_line_count() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo 0
        return
    fi
    wc -l < "$LOG_FILE" | tr -d ' '
}

stop_training() {
    if [[ ! -f "$PID_FILE" ]]; then
        return
    fi
    local pid
    pid=$(tr -d ' \t\r\n' < "$PID_FILE" || true)
    if [[ -z "$pid" ]]; then
        rm -f "$PID_FILE"
        return
    fi
    if kill -0 "$pid" 2>/dev/null; then
        printf '%s\n' "  [SMART] Останавливаем обучение (PID $pid) — checkpoint будет сохранён..." | smart_sink
        kill -TERM "$pid" 2>/dev/null || true
        local _
        for _ in {1..30}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
    fi
    rm -f "$PID_FILE"
}

on_smart_interrupt() {
    echo ""
    echo "  [SMART] Прерывание — останавливаем обучение..."
    stop_training
    echo "  [SMART] Готово. Resume: ./scripts/jgpt-smart.sh"
    exit 0
}

trap on_smart_interrupt INT TERM

# ─── Основной цикл ────────────────────────────────────────────

DOWNGRADE_COUNT=0
UPGRADE_COUNT=0
CYCLE_COUNT=0
UPGRADE_STABLE_EVALS=0

banner() {
    {
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo " JGPT Smart Training  |  $(date '+%Y-%m-%d %H:%M:%S')"
        echo " Пресет    : ${PRESETS[$CURRENT_IDX]}  (idx=$CURRENT_IDX)"
        echo " Downgrade : $DOWNGRADE_COUNT  |  Upgrade : $UPGRADE_COUNT  |  auto_upgrade : ${JGPT_SMART_UPGRADE:-0}"
        echo " Лог       : $LOG_FILE"
        echo "════════════════════════════════════════════════════════════"
        echo ""
    } | smart_sink
}

while true; do
    apply_preset "$CURRENT_IDX"
    PRESET_NAME="$APPLIED_PRESET_NAME"
    export JGPT_STATS_PRESET="${PRESETS[$CURRENT_IDX]}"
    export JGPT_STATS_PRESET_IDX="$CURRENT_IDX"
    UPGRADE_STABLE_EVALS=0
    STICKY_ACTIVE=0
    if [[ "${JGPT_SMART_STICKY_PRESET:-0}" == "1" ]]; then
        STICKY_ACTIVE=1
    fi
    banner

    smart_log_lines=$(log_file_line_count)
    LOG_START_LINE=$((smart_log_lines + 1))
    rm -f "$PID_FILE"
    # Subshell пишет BASHPID до exec — PID совпадает с mvn/java после exec.
    # JGPT_* из пресета наследуются; cmake + e2e — функциями выше.
    (
        echo "$BASHPID" > "$PID_FILE"
        jgpt__maven_opts
        jgpt__export_train_env
        # Смена пресета меняет totalTrainingSteps; без этого globalStep из чекпоина может быть > плана → мгновенный exit 0.
        export JGPT_IF_STEP_BEYOND_PLAN="${JGPT_IF_STEP_BEYOND_PLAN:-restart_schedule}"
        jgpt_e2e_train_overrides
        jgpt_cmake_build_cuda
        jgpt_resolve_mvn_command e2e allbooks
        exec "${JGPT_EXEC_CMD[@]}"
    ) 2>&1 | tee -a "$LOG_FILE" &

    for ((smart_pid_wait = 1; smart_pid_wait <= 50; smart_pid_wait++)); do
        [[ -f "$PID_FILE" ]] && break
        sleep 0.1
    done

    if [[ ! -f "$PID_FILE" ]]; then
        echo "  [SMART] [ERROR] Не удалось получить PID обучения"
        break
    fi
    TRAIN_PID=$(tr -d ' \t\r\n' < "$PID_FILE")
    printf '%s\n' "  [SMART] Обучение запущено (PID=$TRAIN_PID, пресет=$PRESET_NAME)" | smart_sink

    STOP_REASON=""

    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep "$MONITOR_INTERVAL"

        kill -0 "$TRAIN_PID" 2>/dev/null || break

        OOM_COUNT=$(count_pattern_from_line "$LOG_FILE" \
            "cudaMalloc failed|out of memory|OutOfMemoryError" "$LOG_START_LINE")
        if [[ "$OOM_COUNT" -ge "$OOM_THRESHOLD" ]]; then
            STOP_REASON="OOM ($OOM_COUNT раз)"
            break
        fi

        FP16_STUCK=$(count_pattern_from_line "$LOG_FILE" \
            "масштаб loss.*1\.000×" "$LOG_START_LINE")
        if [[ "$FP16_STUCK" -ge "$FP16_STUCK_THRESHOLD" ]]; then
            STOP_REASON="FP16 scale=1.0× залип ($FP16_STUCK шагов пропущено)"
            break
        fi

        IDLE=$(last_step_time "$LOG_START_LINE")
        if [[ "$IDLE" -gt "$HANG_SECONDS" ]] && [[ "$LOG_START_LINE" -gt 1 ]]; then
            STOP_REASON="Зависание (нет шагов $IDLE сек)"
            break
        fi

        STEP_LOG_COUNT=$(count_pattern_from_line "$LOG_FILE" "\\[STEP\\]" "$LOG_START_LINE")
        PLATEAU=$(count_plateau_evals "$LOG_FILE" "$LOG_START_LINE")
        if [[ "$STEP_LOG_COUNT" -ge "$PLATEAU_MIN_STEP_LINES" ]] \
                && [[ "$PLATEAU" -ge "$PLATEAU_THRESHOLD" ]]; then
            STOP_REASON="Плато eval_loss ($PLATEAU eval подряд без улучшения)"
            break
        fi

        if [[ "${JGPT_SMART_UPGRADE:-0}" == "1" ]] && [[ "$CURRENT_IDX" -gt 0 ]]; then
            UPGRADE_STABLE_EVALS=$(count_eval_best_improvements "$LOG_FILE" "$LOG_START_LINE")
            if [[ "$UPGRADE_STABLE_EVALS" -ge "$STABLE_EVALS_FOR_UPGRADE" ]]; then
                STOP_REASON="UPGRADE"
                break
            fi
        fi
    done

    if [[ -n "$STOP_REASON" ]]; then
        stop_training
    fi

    EXIT_CODE=0
    wait "$TRAIN_PID" 2>/dev/null || EXIT_CODE=$?
    rm -f "$PID_FILE"

    if [[ -z "$STOP_REASON" ]]; then
        OOM_COUNT=$(count_pattern_from_line "$LOG_FILE" \
            "cudaMalloc failed|out of memory|OutOfMemoryError" "$LOG_START_LINE")
        if [[ "$OOM_COUNT" -ge "$OOM_THRESHOLD" ]]; then
            STOP_REASON="OOM (мгновенный крэш, $OOM_COUNT раз)"
        fi
    fi

    if [[ -z "$STOP_REASON" ]] && [[ "$EXIT_CODE" -eq 0 ]]; then
        CYCLE_COUNT=$((CYCLE_COUNT + 1))
        local_n=${#PRESETS[@]}
        if [[ "$STICKY_ACTIVE" -eq 1 ]]; then
            {
                echo ""
                echo "  [SMART] ✓ Сегмент завершён штатно (пресет=$PRESET_NAME, лучший eval — см. лог AllBooksTrain)"
                echo "  [SMART] ↻ Sticky preset: перезапуск на том же пресете ${PRESETS[$CURRENT_IDX]} (idx=$CURRENT_IDX), без перехода по кругу"
            } | smart_sink
        else
            NEXT_IDX=$(( (CURRENT_IDX + 1) % local_n ))
            {
                echo ""
                echo "  [SMART] ✓ Сегмент завершён штатно (пресет=$PRESET_NAME, лучший eval — см. лог AllBooksTrain)"
                echo "  [SMART] ⟳ Круг #$CYCLE_COUNT: следующий пресет по кругу — ${PRESETS[$NEXT_IDX]} (idx=$NEXT_IDX), resume из checkpoint"
            } | smart_sink
            CURRENT_IDX=$NEXT_IDX
        fi
        sleep 3
        continue
    fi

    if [[ "$STOP_REASON" == "UPGRADE" ]]; then
        if [[ "$STICKY_ACTIVE" -eq 1 ]]; then
            {
                echo ""
                echo "  [SMART] ↻ Sticky preset: upgrade отключён, остаёмся на ${PRESETS[$CURRENT_IDX]} (idx=$CURRENT_IDX)"
            } | smart_sink
            sleep 3
            continue
        fi
        NEW_IDX=$((CURRENT_IDX - 1))
        UPGRADE_COUNT=$((UPGRADE_COUNT + 1))
        {
            echo ""
            echo "  [SMART] ↑ Upgrade #$UPGRADE_COUNT: ${PRESETS[$CURRENT_IDX]} → ${PRESETS[$NEW_IDX]}"
            echo "           Число новых лучших eval в сегменте: $UPGRADE_STABLE_EVALS (порог: $STABLE_EVALS_FOR_UPGRADE)"
        } | smart_sink
        CURRENT_IDX=$NEW_IDX
        sleep 3
        continue
    fi

    if [[ -n "$STOP_REASON" ]]; then
        {
            echo ""
            echo "  [SMART] ⚠ Проблема обнаружена: $STOP_REASON"
        } | smart_sink

        if [[ "$STICKY_ACTIVE" -eq 1 ]]; then
            printf '%s\n' "  [SMART] ↻ Sticky preset: downgrade отключён, перезапуск того же пресета ${PRESETS[$CURRENT_IDX]} (idx=$CURRENT_IDX)" | smart_sink
            UPGRADE_STABLE_EVALS=0
            sleep 3
            continue
        fi

        NEW_IDX=$((CURRENT_IDX + 1))
        if [[ "$NEW_IDX" -ge "${#PRESETS[@]}" ]]; then
            printf '%s\n' "  [SMART] Был последний пресет (${PRESETS[$CURRENT_IDX]}) — переход по кругу на ${PRESETS[0]}" | smart_sink
            NEW_IDX=0
        fi
        DOWNGRADE_COUNT=$((DOWNGRADE_COUNT + 1))
        printf '%s\n' "  [SMART] ↓ Downgrade #$DOWNGRADE_COUNT: ${PRESETS[$CURRENT_IDX]} → ${PRESETS[$NEW_IDX]}" | smart_sink
        CURRENT_IDX=$NEW_IDX
        UPGRADE_STABLE_EVALS=0
        sleep 3
        continue
    fi

    if [[ "$EXIT_CODE" -ne 0 ]]; then
        {
            echo "  [SMART] ✗ Процесс завершился с кодом $EXIT_CODE"
            echo "  Последние строки лога:"
            tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/    /' || true
        } | smart_sink

        if [[ "$STICKY_ACTIVE" -eq 1 ]]; then
            printf '%s\n' "  [SMART] ↻ Sticky preset: смена пресета отключена, повтор на ${PRESETS[$CURRENT_IDX]} (idx=$CURRENT_IDX)" | smart_sink
            sleep 3
            continue
        fi

        NEW_IDX=$((CURRENT_IDX + 1))
        if [[ "$NEW_IDX" -ge "${#PRESETS[@]}" ]]; then
            printf '%s\n' "  [SMART] Был последний пресет — переход по кругу на ${PRESETS[0]}" | smart_sink
            NEW_IDX=0
        fi
        DOWNGRADE_COUNT=$((DOWNGRADE_COUNT + 1))
        printf '%s\n' "  [SMART] ↓ Downgrade #$DOWNGRADE_COUNT → ${PRESETS[$NEW_IDX]}" | smart_sink
        CURRENT_IDX=$NEW_IDX
        sleep 3
        continue
    fi
done

echo ""
echo "  [SMART] Downgrade: $DOWNGRADE_COUNT  |  Upgrade: $UPGRADE_COUNT  |  Кругов штатного продолжения: $CYCLE_COUNT"
echo "  [SMART] Финальный пресет: ${PRESETS[$CURRENT_IDX]}"
