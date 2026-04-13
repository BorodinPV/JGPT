cmake_minimum_required(VERSION 3.18)
project(jgpt_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Найти Java
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Для Ubuntu: системные пути CUDA
# Заголовки в /usr/include, библиотеки в /usr/lib/x86_64-linux-gnu
include_directories(/usr/include)
link_directories(/usr/lib/x86_64-linux-gnu)

# Корень проекта JGPT/: из src/main/cpp → три уровня вверх
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)

# Часть JNI/ядер (TensorOpsGPU и др.) вынесена в отдельную .so — ускоряет инкрементальную сборку и явно
# разделяет «extra» от основного модуля. libjgpt_cuda.so линкуется с libjgpt_cuda_extra.so; при загрузке
# из каталога build/ нужен RUNPATH $ORIGIN (см. ниже).
add_library(jgpt_cuda_extra SHARED jgpt_cuda_extra.cu)
add_library(jgpt_cuda SHARED jgpt_cuda.cu)

set(CUDA_LIBS
    cublas
    cudart
    cuda
    ${JNI_LIBRARIES}
)

target_link_libraries(jgpt_cuda_extra PRIVATE ${CUDA_LIBS})
target_link_libraries(jgpt_cuda PRIVATE jgpt_cuda_extra ${CUDA_LIBS})

foreach (TGT jgpt_cuda jgpt_cuda_extra)
    set_target_properties(${TGT} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_ROOT}/build"
        CUDA_ARCHITECTURES "86"
    )
    target_compile_options(${TGT} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -diag-suppress=550
            -diag-suppress=177
        >
    )
endforeach ()

# libjgpt_cuda.so подтягивает libjgpt_cuda_extra.so из того же каталога (build/), плюс системный CUDA.
set_target_properties(jgpt_cuda PROPERTIES
    BUILD_RPATH "$ORIGIN;/usr/lib/x86_64-linux-gnu"
    INSTALL_RPATH "$ORIGIN;/usr/lib/x86_64-linux-gnu"
    BUILD_WITH_INSTALL_RPATH TRUE
)
set_target_properties(jgpt_cuda_extra PROPERTIES
    BUILD_RPATH "/usr/lib/x86_64-linux-gnu"
    INSTALL_RPATH "/usr/lib/x86_64-linux-gnu"
    BUILD_WITH_INSTALL_RPATH TRUE
)
