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

# Исходники
add_library(jgpt_cuda SHARED jgpt_cuda.cu jgpt_cuda_extra.cu)

# Линковка с явными библиотеками для Ubuntu
target_link_libraries(jgpt_cuda
    cublas          # /usr/lib/x86_64-linux-gnu/libcublas.so
    cudart          # /usr/lib/x86_64-linux-gnu/libcudart.so
    cuda            # /usr/lib/x86_64-linux-gnu/libcuda.so
    ${JNI_LIBRARIES}
)

# Корень проекта JGPT/: из src/main/cpp → три уровня вверх
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
set_target_properties(jgpt_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_ROOT}/build"
    CUDA_ARCHITECTURES "86"  # RTX 3080 = Ampere
)

# Подавить предупреждения о неиспользуемых переменных/функциях в .cu файлах
target_compile_options(jgpt_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -diag-suppress=550   # variable set but never used
        -diag-suppress=177   # function declared but never referenced
    >
)

# Runtime path
set_target_properties(jgpt_cuda PROPERTIES
    INSTALL_RPATH "/usr/lib/x86_64-linux-gnu"
    BUILD_WITH_INSTALL_RPATH TRUE
)
