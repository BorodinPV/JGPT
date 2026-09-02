cmake_minimum_required(VERSION 3.24)
project(jgpt_cuda LANGUAGES CXX CUDA)

# Linux: CUDA 12.x host compiler is usually GCC ≤ 13. GCC 14+ needs this nvcc flag.
if(UNIX AND NOT APPLE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(JNI REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Ubuntu distro CUDA. Windows Toolkit comes from find_package(CUDAToolkit).
if(UNIX AND NOT APPLE)
    include_directories(/usr/include)
    link_directories(/usr/lib/x86_64-linux-gnu)
endif()

get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)

# native = GPU of the machine that compiles (RTX 3080 → 86, RTX 2060 → 75).
# Override: cmake -DCMAKE_CUDA_ARCHITECTURES=86
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(JGPT_CUDA_LIBS
    CUDA::cublas
    CUDA::cudart
    CUDA::cuda_driver
    ${JNI_LIBRARIES}
)
if(TARGET CUDA::cublasLt)
    list(APPEND JGPT_CUDA_LIBS CUDA::cublasLt)
endif()

set(JGPT_CUDNN_INCLUDE_DIR "")
set(JGPT_CUDNN_LIBRARY "")
set(JGPT_CUDNN_LIB_DIR "")
set(_JGPT_CUDNN_WHEEL "${PROJECT_ROOT}/third_party/cudnn/nvidia/cudnn")
if(EXISTS "${_JGPT_CUDNN_WHEEL}/include/cudnn.h")
    set(JGPT_CUDNN_INCLUDE_DIR "${_JGPT_CUDNN_WHEEL}/include")
endif()
# Linux pip wheel ships ELF libcudnn.so.9. MSVC link of that file is LNK1107.
if(WIN32)
    foreach(_cand
            "${_JGPT_CUDNN_WHEEL}/lib/x64/cudnn.lib"
            "${_JGPT_CUDNN_WHEEL}/lib/cudnn.lib"
            "${_JGPT_CUDNN_WHEEL}/lib/x64/cudnn64_9.lib")
        if(EXISTS "${_cand}")
            set(JGPT_CUDNN_LIBRARY "${_cand}")
            break()
        endif()
    endforeach()
    if(NOT JGPT_CUDNN_LIBRARY)
        file(GLOB _JGPT_CUDNN_SYS_LIBS
            "$ENV{ProgramFiles}/NVIDIA/CUDNN/*/lib/x64/cudnn.lib"
            "$ENV{ProgramFiles}/NVIDIA/CUDNN/*/lib/x64/*/cudnn.lib"
            "$ENV{ProgramFiles}/NVIDIA/CUDNN/*/lib/cudnn.lib")
        if(_JGPT_CUDNN_SYS_LIBS)
            list(SORT _JGPT_CUDNN_SYS_LIBS)
            list(REVERSE _JGPT_CUDNN_SYS_LIBS)
            list(GET _JGPT_CUDNN_SYS_LIBS 0 JGPT_CUDNN_LIBRARY)
        endif()
    endif()
    if(NOT JGPT_CUDNN_LIBRARY)
        find_library(JGPT_CUDNN_LIBRARY NAMES cudnn cudnn64_9
            HINTS "${_JGPT_CUDNN_WHEEL}" "$ENV{CUDNN_ROOT}" "$ENV{CUDNN_HOME}"
            PATH_SUFFIXES lib lib/x64 lib64)
    endif()
    if(NOT JGPT_CUDNN_LIBRARY AND EXISTS "${_JGPT_CUDNN_WHEEL}/lib/libcudnn.so.9")
        message(STATUS "JGPT: ignoring Linux libcudnn.so.9 on Windows (need cudnn.lib). scripts/windows/fetch-cudnn.ps1")
    endif()
else()
    if(EXISTS "${_JGPT_CUDNN_WHEEL}/lib/libcudnn.so.9")
        set(JGPT_CUDNN_LIBRARY "${_JGPT_CUDNN_WHEEL}/lib/libcudnn.so.9")
    endif()
    if(NOT JGPT_CUDNN_LIBRARY)
        set(_JGPT_CUDNN_HINTS "$ENV{CUDNN_ROOT}" "$ENV{CUDNN_HOME}" /usr)
        find_library(JGPT_CUDNN_LIBRARY NAMES cudnn HINTS ${_JGPT_CUDNN_HINTS} PATH_SUFFIXES lib lib64)
    endif()
endif()
if(NOT JGPT_CUDNN_INCLUDE_DIR)
    set(_JGPT_CUDNN_HINTS "$ENV{CUDNN_ROOT}" "$ENV{CUDNN_HOME}" /usr "$ENV{ProgramFiles}/NVIDIA/CUDNN")
    find_path(JGPT_CUDNN_INCLUDE_DIR cudnn.h HINTS ${_JGPT_CUDNN_HINTS} PATH_SUFFIXES include)
endif()
if(JGPT_CUDNN_LIBRARY)
    get_filename_component(JGPT_CUDNN_LIB_DIR "${JGPT_CUDNN_LIBRARY}" DIRECTORY)
endif()
set(JGPT_CUDNN_FRONTEND_INCLUDE "")
if(EXISTS "${PROJECT_ROOT}/third_party/cudnn-frontend/include/cudnn_frontend.h")
    set(JGPT_CUDNN_FRONTEND_INCLUDE "${PROJECT_ROOT}/third_party/cudnn-frontend/include")
endif()
set(JGPT_HAS_CUDNN 0)
if(JGPT_CUDNN_INCLUDE_DIR AND JGPT_CUDNN_LIBRARY AND JGPT_CUDNN_FRONTEND_INCLUDE)
    set(JGPT_HAS_CUDNN 1)
    message(STATUS "JGPT cuDNN SDPA: ${JGPT_CUDNN_LIBRARY}")
else()
    if(WIN32)
        message(STATUS "JGPT cuDNN SDPA: not found (need include=${JGPT_CUDNN_INCLUDE_DIR} lib=${JGPT_CUDNN_LIBRARY} fe=${JGPT_CUDNN_FRONTEND_INCLUDE}). scripts/windows/fetch-cudnn.ps1")
    else()
        message(STATUS "JGPT cuDNN SDPA: not found (need include=${JGPT_CUDNN_INCLUDE_DIR} lib=${JGPT_CUDNN_LIBRARY} fe=${JGPT_CUDNN_FRONTEND_INCLUDE}). scripts/linux/fetch-cudnn.sh")
    endif()
endif()

set(JGPT_EXTRA_SRCS jgpt_cuda_extra.cu jgpt_cudnn_convert.cu jgpt_cudnn_sdpa.cpp)

# Two .so on Linux (faster incremental rebuilds). One .dll on Windows:
# extra ↔ main has a circular symbol graph; MSVC cannot link that as two DLLs.
if(WIN32)
    add_library(jgpt_cuda SHARED jgpt_cuda.cu ${JGPT_EXTRA_SRCS})
    target_link_libraries(jgpt_cuda PRIVATE ${JGPT_CUDA_LIBS})
    set(JGPT_CUDA_TARGETS jgpt_cuda)
else()
    add_library(jgpt_cuda_extra SHARED ${JGPT_EXTRA_SRCS})
    add_library(jgpt_cuda SHARED jgpt_cuda.cu)
    target_link_libraries(jgpt_cuda_extra PRIVATE ${JGPT_CUDA_LIBS})
    target_link_libraries(jgpt_cuda PRIVATE jgpt_cuda_extra ${JGPT_CUDA_LIBS})
    set(JGPT_CUDA_TARGETS jgpt_cuda jgpt_cuda_extra)
endif()

foreach (TGT ${JGPT_CUDA_TARGETS})
    set_target_properties(${TGT} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_ROOT}/build"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_ROOT}/build"
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_ROOT}/build"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_ROOT}/build"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_ROOT}/build"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_ROOT}/build"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_ROOT}/build"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_ROOT}/build"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_ROOT}/build"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${PROJECT_ROOT}/build"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${PROJECT_ROOT}/build"
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${TGT} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -diag-suppress=550
            -diag-suppress=177
        >
    )
    if(UNIX AND NOT APPLE)
        target_compile_options(${TGT} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
        )
    endif()
    if(DEFINED ENV{JGPT_FA_TILE_SIZE})
        target_compile_definitions(${TGT} PRIVATE JGPT_FA_TILE_SIZE=$ENV{JGPT_FA_TILE_SIZE})
    endif()
    if(JGPT_HAS_CUDNN)
        target_compile_definitions(${TGT} PRIVATE JGPT_HAS_CUDNN=1)
        target_include_directories(${TGT} PRIVATE
            "${JGPT_CUDNN_INCLUDE_DIR}"
            "${JGPT_CUDNN_FRONTEND_INCLUDE}"
        )
    endif()
endforeach ()

if(JGPT_HAS_CUDNN)
    if(TARGET jgpt_cuda_extra)
        target_link_libraries(jgpt_cuda_extra PRIVATE "${JGPT_CUDNN_LIBRARY}")
    else()
        target_link_libraries(jgpt_cuda PRIVATE "${JGPT_CUDNN_LIBRARY}")
    endif()
endif()

if(UNIX AND NOT APPLE)
    set(_JGPT_RPATH "$ORIGIN;/usr/lib/x86_64-linux-gnu")
    if(JGPT_CUDNN_LIB_DIR)
        set(_JGPT_RPATH "${JGPT_CUDNN_LIB_DIR};${_JGPT_RPATH}")
    endif()
    set_target_properties(jgpt_cuda PROPERTIES
        BUILD_RPATH "${_JGPT_RPATH}"
        INSTALL_RPATH "${_JGPT_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    set_target_properties(jgpt_cuda_extra PROPERTIES
        BUILD_RPATH "${_JGPT_RPATH}"
        INSTALL_RPATH "${_JGPT_RPATH}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Java System.load does not put the DLL directory on the Windows loader search path.
# CUDA 13+ ships runtime DLLs in bin/x64 (not bin/).
if(WIN32 AND CUDAToolkit_BIN_DIR)
    file(GLOB _JGPT_CUDA_RUNTIME_DLLS
        "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
        "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
        "${CUDAToolkit_BIN_DIR}/cublasLt64_*.dll"
        "${CUDAToolkit_BIN_DIR}/nvJitLink*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/cudart64_*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/cublasLt64_*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/nvJitLink*.dll"
    )
    if(_JGPT_CUDA_RUNTIME_DLLS)
        add_custom_command(TARGET jgpt_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_JGPT_CUDA_RUNTIME_DLLS}
                "${PROJECT_ROOT}/build"
            COMMENT "Copy CUDA runtime DLLs into ${PROJECT_ROOT}/build"
        )
    endif()
endif()

if(WIN32 AND JGPT_HAS_CUDNN)
    set(_JGPT_CUDNN_DLL_ROOT "")
    if(JGPT_CUDNN_INCLUDE_DIR)
        get_filename_component(_JGPT_CUDNN_DLL_ROOT "${JGPT_CUDNN_INCLUDE_DIR}" DIRECTORY)
    endif()
    file(GLOB _JGPT_CUDNN_RUNTIME_DLLS
        "${_JGPT_CUDNN_DLL_ROOT}/bin/cudnn*.dll"
        "${_JGPT_CUDNN_DLL_ROOT}/bin/*/cudnn*.dll"
        "${JGPT_CUDNN_LIB_DIR}/../bin/cudnn*.dll"
        "${JGPT_CUDNN_LIB_DIR}/cudnn*.dll"
    )
    if(_JGPT_CUDNN_RUNTIME_DLLS)
        add_custom_command(TARGET jgpt_cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_JGPT_CUDNN_RUNTIME_DLLS}
                "${PROJECT_ROOT}/build"
            COMMENT "Copy cuDNN DLLs into ${PROJECT_ROOT}/build"
        )
    endif()
endif()
