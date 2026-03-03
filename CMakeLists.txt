cmake_minimum_required(VERSION 3.17)

project(tutorialombackend LANGUAGES C CXX)


set(ASCEND_DIR /usr/local/Ascend)
set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)
set(ATC_INCLUDE_DIR ${ASCEND_PATH}/${ARCH}-linux/include)
set(ONNXRUNTIME_ROOT /opt/onnxruntime)

message("current arch: ${ARCH}")

option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
# 需根据实际路径填写
set(ONNXRUNTIME_ROOT ${ONNXRUNTIME_ROOT})

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-std=c++17)
add_compile_options(-Wno-unused-parameter)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SKIP_RPATH TRUE)

if (USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-all")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-z,relro,-z,now")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pie -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftrapv -s")



# add_definitions(-D_GLIBCXX_USE_C99=1)
#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
repo-common
GIT_REPOSITORY https://github.com/triton-inference-server/common.git
GIT_TAG ${TRITON_COMMON_REPO_TAG}
GIT_SHALLOW ON
)
FetchContent_Declare(
repo-core
GIT_REPOSITORY https://github.com/triton-inference-server/core.git
GIT_TAG ${TRITON_CORE_REPO_TAG}
GIT_SHALLOW ON
)
FetchContent_Declare(
repo-backend
GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
GIT_TAG ${TRITON_BACKEND_REPO_TAG}
GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#

LINK_DIRECTORIES(
$ENV{TRITON_HOME_PATH}/lib
)
INCLUDE_DIRECTORIES(
${ASCEND_DIR}/ascend-toolkit/latest/${ARCH}-linux/include
)

# onnxruntime
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_library(triton-npu-ge-backend SHARED ${SOURCE_FILES})

add_library(
TutorialmindieBackend::triton-npu-ge-backend ALIAS triton-npu-ge-backend
)

target_include_directories(
triton-npu-ge-backend
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/Ascend/code/json/json/include/  # json安装目录
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${ATC_INCLUDE_DIR}/graph
    ${ATC_INCLUDE_DIR}/ge
    ${ATC_INCLUDE_DIR}/parser    
    ${ATC_INCLUDE_DIR}
    ${ASCEND_PATH}/compiler/lib64/stub
    ${ASCEND_PATH}/runtime/include/
)

target_compile_features(triton-npu-ge-backend PRIVATE cxx_std_11)

target_link_directories(triton-npu-ge-backend PRIVATE
    ${ASCEND_PATH}/compiler/lib64/stub
    ${ASCEND_PATH}/${ARCH}-linux/devlib/linux/${ARCH}
)

target_link_libraries(
triton-npu-ge-backend
PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    ascendcl
    graph
    ge_compiler
    fmk_onnx_parser
    fmk_parser
    ge_runner
    onnxruntime
)

set_target_properties(
triton-npu-ge-backend PROPERTIES
OUTPUT_NAME triton_npu_ge
)

# Install

install(
TARGETS
triton-npu-ge-backend
DESTINATION $ENV{TRITON_HOME_PATH}/backends/npu_ge
)

