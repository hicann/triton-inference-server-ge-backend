#!/bin/bash
set -e

BUILD_TYPE="Release"
TRITON_VERSION="r24.02"
INCREMENTAL_COMPILATION="OFF"
COMPUTER_ARCH=$(uname -m)
ONNXRUNTIME_ROOT="/opt/onnxruntime"
echo "COMPUTER_ARCH: ${COMPUTER_ARCH}"

usage() {
  cat <<EOF
Usage: bash build.sh [options]

Options:
  --build-type <type>              Build type, Release or Debug (default: Debug)
  --incremental                    Incremental build (default: clean build directory then full build)
  --tritonserver-version <version> Triton Inference Server version, also used as the
                                   git tag for common/core/backend repos (default: r24.02)
  --tritonserver-home-path <path>  Triton install path, the directory where tritonserver
                                   resides (default: /opt/tritonserver, can also be set via
                                   the TRITON_HOME_PATH environment variable)
  --help                           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      BUILD_TYPE="$2"
      if [[ "$BUILD_TYPE" != "Release" && "$BUILD_TYPE" != "Debug" ]]; then
        echo "Invalid build type: $BUILD_TYPE, valid values are Release or Debug" >&2
        exit 1
      fi
      shift 2
      ;;
    --incremental)
      INCREMENTAL_COMPILATION="ON"
      shift
      ;;
    --tritonserver-version)
      TRITON_VERSION="$2"
      shift 2
      ;;
    --tritonserver-home-path)
      export TRITON_HOME_PATH="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "BUILD_TYPE: ${BUILD_TYPE}"
mkdir -p build && cd build

if [ ${INCREMENTAL_COMPILATION} == "OFF" ]; then
  rm -rf *
fi

echo "Triton version: ${TRITON_VERSION}"

if [ -z "$TRITON_HOME_PATH" ]; then
  TRITON_HOME_PATH="/opt/tritonserver"
  echo "TRITON_HOME_PATH not set, using default path: ${TRITON_HOME_PATH}"
fi

echo "Triton install path: ${TRITON_HOME_PATH}"

if [ ! -d "$TRITON_HOME_PATH" ]; then
  echo "$TRITON_HOME_PATH is not a directory! Please check triton install path."
  exit 1
fi

COMPILE_OPTIONS=""

if tmp_path=$(which python3 2>/dev/null); then
  PYTHON_PATH="$tmp_path"
elif tmp_path=$(which python 2>/dev/null); then
  PYTHON_PATH="$tmp_path"
else
  echo "python3 or python not found, please install Python first." >&2
  exit 1
fi
echo "Python: ${PYTHON_PATH}"

if [ $(${PYTHON_PATH} -c 'import torch; print(torch.compiled_with_cxx11_abi())') == "True" ]; then
  USE_CXX11_ABI=ON
else
  USE_CXX11_ABI=OFF
fi

COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=$USE_CXX11_ABI"
COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=$BUILD_TYPE"
COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install"
COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTRITON_COMMON_REPO_TAG=$TRITON_VERSION"
COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTRITON_BACKEND_REPO_TAG=$TRITON_VERSION"
COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTRITON_CORE_REPO_TAG=$TRITON_VERSION"
COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTRITON_ENABLE_GPU=OFF"
COMPILE_OPTIONS="${COMPILE_OPTIONS} -DPython_EXECUTABLE=$PYTHON_PATH"
COMPILE_OPTIONS="${COMPILE_OPTIONS} -DARCH=${COMPUTER_ARCH}"

echo $COMPILE_OPTIONS
cmake $COMPILE_OPTIONS ..
make install
