#!/bin/bash
set -e

BUILD_TYPE="Release"
TRITON_VERSION="r24.02"
GTEST_VERSION="v1.15.2"
DT_MODE="OFF"
INCREMENTAL_COMPILATION="OFF"
COMPUTER_ARCH=$(uname -m)
ONNXRUNTIME_ROOT="/opt/onnxruntime"
echo "COMPUTER_ARCH: ${COMPUTER_ARCH}"

while getopts "dtev:p:" opt; do
  case ${opt} in
    d)
      BUILD_TYPE="Debug"
      ;;
    t)
      DT_MODE="ON"
      ;;
    e)
      INCREMENTAL_COMPILATION="ON"
      ;;
    v)
      TRITON_VERSION=$OPTARG
      ;;
    p)
      export TRITON_HOME_PATH=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$opt" >&2
      exit 1
      ;;
  esac
done
echo "BUILD_TYPE: ${BUILD_TYPE}"
mkdir -p build && cd build

if [ ${INCREMENTAL_COMPILATION} == "OFF" ]; then
  rm -rf *
fi

if [ ${DT_MODE} == "ON" ]; then
  cmake -DDT_TEST=ON -DGOOGLETEST_REPO_TAG=$GTEST_VERSION ..
  make -j$(nproc)
else
  echo "Triton version: ${TRITON_VERSION}"

  if [ -z "$TRITON_HOME_PATH" ]; then
    echo "env TRITON_HOME_PATH is null, please set env or use -p to tell us where triton is installed."
    exit 1
  fi

  echo "Triton install path: ${TRITON_HOME_PATH}"

  if [ ! -d "$TRITON_HOME_PATH" ]; then
    echo "$TRITON_HOME_PATH is not a directory! Please check triton install path."
    exit 1
  fi

  COMPILE_OPTIONS=""

  if [ $(python3 -c 'import torch; print(torch.compiled_with_cxx11_abi())') == "True" ]; then
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
  COMPILE_OPTIONS="${COMPILE_OPTIONS} -DPython_EXECUTABLE=$(which python)"
  COMPILE_OPTIONS="${COMPILE_OPTIONS} -DARCH=${COMPUTER_ARCH}"
  echo $COMPILE_OPTIONS
  cmake $COMPILE_OPTIONS ..
  make -j12 install
fi
