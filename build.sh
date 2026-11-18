#!/bin/bash
set -e

BUILD_TYPE="Release"
TRITON_VERSION="r24.02"
INCREMENTAL_COMPILATION="OFF"
COMPUTER_ARCH=$(uname -m)
ONNXRUNTIME_ROOT="/opt/onnxruntime"
CXX11_ABI_OVERRIDE=""
echo "COMPUTER_ARCH: ${COMPUTER_ARCH}"

usage() {
  cat <<EOF
Usage: bash build.sh [options]

Options:
  --build-type <type>              Build type, Release or Debug (default: Release)
  --incremental                    Incremental build (default: clean build directory then full build)
  --tritonserver-version <version> Triton Inference Server version, also used as the
                                   git tag for common/core/backend repos (default: r24.02)
  --tritonserver-home-path <path>  Triton install path, the directory where tritonserver
                                   resides (default: /opt/tritonserver, can also be set via
                                   the TRITON_HOME_PATH environment variable)
  --onnxruntime-path <path>        onnxruntime install path (default: /opt/onnxruntime)
  --cxx11-abi <mode>               PyTorch C++ ABI: on|off|auto (default: auto, probe torch)
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
    --onnxruntime-path)
      ONNXRUNTIME_ROOT="$2"
      shift 2
      ;;
    --cxx11-abi)
      case "${2-}" in
        auto)
          CXX11_ABI_OVERRIDE=""
          ;;
        on|ON|On|true|TRUE|True|1)
          CXX11_ABI_OVERRIDE=ON
          ;;
        off|OFF|Off|false|FALSE|False|0)
          CXX11_ABI_OVERRIDE=OFF
          ;;
        *)
          echo "Invalid --cxx11-abi value: ${2-} (expected on|off|auto)" >&2
          exit 1
          ;;
      esac
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

# Resolve the source directory before creating or cleaning build artifacts.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [ ! -f "${PROJECT_DIR}/CMakeLists.txt" ]; then
  echo "CMakeLists.txt not found beside build.sh: ${PROJECT_DIR}" >&2
  exit 1
fi
if [ -L "${PROJECT_DIR}/build" ]; then
  echo "Refusing to use a symbolic link as the build directory" >&2
  exit 1
fi

cd -- "${PROJECT_DIR}"
echo "BUILD_TYPE: ${BUILD_TYPE}"
mkdir -p build
cd -- build

if [ ${INCREMENTAL_COMPILATION} == "OFF" ]; then
  rm -rf *
fi

echo "Triton version: ${TRITON_VERSION}"

if [ -z "$TRITON_HOME_PATH" ]; then
  export TRITON_HOME_PATH="/opt/tritonserver"
  echo "TRITON_HOME_PATH not set, using default path: ${TRITON_HOME_PATH}"
fi

echo "Triton install path: ${TRITON_HOME_PATH}"

if [ ! -d "$TRITON_HOME_PATH" ]; then
  echo "$TRITON_HOME_PATH is not a directory! Please check triton install path."
  exit 1
fi

COMPILE_OPTIONS=()

if tmp_path=$(which python3 2>/dev/null); then
  PYTHON_PATH="$tmp_path"
elif tmp_path=$(which python 2>/dev/null); then
  PYTHON_PATH="$tmp_path"
else
  echo "python3 or python not found, please install Python first." >&2
  exit 1
fi
echo "Python: ${PYTHON_PATH}"

# PyTorch C++ ABI detection: explicit override > probing torch.
# A failed or ambiguous probe must not be silent (issue #65), but it must not
# break builds that do not need torch either: warn loudly, fall back to the
# documented default, and let the caller pin the value with --cxx11-abi.
DEFAULT_CXX11_ABI="${DEFAULT_CXX11_ABI:-OFF}"
if [ -n "${CXX11_ABI_OVERRIDE:-}" ]; then
  USE_CXX11_ABI=$CXX11_ABI_OVERRIDE
  echo "PyTorch C++ ABI: USE_CXX11_ABI=${USE_CXX11_ABI} (set by --cxx11-abi)"
elif TORCH_CXX11_ABI=$("${PYTHON_PATH}" -c 'import torch; print(torch.compiled_with_cxx11_abi())' 2>/dev/null); then
  case "$TORCH_CXX11_ABI" in
    True)
      USE_CXX11_ABI=ON
      ;;
    False)
      USE_CXX11_ABI=OFF
      ;;
    *)
      echo "WARNING: unexpected PyTorch C++ ABI query result '${TORCH_CXX11_ABI}'." >&2
      echo "WARNING: falling back to USE_CXX11_ABI=${DEFAULT_CXX11_ABI}; pass --cxx11-abi on|off to override." >&2
      USE_CXX11_ABI=$DEFAULT_CXX11_ABI
      ;;
  esac
else
  echo "WARNING: failed to query the PyTorch C++ ABI with ${PYTHON_PATH} (is torch installed in this interpreter?)." >&2
  echo "WARNING: falling back to USE_CXX11_ABI=${DEFAULT_CXX11_ABI}; pass --cxx11-abi on|off to override." >&2
  USE_CXX11_ABI=$DEFAULT_CXX11_ABI
fi

COMPILE_OPTIONS+=("-DUSE_CXX11_ABI=$USE_CXX11_ABI")
COMPILE_OPTIONS+=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE")
COMPILE_OPTIONS+=("-DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install")
COMPILE_OPTIONS+=("-DTRITON_COMMON_REPO_TAG=$TRITON_VERSION")
COMPILE_OPTIONS+=("-DTRITON_BACKEND_REPO_TAG=$TRITON_VERSION")
COMPILE_OPTIONS+=("-DTRITON_CORE_REPO_TAG=$TRITON_VERSION")
COMPILE_OPTIONS+=("-DTRITON_ENABLE_GPU=OFF")
COMPILE_OPTIONS+=("-DPython_EXECUTABLE=$PYTHON_PATH")
COMPILE_OPTIONS+=("-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT")
COMPILE_OPTIONS+=("-DARCH=${COMPUTER_ARCH}")

printf '%s\n' "${COMPILE_OPTIONS[@]}"
cmake "${COMPILE_OPTIONS[@]}" ..
make install
