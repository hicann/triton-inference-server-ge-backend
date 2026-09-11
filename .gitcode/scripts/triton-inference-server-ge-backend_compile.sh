#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
# Color
Red='\e[0;31m'          # Red
Green='\e[0;32m'        # Green
BRed='\e[1;31m'         # Red
BGreen='\e[1;32m'       # Green
BCyan='\e[1;36m'        # Cyan
Purple='\e[0;35m'       # Purple
BPurple='\e[1;35m'      # Bold Purple
Color_Off='\e[0m'       # Text Reset
Now=`date +"%Y-%m-%d %H:%M:%S"`

function LOG_DO() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BPurple}[Command]${Color_Off} ${date_time} ${Purple}$*${Color_Off}"
    "$@"
}

# Log error
function LOG_ERROR() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BRed}[ERROR] ${date_time} ${1}${Color_Off}"
}

# Log info
function LOG_INFO() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "${BGreen}[INFO] ${date_time} ${1}${Color_Off}"
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    local log_flag=${4:-"true"}
    local log_path=${5}
    if [ "${actual_value}" != "${expect_value}" ]; then
        if [ -n "${log_path}" ] && [ -f "${log_path}" ]; then
            cat "${log_path}"
        fi
        LOG_ERROR "${assert_msg} is failed."
        exit 1
    else
        if [ "${log_flag}" = "true" ]; then
            LOG_INFO "${assert_msg} is success."
        fi
    fi
}

gcc --version
echo "Build ${REPOSITORY_NAME}."
cd ${WORKSPACE}/ || exit
pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu --default-timeout=300 --retries=5
export TRITON_HOME_PATH="/opt/tritonserver"
git config --global url."https://gitcode.com/guoyiwei1111/triton-inference-server-common.git".insteadOf "https://github.com/triton-inference-server/common.git"
git config --global url."https://gitcode.com/guoyiwei1111/triton-inference-server-core.git".insteadOf "https://github.com/triton-inference-server/core.git"
git config --global url."https://gitcode.com/guoyiwei1111/triton-inference-server-backend.git".insteadOf "https://github.com/triton-inference-server/backend.git"
LOG_DO bash build.sh
DP_ASSERT_EQUAL "$?" "0" "Build"