#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>

// Triton 核心头文件
#include "triton/core/tritonserver.h"

typedef int RET_CODE;
#define RET_ERR (-1)
#define RET_OK 0

#define LOG_ERROR(msg) TRITONSERVER_LogMessage(TRITONSERVER_LOG_ERROR, __FILE__, __LINE__, msg)

#define LOG_WARNING(msg) TRITONSERVER_LogMessage(TRITONSERVER_LOG_WARN, __FILE__, __LINE__, msg)

#define LOG_INFO(msg) TRITONSERVER_LogMessage(TRITONSERVER_LOG_INFO, __FILE__, __LINE__, msg)

// 检查宏
#define CHECK_RET_ERR(func_call)                                                                \
    do {                                                                                        \
        RET_CODE ret = (func_call);                                                             \
        if (ret == RET_ERR) {                                                                   \
            char msg[256];                                                                      \
            snprintf(msg, sizeof(msg), "Error in %s at %s:%d", #func_call, __FILE__, __LINE__); \
            LOG_ERROR(msg);                                                                     \
            return RET_ERR;                                                                     \
        }                                                                                       \
    } while (0)
// 类型安全的检查宏
#define CHECK_EQ_RETURN(func_call, error_value, return_code)                                                     \
    do {                                                                                                         \
        auto ret = (func_call);                                                                                  \
        if (ret == (error_value)) {                                                                              \
            std::ostringstream oss;                                                                              \
            oss << "Error in " << #func_call << " at " << __FILE__ << ":" << __LINE__ << " (ret=" << ret << ")"; \
            LOG_ERROR(oss.str().c_str());                                                                        \
            return return_code;                                                                                  \
        }                                                                                                        \
    } while (0)

#define CHECK_NE_RETURN(func_call, success_value, return_code)                                           \
    do {                                                                                                 \
        auto ret = (func_call);                                                                          \
        if (ret != (success_value)) {                                                                    \
            std::ostringstream oss;                                                                      \
            oss << "Error in " << #func_call << " at " << __FILE__ << ":" << __LINE__ << " (get=" << ret \
                << ", expected=" << success_value << ")";                                                \
            LOG_ERROR(oss.str().c_str());                                                                \
            return return_code;                                                                          \
        }                                                                                                \
    } while (0)

#endif