/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef MODEL_INSTANCE_STATE_H
#define MODEL_INSTANCE_STATE_H
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <regex>
#include <unistd.h>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonbackend.h"
#include "nlohmann/json.hpp"
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "all_ops.h"
#include "acl/acl.h"
#include "onnx_parser.h"
#include "tensorflow_parser.h"
#include "model_state.h"
#include "scheduler.h"
#include "inference.h"

using namespace ge;
using json = nlohmann::json;
using namespace std;

namespace triton::backend::npu_ge {

static std::atomic<int> next_id{0};
static std::atomic<int> lock_id{0};
static std::atomic<int> notify_id{0};

class ModelInstanceState : public BackendModelInstance {
public:
    static TRITONSERVER_Error *Create(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance,
                                      ModelInstanceState **state);
    ModelInstanceState(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance);
    virtual ~ModelInstanceState();
    int ProcessRequests(TRITONBACKEND_Request **requests, const uint32_t request_count);

    int Init();

    // 静态工具方法
    static std::string GetEnvVar(const std::string &name);
    static void LoadGeConfig(std::string &env_value, std::map<ge::AscendString, ge::AscendString> &configMap);

    void ConfigureParserOptions(std::map<ge::AscendString, ge::AscendString> &parser_options);
    bool ParseGraph(std::map<ge::AscendString, ge::AscendString> &parser_options, ge::Graph &graph1, std::mutex &mu);
    bool AddAndCompileGraph(ge::Session *session_, int graph_id, ge::Graph &graph1, ge::Status &ret);
    bool CreateAndLoadStream(ge::Session *session_, int graph_id, aclrtStream &stream_, ge::Status &ret,
                             aclError &aclRet);
    void ConfigureDumpOptions(std::map<ge::AscendString, ge::AscendString> &options);
    bool SetDeviceAndContext(int dev_id, aclrtContext &context_);
    void CreateSessions(int dev_index, int device_exec_block,
                        const std::map<ge::AscendString, ge::AscendString> &options,
                        std::vector<ge::Session *> &sessions);
    void CreateThreads(int dev_id, int dev_index, int device_exec_block, std::mutex &mu, aclrtContext context_,
                       const std::vector<ge::Session *> &sessions, std::vector<std::thread> &threads);
    void RecordInitFailure(int graph_id);
    void RecordInitException();
    int InitializeGlobalResources();
    int InitializeDeviceThreads(const std::vector<int> &dev_ids_, int lock);
    void HandleInitFailure();
    void InitializeGraphSession(int graph_id, int dev_id, aclrtContext context_, std::mutex &mu, ge::Session *session);
    void ConfigureGeOptions(std::map<ge::AscendString, ge::AscendString> &options);

private:
    // 初始化相关方法
    int InitGEEnvironment();
    int InitDevices(std::vector<int> &dev_ids_);
    int InitGraphSession(int dev_id, int graph_id, aclrtContext context, std::mutex &mu, ge::Session *session_);
    void StaticModeConfig(std::map<ge::AscendString, ge::AscendString> &parser_options);
    void StaticModeConfigOne(std::map<ge::AscendString, ge::AscendString> &parser_options);

    int CalculateDeviceExecBlock(int device_count);
    void CreateDeviceThreads(int dev_id, int dev_index, int device_exec_block, std::vector<std::thread> &threads,
                             std::mutex &mu);
    void JoinAllThreads(std::vector<std::thread> &threads);
    ModelState *model_state_;
    std::string model_config_path_;
    std::string model_path_;
    std::string acl_path_;
    int thread_id_ = -1;

    std::atomic<bool> init_failed_{false};
    std::exception_ptr init_exception_{nullptr};
    std::mutex exception_mutex_;
    Inference *inference_ = nullptr;
};
}

#endif
