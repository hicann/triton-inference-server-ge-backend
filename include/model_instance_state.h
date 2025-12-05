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

#include "model_state.h"
#include "scheduler.h"

using namespace ge;
using json = nlohmann::json;
using namespace std;

namespace triton::backend::npu_ge {

static std::atomic<int> next_id{0};
static std::atomic<int> lock_id{0};

class ModelInstanceState : public BackendModelInstance {
public:
    static TRITONSERVER_Error *Create(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance,
                                      ModelInstanceState **state);
    ModelInstanceState(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance);
    virtual ~ModelInstanceState();
    int ProcessRequests(TRITONBACKEND_Request **requests, const uint32_t request_count);

    int Init();

    // 工具方法
    int GetValueByKey(size_t key) const;
    ge::DataType TransToGeData(size_t key) const;

    void FreeDevResources(std::vector<void *> indev_buffer_, std::vector<void *> outdev_buffer_);
    void FreeHostResources(std::vector<void *> inhost_buffer_, std::vector<void *> outhost_buffer_);

    // 静态工具方法
    static std::string GetEnvVar(const std::string &name);
    static void LoadGeConfig(std::string &env_value, std::map<ge::AscendString, ge::AscendString> &configMap);

private:
    // 初始化相关方法
    void InitTypeMappings();
    int InitGEEnvironment();
    int InitDevices(std::vector<int> &dev_ids_);
    int InitGraphSession(int dev_id, int graph_id, aclrtContext context, std::mutex &mu, ge::Session *session_);
    void StaticModeConfig(std::map<ge::AscendString, ge::AscendString> &parser_options);
    void StaticModeConfigOne(std::map<ge::AscendString, ge::AscendString> &parser_options);

    int CalculateDeviceExecBlock(int device_count);
    void CreateDeviceThreads(int dev_id, int dev_index, int device_exec_block, std::vector<std::thread> &threads,
                             std::mutex &mu);
    void JoinAllThreads(std::vector<std::thread> &threads);
    void FreeResources();

    // 请求处理相关方法
    int ProcessSingleRequest(TRITONBACKEND_Request *request, std::vector<Scheduler::Instance *> instances);

    void CalculateBatchDistribution(int batch_total, size_t instance_count, std::vector<int> &input_offset,
                                    std::vector<int> &batch_result);

    int ExtractInputInfo(TRITONBACKEND_Request *request, std::vector<void *> &inhost_buffer_,
                         std::vector<int> &inhost_line_size_, int &batch_total, std::vector<int> &input_offset,
                         std::vector<int> &batch_result, size_t instance_count);

    int PrepareOutputBuffers(std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outsize,
                             std::vector<int> &outhost_line_size_, int batch_total);

    void GetExecutionBatchParams(int batch, int &exec_batch, int &cycle_count);

    int ProcessSingleInstance(Scheduler::Instance *instance, int instance_index, int instance_count,
                              std::vector<void *> &inhost_buffer_, std::vector<int> &inhost_line_size_,
                              std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
                              const std::vector<int> &input_offset, const std::vector<int> &batch_result,
                              int &success_count, int &invalid_count);

    int CheckInferenceResult(int success_count, int invalid_count, int total_instances);

    int ExecuteInference(std::vector<Scheduler::Instance *> instances, std::vector<void *> &inhost_buffer_,
                         std::vector<int> &inhost_line_size_, std::vector<void *> &outhost_buffer_,
                         std::vector<int> &outhost_line_size_, const std::vector<int> &input_offset,
                         const std::vector<int> &batch_result);

    int ExecuteInferenceParallel(std::vector<Scheduler::Instance *> instances, std::vector<void *> &inhost_buffer_,
                                 std::vector<int> &inhost_line_size_, std::vector<void *> &outhost_buffer_,
                                 std::vector<int> &outhost_line_size_, const std::vector<int> &input_offset,
                                 const std::vector<int> &batch_result);

    int ExecuteSingleInference(Scheduler::Instance *instance, int instance_index, int instance_count,
                               std::vector<void *> &inhost_buffer_, std::vector<int> &inhost_line_size_,
                               std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
                               const std::vector<int> &input_offset, const std::vector<int> &batch_result);

    int PrepareInputTensors(Scheduler::Instance *instance, std::vector<void *> &inhost_buffer_,
                            std::vector<int> &inhost_line_size_, int exec_batch, int instance_index,
                            std::vector<void *> &indev_buffer_, std::vector<gert::Tensor> &inputs);

    int BuildInputTensor(size_t input_index, int exec_batch, void *dev_buffer, gert::Tensor &tensor);

    int PrepareOutputTensors(std::vector<int> &outhost_line_size_, int exec_batch, std::vector<void *> &outdev_buffer_,
                             std::vector<gert::Tensor> &outputs);

    int BuildOutputTensor(size_t output_index, int exec_batch, void *out_dev_buffer, gert::Tensor &tensor);

    int ExecuteInferenceCycle(Scheduler::Instance *instance, std::vector<void *> &inhost_buffer_,
                              std::vector<int> &inhost_line_size_, std::vector<void *> &outhost_buffer_,
                              std::vector<int> &outhost_line_size_, std::vector<gert::Tensor> &inputs,
                              std::vector<gert::Tensor> &outputs, int exec_batch, int cycle_count, int instance_index,
                              const std::vector<int> &input_offset);

    int ExecuteInference(Scheduler::Instance *instance, std::vector<gert::Tensor> &inputs,
                         std::vector<gert::Tensor> &outputs);

    int CopyOutputToHost(Scheduler::Instance *instance, std::vector<void *> &outhost_buffer_,
                         std::vector<int> &outhost_line_size_, int batch, int batch_front, int instance_index,
                         std::vector<gert::Tensor> &outputs);

    int BuildResponse(TRITONBACKEND_Request *request, std::vector<void *> &outhost_buffer_,
                      std::vector<int64_t> &outsize, int batch_total);

    int BuildSingleOutput(TRITONBACKEND_Response *response, size_t output_index, void *outhost_buffer,
                          int64_t buffer_size, int batch_total);

    std::unique_ptr<int64_t[]> CreateOutputShape(size_t output_index, uint32_t &size, int batch_total);

    ModelState *model_state_;
    std::string model_config_path_;
    std::string model_path_;
    std::string acl_path_;
    int thread_id_ = -1;
    std::map<size_t, size_t> size_by_type_;
    std::map<size_t, ge::DataType> ge_data_trans_;
    std::atomic<bool> init_failed_{false};
    std::exception_ptr init_exception_{nullptr};
    std::mutex exception_mutex_;
    ge::Session *session_;
};
}

#endif
