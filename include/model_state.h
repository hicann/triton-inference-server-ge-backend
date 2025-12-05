/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#ifndef MODEL_STATE_H
#define MODEL_STATE_H

#include <atomic>
#include <string>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstdlib>
#include <unistd.h>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_model.h"
#include "triton/core/tritonbackend.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/triton_json.h"
#include "nlohmann/json.hpp"

#include "scheduler.h"

namespace triton::backend::npu_ge {

class ModelState : public BackendModel {
public:
    static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model, ModelState **state);
    virtual ~ModelState();
    // Get backend state

    ModelState(TRITONBACKEND_Model *triton_model);
    enum class InferMode { DYNAMICMODEL, STATICMODEL };
    struct ClientTensor {
        std::string name_;
        size_t dtype_;
        std::vector<int64_t> dims_;
    };
    // 获取方法
    std::string GetModelPath()
    {
        return model_file_;
    }
    const char *GetRuntimeModelPath() const
    {
        return runtime_modeldir_;
    }
    const std::vector<int> &GetDeviceIds() const
    {
        return device_ids_;
    }
    const std::string &GetDeviceIdsStr() const
    {
        return device_ids_str_;
    }
    int GetDeviceExecBlock() const
    {
        return device_exec_block_;
    }
    int GetInstanceExecBlock() const
    {
        return instance_exec_block_;
    }
    int GetTritonThreadCount() const
    {
        return triton_thread_count_;
    }
    bool GetDeviceLB() const
    {
        return device_lb_;
    }
    InferMode GetInferMode() const
    {
        return static_cast<InferMode>(infer_mode_);
    }

    const std::unordered_map<std::string, std::string> &GetGeConfig() const
    {
        return ge_map_;
    }
    const std::unordered_map<std::string, std::string> &GetConfig() const
    {
        return process_config_map_;
    }

    size_t GetInputCount() const
    {
        return input_count_;
    }
    size_t GetOutputCount() const
    {
        return output_count_;
    }
    const std::vector<ClientTensor> &GetInputClientTensors() const
    {
        return input_client_tensor_;
    }
    const std::vector<ClientTensor> &GetOutputClientTensors() const
    {
        return output_client_tensor_;
    }

    Scheduler *GetScheduler()
    {
        return &scheduler_;
    }

    // 设置方法
    void SetInferMode(int mode)
    {
        infer_mode_ = static_cast<InferMode>(mode);
    }

    // 配置解析方法
    void ParseGeConfig(const std::string &json_str);
    TRITONSERVER_Error *ParseModelConfig();
    void DisposeConfig();

    bool GetGeStaticMode() const
    {
        return ge_static_mode_;
    }

    bool GetDumpData() const
    {
        return enable_dump_data_;
    }

    void SetGeStaticMode(bool mode)
    {
        ge_static_mode_ = mode;
    }
    // 打印方法
    void PrintModelConfig();
    void PrintClientTensors(const std::vector<ClientTensor> &tensors, const std::string &tensor_type);

private:
    // 私有辅助方法
    void ParseCmdlineConfig(const nlohmann::json &cmdline, const std::vector<std::string> &npu_ge_config);
    void SetDumpGraph(const std::string &path);
    void SetProfiling(const std::string &type, const std::string &path, const std::string &aic_metrics);
    TRITONSERVER_Error *ParseTensorInfo(common::TritonJson::Value &tensor, ClientTensor &client_tensor, bool is_input);
    TRITONSERVER_Error *ParseInputTensors(common::TritonJson::Value &inputs);
    TRITONSERVER_Error *ParseOutputTensors(common::TritonJson::Value &outputs);

    void PrintTensorInfo(const ClientTensor &tensor, const std::string &prefix);

    void ProcessDeviceIdsConfig();
    void ProcessDeviceExecBlocksConfig();
    void ProcessInstanceExecBlocksConfig();

    void ParseInstanceGroupConfig();
    void ParseModelParametersConfig();

    void ParseParameterValue(common::TritonJson::Value &params, const std::string &key, std::string &output_value,
                             std::vector<int> &output_ids);
    void ParseParameterValue(common::TritonJson::Value &params, const std::string &key, int &output_value);

    void FindModelFile();
    void InitializeBackendConfig(TRITONBACKEND_Model *triton_model);
    void DetermineInferMode();

    std::string FindFirstOnnxFile(const std::string &path);
    std::string CheckAndReturnOnnxFile(const std::string &filePath);
    std::string SearchOnnxFileInDirectory(const std::string &path);

    const char *runtime_modeldir_ = nullptr;
    std::string model_file_ = "";

    std::vector<int> device_ids_;
    std::string device_ids_str_;
    std::map<int, ge::Session *> session_map_ = {};
    Scheduler scheduler_;
    bool device_lb_ = false;
    int device_exec_block_ = -1;
    int instance_exec_block_ = 1;
    int triton_thread_count_ = 0;
    std::unordered_map<std::string, std::string> ge_map_;
    std::unordered_map<std::string, std::string> process_config_map_;
    std::string model_name_;

    size_t input_count_;
    size_t output_count_;
    std::unordered_map<std::string, std::string> parameters_;
    std::vector<ClientTensor> input_client_tensor_;
    std::vector<ClientTensor> output_client_tensor_;
    InferMode infer_mode_ = InferMode::DYNAMICMODEL;
    bool ge_static_mode_ = false;
    bool enable_dump_data_ = false;
};

}

#endif
