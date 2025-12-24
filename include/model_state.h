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
#include <tuple>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <unordered_set>
#include <vector>
#include <string>
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

    enum class ModelMode {
        MAX_BATCH,  // 自动加上batch维度 pbtxt中max_batch_size>0
        MAX_BATCH_HAVE_UNKNOW_DIM,
        NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE,  // max_batch_size==0 所有张量维度0一样且不是-1
        NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM,  // max_batch_size==0 所有张量维度0一样且不是-1 其余维度包含-1(未知)
        NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE,                  // max_batch_size==0 所有张量维度0一样且是-1
        NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE_HAVE_UNKNOW_DIM,  // max_batch_size==0 所有张量维度0一样且是-1
        NO_MAX_BATCH_FIRST_NOT_SAME,  // max_batch_size==0 所有张量维度0不全一样且所有张量形状确认
        NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE,  //  max_batch_size==0 所有张量维度0不全一样且存在未知维度 两种情况 1. 首维度都是-1   2. 其他
        TENSOR_ZERO
    };

    enum class ModelType { NONE, ONNX, TENSORFLOW, CAFFE, OM, AIR };

    // 友元函数声明
    friend std::ostream &operator<<(std::ostream &os, ModelState::ModelMode mode);

    struct Express {
        std::string expressName;                                  // 表达式
        std::map<std::string, std::pair<size_t, size_t>> dimMap;  // 表达式中张量  第几张量第几维度索引
    };

    void SetModelMode();
    ModelMode GetModelNode()
    {
        return model_mode_;
    }

    ModelType GetModelType()
    {
        return model_type_;
    }

    struct OnnxTensorInfo {
        std::string name_;
        size_t dtype_;
        std::vector<int64_t> dims_;
        std::vector<std::string> dim_name_;
    };

    std::string GetModelName()
    {
        return model_name_;
    }
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
    const std::vector<OnnxTensorInfo> &GetInputClientTensors() const
    {
        return input_client_tensor_;
    }
    const std::vector<OnnxTensorInfo> &GetOutputClientTensors() const
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
    int GetInitStatus() const
    {
        return init_thread_tag_;
    }

    void SetInitStatus(int mode)
    {
        init_thread_tag_ = mode;
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

    bool GetDynamicBatchSupport()
    {
        return can_dynamic_batching;
    }
    const auto &GetInToOutMap() const noexcept
    {
        return input_dim_map_output_dim;
    }
    // 打印方法
    void PrintModelConfig();
    void PrintClientTensors(const std::vector<OnnxTensorInfo> &tensors, const std::string &tensor_type);
    struct DynamicBatching {
        std::vector<int64_t> preferred_batch_sizes;
        int64_t max_queue_delay_microseconds_;
    };
    DynamicBatching &GetDynamicmode()
    {
        return dynamic_batching_;
    }
    std::string ProcessEntry(const std::filesystem::directory_entry &entry, const std::string &extension);

private:
    // 私有辅助方法
    void ParseCmdlineConfig(const nlohmann::json &cmdline, const std::vector<std::string> &npu_ge_config);
    void SetDumpGraph(const std::string &path);
    void SetProfiling(const std::string &type, const std::string &path, const std::string &aic_metrics);
    TRITONSERVER_Error *ParseTensorInfo(common::TritonJson::Value &tensor, OnnxTensorInfo &client_tensor,
                                        bool is_input);
    TRITONSERVER_Error *ParseInputTensors(common::TritonJson::Value &inputs);
    TRITONSERVER_Error *ParseOutputTensors(common::TritonJson::Value &outputs);
    TRITONSERVER_Error *ParseDynamicBatching();
    void PrintTensorInfo(const OnnxTensorInfo &tensor, const std::string &prefix);

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

    std::string FindFirstFile(const std::string &path, const std::string &extension);
    std::string CheckAndReturnFile(const std::string &filePath, const std::string &extension);
    std::string SearchFileInDirectory(const std::string &path, const std::string &extension);

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

    std::unordered_map<std::string, std::string> parameters_;
    InferMode infer_mode_ = InferMode::DYNAMICMODEL;
    bool ge_static_mode_ = false;
    bool enable_dump_data_ = false;
    DynamicBatching dynamic_batching_;

    void ParseOnnxInfo();
    TRITONSERVER_Error *CheckConfigIO();
    TRITONSERVER_Error *AutoCompleteConfig();
    TRITONSERVER_Error *AutoCompleteIO(const char *key, const std::vector<OnnxTensorInfo> &io_infos);
    TRITONSERVER_Error *AutoCompleteMaxBatch();
    TRITONSERVER_Error *PreNegativeOne();

    bool CheckDynamicBatch();

    size_t input_count_;
    size_t output_count_;
    std::vector<OnnxTensorInfo> input_client_tensor_;  // 记录配置文件
    std::vector<OnnxTensorInfo> output_client_tensor_;
    TRITONSERVER_Error *CreateInputdimToOutputdim();
    std::unordered_set<std::string> ExtractVariables(const std::string &expr);
    std::string RemoveSpaces(const std::string &str);
    bool IsValidVariableStart(char c);
    bool IsValidVariableChar(char c);
    TRITONSERVER_Error *FindOutputDim(std::unordered_set<std::string> &umap1, Express &ex);

    size_t onnx_input_count_;
    size_t onnx_output_count_;
    std::vector<OnnxTensorInfo> input_onnx_tensor_;  // 记录模型读取
    std::vector<OnnxTensorInfo> output_onnx_tensor_;
    void LogInputDimMapOutputDim();
    int GetFirstDimNum();
    TRITONSERVER_Error *CompareInputOutput();
    bool FindMapResult(std::string &name, std::tuple<size_t, size_t, std::string> &t1);
    void GetnegativeOne(size_t index, std::vector<int64_t> &dims_);
    bool ContainNegativeOneFromTensor(size_t index = 1);
    TRITONSERVER_Error *CheckModelMode();
    TRITONSERVER_Error *InputdimToOutputdimMap(ModelState **state);
    std::vector<std::string> GetOnnxSymbolicDimension(const Ort::TypeInfo &type_info);
    void AddDimName();
    void AddNegativeOneInfo();
    void ChangeOutputdim();
    void CompareOnnxAndTxt();
    std::string PrintModelMode(ModelState::ModelMode mode);
    bool can_dynamic_batching = false;
    TRITONSERVER_Error *ProcessFindOutputDim(std::unordered_set<std::string> &umap1, Express &ex);
    TRITONSERVER_Error *FindAndSetDim(Express &ex, const std::string &findname);
    std::vector<std::string> ProcessSymbolicDimensions(const Ort::TypeInfo &type_info);
    void LogSymbolicDimensions(const std::vector<std::string> &result);

    int FirstDimNameSame();
    ModelState::ModelType model_type_ = ModelState::ModelType::NONE;

    template <typename T>
    bool ContainNegativeOne(const std::vector<T> &output_tensors, size_t index)
    {
        bool flag = false;
        for (size_t i = 0; i < output_tensors.size(); i++) {
            auto out = output_tensors[i].dims_;
            for (size_t j = index; j < out.size(); j++) {
                if (out[j] == -1) {
                    flag = true;
                }
            }
        }
        return flag;
    }
    std::map<std::pair<size_t, size_t>, Express> input_dim_map_output_dim;
    std::vector<std::pair<size_t, size_t>> negativeOne;
    ModelMode model_mode_ = ModelMode::TENSOR_ZERO;
    int init_thread_tag_ = -1;
};

template <typename T1, typename T2>
void PrintVector(const std::vector<T1> &vec, const std::string &name, const std::vector<T2> &dimname)
{
    std::string message{"["};
    for (auto &v : vec) {
        message += std::to_string(v);
        message += ", ";
    }
    message.erase(message.end() - 2, message.end());
    message += "]\n";
    for (auto &v : dimname) {
        message += v;
        message += ", ";
    }
    message.erase(message.end() - 2, message.end());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (name + ": " + message).c_str());
}

}

#endif
