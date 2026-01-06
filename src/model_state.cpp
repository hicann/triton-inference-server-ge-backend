/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "model_state.h"

using json = nlohmann::json;
using namespace std;
namespace triton::backend::npu_ge {

// TRITONSERVER_TYPE对应索引
std::map<std::string, size_t> data_type_map = {
    {"TYPE_FP32", 11},   // TRITONSERVER_TYPE_FP32 11
    {"TYPE_FP16", 10},   // TRITONSERVER_TYPE_FP16 10
    {"TYPE_INT8", 6},    // TRITONSERVER_TYPE_INT8 6
    {"TYPE_INT16", 7},   // TRITONSERVER_TYPE_INT16 7
    {"TYPE_INT32", 8},   // TRITONSERVER_TYPE_INT32 8
    {"TYPE_INT64", 9},   // TRITONSERVER_TYPE_INT64 9
    {"TYPE_UINT8", 2},   // TRITONSERVER_TYPE_UINT8 2
    {"TYPE_UINT16", 3},  // TRITONSERVER_TYPE_UINT16 3
    {"TYPE_UINT32", 4},  // TRITONSERVER_TYPE_UINT32 4
    {"TYPE_UINT64", 5},  // TRITONSERVER_TYPE_UINT64 5
    {"TYPE_BOOL", 1},    // TRITONSERVER_TYPE_BOOL 1
    {"TYPE_STRING", 13}  // TRITONSERVER_TYPE_BYTES 13
};

std::map<size_t, string> GeDataTypeToModelConfigDataTypeMap = {
    {11, "TYPE_FP32"},  {10, "TYPE_FP16"},  {6, "TYPE_INT8"},  {7, "TYPE_INT16"},
    {8, "TYPE_INT32"},  {9, "TYPE_INT64"},  {2, "TYPE_UINT8"}, {3, "TYPE_UINT16"},
    {4, "TYPE_UINT32"}, {5, "TYPE_UINT64"}, {1, "TYPE_BOOL"},  {13, "TYPE_STRING"}};

size_t GetType(ONNXTensorElementDataType type)
{
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return data_type_map["TYPE_FP32"];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return data_type_map["TYPE_UINT8"];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return data_type_map["TYPE_INT8"];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return data_type_map["TYPE_UINT16"];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return data_type_map["TYPE_INT16"];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return data_type_map["TYPE_INT32"];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return data_type_map["TYPE_INT64"];
        default:
            return static_cast<size_t>(TRITONSERVER_TYPE_INVALID);
    }
}

std::vector<std::string> Split(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

void ModelState::SetDumpGraph(const std::string &path)
{
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("use dump_graph ")).c_str());
    // Validate path safety (prevent command injection)
    if (path.empty() || path.find(";") != std::string::npos || path.find("&") != std::string::npos ||
        path.find("|") != std::string::npos) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Error: Path contains unsafe characters or is empty");
    }
    // Set graph dump path
    setenv("DUMP_GRAPH_PATH", path.c_str(), 1);
    // Set print model flag
    setenv("PRINT_MODEL", "1", 1);
    // Set GE graph dump level
    setenv("DUMP_GE_GRAPH", "2", 1);
    // Set graph dump level
    setenv("DUMP_GRAPH_LEVEL", "2", 1);
    // Clean up dump directory
    std::string cleanup_cmd = "rm -rf " + path + "/* 2>/dev/null";
    int result = system(cleanup_cmd.c_str());
    if (result != 0) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            (std::string("Warning: Problem occurred while cleaning directory, command: ") + cleanup_cmd).c_str());
        // Don't return false here because the directory might just be empty
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Successfully cleaned directory: ") + path).c_str());
    }
}

void ModelState::SetProfiling(const std::string &type = "true", const std::string &path = "",
                              const std::string &aic_metrics = "PipeUtilization")
{
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("use profiling ")).c_str());
    setenv("PROFILING_MODE", type.c_str(), 1);
    // 如果提供了路径，设置性能分析选项
    if (!path.empty()) {
        // 构建PROFILING_OPTIONS JSON字符串
        std::string profiling_options = "{";
        profiling_options += "\"output\":\"" + path + "\"";
        profiling_options += ", \"training_trace\":\"on\"";
        profiling_options += ", \"task_trace\":\"on\"";
        profiling_options += ", \"aicpu\":\"on\"";
        profiling_options += ", \"fp_point\":\"\"";
        profiling_options += ", \"bp_point\":\"\"";
        profiling_options += ", \"aic_metrics\":\"" + aic_metrics + "\"";
        profiling_options += ", \"runtime_api\":\"on\"";
        profiling_options += "}";
        setenv("PROFILING_OPTIONS", profiling_options.c_str(), 1);
    }
    pid_t pid = getpid();  // 获取当前进程ID，返回值为 pid_t 类型
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Current Process ID: ") + std::to_string(pid)).c_str());
}

// 解析GE配置
void ModelState::ParseGeConfig(const std::string &json_str)
{
    std::vector<std::string> npu_ge_config = {"device_ids",   "device_exec_blocks", "instance_exec_blocks",
                                              "static_model", "dump_graph",         "profiling",
                                              "dump_data"};
    try {
        json j = json::parse(json_str);
        if (j.contains("cmdline") && j["cmdline"].is_object()) {
            ParseCmdlineConfig(j["cmdline"], npu_ge_config);
        }
    } catch (const json::parse_error &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("JSON parsing error: ") + e.what()).c_str());
    } catch (const exception &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("Error occurred: ") + e.what()).c_str());
    }
}

void ModelState::SetOptions(std::string key, std::string value)
{
    vector<string> options = {"global.", "session.", "graph."};
    for (auto &option : options) {
        if (key.find(option) != std::string::npos) {
            string front = key.substr(0, option.size() - 1);
            string back = key.substr(option.size());
            geOption_[front][back] = value;
        }
    }
}

// 解析命令行配置
void ModelState::ParseCmdlineConfig(const json &cmdline, const std::vector<std::string> &npu_ge_config)
{
    std::string ge_pre = "ge.";
    for (auto &[key, value] : cmdline.items()) {
        if (key.size() >= ge_pre.size() && key.substr(0, ge_pre.size()) == ge_pre && value.is_string()) {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                        (std::string("ge config ") + "key:" + key + " value: " + value.get<string>()).c_str());
            ge_map_[key] = value.get<string>();
        }
        if (std::find(npu_ge_config.begin(), npu_ge_config.end(), key) != npu_ge_config.end()) {
            process_config_map_[key] = value.get<string>();
            LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                        (std::string("npu_ge config ") + "key: " + key + " value: " + value.get<string>()).c_str());
            if (key == "static_model" && value.get<string>() == "1") {
                LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("use ge static_model ")).c_str());
                SetGeStaticMode(true);
            }
            if (key == "dump_graph" && value.get<string>() == "1") {
                SetDumpGraph("./dump_graph");
            }
            if (key == "dump_data" && value.get<string>() == "1") {
                enable_dump_data_ = true;
            }
            if (key == "profiling") {
                SetProfiling(value.get<string>(), "./profiling", "PipeUtilization");
            }
        }
        SetOptions(key, value.get<string>());
    }
}

// 解析单个张量信息
TRITONSERVER_Error *ModelState::ParseTensorInfo(common::TritonJson::Value &tensor, OnnxTensorInfo &client_tensor,
                                                bool is_input)
{
    std::string name;
    RETURN_IF_ERROR(tensor.MemberAsString("name", &name));
    client_tensor.name_ = name;

    std::string data_type;
    RETURN_IF_ERROR(tensor.MemberAsString("data_type", &data_type));
    client_tensor.dtype_ = data_type_map[data_type];

    common::TritonJson::Value dims;
    RETURN_IF_ERROR(tensor.MemberAsArray("dims", &dims));

    std::string log_prefix = is_input ? "Input" : "Output";
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("  ") + log_prefix + " Name: " + name + ", Data Type: " + data_type + ", Dims: [").c_str());

    for (size_t j = 0; j < dims.ArraySize(); j++) {
        int64_t dim;
        RETURN_IF_ERROR(dims.IndexAsInt(j, &dim));
        client_tensor.dims_.push_back(dim);
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("  dim") + std::to_string(j) + ": " + std::to_string(dim)).c_str());
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("]")).c_str());

    return nullptr;
}

// 解析输入张量
TRITONSERVER_Error *ModelState::ParseInputTensors(common::TritonJson::Value &inputs)
{
    input_count_ = inputs.ArraySize();
    input_client_tensor_.resize(input_count_);

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Inputs:")).c_str());
    for (size_t i = 0; i < inputs.ArraySize(); i++) {
        common::TritonJson::Value input;
        RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));
        RETURN_IF_ERROR(ParseTensorInfo(input, input_client_tensor_[i], true));
    }
    return nullptr;
}

// 解析输出张量
TRITONSERVER_Error *ModelState::ParseOutputTensors(common::TritonJson::Value &outputs)
{
    output_count_ = outputs.ArraySize();
    output_client_tensor_.resize(output_count_);

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Outputs:")).c_str());
    for (size_t i = 0; i < outputs.ArraySize(); i++) {
        common::TritonJson::Value output;
        RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));
        RETURN_IF_ERROR(ParseTensorInfo(output, output_client_tensor_[i], false));
    }
    return nullptr;
}

TRITONSERVER_Error *ModelState::ParseDynamicBatching()
{
    triton::common::TritonJson::Value value;
    bool found_dynamic_batching = ModelConfig().Find("dynamic_batching", &value);
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("found_dynamic_batching :") + std::to_string(static_cast<int>(found_dynamic_batching))).c_str());
    triton::common::TritonJson::Value outputs;
    if (found_dynamic_batching) {
        std::string max_queue_delay_microseconds;
        int64_t delay_microseconds = 0;
        RETURN_IF_ERROR(value.MemberAsInt("max_queue_delay_microseconds", &delay_microseconds));
        dynamic_batching_.max_queue_delay_microseconds_ = delay_microseconds;
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("max_queue_delay_microseconds :") +
                                               std::to_string(static_cast<int>(delay_microseconds)))
                                                  .c_str());

        common::TritonJson::Value dims;
        RETURN_IF_ERROR(value.MemberAsArray("preferred_batch_size", &dims));
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string(" preferred_batch_size:  Dims: [")).c_str());
        for (size_t j = 0; j < dims.ArraySize(); j++) {
            int64_t dim;
            RETURN_IF_ERROR(dims.IndexAsInt(j, &dim));

            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (std::string("  dim") + std::to_string(j) + ": " + std::to_string(dim)).c_str());
            dynamic_batching_.preferred_batch_sizes.emplace_back(dim);
        }
        sort(dynamic_batching_.preferred_batch_sizes.begin(), dynamic_batching_.preferred_batch_sizes.end());
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("]")).c_str());
    }
    return nullptr;
}

TRITONSERVER_Error *ModelState::ParseModelConfig()
{
    common::TritonJson::Value inputs;
    common::TritonJson::Value outputs;
    model_config_.MemberAsArray("input", &inputs);
    model_config_.MemberAsArray("output", &outputs);
    model_config_.MemberAsString("name", &model_name_);
    ParseInputTensors(inputs);
    ParseOutputTensors(outputs);
    ParseDynamicBatching();
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("max_batch_size ") + std::to_string(MaxBatchSize())).c_str());

    return nullptr;
}

// 打印张量信息
void ModelState::PrintTensorInfo(const OnnxTensorInfo &tensor, const std::string &prefix)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (prefix + "Name: " + tensor.name_).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (prefix + "Data Type: " + std::to_string(tensor.dtype_)).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (prefix + "Dimensions: ").c_str());

    if (tensor.dims_.empty()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (prefix + "  None").c_str());
    } else {
        for (size_t i = 0; i < tensor.dims_.size(); ++i) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (prefix + "  dim" + std::to_string(i) + ": " + std::to_string(tensor.dims_[i])).c_str());
        }
    }
}

void ModelState::PrintClientTensors(const std::vector<OnnxTensorInfo> &tensors, const std::string &tensor_type)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("=== ") + tensor_type + " Tensors ===").c_str());
    for (const auto &tensor : tensors) {
        PrintTensorInfo(tensor, "  ");
    }
}

void ModelState::PrintModelConfig()
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("PrintModelConfig ::::::  start")).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("runtime_modeldir_ :") + std::string(runtime_modeldir_)).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("device_ids_ :") + device_ids_str_).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("input_count_ ") + std::to_string(static_cast<int>(input_count_))).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("output_count_ ") + std::to_string(static_cast<int>(output_count_))).c_str());
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("can_dynamic_batching: ") + std::to_string(static_cast<int>(can_dynamic_batching))).c_str());

    for (const auto &[key, value] : parameters_) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Key: ") + key + ", Value: " + value).c_str());
    }

    PrintClientTensors(input_client_tensor_, "Input");
    PrintClientTensors(output_client_tensor_, "Output");

    if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
        triton::common::TritonJson::WriteBuffer buffer;
        ModelConfig().PrettyWrite(&buffer);
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Default Config:\n") + buffer.Contents()).c_str());
    }
}

// 处理设备ID配置
void ModelState::ProcessDeviceIdsConfig()
{
    if (GetConfig().count("device_ids")) {
        std::vector<std::string> dev_ids_str = Split(GetConfig().at("device_ids"), ',');
        for (const auto &dev_id_str : dev_ids_str) {
            device_ids_.push_back(std::stoi(dev_id_str));
        }
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("backend-config set device_ids ") + GetConfig().at("device_ids")).c_str());
    }
}

// 处理设备执行块配置
void ModelState::ProcessDeviceExecBlocksConfig()
{
    if (GetConfig().count("device_exec_blocks")) {
        device_exec_block_ = std::stoi(GetConfig().at("device_exec_blocks"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("backend-config set device_exec_blocks ") + GetConfig().at("device_exec_blocks")).c_str());
    }
}

// 处理实例执行块配置
void ModelState::ProcessInstanceExecBlocksConfig()
{
    if (GetConfig().count("instance_exec_blocks")) {
        instance_exec_block_ = std::stoi(GetConfig().at("instance_exec_blocks"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string("backend-config set instance_exec_block_ ") + GetConfig().at("instance_exec_blocks")).c_str());
    }
}

void ModelState::DisposeConfig()
{
    ProcessDeviceIdsConfig();
    ProcessDeviceExecBlocksConfig();
    ProcessInstanceExecBlocksConfig();
}

// 解析实例组配置
void ModelState::ParseInstanceGroupConfig()
{
    triton::common::TritonJson::Value instance_group;
    if (ModelConfig().Find("instance_group", &instance_group)) {
        for (size_t i = 0; i < instance_group.ArraySize(); ++i) {
            triton::common::TritonJson::Value instance_obj;
            instance_group.IndexAsObject(i, &instance_obj);
            int64_t count_str;
            instance_obj.MemberAsInt("count", &count_str);
            triton_thread_count_ += count_str;
        }
    }
}

// 解析模型参数配置
void ModelState::ParseModelParametersConfig()
{
    triton::common::TritonJson::Value params;
    if (model_config_.Find("parameters", &params)) {
        ParseParameterValue(params, "device_ids", device_ids_str_, device_ids_);
        ParseParameterValue(params, "device_exec_blocks", device_exec_block_);
        ParseParameterValue(params, "instance_exec_blocks", instance_exec_block_);
        int tmp_infer_mode = static_cast<int>(infer_mode_);
        ParseParameterValue(params, "static_model", tmp_infer_mode);
        if (tmp_infer_mode == 1) {
            SetGeStaticMode(true);
        }
        std::string tmp;
        TRITONSERVER_Error *error = GetParameterValue(params, "dump_graph", &tmp);
        if (error == nullptr && tmp == "1") {
            SetDumpGraph("./dump_graph");
        }
        error = GetParameterValue(params, "profiling", &tmp);
        if (error == nullptr && (tmp == "dynamic" || tmp == "true")) {
            SetProfiling(tmp, "./profiling", "PipeUtilization");
        }
        error = GetParameterValue(params, "dump_data", &tmp);
        if (error == nullptr && tmp == "1") {
            enable_dump_data_ = true;
        }
    }
}

void ModelState::PrintGeOptions()
{
    for (const auto &outer_pair : geOption_) {
        const std::string &outer_key = outer_pair.first;
        const std::map<std::string, std::string> &inner_map = outer_pair.second;
        for (const auto &inner_pair : inner_map) {
            const std::string &inner_key = inner_pair.first;
            const std::string &output_value = inner_pair.second;
            // 使用 LOG_MESSAGE 打印日志
            LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                        (std::string("options ") + outer_key + "." + inner_key + " " + output_value).c_str());
        }
    }
}

// 解析参数值（字符串类型）
void ModelState::ParseParameterValue(common::TritonJson::Value &params, const std::string &key,
                                     std::string &output_value, std::vector<int> &output_ids)
{
    TRITONSERVER_Error *error = GetParameterValue(params, key, &output_value);
    if (error == nullptr) {
        output_ids.clear();
        std::vector<std::string> dev_ids_str = Split(output_value, ',');
        for (const auto &dev_id_str : dev_ids_str) {
            output_ids.push_back(std::stoi(dev_id_str));
        }
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("parse ") + key + " " + output_value).c_str());
    } else {
        TRITONSERVER_ErrorDelete(error);
    }
}

// 解析参数值（整数类型）
void ModelState::ParseParameterValue(common::TritonJson::Value &params, const std::string &key, int &output_value)
{
    std::string tmp;
    TRITONSERVER_Error *error = GetParameterValue(params, key, &tmp);
    if (error == nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("parse ") + key + " " + tmp).c_str());
        output_value = std::stoi(tmp);
    } else {
        TRITONSERVER_ErrorDelete(error);
    }
}

// 查找模型文件
void ModelState::FindModelFile()
{
    model_file_ = FindFirstFile(std::string(runtime_modeldir_), ".onnx");
    if (model_file_ != "") {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("find onnx path: ") + model_file_).c_str());
        model_type_ = ModelState::ModelType::ONNX;
        return;
    }
    model_file_ = FindFirstFile(std::string(runtime_modeldir_), ".pb");
    if (model_file_ != "") {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("find tensorflow pb path: ") + model_file_).c_str());
        model_type_ = ModelState::ModelType::TENSORFLOW;
        return;
    }
    TRITONSERVER_Error *error =
        TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "can't find model file in model path!");
    TRITONSERVER_ErrorDelete(error);
}

// 确定推理模式
void ModelState::DetermineInferMode()
{
    if (MaxBatchSize() > 0) {
        SetInferMode(0);
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("use dynamic_batchsize ")).c_str());
    } else {
        SetInferMode(1);
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("use static_batchsize ")).c_str());
    }
}

// 初始化后端配置
void ModelState::InitializeBackendConfig(TRITONBACKEND_Model *triton_model)
{
    TRITONBACKEND_Backend *backend;
    THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_ModelBackend(triton_model, &backend));

    TRITONSERVER_Message *backend_config_message;
    TRITONBACKEND_BackendConfig(backend, &backend_config_message);

    const char *buffer;
    size_t byte_size;
    TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size);

    if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
        triton::common::TritonJson::Value json_root;
        json_root.Parse(buffer, byte_size);
        triton::common::TritonJson::WriteBuffer pretty_buffer;
        json_root.PrettyWrite(&pretty_buffer);

        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("backend configuration:\n") + pretty_buffer.Contents()).c_str());
    }

    TRITONBACKEND_ArtifactType artifact_type;
    THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &runtime_modeldir_));

    ParseGeConfig(std::string(buffer));
}

int ModelState::GetFirstDimNum()
{
    if (input_client_tensor_.size() > 0 && output_client_tensor_.size() > 0) {
        int64_t compare = input_client_tensor_[0].dims_[0];
        int flag = 0;
        for (size_t i = 0; i < input_client_tensor_.size(); i++) {
            if (compare != input_client_tensor_[i].dims_[0]) {
                flag = 1;
                break;
            }
        }
        if (flag == 1) {
            return INT_MAX;
        }
        return compare;
    }
    if (input_onnx_tensor_.size() > 0 && output_onnx_tensor_.size() > 0) {
        int64_t compare = input_onnx_tensor_[0].dims_[0];
        int flag = 0;
        for (size_t i = 0; i < input_onnx_tensor_.size(); i++) {
            if (compare != input_onnx_tensor_[i].dims_[0]) {
                flag = 1;
                break;
            }
        }
        if (flag == 1) {
            return INT_MAX;
        }
        return compare;
    } else {
        return INT_MAX - 1;
    }
}

// 非维度0中包含-1
bool ModelState::ContainNegativeOneFromTensor(size_t index)
{
    if (input_client_tensor_.size() > 0 && output_client_tensor_.size() > 0) {
        return ContainNegativeOne(output_client_tensor_, index);
    }

    if (input_onnx_tensor_.size() > 0 && output_onnx_tensor_.size() > 0) {
        if (MaxBatchSize() > 0) {
            index = 1;
        }
        return ContainNegativeOne(output_onnx_tensor_, index);
    }

    return false;
}

// -1 未读取到有效数值 0 不相等 1 相等
int ModelState::FirstDimNameSame()
{
    if (input_onnx_tensor_.size() > 0 && output_onnx_tensor_.size() > 0) {
        string compare = input_onnx_tensor_[0].dim_name_[0];
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("compare") + compare).c_str());
        int flag = 0;
        for (size_t i = 0; i < input_onnx_tensor_.size(); i++) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_VERBOSE,
                (std::string("input_onnx_tensor_[i].dim_name_[0] ") + input_onnx_tensor_[i].dim_name_[0]).c_str());
            if (compare != input_onnx_tensor_[i].dim_name_[0]) {
                flag = 1;
                break;
            }
        }
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("flag ") + std::to_string(flag)).c_str());
        if (flag == 1) {
            return 0;
        }
        return 1;
    }
    return -1;
}

void ModelState::SetModelMode()
{
    if (MaxBatchSize() > 0) {
        if (ContainNegativeOneFromTensor(0)) {
            model_mode_ = ModelMode::MAX_BATCH_HAVE_UNKNOW_DIM;
        } else {
            model_mode_ = ModelMode::MAX_BATCH;
        }
        return;
    }
    int res = GetFirstDimNum();
    if (res == INT_MAX) {
        if (ContainNegativeOneFromTensor(0)) {
            model_mode_ = ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE;
        } else {
            model_mode_ = ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME;
        }
    } else if (res == INT_MAX - 1) {
        model_mode_ = ModelMode::TENSOR_ZERO;
    } else if (res == -1) {
        // 判断-1的dim_name_是否一致
        int ret = FirstDimNameSame();
        if (ret == -1) {
            model_mode_ = ModelMode::TENSOR_ZERO;
        } else if (ret == 0) {
            model_mode_ = ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE;
        } else if (ContainNegativeOneFromTensor()) {
            model_mode_ = ModelMode::NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE_HAVE_UNKNOW_DIM;
        } else {
            model_mode_ = ModelMode::NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE;
        }
    } else {
        if (ContainNegativeOneFromTensor()) {
            model_mode_ = ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM;
        } else {
            model_mode_ = ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE;
        }
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Current mode ") + PrintModelMode(model_mode_)).c_str());
}

ModelState::ModelState(TRITONBACKEND_Model *triton_model) : BackendModel(triton_model), scheduler_()
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ModelState ::::::  start")).c_str());

    InitializeBackendConfig(triton_model);
    DisposeConfig();
    ParseModelConfig();
    ParseInstanceGroupConfig();
    ParseModelParametersConfig();
    FindModelFile();
    DetermineInferMode();
    ParseOnnxInfo();
    if (model_type_ == ModelState::ModelType::ONNX) {
        ParseOnnxInfo();
    }
    SetModelMode();
}

// 检查是否为文件并返回路径
std::string ModelState::CheckAndReturnFile(const std::string &filePath, const std::string &extension)
{
    if (filePath.size() >= extension.size() && filePath.substr(filePath.size() - extension.size()) == extension) {
        return filePath;  // 返回副本，安全
    }
    return "";  // 返回空字符串表示未找到
}

// 在目录中搜索ONNX文件
std::string ModelState::SearchFileInDirectory(const std::string &path, const std::string &extension)
{
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        std::string result = ProcessEntry(entry, extension);
        if (!result.empty()) {
            return result;
        }
    }
    return "";  // 返回空字符串表示未找到
}

std::string ModelState::ProcessEntry(const std::filesystem::directory_entry &entry, const std::string &extension)
{
    try {
        if (entry.is_directory()) {
            std::string result = FindFirstFile(entry.path().string(), extension);
            if (!result.empty()) {
                return result;
            }
        } else if (entry.is_regular_file()) {
            std::string result = CheckAndReturnFile(entry.path().string(), extension);
            if (!result.empty()) {
                return result;
            }
        }
    } catch (const std::filesystem::filesystem_error &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ("Error accessing file: " + std::string(e.what())).c_str());
        // 返回空字符串表示处理失败，但不抛出异常
        return "";
    } catch (const std::exception &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ("General error: " + std::string(e.what())).c_str());
        return "";
    }
    return "";
}

// 递归查找第一个.onnx文件
std::string ModelState::FindFirstFile(const std::string &path, const std::string &extension)
{
    try {
        if (!std::filesystem::exists(path)) {
            return "";
        }

        if (!std::filesystem::is_directory(path)) {
            return CheckAndReturnFile(path, extension);
        }

        return SearchFileInDirectory(path, extension);
    } catch (const std::filesystem::filesystem_error &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ("File system error: " + std::string(e.what())).c_str());
    }

    return "";
}

// 移除字符串中的所有空格
std::string ModelState::RemoveSpaces(const std::string &str)
{
    std::string result;
    for (char c : str) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            result += c;
        }
    }
    return result;
}

// 检查字符是否可以作为变量名的开头
bool ModelState::IsValidVariableStart(char c)
{
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

// 检查字符是否可以作为变量名的一部分
bool ModelState::IsValidVariableChar(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::unordered_set<std::string> ModelState::ExtractVariables(const std::string &expr)
{
    std::unordered_set<std::string> variables;
    std::string cleanedExpr = RemoveSpaces(expr);

    size_t i = 0;
    while (i < cleanedExpr.size()) {
        // 检查是否可能是变量名的开始
        if (IsValidVariableStart(cleanedExpr[i])) {
            std::string varName;

            // 提取完整的变量名
            while (i < cleanedExpr.size() && IsValidVariableChar(cleanedExpr[i])) {
                varName += cleanedExpr[i];
                i++;
            }

            // 添加到集合中
            if (!varName.empty()) {
                variables.insert(varName);
            }
        } else {
            i++;
        }
    }

    return variables;
}

// 根据记录和pbtxt的获取值进行对比

TRITONSERVER_Error *ModelState::FindOutputDim(std::unordered_set<std::string> &umap1, Express &ex)
{
    return ProcessFindOutputDim(umap1, ex);
}

TRITONSERVER_Error *ModelState::ProcessFindOutputDim(std::unordered_set<std::string> &umap1, Express &ex)
{
    // 处理每个查找名称
    for (auto it = umap1.begin(); it != umap1.end(); it++) {
        string findname = *it;
        TRITONSERVER_Error *error = FindAndSetDim(ex, findname);
        if (error != nullptr) {
            return error;
        }
    }
    return nullptr;
}

TRITONSERVER_Error *ModelState::FindAndSetDim(Express &ex, const string &findname)
{
    int flag = 0;
    for (size_t i = 0; i < input_client_tensor_.size(); i++) {
        std::vector<std::string> v2 = input_client_tensor_[i].dim_name_;

        for (size_t j = 0; j < v2.size(); j++) {
            if (v2[j] == findname) {
                ex.dimMap[findname] = make_pair(i, j);
                flag = 1;
                break;
            }
        }
        if (flag == 1) {
            break;
        }
    }

    if (flag == 0) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, (findname + " have not find ").c_str());
    }
    return nullptr;
}

void ModelState::LogInputDimMapOutputDim()
{
    for (const auto &[input_tensor_dim, express] : input_dim_map_output_dim) {
        // Output input dimension and corresponding expression information
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("Input mapping: Tensor ") + std::to_string(input_tensor_dim.first) + " dimension " +
                     std::to_string(input_tensor_dim.second) + " -> Expression: " + express.expressName)
                        .c_str());

        // Output dimension mapping of each tensor in the expression
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("  Tensor dimension mapping in expression '") + express.expressName + "':").c_str());

        for (const auto &[tensor_name, tensor_dim] : express.dimMap) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (std::string("    Tensor '") + tensor_name + "' -> Tensor " + std::to_string(tensor_dim.first) +
                         " dimension " + std::to_string(tensor_dim.second))
                            .c_str());
        }

        // Separator line
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "----------------------------------------");
    }
}

void ModelState::AddNegativeOneInfo()
{
    for (size_t i = 0; i < output_count_; i++) {
        auto &onnxInfo = output_client_tensor_[i].dims_;
        for (size_t j = 0; j < onnxInfo.size(); j++) {
            if (onnxInfo[j] == -1) {
                negativeOne.push_back({i, j});
            }
        }
    }
}

TRITONSERVER_Error *ModelState::CreateInputdimToOutputdim()
{
    AddNegativeOneInfo();
    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (std::string("negativeOne.size() ") + std::to_string(negativeOne.size())).c_str());
    for (size_t i = 0; i < negativeOne.size(); i++) {
        pair<size_t, size_t> p1 = negativeOne[i];
        string temp = output_client_tensor_[p1.first].dim_name_[p1.second];
        if (temp == "") {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (std::string("Tensor ") + to_string(p1.first) + ": " + output_client_tensor_[i].name_ +
                         " dimension " + to_string(p1.second) + " key is empty")
                            .c_str());
            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, (std::string("-1 dim has no name")).c_str());
        }
        if (output_client_tensor_[p1.first].dims_[p1.second] != -1) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (std::string("Tensor ") + to_string(p1.first) + ": " + output_client_tensor_[i].name_ +
                         " dimension " + to_string(p1.second) + " is not -1")
                            .c_str());
            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, (std::string("-1 dim find error")).c_str());
        }
        unordered_set<std::string> umap1 = ExtractVariables(temp);
        Express ex;
        ex.expressName = temp;
        RETURN_IF_ERROR(FindOutputDim(umap1, ex));
        input_dim_map_output_dim[{p1.first, p1.second}] = ex;
    }
    LogInputDimMapOutputDim();
    return nullptr;
}

TRITONSERVER_Error *ModelState::CheckModelMode()
{
    if (model_mode_ == ModelMode::TENSOR_ZERO) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                                     (std::string("not support model_mode_") + PrintModelMode(model_mode_)).c_str());
    }
    return nullptr;
}

TRITONSERVER_Error *ModelState::InputdimToOutputdimMap(ModelState **state)
{
    if (model_mode_ == ModelMode::MAX_BATCH || model_mode_ == ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE ||
        model_mode_ == ModelMode::NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE ||
        model_mode_ == ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME) {
        // No need to process, keep input_client_tensor_ settings
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Current mode ") + PrintModelMode(model_mode_)).c_str());
    }
    if (model_mode_ == ModelMode::MAX_BATCH_HAVE_UNKNOW_DIM ||
        model_mode_ == ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM ||
        model_mode_ == ModelMode::NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE_HAVE_UNKNOW_DIM ||
        model_mode_ == ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Current mode ") + PrintModelMode(model_mode_)).c_str());
        RETURN_IF_ERROR((*state)->CreateInputdimToOutputdim());
    }
    return nullptr;
}

void ModelState::ChangeOutputdim()
{
    int index = 0;
    ModelState::ModelMode modelmode = GetModelNode();
    if (modelmode == ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE ||
        modelmode ==
            ModelMode::
                NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM) {  // 2 NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE 3 NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM
        index = 1;
    }
    for (size_t i = 0; i < output_client_tensor_.size(); i++) {
        output_client_tensor_[i].dim_name_[index] = "((seq_len*1)-0)/1";
    }
}

// 对比onnx的信息核配置信息 修改与onnx不一样的地方
void ModelState::CompareOnnxAndTxt()
{
    int count = 0;
    if (input_count_ != input_onnx_tensor_.size()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("input_count_ mismatch ") + to_string(input_count_) + " " +
                                               to_string(input_onnx_tensor_.size()))
                                                  .c_str());
        return;
    }
    if (output_count_ != output_onnx_tensor_.size()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("output_count_ mismatch") + to_string(output_count_) + " " +
                                               to_string(output_onnx_tensor_.size()))
                                                  .c_str());
        return;
    }
    for (size_t i = 0; i < input_count_; i++) {
        std::vector<int64_t> &v1 = input_client_tensor_[i].dims_;
        std::vector<int64_t> &v2 = input_onnx_tensor_[i].dims_;
        for (size_t j = 0; j < v1.size(); j++) {
            if (v1[j] != v2[j] && v2[j] != -1 && v1[j] == -1) {
                v1[j] = v2[j];
                count++;
                LOG_MESSAGE(TRITONSERVER_LOG_WARN, (std::string("replace ") + "input tensor " + to_string(i) +
                                                    " dimension " + to_string(j) + " replaced with " + to_string(v2[j]))
                                                       .c_str());
            }
        }
    }
    for (size_t i = 0; i < output_count_; i++) {
        std::vector<int64_t> &v1 = output_client_tensor_[i].dims_;
        std::vector<int64_t> &v2 = output_onnx_tensor_[i].dims_;
        for (size_t j = 0; j < v1.size(); j++) {
            if (v1[j] != v2[j] && v2[j] != -1 && v1[j] == -1) {
                v1[j] = v2[j];
                count++;
                LOG_MESSAGE(TRITONSERVER_LOG_WARN, (std::string("replace ") + "output tensor " + to_string(i) +
                                                    " dimension " + to_string(j) + " replaced with " + to_string(v2[j]))
                                                       .c_str());
            }
        }
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Replaced ") + to_string(count) + " dimensions").c_str());
    if (count > 0) {
        SetModelMode();
    }
}

std::string ModelState::PrintModelMode(ModelState::ModelMode mode)
{
    switch (mode) {
        case ModelState::ModelMode::MAX_BATCH:
            return "MAX_BATCH";
        case ModelState::ModelMode::MAX_BATCH_HAVE_UNKNOW_DIM:
            return "MAX_BATCH_HAVE_UNKNOW_DIM";
        case ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE:
            return "NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE";
        case ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM:
            return "NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM";
        case ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE:
            return "NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE";
        case ModelState::ModelMode::NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE_HAVE_UNKNOW_DIM:
            return "NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE_HAVE_UNKNOW_DIM";
        case ModelState::ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME:
            return "NO_MAX_BATCH_FIRST_NOT_SAME";
        case ModelState::ModelMode::TENSOR_ZERO:
            return "TENSOR_ZERO";
        case ModelState::ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE:
            return "NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE";
        default:
            return "UNKNOWN_MODEL_MODE: " + to_string(static_cast<int>(mode));
    }
}

TRITONSERVER_Error *ModelState::Create(TRITONBACKEND_Model *triton_model, ModelState **state)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Create MindIE ModelState:")).c_str());
    try {
        *state = new ModelState(triton_model);
    } catch (const BackendModelException &ex) {
        RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                             std::string("unexpected nullptr in BackendModelException"));
        RETURN_IF_ERROR(ex.err_);
    }
    RETURN_IF_ERROR((*state)->CheckModelMode());

    triton::common::TritonJson::Value parameters;
    (*state)->ModelConfig().Find("parameters", &parameters);
    vector<string> member_names;
    TRITONJSON_STATUSTYPE status = parameters.Members(&member_names);
    if (status != TRITONJSON_STATUSSUCCESS) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to obtain JSON object member name");
    }

    for (size_t i = 0; i < member_names.size(); i++) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("Member name: " + member_names[i]).c_str());
        string tmp_str;
        TRITONSERVER_Error *error = GetParameterValue(parameters, member_names[i], &tmp_str);
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("Member value: " + tmp_str).c_str());
        if (error == nullptr) {
            (*state)->SetOptions(member_names[i], tmp_str);
        } else {
            TRITONSERVER_ErrorDelete(error);
        }
    }

    // 提前将模型的配置的输出维度配置好
    // 1. MAX_BATCH [512] 按照模型解析出来的记录
    // 2. MAX_BATCH_HAVE_UNKNOW_DIM [-1] 需要建立映射 map<pair<int, int>, structmapoutput>
    // 3. NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE 静态图 按照模型解析出来的记录
    // 4. NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM  需要建立映射 map<pair<int, int>, structmapoutput>
    // 5. NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE, 按照模型解析出来的记录
    // 6. NO_MAX_BATCH_FIRST_SAME_NEGATIVE_ONE_HAVE_UNKNOW_DIM 需要建立映射 map<pair<int, int>, structmapoutput>
    bool auto_complete_config = false;
    RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(triton_model, &auto_complete_config));
    if (auto_complete_config) {
        if ((*state)->GetModelType() == ModelState::ModelType::ONNX) {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, "start auto complete onnx config");
            RETURN_IF_ERROR((*state)->AutoCompleteConfig());
            ModelMode modelmode = (*state)->GetModelNode();
            if (modelmode != ModelMode::MAX_BATCH && modelmode != ModelMode::MAX_BATCH_HAVE_UNKNOW_DIM &&
                modelmode != ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME &&
                modelmode != ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE) {
                RETURN_IF_ERROR((*state)->PreNegativeOne());
            }
            RETURN_IF_ERROR((*state)->SetModelConfig());
        }
    }
    (*state)->CheckDynamicBatch();
    (*state)->PrintModelConfig();
    (*state)->PrintGeOptions();
    RETURN_IF_ERROR((*state)->InputdimToOutputdimMap(state));

    return nullptr;  // success
}

ModelState::~ModelState()
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ModelState ~ start: ")).c_str());
}

std::vector<std::string> ModelState::GetOnnxSymbolicDimension(const Ort::TypeInfo &type_info)
{
    std::vector<std::string> result;
    try {
        result = ProcessSymbolicDimensions(type_info);
    } catch (const Ort::Exception &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("GetSymbolicDimensions() not available:  ") + e.what()).c_str());
    }
    return result;
}

std::vector<std::string> ModelState::ProcessSymbolicDimensions(const Ort::TypeInfo &type_info)
{
    std::vector<std::string> result;
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<const char *> raw_dims = tensor_info.GetSymbolicDimensions();

    result.reserve(raw_dims.size());
    for (const char *dim : raw_dims) {
        result.push_back(dim ? std::string(dim) : "unknown");
    }

    LogSymbolicDimensions(result);
    return result;
}

void ModelState::LogSymbolicDimensions(const std::vector<std::string> &result)
{
    if (!result.empty()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("  Symbolic dimensions: ")).c_str());
        for (size_t j = 0; j < result.size(); j++) {
            if (j > 0) {
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string(", ")).c_str());
            }
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (std::string("Dimension") + std::to_string(j) + ": '" + result[j] + "'").c_str());
        }
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("  No symbolic dimension information")).c_str());
    }
}
bool ModelState::FindMapResult(std::string &name, std::tuple<size_t, size_t, std::string> &t1)
{
    size_t len = input_onnx_tensor_.size();
    if (name == "") {
        return false;
    }
    for (size_t i = 0; i < len; i++) {
        OnnxTensorInfo &in = input_onnx_tensor_[i];
        vector<std::string> &in_dim_name = in.dim_name_;
        for (size_t j = 1; j < in_dim_name.size(); j++) {
            if (in_dim_name[j].find(name) != string::npos) {
                t1 = (make_tuple(i, j, in_dim_name[j]));
                return true;
            }
        }
    }
    return false;
}

void ModelState::ParseOnnxInfo()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelIOInfo");

    // 创建会话
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_file_.c_str(), session_options);

    // 获取输入输出数量
    onnx_input_count_ = session.GetInputCount();
    onnx_output_count_ = session.GetOutputCount();

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, model_name_.c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, ("onnx_input_count_: " + to_string(onnx_input_count_)).c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, ("onnx_output_count_: " + to_string(onnx_output_count_)).c_str());

    input_onnx_tensor_.resize(onnx_input_count_);
    output_onnx_tensor_.resize(onnx_output_count_);

    for (size_t i = 0; i < onnx_input_count_; ++i) {
        // 读取数据名称
        input_onnx_tensor_[i].name_ = session.GetInputNames()[i];

        // 获取数据类型
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        input_onnx_tensor_[i].dtype_ = GetType(type);

        // 获取数据形状
        input_onnx_tensor_[i].dims_ = tensor_info.GetShape();
        input_onnx_tensor_[i].dim_name_ = GetOnnxSymbolicDimension(type_info);
    }

    for (size_t i = 0; i < onnx_output_count_; ++i) {
        // 读取数据名称
        output_onnx_tensor_[i].name_ = session.GetOutputNames()[i];

        // 获取数据类型
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        output_onnx_tensor_[i].dtype_ = GetType(type);

        // 获取数据形状
        output_onnx_tensor_[i].dims_ = tensor_info.GetShape();
        output_onnx_tensor_[i].dim_name_ = GetOnnxSymbolicDimension(type_info);
    }
}

TRITONSERVER_Error *ModelState::CheckConfigIO()
{
    if (input_count_ == 0 || output_count_ == 0) {
        return nullptr;
    }

    for (size_t i = 0; i < input_count_; ++i) {
        for (size_t j = 0; j < input_client_tensor_[i].dims_.size(); ++j) {
            if (!(input_client_tensor_[i].dims_[j] == input_onnx_tensor_[i].dims_[j] ||
                  input_onnx_tensor_[i].dims_[j] == -1)) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_UNSUPPORTED,
                    (input_client_tensor_[i].name_ + " have invalid input dim at: " + std::to_string(j),
                     "it should be " + std::to_string(input_onnx_tensor_[i].dims_[j]))
                        .c_str());
            }
        }
    }

    for (size_t i = 0; i < output_count_; ++i) {
        for (size_t j = 0; j < output_client_tensor_[i].dims_.size(); ++j) {
            if (!(output_client_tensor_[i].dims_[j] == output_onnx_tensor_[i].dims_[j] ||
                  output_onnx_tensor_[i].dims_[j] == -1)) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_UNSUPPORTED,
                    (output_client_tensor_[i].name_ + " have invalid output dim at: " + std::to_string(j),
                     "it should be " + std::to_string(output_onnx_tensor_[i].dims_[j]))
                        .c_str());
            }
        }
    }

    return nullptr;
}

TRITONSERVER_Error *ModelState::PreNegativeOne()
{
    for (size_t i = 0; i < input_count_; ++i) {
        if (input_client_tensor_[i].dims_[0] == -1) {
            input_client_tensor_[i].dims_.erase(input_client_tensor_[i].dims_.begin());
            input_client_tensor_[i].dim_name_.erase(input_client_tensor_[i].dim_name_.begin());
        }
    }
    for (size_t i = 0; i < output_count_; ++i) {
        if (output_client_tensor_[i].dims_[0] == -1) {
            output_client_tensor_[i].dims_.erase(output_client_tensor_[i].dims_.begin());
            output_client_tensor_[i].dim_name_.erase(output_client_tensor_[i].dim_name_.begin());
        }
    }

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "before input and output config:");
    for (const auto &tensor : input_client_tensor_) {
        PrintVector(tensor.dims_, tensor.name_, tensor.dim_name_);
    }
    for (const auto &tensor : output_client_tensor_) {
        PrintVector(tensor.dims_, tensor.name_, tensor.dim_name_);
    }

    return nullptr;
}

void ModelState::AddDimName()
{
    for (size_t i = 0; i < input_client_tensor_.size(); i++) {
        if (input_client_tensor_[i].dim_name_.size() == 0) {
            input_client_tensor_[i].dim_name_ = input_onnx_tensor_[i].dim_name_;
        }
    }
    for (size_t i = 0; i < output_client_tensor_.size(); i++) {
        if (output_client_tensor_[i].dim_name_.size() == 0) {
            output_client_tensor_[i].dim_name_ = output_onnx_tensor_[i].dim_name_;
        }
    }
}

TRITONSERVER_Error *ModelState::AutoCompleteConfig()
{
    if (input_count_ > 0 && output_count_ > 0) {
        AddDimName();
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "no need auto config");
        return nullptr;  // success
    }

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Start AutoCompleteConfig!");
    ModelState::ModelMode modelmode = GetModelNode();
    if (input_count_ == 0) {
        RETURN_IF_ERROR(AutoCompleteIO("input", input_onnx_tensor_));
        input_client_tensor_ = input_onnx_tensor_;
        input_count_ = input_onnx_tensor_.size();

        if (modelmode == ModelMode::MAX_BATCH || modelmode == ModelMode::MAX_BATCH_HAVE_UNKNOW_DIM) {
            for (size_t i = 0; i < input_count_; i++) {
                input_client_tensor_[i].dims_.erase(input_client_tensor_[i].dims_.begin());
            }
        }
    }

    if (output_count_ == 0) {
        RETURN_IF_ERROR(AutoCompleteIO("output", output_onnx_tensor_));
        output_client_tensor_ = output_onnx_tensor_;
        output_count_ = output_onnx_tensor_.size();
        if (modelmode == ModelMode::MAX_BATCH || modelmode == ModelMode::MAX_BATCH_HAVE_UNKNOW_DIM) {
            for (size_t i = 0; i < output_count_; i++) {
                output_client_tensor_[i].dims_.erase(output_client_tensor_[i].dims_.begin());
            }
        }
    }

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "after input and output config:");
    for (const auto &tensor : input_client_tensor_) {
        PrintVector(tensor.dims_, tensor.name_, tensor.dim_name_);
    }
    for (const auto &tensor : output_client_tensor_) {
        PrintVector(tensor.dims_, tensor.name_, tensor.dim_name_);
    }

    // 打印更新后的ModelConfig
    if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
        triton::common::TritonJson::WriteBuffer buffer;
        RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("AutoCompleteConfig:\n") + buffer.Contents()).c_str());
    }

    return nullptr;
}

TRITONSERVER_Error *ModelState::AutoCompleteIO(const char *key, const std::vector<OnnxTensorInfo> &io_infos)
{
    triton::common::TritonJson::Value existing_ios;
    bool found_ios = ModelConfig().Find(key, &existing_ios);

    triton::common::TritonJson::Value ios(ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);

    for (const auto &tensor : io_infos) {
        triton::common::TritonJson::Value io(ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);

        RETURN_IF_ERROR(io.AddString("name", tensor.name_));

        RETURN_IF_ERROR(io.AddString("data_type", GeDataTypeToModelConfigDataTypeMap[tensor.dtype_]));

        triton::common::TritonJson::Value dims(ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);

        if (MaxBatchSize() != 0) {
            for (size_t i = 1; i < tensor.dims_.size(); ++i) {
                RETURN_IF_ERROR(dims.AppendInt(tensor.dims_[i]));
            }
        } else {
            for (const auto dim : tensor.dims_) {
                RETURN_IF_ERROR(dims.AppendInt(dim));
            }
        }

        RETURN_IF_ERROR(io.Add("dims", std::move(dims)));
        RETURN_IF_ERROR(ios.Append(std::move(io)));
    }
    if (found_ios) {
        existing_ios.Swap(ios);
    } else {
        ModelConfig().Add(key, std::move(ios));
    }

    return nullptr;
}

bool ModelState::CheckDynamicBatch()
{
    bool can_support_batching = true;
    ModelMode modelmode = GetModelNode();
    if (modelmode == ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE ||
        modelmode == ModelMode::NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM ||
        modelmode == ModelMode::NO_MAX_BATCH_FIRST_NOT_SAME ||
        modelmode ==
            ModelMode::
                NO_MAX_BATCH_FIRST_NOT_SAME_EXIST_NEGATIVE) {  // 2 NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE 3 NO_MAX_BATCH_FIRST_SAME_NOT_NEGATIVE_ONE_HAVE_UNKNOW_DIM
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("can_dynamic_batching: ") + std::to_string(static_cast<int>(can_dynamic_batching))).c_str());
        can_dynamic_batching = false;
        return false;
    }
    for (const auto &tensor : input_onnx_tensor_) {
        const auto &dims = tensor.dims_;
        if ((dims.size() == 0) || (dims[0] != -1)) {
            can_support_batching = false;
        }
    }
    for (const auto &tensor : output_onnx_tensor_) {
        const auto &dims = tensor.dims_;
        if ((dims.size() == 0) || (dims[0] != -1)) {
            can_support_batching = false;
        }
    }
    // 设置flag
    can_dynamic_batching = can_support_batching;
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("can_dynamic_batching: ") + std::to_string(static_cast<int>(can_dynamic_batching))).c_str());
    return can_support_batching;
}
}