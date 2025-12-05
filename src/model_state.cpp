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
    // 验证路径安全性（防止命令注入）
    if (path.empty() || path.find(";") != std::string::npos || path.find("&") != std::string::npos ||
        path.find("|") != std::string::npos) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "错误: 路径包含不安全字符或为空");
    }
    // 设置图转储路径
    setenv("DUMP_GRAPH_PATH", path.c_str(), 1);
    // 设置打印模型标志
    setenv("PRINT_MODEL", "1", 1);
    // 设置GE图转储级别
    setenv("DUMP_GE_GRAPH", "2", 1);
    // 设置图转储级别
    setenv("DUMP_GRAPH_LEVEL", "2", 1);
    // 清理转储目录
    std::string cleanup_cmd = "rm -rf " + path + "/* 2>/dev/null";
    int result = system(cleanup_cmd.c_str());
    if (result != 0) {
        std::cout << "注意: 清理目录时可能出现问题，命令: " << cleanup_cmd << std::endl;
        // 这里不返回false，因为可能只是目录为空
    } else {
        std::cout << "成功清理目录: " << path << std::endl;
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
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("当前进程ID: ") + std::to_string(pid)).c_str());
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
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("JSON解析错误: ") + e.what()).c_str());
    } catch (const exception &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("发生错误: ") + e.what()).c_str());
    }
}

// 解析命令行配置
void ModelState::ParseCmdlineConfig(const json &cmdline, const std::vector<std::string> &npu_ge_config)
{
    std::string ge_pre = "ge.";
    for (const auto &[key, value] : cmdline.items()) {
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
    }
}

// 解析单个张量信息
TRITONSERVER_Error *ModelState::ParseTensorInfo(common::TritonJson::Value &tensor, ClientTensor &client_tensor,
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

TRITONSERVER_Error *ModelState::ParseModelConfig()
{
    common::TritonJson::Value inputs;
    common::TritonJson::Value outputs;

    RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
    RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));
    RETURN_IF_ERROR(model_config_.MemberAsString("name", &model_name_));

    RETURN_IF_ERROR(ParseInputTensors(inputs));
    RETURN_IF_ERROR(ParseOutputTensors(outputs));

    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("max_batch_size ") + std::to_string(MaxBatchSize())).c_str());

    return nullptr;
}

// 打印张量信息
void ModelState::PrintTensorInfo(const ClientTensor &tensor, const std::string &prefix)
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

void ModelState::PrintClientTensors(const std::vector<ClientTensor> &tensors, const std::string &tensor_type)
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

    for (const auto &[key, value] : parameters_) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("Key: ") + key + ", Value: " + value).c_str());
    }

    PrintClientTensors(input_client_tensor_, "Input");
    PrintClientTensors(output_client_tensor_, "Output");
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

// 查找ONNX模型文件
void ModelState::FindModelFile()
{
    model_file_ = FindFirstOnnxFile(std::string(runtime_modeldir_));
    if (model_file_ == "") {
        TRITONSERVER_Error *error =
            TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "can't find onnx file in model path!");
        TRITONSERVER_ErrorDelete(error);
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("find onnx path: ") + model_file_).c_str());
    }
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
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("backend configuration:\n") + buffer).c_str());

    TRITONBACKEND_ArtifactType artifact_type;
    THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_ModelRepository(triton_model, &artifact_type, &runtime_modeldir_));

    ParseGeConfig(std::string(buffer));
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
    PrintModelConfig();
}

// 检查是否为ONNX文件并返回路径
std::string ModelState::CheckAndReturnOnnxFile(const std::string &filePath)
{
    std::string onnx_extension = ".onnx";
    if (filePath.size() >= onnx_extension.size() &&
        filePath.substr(filePath.size() - onnx_extension.size()) == onnx_extension) {
        return filePath;  // 返回副本，安全
    }
    return "";  // 返回空字符串表示未找到
}

// 在目录中搜索ONNX文件
std::string ModelState::SearchOnnxFileInDirectory(const std::string &path)
{
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        try {
            if (entry.is_directory()) {
                std::string result = FindFirstOnnxFile(entry.path().string());
                if (!result.empty()) {
                    return result;
                }
            } else if (entry.is_regular_file()) {
                std::string result = CheckAndReturnOnnxFile(entry.path().string());
                if (!result.empty()) {
                    return result;
                }
            }
        } catch (const std::filesystem::filesystem_error &e) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ("访问文件时出错: " + std::string(e.what())).c_str());
            continue;
        }
    }
    return "";  // 返回空字符串表示未找到
}

// 递归查找第一个.onnx文件
std::string ModelState::FindFirstOnnxFile(const std::string &path)
{
    try {
        if (!std::filesystem::exists(path)) {
            return "";
        }

        if (!std::filesystem::is_directory(path)) {
            return CheckAndReturnOnnxFile(path);
        }

        return SearchOnnxFileInDirectory(path);
    } catch (const std::filesystem::filesystem_error &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ("文件系统错误: " + std::string(e.what())).c_str());
    }

    return "";
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
    return nullptr;  // success
}

ModelState::~ModelState()
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ModelState ~ start: ")).c_str());
}

}