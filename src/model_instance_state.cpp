/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */
#include "utils.h"
#include "model_instance_state.h"

#define TYPE_32_BYTE 4
#define TYPE_64_BYTE 8
#define TYPE_16_BYTE 2
#define TYPE_8_BYTE 1
#define BYTE_PTR (char *)

using namespace ge;
using namespace std;
using json = nlohmann::json;

namespace triton::backend::npu_ge {

// 读取环境变量的函数
std::string ModelInstanceState::GetEnvVar(const std::string &name)
{
    const char *value = std::getenv(name.c_str());
    return (value != nullptr) ? std::string(value) : "";
}

void ModelInstanceState::LoadGeConfig(std::string &env_value, std::map<ge::AscendString, ge::AscendString> &configMap)
{
    try {
        // 解析JSON
        json jsonData = json::parse(env_value);
        // 验证JSON类型
        if (!jsonData.is_object()) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("错误: JSON文件内容不是对象格式")).c_str());
            return;
        }

        // 遍历JSON对象
        for (auto &element : jsonData.items()) {
            std::string key = element.key();
            std::string value;

            // 根据值的类型进行适当转换
            if (element.value().is_string()) {
                value = element.value().get<std::string>();
            } else if (element.value().is_number()) {
                value = std::to_string(element.value().get<double>());
            } else if (element.value().is_boolean()) {
                value = element.value().get<bool>() ? "true" : "false";
            } else if (element.value().is_null()) {
                value = "null";
            } else {
                // 对于数组或对象等复杂类型，使用JSON字符串表示
                value = element.value().dump();
            }
            configMap[ge::AscendString(key.c_str())] = ge::AscendString(value.c_str());
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("key: ") + key + " value: " + value).c_str());
        }
        return;
    } catch (const json::parse_error &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("JSON解析错误: ") + e.what()).c_str());
        return;
    } catch (const std::exception &e) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("错误: ") + e.what()).c_str());
        return;
    }
}

void ModelInstanceState::FreeResources() {}

// 构建静态模式配置
void ModelInstanceState::StaticModeConfig(std::map<ge::AscendString, ge::AscendString> &parser_options)
{
    string ir_option;
    for (int i = 0; i < model_state_->GetInputClientTensors().size(); i++) {
        ir_option += model_state_->GetInputClientTensors()[i].name_;
        ir_option += ":";
        for (int j = 0; j < model_state_->GetInputClientTensors()[i].dims_.size() - 1; j++) {
            ir_option += std::to_string(model_state_->GetInputClientTensors()[i].dims_[j]);
            ir_option += ",";
        }
        ir_option += std::to_string(
            model_state_->GetInputClientTensors()[i].dims_[model_state_->GetInputClientTensors()[i].dims_.size() - 1]);
        ir_option += ";";
    }
    ir_option = ir_option.substr(0, ir_option.size() - 1);
    parser_options[ge::AscendString(ge::ir_option::INPUT_SHAPE)] = ge::AscendString(ir_option.c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("ir_option: ") + ir_option).c_str());
}

void ModelInstanceState::StaticModeConfigOne(std::map<ge::AscendString, ge::AscendString> &parser_options)
{
    string ir_option;
    for (size_t i = 0; i < model_state_->GetInputClientTensors().size(); i++) {
        ir_option += model_state_->GetInputClientTensors()[i].name_;
        ir_option += ":1,";
        for (size_t j = 0; j < model_state_->GetInputClientTensors()[i].dims_.size() - 1; j++) {
            ir_option += std::to_string(model_state_->GetInputClientTensors()[i].dims_[j]);
            ir_option += ",";
        }
        ir_option += std::to_string(
            model_state_->GetInputClientTensors()[i].dims_[model_state_->GetInputClientTensors()[i].dims_.size() - 1]);
        ir_option += ";";
    }
    ir_option = ir_option.substr(0, ir_option.size() - 1);
    parser_options[ge::AscendString(ge::ir_option::INPUT_SHAPE)] = ge::AscendString(ir_option.c_str());
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("ir_option: ") + ir_option).c_str());
}

// 初始化类型映射
void ModelInstanceState::InitTypeMappings()
{
    size_by_type_ = {{TRITONSERVER_TYPE_BOOL, TYPE_8_BYTE},    {TRITONSERVER_TYPE_UINT8, TYPE_8_BYTE},
                     {TRITONSERVER_TYPE_UINT16, TYPE_16_BYTE}, {TRITONSERVER_TYPE_UINT32, TYPE_32_BYTE},
                     {TRITONSERVER_TYPE_UINT64, TYPE_64_BYTE}, {TRITONSERVER_TYPE_INT8, TYPE_8_BYTE},
                     {TRITONSERVER_TYPE_INT16, TYPE_16_BYTE},  {TRITONSERVER_TYPE_INT32, TYPE_32_BYTE},
                     {TRITONSERVER_TYPE_INT64, TYPE_64_BYTE},  {TRITONSERVER_TYPE_FP16, TYPE_16_BYTE},
                     {TRITONSERVER_TYPE_FP32, TYPE_32_BYTE},   {TRITONSERVER_TYPE_FP64, TYPE_64_BYTE},
                     {TRITONSERVER_TYPE_BYTES, TYPE_8_BYTE},   {TRITONSERVER_TYPE_BF16, TYPE_16_BYTE}};

    ge_data_trans_ = {{TRITONSERVER_TYPE_BOOL, ge::DT_BOOL},     {TRITONSERVER_TYPE_UINT8, ge::DT_UINT8},
                      {TRITONSERVER_TYPE_UINT16, ge::DT_UINT16}, {TRITONSERVER_TYPE_UINT32, ge::DT_UINT32},
                      {TRITONSERVER_TYPE_UINT64, ge::DT_UINT64}, {TRITONSERVER_TYPE_INT8, ge::DT_INT8},
                      {TRITONSERVER_TYPE_INT16, ge::DT_INT16},   {TRITONSERVER_TYPE_INT32, ge::DT_INT32},
                      {TRITONSERVER_TYPE_INT64, ge::DT_INT64},   {TRITONSERVER_TYPE_FP16, ge::DT_FLOAT16},
                      {TRITONSERVER_TYPE_FP32, ge::DT_FLOAT},    {TRITONSERVER_TYPE_BYTES, ge::DT_STRING},
                      {TRITONSERVER_TYPE_BF16, ge::DT_BF16}};
}

// 初始化GE环境
int ModelInstanceState::InitGEEnvironment()
{
    aclError retInit = aclInit(nullptr);
    if (retInit != ACL_ERROR_NONE) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclInit failed, ret is: ") + std::to_string(retInit)).c_str());
        return RET_ERR;
    }

    std::map<ge::AscendString, ge::AscendString> ge_init_options;
    std::string ge_json(model_state_->GetRuntimeModelPath());
    ge_json += "/../ge.json";
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, ge_json.c_str());

    std::string env_value = GetEnvVar("GE_NPU_CONFIG");
    if (!env_value.empty()) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("GE_NPU_CONFIG = ") + env_value).c_str());
        LoadGeConfig(env_value, ge_init_options);
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, "No GE_NPU_CONFIG parameters");
        if (!model_state_->GetGeConfig().empty()) {
            for (auto &pair : model_state_->GetGeConfig()) {
                ge_init_options[ge::AscendString((pair.first).c_str())] = ge::AscendString((pair.second).c_str());
                LOG_MESSAGE(
                    TRITONSERVER_LOG_INFO,
                    (std::string("ge_init_options: ") + "key: " + pair.first + " value: " + pair.second).c_str());
            }
        }
    }

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                (std::string("ge_init_options.size(): ") + std::to_string(ge_init_options.size())).c_str());

    ge::Status ret = ge::GEInitialize(ge_init_options);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string(" GEInitialize failed.") + std::to_string(ret)).c_str());
        aclFinalize();
        return RET_ERR;
    }
    return RET_OK;
}

// 初始化设备
int ModelInstanceState::InitDevices(std::vector<int> &dev_ids_)
{
    uint32_t dev_count;
    aclError retInit = aclrtGetDeviceCount(&dev_count);
    if (retInit != ACL_ERROR_NONE) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclrtGetDeviceCount failed, ret is: ") + std::to_string(retInit)).c_str());
        return RET_ERR;
    }

    if (dev_ids_.size() == 0) {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("aclrtGetDeviceCount getDevice count: ") + std::to_string(dev_count)).c_str());
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, "user not set dev_ids ,use all npu card");
        for (int i = 0; i < dev_count; i++) {
            dev_ids_.push_back(i);
        }
    }

    if (dev_ids_.size() == 0) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "no npu card detected!");
        return RET_ERR;
    }
    return RET_OK;
}

// 初始化图会话
int ModelInstanceState::InitGraphSession(int dev_id, int graph_id, aclrtContext context, std::mutex &mu,
                                         ge::Session *session_)
{
    aclError retInit;
    ge::Status ret;
    ge::graphStatus graph_ret;
    aclrtStream stream_ = nullptr;

    aclrtSetCurrentContext(context);
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("graph_id: ") + std::to_string(graph_id) + std::string("start init")).c_str());

    ge::Graph graph1;
    std::map<ge::AscendString, ge::AscendString> parser_options;

    if (model_state_->GetGeStaticMode()) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "using ge static mode.");
        if (model_state_->MaxBatchSize() > 0) {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "generate one batch graph");
            StaticModeConfigOne(parser_options);
        } else {
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "generate fixed batch graph");
            StaticModeConfig(parser_options);
        }
    }

    {
        std::lock_guard<std::mutex> lock(mu);
        graph_ret = ge::aclgrphParseONNX(model_state_->GetModelPath().c_str(), parser_options, graph1);
        if (graph_ret != RET_OK) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("aclgrphParseONNX execute failed, ret is: ") + std::to_string(graph_ret)).c_str());
            return RET_ERR;
        }
    }

    ret = session_->AddGraph(graph_id, graph1);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("session_->AddGraph failed, ret is: ") + std::to_string(graph_ret)).c_str());
        return RET_ERR;
    }

    ret = session_->CompileGraph(graph_id);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("session_->CompileGraph failed, ret is: ") + std::to_string(ret)).c_str());
        return RET_ERR;
    }

    // 创建一个Stream
    aclError aclRet = aclrtCreateStream(&stream_);
    if (aclRet != ACL_SUCCESS) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclrtCreateStream execute failed, code is:") + std::to_string(aclRet)).c_str());
        return RET_ERR;
    }

    std::map<AscendString, AscendString> load_graph_options = {};
    ret = session_->LoadGraph(graph_id, load_graph_options, stream_);
    if (ret != RET_OK) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("session_->LoadGraph failed, ret is: ") + std::to_string(ret)).c_str());
        return RET_ERR;
    }

    model_state_->GetScheduler()->AddInstance(graph_id, context, stream_, session_, dev_id);
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("graph_id: ") + std::to_string(graph_id) + std::string("finished")).c_str());

    return RET_OK;
}

// 计算设备执行块数量
int ModelInstanceState::CalculateDeviceExecBlock(int device_count)
{
    int device_exec_block =
        (model_state_->GetTritonThreadCount() * model_state_->GetInstanceExecBlock() + device_count - 1) / device_count;
    if (model_state_->GetDeviceExecBlock() != -1) {
        device_exec_block = model_state_->GetDeviceExecBlock();
    }
    return device_exec_block;
}

// 创建设备线程
void ModelInstanceState::CreateDeviceThreads(int dev_id, int dev_index, int device_exec_block,
                                             std::vector<std::thread> &threads, std::mutex &mu)
{
    std::map<ge::AscendString, ge::AscendString> options;
    std::string str = std::to_string(dev_id);
    options[ge::AscendString("ge.session_device_id")] = ge::AscendString(str.c_str());
    options[ge::AscendString("ge.constLifecycle")] = ge::AscendString("session");

    if (model_state_->GetDumpData()) {
        options[ge::AscendString("ge.exec.enableDump")] = ge::AscendString("1");
        options[ge::AscendString("ge.exec.dumpPath")] = ge::AscendString("./dump_data");
        options[ge::AscendString("ge.exec.dumpMode")] = ge::AscendString("all");
    }

    aclrtContext context_ = nullptr;
    session_ = new Session(options);
    if (session_ == nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("Create session_ failed.")).c_str());
        return;
    }

    aclError retInit = aclrtSetDevice(dev_id);
    if (retInit != ACL_SUCCESS) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                    (std::string("aclrtSetDevice execute failed, code is: ") + std::to_string(retInit)).c_str());
        delete session_;
        return;
    }

    aclrtGetCurrentContext(&context_);

    for (int j = 0; j < device_exec_block; j++) {
        int graph_id = j + device_exec_block * dev_index;
        threads.emplace_back([this, &mu, graph_id, context_, dev_id]() {
            try {
                int result = InitGraphSession(dev_id, graph_id, context_, mu, session_);
                if (result == RET_ERR) {
                    init_failed_.store(true);
                    // 可以记录哪个线程失败
                    std::lock_guard<std::mutex> lock(exception_mutex_);
                    if (!init_exception_) {
                        init_exception_ = std::make_exception_ptr(
                            std::runtime_error("InitGraphSession failed for graph_id: " + std::to_string(graph_id)));
                    }
                }
            } catch (...) {
                init_failed_.store(true);
                std::lock_guard<std::mutex> lock(exception_mutex_);
                if (!init_exception_) {
                    init_exception_ = std::current_exception();
                }
            }
        });
    }
}

// 等待所有线程完成
void ModelInstanceState::JoinAllThreads(std::vector<std::thread> &threads)
{
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

int ModelInstanceState::Init()
{
    InitTypeMappings();

    // 使用静态 thread_local 变量，确保只初始化一次
    thread_id_ = next_id.fetch_add(1);
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("Thread initializing... : thread id ") + std::to_string(thread_id_)).c_str());

    if (thread_id_ == 0) {
        if (InitGEEnvironment() != RET_OK) {
            return RET_ERR;
        }

        std::vector<int> dev_ids_ = model_state_->GetDeviceIds();
        if (InitDevices(dev_ids_) != RET_OK) {
            return RET_ERR;
        }

        int device_exec_block_ = CalculateDeviceExecBlock(dev_ids_.size());
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
                    (std::string("final device exec block count: ") + std::to_string(device_exec_block_)).c_str());

        std::vector<std::thread> threads;
        std::mutex mu;

        for (int i = 0; i < dev_ids_.size(); i++) {
            CreateDeviceThreads(dev_ids_[i], i, device_exec_block_, threads, mu);
        }

        // 等待所有线程完成
        JoinAllThreads(threads);
        // 检查是否有初始化失败
        if (init_failed_.load()) {
            if (init_exception_) {
                try {
                    std::rethrow_exception(init_exception_);
                } catch (const std::exception &e) {
                    std::cerr << "Init failed with exception: " << e.what() << std::endl;
                }
            }
            return RET_ERR;
        }

        return RET_OK;
    }

    return RET_OK;
}

TRITONSERVER_Error *ModelInstanceState::Create(ModelState *model_state,
                                               TRITONBACKEND_ModelInstance *triton_model_instance,
                                               ModelInstanceState **state)
{
    try {
        *state = new ModelInstanceState(model_state, triton_model_instance);
    } catch (const BackendModelInstanceException &ex) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNKNOWN, "unexpected nullptr in BackendModelInstanceException");
    }
    int ret = (*state)->Init();
    if (ret != RET_OK) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNKNOWN, "init instance error");
    }
    return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
}

ModelInstanceState::~ModelInstanceState()
{
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("ModelInstanceState 释放开始 ")).c_str());
    FreeResources();
    next_id.fetch_sub(1);
    if (thread_id_ == 0) {
        while (1) {
            sleep(1);
            if (next_id.load() == 0) {
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "wait sub ==0 try finalize");
                ge::GEFinalize();
                delete session_;
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, " GEFinalize");
                aclFinalize();
                LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, " aclFinalize");
                break;
            }
        }
    }
}

int ModelInstanceState::GetValueByKey(size_t key) const
{
    auto it = size_by_type_.find(key);
    if (it != size_by_type_.end()) {
        return it->second;
    } else {
        std::cerr << "键 " << key << " 不存在" << std::endl;
        return RET_ERR;
    }
}

ge::DataType ModelInstanceState::TransToGeData(size_t key) const
{
    auto it = ge_data_trans_.find(key);
    if (it != ge_data_trans_.end()) {
        return it->second;
    } else {
        std::cerr << "键 " << key << " 不存在" << std::endl;
        return ge::DT_MAX;
    }
}

void ModelInstanceState::FreeDevResources(std::vector<void *> indev_buffer_, std::vector<void *> outdev_buffer_)
{
    for (size_t i = 0; i < indev_buffer_.size(); i++) {
        aclrtFree(indev_buffer_[i]);
    }
    indev_buffer_.clear();

    for (size_t i = 0; i < outdev_buffer_.size(); i++) {
        aclrtFree(outdev_buffer_[i]);
    }
    outdev_buffer_.clear();
}

void ModelInstanceState::FreeHostResources(std::vector<void *> inhost_buffer_, std::vector<void *> outhost_buffer_)
{
    for (size_t i = 0; i < inhost_buffer_.size(); i++) {
        aclrtFreeHost(inhost_buffer_[i]);
    }

    inhost_buffer_.clear();
    for (size_t i = 0; i < outhost_buffer_.size(); i++) {
        aclrtFreeHost(outhost_buffer_[i]);
    }
    outhost_buffer_.clear();
}

// 计算批次分配
void ModelInstanceState::CalculateBatchDistribution(int batch_total, size_t instance_count,
                                                    std::vector<int> &input_offset, std::vector<int> &batch_result)
{
    input_offset.resize(instance_count, 0);
    batch_result.resize(instance_count, 0);

    int base = batch_total / instance_count;
    int remainder = batch_total % instance_count;

    for (int i = 0; i < instance_count; ++i) {
        batch_result[i] = base;
        if (i < remainder) {
            batch_result[i] += 1;
        }
    }

    for (int i = 1; i < instance_count; ++i) {
        input_offset[i] = input_offset[i - 1] + batch_result[i - 1];
    }
}

// 提取输入信息
int ModelInstanceState::ExtractInputInfo(TRITONBACKEND_Request *request, std::vector<void *> &inhost_buffer_,
                                         std::vector<int> &inhost_line_size_, int &batch_total,
                                         std::vector<int> &input_offset, std::vector<int> &batch_result,
                                         size_t instance_count)
{
    aclError acl_ret;

    for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
        TRITONBACKEND_Input *input;
        TRITONBACKEND_RequestInputByIndex(request, j, &input);

        TRITONSERVER_DataType datatype;
        const int64_t *shape;
        uint32_t dims_count;
        const char *input_name;
        TRITONBACKEND_InputProperties(input, &input_name, &datatype, &shape, &dims_count, nullptr, nullptr);

        if (j == 0) {
            batch_total = shape[0];
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                        (std::string("batch_total is:") + std::to_string(batch_total)).c_str());

            CalculateBatchDistribution(batch_total, instance_count, input_offset, batch_result);
        }

        const void *buffer;
        uint64_t buffer_size;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        TRITONBACKEND_InputBuffer(input, 0, &buffer, &buffer_size, &memory_type, &memory_type_id);

        void *host_buffer = nullptr;
        acl_ret = aclrtMallocHost(&host_buffer, buffer_size);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc host buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        inhost_buffer_.push_back(host_buffer);
        std::copy(BYTE_PTR buffer, BYTE_PTR buffer + buffer_size, BYTE_PTR host_buffer);

        size_t sample_element_size = GetValueByKey(model_state_->GetInputClientTensors()[j].dtype_);
        int per_buffer_size = sample_element_size;

        for (size_t k = 1; k < dims_count; k++) {
            per_buffer_size *= shape[k];
        }
        inhost_line_size_.push_back(per_buffer_size);
    }
    return RET_OK;
}

// 准备输出缓冲区
int ModelInstanceState::PrepareOutputBuffers(std::vector<void *> &outhost_buffer_, std::vector<int64_t> &outsize,
                                             std::vector<int> &outhost_line_size_, int batch_total)
{
    aclError acl_ret;

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        size_t sample_element_size = GetValueByKey(model_state_->GetOutputClientTensors()[j].dtype_);
        int64_t total_out_size = static_cast<int64_t>(sample_element_size);
        vector<int64_t> dims_out = model_state_->GetOutputClientTensors()[j].dims_;

        if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL) {
            total_out_size *= batch_total;
        }

        for (auto dim : dims_out) {
            total_out_size *= dim;
            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("out dim: ") + to_string(dim)).c_str());
        }

        void *outhost_buffer = nullptr;
        acl_ret = aclrtMallocHost(&outhost_buffer, total_out_size);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        outhost_buffer_.push_back(outhost_buffer);
        outsize.push_back(total_out_size);

        int per_buffer_size = sample_element_size;
        for (size_t k = 0; k < dims_out.size(); k++) {
            per_buffer_size *= dims_out[k];
        }
        outhost_line_size_.push_back(per_buffer_size);
    }
    return RET_OK;
}

// 获取执行批次参数
void ModelInstanceState::GetExecutionBatchParams(int batch, int &exec_batch, int &cycle_count)
{
    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL) {
        // 动态batch，ge静态生成，固定batch=1
        if (model_state_->GetGeStaticMode()) {
            exec_batch = 1;
            cycle_count = batch;
        } else {
            exec_batch = batch;
            cycle_count = 1;
        }
    } else {
        exec_batch = batch;
        cycle_count = 1;
    }
}

// 准备输入张量
int ModelInstanceState::PrepareInputTensors(Scheduler::Instance *instance, std::vector<void *> &inhost_buffer_,
                                            std::vector<int> &inhost_line_size_, int exec_batch, int instance_index,
                                            std::vector<void *> &indev_buffer_, std::vector<gert::Tensor> &inputs)
{
    aclError acl_ret;
    inputs.resize(model_state_->GetInputCount());

    for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("current instance_index") + std::to_string(instance_index) +
                                               " : " + std::to_string(exec_batch))
                                                  .c_str());

        int buffer_size = exec_batch * inhost_line_size_[j];
        void *dev_buffer = nullptr;
        acl_ret = aclrtMalloc(&dev_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (acl_ret != ACL_ERROR_NONE) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc dev buffer failed, ret is:") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }
        indev_buffer_.push_back(dev_buffer);

        // 构建输入张量
        if (BuildInputTensor(j, exec_batch, dev_buffer, inputs[j]) != RET_OK) {
            return RET_ERR;
        }
    }
    return RET_OK;
}

// 构建输入张量
int ModelInstanceState::BuildInputTensor(size_t input_index, int exec_batch, void *dev_buffer, gert::Tensor &tensor)
{
    std::initializer_list<int64_t> origin_shape_list;
    std::initializer_list<int64_t> storage_shape_list;
    gert::StorageShape input_shape(origin_shape_list, storage_shape_list);
    gert::Shape &origin_shape = input_shape.MutableOriginShape();
    gert::Shape &storage_shape = input_shape.MutableStorageShape();

    vector<int64_t> dims_in = model_state_->GetInputClientTensors()[input_index].dims_;

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL) {
        origin_shape.AppendDim(exec_batch);
        storage_shape.AppendDim(exec_batch);
    }

    for (size_t i = 0; i < dims_in.size(); i++) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("input append dim is ") + std::to_string(dims_in[i])).c_str());
        origin_shape.AppendDim(dims_in[i]);
        storage_shape.AppendDim(dims_in[i]);
    }

    ge::DataType input_dtype = TransToGeData(model_state_->GetInputClientTensors()[input_index].dtype_);
    if (input_dtype == ge::DT_MAX) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("TransToGeData failed, ret is:") +
                                             std::to_string(model_state_->GetInputClientTensors()[input_index].dtype_))
                                                .c_str());
        return RET_ERR;
    }

    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("set input_dtype is ") + std::to_string(static_cast<int>(input_dtype))).c_str());

    gert::StorageFormat input_format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    tensor = {input_shape, input_format, gert::kOnDeviceHbm, input_dtype, dev_buffer};

    return RET_OK;
}

// 准备输出张量
int ModelInstanceState::PrepareOutputTensors(std::vector<int> &outhost_line_size_, int exec_batch,
                                             std::vector<void *> &outdev_buffer_, std::vector<gert::Tensor> &outputs)
{
    aclError acl_ret;
    outputs.resize(model_state_->GetOutputCount());

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        int total_out_size = exec_batch * outhost_line_size_[j];
        void *out_dev_buffer = nullptr;
        acl_ret = aclrtMalloc(&out_dev_buffer, total_out_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (acl_ret != ACL_ERROR_NONE) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("malloc host buffer failed, ret is:") + std::to_string(static_cast<int>(acl_ret)))
                            .c_str());
            return RET_ERR;
        }
        outdev_buffer_.push_back(out_dev_buffer);

        // 构建输出张量
        if (BuildOutputTensor(j, exec_batch, out_dev_buffer, outputs[j]) != RET_OK) {
            return RET_ERR;
        }
    }
    return RET_OK;
}

// 构建输出张量
int ModelInstanceState::BuildOutputTensor(size_t output_index, int exec_batch, void *out_dev_buffer,
                                          gert::Tensor &tensor)
{
    std::initializer_list<int64_t> origin_shape_list_out;
    std::initializer_list<int64_t> storage_shape_listt_out;
    gert::StorageShape output_shape(origin_shape_list_out, storage_shape_listt_out);

    gert::Shape &origin_shape_out = output_shape.MutableOriginShape();
    gert::Shape &storage_shape_out = output_shape.MutableStorageShape();

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL) {
        origin_shape_out.AppendDim(exec_batch);
        storage_shape_out.AppendDim(exec_batch);
    }

    vector<int64_t> dims_out = model_state_->GetOutputClientTensors()[output_index].dims_;
    for (auto dim : dims_out) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("output append dim is ") + std::to_string(dim)).c_str());
        origin_shape_out.AppendDim(dim);
        storage_shape_out.AppendDim(dim);
    }

    ge::DataType output_dtype = TransToGeData(model_state_->GetOutputClientTensors()[output_index].dtype_);
    if (output_dtype == ge::DT_MAX) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("TransToGeData failed, ret is:") + std::to_string(static_cast<int>(output_dtype))).c_str());
        return RET_ERR;
    }

    gert::StorageFormat output_format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    tensor = {output_shape, output_format, gert::kOnDeviceHbm, output_dtype, out_dev_buffer};

    return RET_OK;
}

// 执行推理循环
int ModelInstanceState::ExecuteInferenceCycle(Scheduler::Instance *instance, std::vector<void *> &inhost_buffer_,
                                              std::vector<int> &inhost_line_size_, std::vector<void *> &outhost_buffer_,
                                              std::vector<int> &outhost_line_size_, std::vector<gert::Tensor> &inputs,
                                              std::vector<gert::Tensor> &outputs, int exec_batch, int cycle_count,
                                              int instance_index, const std::vector<int> &input_offset)
{
    aclError acl_ret;
    ge::Status ret;

    for (int k = 0; k < cycle_count; k++) {
        // 复制输入数据到设备
        for (size_t j = 0; j < model_state_->GetInputCount(); j++) {
            int buffer_size = exec_batch * inhost_line_size_[j];
            void *src_ptr =
                (void *)(BYTE_PTR inhost_buffer_[j] + (input_offset[instance_index] + k) * inhost_line_size_[j]);

            acl_ret = aclrtMemcpy(inputs[j].GetAddr(), buffer_size, src_ptr, buffer_size, ACL_MEMCPY_HOST_TO_DEVICE);
            if (acl_ret != ACL_SUCCESS) {
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                            (std::string("aclrtMemcpy is failed, ret code is ") + std::to_string(acl_ret)).c_str());
                return RET_ERR;
            }
        }

        // 开始推理
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "start infer!: ");

        ret = instance->session->ExecuteGraphWithStreamAsync(instance->index, instance->stream, inputs, outputs);
        if (ret != RET_OK) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("execute model failed, ret is: ") + std::to_string(ret)).c_str());
            return RET_ERR;
        }

        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("aclrtSynchronizeStream: ")).c_str());
        acl_ret = aclrtSynchronizeStream(instance->stream);
        if (acl_ret != ACL_SUCCESS) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                (std::string("aclrtSynchronizeStream is failed, ret code is ") + std::to_string(acl_ret)).c_str());
            return RET_ERR;
        }

        // 复制输出数据到主机
        for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
            int total_out_size = exec_batch * outhost_line_size_[j];
            void *outdevBuffer = outputs[j].GetAddr();
            void *dst_ptr =
                (void *)(BYTE_PTR outhost_buffer_[j] + (input_offset[instance_index] + k) * outhost_line_size_[j]);

            aclrtMemcpy(dst_ptr, total_out_size, outdevBuffer, total_out_size, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }

    return RET_OK;
}

// 执行单个推理实例
int ModelInstanceState::ExecuteSingleInference(Scheduler::Instance *instance, int instance_index, int instance_count,
                                               std::vector<void *> &inhost_buffer_, std::vector<int> &inhost_line_size_,
                                               std::vector<void *> &outhost_buffer_,
                                               std::vector<int> &outhost_line_size_,
                                               const std::vector<int> &input_offset,
                                               const std::vector<int> &batch_result)
{
    int batch = batch_result[instance_index];
    if (batch == 0) {
        return RET_OK;  // 无需执行
    }

    int exec_batch;
    int cycle_count;
    GetExecutionBatchParams(batch, exec_batch, cycle_count);

    aclrtSetCurrentContext(instance->context);

    std::vector<void *> indev_buffer_;
    std::vector<void *> outdev_buffer_;
    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;

    // 准备输入张量
    if (PrepareInputTensors(instance, inhost_buffer_, inhost_line_size_, exec_batch, instance_index, indev_buffer_,
                            inputs) != RET_OK) {
        FreeDevResources(indev_buffer_, outdev_buffer_);
        return RET_ERR;
    }

    // 准备输出张量
    if (PrepareOutputTensors(outhost_line_size_, exec_batch, outdev_buffer_, outputs) != RET_OK) {
        FreeDevResources(indev_buffer_, outdev_buffer_);
        return RET_ERR;
    }

    // 执行推理循环
    if (ExecuteInferenceCycle(instance, inhost_buffer_, inhost_line_size_, outhost_buffer_, outhost_line_size_, inputs,
                              outputs, exec_batch, cycle_count, instance_index, input_offset) != RET_OK) {
        FreeDevResources(indev_buffer_, outdev_buffer_);
        return RET_ERR;
    }

    FreeDevResources(indev_buffer_, outdev_buffer_);
    return RET_OK;
}

// 处理单个推理实例的公共方法
int ModelInstanceState::ProcessSingleInstance(
    Scheduler::Instance *instance, int instance_index, int instance_count, std::vector<void *> &inhost_buffer_,
    std::vector<int> &inhost_line_size_, std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
    const std::vector<int> &input_offset, const std::vector<int> &batch_result, int &success_count, int &invalid_count)
{
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("instance_index: ") + std::to_string(instance_index) +
                                           " :batch: " + std::to_string(batch_result[instance_index]))
                                              .c_str());

    int batch = batch_result[instance_index];
    if (batch == 0) {
        invalid_count++;
        return RET_OK;  // 无效实例，但不算错误
    }

    int result = ExecuteSingleInference(instance, instance_index, instance_count, inhost_buffer_, inhost_line_size_,
                                        outhost_buffer_, outhost_line_size_, input_offset, batch_result);
    if (result == RET_OK) {
        success_count++;
    }

    return result;
}

// 并行执行推理
int ModelInstanceState::ExecuteInferenceParallel(
    std::vector<Scheduler::Instance *> instances, std::vector<void *> &inhost_buffer_,
    std::vector<int> &inhost_line_size_, std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
    const std::vector<int> &input_offset, const std::vector<int> &batch_result)
{
    if (instances.empty()) {
        return RET_OK;  // 没有实例，直接返回成功
    }

    std::vector<std::thread> threads;
    std::mutex mu;
    std::atomic<int> success_count{0};
    std::atomic<int> invalid_count{0};

    for (int instance_index = 0; instance_index < instances.size(); instance_index++) {
        threads.emplace_back([this, &mu, &success_count, &invalid_count, instance = instances[instance_index],
                              instance_index, count = instances.size(), &inhost_buffer_, &inhost_line_size_,
                              &outhost_buffer_, &outhost_line_size_, &input_offset, &batch_result]() {
            // 使用局部变量来收集结果，然后原子更新
            int local_success = 0;
            int local_invalid = 0;

            ProcessSingleInstance(instance, instance_index, count, inhost_buffer_, inhost_line_size_, outhost_buffer_,
                                  outhost_line_size_, input_offset, batch_result, local_success, local_invalid);

            // 原子更新计数
            if (local_invalid > 0) {
                invalid_count += local_invalid;
            }
            if (local_success > 0) {
                success_count += local_success;
            }
        });
    }

    JoinAllThreads(threads);
    model_state_->GetScheduler()->SetInstanceStatus(instances, Scheduler::Status::IDLE);

    return CheckInferenceResult(success_count, invalid_count, instances.size());
}

// 串行执行推理（不使用线程）
int ModelInstanceState::ExecuteInference(std::vector<Scheduler::Instance *> instances,
                                         std::vector<void *> &inhost_buffer_, std::vector<int> &inhost_line_size_,
                                         std::vector<void *> &outhost_buffer_, std::vector<int> &outhost_line_size_,
                                         const std::vector<int> &input_offset, const std::vector<int> &batch_result)
{
    if (instances.empty()) {
        return RET_OK;  // 没有实例，直接返回成功
    }

    int success_count = 0;
    int invalid_count = 0;

    for (int instance_index = 0; instance_index < instances.size(); instance_index++) {
        ProcessSingleInstance(instances[instance_index], instance_index, instances.size(), inhost_buffer_,
                              inhost_line_size_, outhost_buffer_, outhost_line_size_, input_offset, batch_result,
                              success_count, invalid_count);
    }

    model_state_->GetScheduler()->SetInstanceStatus(instances, Scheduler::Status::IDLE);

    return CheckInferenceResult(success_count, invalid_count, instances.size());
}

// 检查推理结果的公共方法
int ModelInstanceState::CheckInferenceResult(int success_count, int invalid_count, int total_instances)
{
    if (success_count != (total_instances - invalid_count) || invalid_count == total_instances) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "infer output failed!");
        return RET_ERR;
    }
    return RET_OK;
}

// 构建响应
int ModelInstanceState::BuildResponse(TRITONBACKEND_Request *request, std::vector<void *> &outhost_buffer_,
                                      std::vector<int64_t> &outsize, int batch_total)
{
    TRITONBACKEND_Response *response;
    TRITONBACKEND_ResponseNew(&response, request);

    for (size_t j = 0; j < model_state_->GetOutputCount(); j++) {
        if (BuildSingleOutput(response, j, outhost_buffer_[j], outsize[j], batch_total) != RET_OK) {
            TRITONBACKEND_ResponseDelete(response);
            return RET_ERR;
        }
    }

    TRITONSERVER_Error *error = TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
    if (error != nullptr) {
        TRITONBACKEND_ResponseDelete(response);
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ResponseSend failed, ")).c_str());
        return RET_ERR;
    }

    return RET_OK;
}

// 构建单个输出
int ModelInstanceState::BuildSingleOutput(TRITONBACKEND_Response *response, size_t output_index, void *outhost_buffer,
                                          int64_t buffer_size, int batch_total)
{
    string output_name = model_state_->GetOutputClientTensors()[output_index].name_;
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("current  output_name ") + output_name).c_str());

    TRITONBACKEND_Output *output = nullptr;
    uint32_t size = static_cast<uint32_t>(model_state_->GetOutputClientTensors()[output_index].dims_.size());

    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL) {
        size += 1;
    }

    std::unique_ptr<int64_t[]> shape_out = CreateOutputShape(output_index, size, batch_total);
    if (!shape_out) {
        return RET_ERR;
    }

    TRITONSERVER_Error *error = TRITONBACKEND_ResponseOutput(
        response, &output, model_state_->GetOutputClientTensors()[output_index].name_.c_str(),
        static_cast<TRITONSERVER_DataType>(model_state_->GetOutputClientTensors()[output_index].dtype_),
        shape_out.get(), size);

    if (error != nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("TRITONBACKEND_ResponseOutput failed, ")).c_str());
        return RET_ERR;
    }

    void *buffer_out;
    TRITONSERVER_MemoryType memory_type_out;
    int64_t memory_type_id_out;
    error = TRITONBACKEND_OutputBuffer(output, &buffer_out, buffer_size, &memory_type_out, &memory_type_id_out);
    if (error != nullptr) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_OutputBuffer failed, ")).c_str());
        return RET_ERR;
    }
    std::copy(BYTE_PTR outhost_buffer, BYTE_PTR outhost_buffer + buffer_size, BYTE_PTR buffer_out);
    return RET_OK;
}

// 创建输出形状数组
std::unique_ptr<int64_t[]> ModelInstanceState::CreateOutputShape(size_t output_index, uint32_t &size, int batch_total)
{
    std::unique_ptr<int64_t[]> shape_out = std::make_unique<int64_t[]>(size);
    if (model_state_->GetInferMode() == ModelState::InferMode::DYNAMICMODEL) {
        shape_out[0] = batch_total;
        for (size_t k = 0; k < model_state_->GetOutputClientTensors()[output_index].dims_.size(); k++) {
            shape_out[k + 1] = model_state_->GetOutputClientTensors()[output_index].dims_[k];
        }
    } else {
        for (size_t k = 0; k < size; k++) {
            shape_out[k] = model_state_->GetOutputClientTensors()[output_index].dims_[k];
        }
    }

    return shape_out;
}

// 处理单个请求
int ModelInstanceState::ProcessSingleRequest(TRITONBACKEND_Request *request,
                                             std::vector<Scheduler::Instance *> instances)
{
    vector<int64_t> outsize;
    std::vector<void *> outhost_buffer_;
    std::vector<void *> inhost_buffer_;
    std::vector<int> inhost_line_size_;
    std::vector<int> outhost_line_size_;
    std::vector<int> input_offset;
    std::vector<int> batch_result;

    int batch_total = 1;

    // 设置当前上下文
    aclrtContext nowContext;
    aclrtGetCurrentContext(&nowContext);
    if (nowContext != instances[0]->context) {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "process request setcurrentContext ");
        aclrtSetCurrentContext(instances[0]->context);
    }

    // 提取输入信息
    if (ExtractInputInfo(request, inhost_buffer_, inhost_line_size_, batch_total, input_offset, batch_result,
                         instances.size()) != RET_OK) {
        FreeHostResources(inhost_buffer_, outhost_buffer_);
        return RET_ERR;
    }

    // 准备输出缓冲区
    if (PrepareOutputBuffers(outhost_buffer_, outsize, outhost_line_size_, batch_total) != RET_OK) {
        FreeHostResources(inhost_buffer_, outhost_buffer_);
        return RET_ERR;
    }
    int executeRes = 0;
    if (instances.size() == 1) {
        executeRes = ExecuteInference(instances, inhost_buffer_, inhost_line_size_, outhost_buffer_, outhost_line_size_,
                                      input_offset, batch_result);
    } else {
        executeRes = ExecuteInferenceParallel(instances, inhost_buffer_, inhost_line_size_, outhost_buffer_,
                                              outhost_line_size_, input_offset, batch_result);
    }
    if (executeRes != RET_OK) {
        FreeHostResources(inhost_buffer_, outhost_buffer_);
        return RET_ERR;
    }

    // 构建响应
    if (BuildResponse(request, outhost_buffer_, outsize, batch_total) != RET_OK) {
        FreeHostResources(inhost_buffer_, outhost_buffer_);
        return RET_ERR;
    }

    FreeHostResources(inhost_buffer_, outhost_buffer_);
    return RET_OK;
}

int ModelInstanceState::ProcessRequests(TRITONBACKEND_Request **requests, const uint32_t request_count)
{
    for (size_t i = 0; i < request_count; i++) {
        std::vector<Scheduler::Instance *> instances = model_state_->GetScheduler()->GetIdleInstances(
            nullptr, model_state_->GetInstanceExecBlock(), model_state_->GetDeviceLB());
        if (ProcessSingleRequest(requests[i], instances) != RET_OK) {
            return RET_ERR;
        }
        LOG_IF_ERROR(TRITONBACKEND_RequestRelease(requests[i], TRITONSERVER_REQUEST_RELEASE_ALL),
                     "failed releasing request");
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("ProcessRequests end. ")).c_str());
    return RET_OK;
}

}